# turbotwix

A from-scratch, minimal-dependency (numpy only) reader for Siemens MRI raw data
(`.dat` / TWIX) files, built for **non-Cartesian acquisitions** and for files far larger
than RAM.

Reads VB and VD/VE files. It does not implement PMU decoding, slice-geometry parsing, or
ramp-sampling regridding.

**One shape per measurement.** Every ADC line in a measurement must report the same
`(ncha, ncol)`; blocks that carry no samples (ACQEND, PMU/SYNCDATA) are skipped rather
than returned. Measurements that genuinely mix shapes — a coil-sensitivity adjustment
storing body-coil (`ncha=2`) and array-coil (`ncha=44`) images together — raise
`UnsupportedLayoutError`; use pymapvbvd or twixtools for those.

## The data model

A TWIX measurement is a **list of acquisition lines**, each with metadata and a
`(ncha, ncol)` block of samples. turbotwix hands you exactly that: a queryable line
table, and reads that return `(n_lines, ncha, ncol)`.

It does *not* return a dense array indexed by the 14 loop counters, which is what
pymapvbvd and twixtools do. For a spiral or radial acquisition those counters index
shots, interleaves or spokes — there is no k-space grid to fold onto — and building one
anyway costs memory proportional to the *nominal* matrix rather than the acquired data,
forces a policy on duplicate indices, and makes partial reads impossible. Folding onto a
grid is available when you want it (`to_dense`), never implicit.

## Usage

```python
import turbotwix as tw

f = tw.open_twix("meas.dat")
f  # TwixFile(..., measurements=['AdjCoilSens', 'bold_spiral...'])
m = f[-1]  # every measurement is returned; picking one is your call

lines = m.lines  # one strided read of the line headers
lines.image  # LineTable: imaging lines only
lines.noise  # noise-calibration lines, for pre-whitening
len(lines.image), lines.shape  # 4800, (44, 15000)

samples = m.read(lines.image)  # (4800, 44, 15000) complex64
```

Selections are boolean queries and compose, so a partial read is just a smaller
selection — nothing else is touched on disk:

```python
img = m.lines.image
rep0 = img[img.counter("Rep") == 0]  # one volume's shots
vol = m.read(rep0)  # (40, 44, 15000), 201 MiB
shot = m.read(img[5:6])  # one shot, whatever the file's size

lines.has(tw.Flag.REFLECT)  # any of the 64 eval-info bits, by name
lines.image.headers()  # full scan headers on demand: SliceData,
# IceProgramPara, timestamps, centre indices
```

Read into your own buffer to bound memory on files that do not fit in RAM — the copy goes
straight from the mapped file into it, with no intermediate buffer:

```python
buf = np.empty((len(rep0), 44, 15000), dtype=np.complex64)
for r in np.unique(img.counter("Rep")):
    sel = img[img.counter("Rep") == r]
    m.read(sel, out=buf[: len(sel)])
    ...
```

Every line in a measurement has the same `(ncha, ncol)`, so `read` never has to guess what
to do with a mixed selection, and a selection out of an interleaved acquisition (spiral
shots with PMU blocks between them) is one strided view per line rather than a gather.

### Cartesian data

```python
lines = m.lines.image
dense = tw.to_dense(m.read(lines), lines, ("Lin", "Par"))  # (Lin, Par, Cha, Col)
```

`to_dense` raises if several lines land on the same grid position — that normally means
a counter is missing from `dims`, not that the data wants averaging — and names the
counters responsible. Pass `reduce="mean"`, `"sum"` or `"last"` to opt in.
`remove_oversampling(samples)` is a separate function, not a read flag: it is signal
processing (an FFT round-trip), and along a non-Cartesian readout it is not meaningful.

## How it goes fast

1. **One mmap**, no per-line `seek()`+`read()` syscalls.
2. **Two line-table paths, one per regime.** The stream is self-describing, so building
   the table looks inherently sequential. For a *single uniform run* — what Cartesian
   scans produce, often millions of small lines — it is not: the stride from the first
   header holds throughout, so the count is a division, the offsets an `arange`, and the
   flags and counters one strided read of all N headers (0.09 ms on the bundled sample).
   The hypothesis is verified exactly — same `(ncha, ncol)` everywhere, no
   ACQEND/SYNCDATA mid-run, `ScanCounter` advancing by a constant step, which is what
   proves those N headers are headers and not sample data read at a wrong stride — and
   simply declines otherwise. Everything else (interleaved PMU blocks, several runs) is
   walked block by block, which is affordable precisely where it is needed: interleaved
   acquisitions have big lines, so a 25 GiB spiral measurement is ~7.6k blocks and ~32 ms.
3. **Strided copies, no index arrays, no intermediate buffer.** A selection of evenly
   spaced lines is one strided view of the mapped file and one `copyto`; an irregular
   selection is one view per line. Nothing is batched, because nothing needs a
   temporary — so `out=` gives a mmap-to-mmap copy with peak memory equal to the output.
4. **48-byte line rows.** The table keeps offset, flags, shape and counters; the full
   192-byte header is re-read on demand. Selections over the table are ~4x cheaper and
   a multi-million-line table stays in the tens of MB.
5. **Lazy protocol.** ~800 KiB of header text across six buffers costs ~40 ms to
   regex-parse. Splitting it costs 0.04 ms; each buffer is parsed on first access.
6. **Typed protocol values.** Types come from the value's own syntax (quoting, `0x`, a
   decimal point) and from XProtocol's declared tags, not from guessing at the key's
   Hungarian prefix — which returns `"10000"` as a string and `True` for a flag holding
   `"0"`.

## Correctness

`tests/test_parity.py` checks the extracted samples against pymapvbvd and twixtools:
line for line and sample for sample against twixtools' `mdb.data`, and `to_dense` against
pymapvbvd's k-space array (including its `remove_os` path). Install the references with
`uv sync --group parity`; the tests skip otherwise.

## Benchmarks

`benchmarks/bench_read.py` times reading the image data of a `.dat` file with turbotwix,
pymapvbvd and twixtools, each in its own subprocess for isolated peak-RSS measurement.

```
uv sync --group parity
python benchmarks/bench_read.py YOUR_FILE.dat --libs turbotwix turbotwix-lines pymapvbvd twixtools
```

Measured on a 1 GB Cartesian file, page-cache-resident (so this reflects CPU overhead,
not storage). It was generated by a synthetic-`.dat` writer that has since been removed —
turbotwix reads, it does not write — so reproduce with a file of your own. The turbotwix
rows predate the uniform-run fast path and the removal of read batching, so they are
pessimistic:

| library | time (s) | peak RSS (MB) |
|---|---|---|
| turbotwix (lines) | **0.52** | 2071 |
| turbotwix (+ `to_dense`) | 0.80 | 3087 |
| pymapvbvd | 3.13 | 602 |
| twixtools | 2.21 | 1181 |

Peak RSS is higher than the references mostly because it counts mmap-resident file pages
(read-only, file-backed, trivially evictable under pressure) on top of the output array.
The `to_dense` row shows what the dense hypercube costs when you actually want it: 1 GB
more memory and 50% more time, on data that is *densely* sampled — for undersampled or
non-Cartesian data the gap widens with the sampling ratio. What this benchmark does not
exercise is the case the design is really for: a selection out of a file far larger than
RAM. On a synthetic 25 GiB interleaved spiral measurement (5.04 MiB lines, a PMU block
after every second one) the line table costs ~32 ms and one shot ~4 ms, where the
reference readers must assemble the whole array first.

## Known limitations

- A measurement whose ADC lines change `(ncha, ncol)` raises `UnsupportedLayoutError`;
  so does one whose line offsets are not 8-byte aligned.
- PMU/SYNCDATA blocks are skipped, not decoded, so they do not appear in the line table.
- No ramp-sampling regridding or slice-geometry parsing.
- The line table is rebuilt on every `open_twix`; there is no on-disk index cache
  (building it costs one strided read of the headers, so it has not been worth one).

## Development

```
uv sync                    # numpy + dev tools only
uv run pytest
uv run ruff check .
uv run ty check src
uv sync --group parity     # to also run tests/test_parity.py against real references
```
