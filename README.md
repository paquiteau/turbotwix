# turbotwix

A from-scratch, minimal-dependency (numpy only) reader for Siemens MRI raw data
(`.dat` / TWIX) files, built for **non-Cartesian acquisitions** and for files far larger
than RAM.

Reads VB and VD/VE files. It does not implement PMU decoding, slice-geometry parsing, or
ramp-sampling regridding.

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

lines = m.lines  # one pass over the line headers
lines.image  # LineTable: imaging lines only
lines.noise  # noise-calibration lines, for pre-whitening
len(lines.image), lines.image.shapes  # 4800, {(44, 15000)}

samples = m.read(lines.image)  # (4800, 44, 15000) complex64
```

Selections are boolean queries and compose, so a partial read is just a smaller
selection — nothing else is touched on disk:

```python
img = m.lines.image
rep0 = img[img.counter("Rep") == 0]  # one volume's shots
vol = m.read(rep0)  # (40, 44, 15000), 201 MiB, ~260 ms
shot = m.read(img[5:6])  # one shot out of 23.6 GiB, ~2 ms

lines.has(tw.Flag.REFLECT)  # any of the 64 eval-info bits, by name
lines.image.headers()  # full scan headers on demand: SliceData,
# IceProgramPara, timestamps, centre indices
```

Read into your own buffer to bound memory on files that do not fit in RAM — the data is
filled one batch at a time:

```python
buf = np.empty((len(rep0), 44, 15000), dtype=np.complex64)
for r in np.unique(img.counter("Rep")):
    sel = img[img.counter("Rep") == r]
    m.read(sel, out=buf[: len(sel)])
    ...
```

Some measurements hold several acquisitions that the flags do not separate — a coil
sensitivity adjustment stores body-coil (`ncha=2`) and array-coil (`ncha=44`) images
both as plain ONLINE image lines. Reading a mixed selection raises rather than
zero-padding them together; `by_shape()` splits them:

```python
for (ncha, ncol), sel in m.lines.image.by_shape().items():
    data = m.read(sel)
```

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
2. **Run-vectorized header walk.** The stream is self-describing, so it looks
   inherently sequential — but acquisitions come in runs of identically-shaped lines. The
   walk hypothesizes a stride from one header and verifies vectorized how far it holds,
   copying whole runs out of a strided view. Verification is exactly the condition the
   sequential walk tests, so the result is identical by construction. Windows grow
   geometrically and are capped by a byte budget: a wrong hypothesis costs O(window),
   never O(file). On interleaved sequences (a spiral file alternating 5 MiB image lines
   with 2 KiB PMU blocks, median run length 1) probing cannot pay, so the walk notices
   and stands down to line-at-a-time, re-arming periodically.
3. **Strided copies, no index arrays.** A batch of evenly spaced lines is one strided
   view and a memcpy; an irregularly spaced batch (the norm when line kinds interleave)
   is one view per line. The index-array gather — an int64 index per complex64 sample,
   as much index traffic as data — is only the fallback for small irregular batches.
4. **48-byte line rows.** The walk keeps offset, flags, shape and counters; the full
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

`benchmarks/make_synthetic_dat.py` generates a structurally-valid synthetic `.dat` at any
size; `benchmarks/bench_read.py` times reading the image data with turbotwix, pymapvbvd
and twixtools, each in its own subprocess for isolated peak-RSS measurement.

```
uv sync --group parity
python benchmarks/make_synthetic_dat.py /tmp/test.dat --size 1GB
python benchmarks/bench_read.py /tmp/test.dat --libs turbotwix turbotwix-lines pymapvbvd twixtools
```

A 1 GB synthetic file, page-cache-resident (so this measures CPU overhead, not storage):

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
non-Cartesian data the gap widens with the sampling ratio. What this synthetic benchmark
does not exercise is the case the design is really for: on the real 23.6 GiB spiral file,
one shot reads in ~2 ms and one repetition (201 MiB) in ~260 ms, where the reference
readers must assemble the whole 23.6 GiB array first.

## Known limitations

- No ramp-sampling regridding, PMU decoding or slice-geometry parsing.
- Reading a selection that mixes `(ncha, ncol)` shapes raises; use `by_shape()`.
- The line table is rebuilt on every `open_twix`; there is no on-disk index cache
  (walking a 23.6 GiB measurement takes ~55 ms, so it has not been worth one).

## Development

```
uv sync                    # numpy + dev tools only
uv run pytest
uv run ruff check .
uv run ty check src
uv sync --group parity     # to also run tests/test_parity.py against real references
```
