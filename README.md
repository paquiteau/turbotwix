# turbotwix

A from-scratch, opinionated, very fast reader for Siemens MRI raw data (`.dat` / TWIX)
files. Reads VB and VD/VE. It does not implement PMU decoding, slice-geometry parsing, or
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

Line offsets must be 8-byte aligned; blocks that carry no samples (ACQEND, PMU/SYNCDATA)
are skipped rather than returned. A measurement may mix `(ncha, ncol)` — an embedded
parallel-imaging reference scan is short and Cartesian where the imaging lines are long,
and a coil-sensitivity adjustment stores body-coil (`ncha=2`) and array-coil (`ncha=44`)
lines together — and the table holds those lines all the same. Only a *read* needs one
shape, since its result is one `(n_lines, ncha, ncol)` array, so reading such a
measurement means selecting first (`lines.image`, `lines.refscan`); asking for the mixed
set raises `UnsupportedLayoutError`.

## Usage

```python
import turbotwix as tw

f = tw.open_twix("meas.dat")
f  # TwixFile(..., measurements=['AdjCoilSens', 'bold_spiral...'])

lines = f.lines  # one strided read of the line headers
lines.image  # LineTable: imaging lines only
lines.noise  # noise-calibration lines, for pre-whitening
len(lines.image), lines.shape  # 4800, (44, 15000)

samples = f.read(lines.image)  # (4800, 44, 15000) complex64
```

`f.lines`, `f.hdr`, `f.read` and `f.to_dense` act on **the last measurement** (`f.scan`),
which is the scan itself — the measurements before it are the adjustments it needed. The
others are still there, by index or by iteration:

```python
len(f)  # 2
f[0].protocol_name  # 'AdjCoilSens'
noise = f[0].lines.noise  # a calibration measurement, explicitly
```

Selections are boolean queries and compose, so a partial read is just a smaller
selection — nothing else is touched on disk:

```python
img = f.lines.image
rep0 = img[img.counter("Rep") == 0]  # one volume's shots
vol = f.read(rep0)  # (40, 44, 15000), 201 MiB
shot = f.read(img[5:6])  # one shot, whatever the file's size

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
    f.read(sel, out=buf[: len(sel)])
    ...
```

The text protocol is parsed per buffer on first access, by attribute or by key:

```python
f.hdr.Meas.alTR[0]
f.scan.protocol_name, f.scan.patient_name
```

Closing is optional — the mapping is released on garbage collection, and nothing
turbotwix returns is a view of it, so nothing you hold can go stale. Use `close()` or the
context manager when you want the descriptor released at a definite point: looping over
many files, or on Windows, where an open mapping blocks renaming and deleting the `.dat`.

```python
with tw.open_twix("meas.dat") as f:
    samples = f.read(f.lines.image)
```

### Cartesian data

```python
dense = f.to_dense(dims=("Lin", "Par"))  # (Lin, Par, Cha, Col)

lines = f.lines.image  # or, with the read in your hands
dense = tw.to_dense(f.read(lines), lines, ("Lin", "Par"))
```

`to_dense` raises if several lines land on the same grid position — that normally means a
counter is missing from `dims`, not that the data wants averaging — and names the counters
responsible. `dims="minimal"` picks the axes for you: the counters that vary, minus those
the others already determine (`minimal_dims`). That packs the grid, at the cost of an axis
you may have wanted to slice on, so naming the dims yourself keeps both the rank and the
memory predictable.

## Correctness

`tests/test_parity.py` checks the extracted samples against pymapvbvd and twixtools:
line for line and sample for sample against twixtools' `mdb.data`, and `to_dense` against
pymapvbvd's k-space array. Install the references with `uv sync --group parity`; the tests
skip otherwise.

## Performance

Reading the image data of a 1 GB page-cache-resident Cartesian file: **0.52 s**, against
3.13 s for pymapvbvd and 2.21 s for twixtools. The case the design is really for is the one
a benchmark like that does not show — a selection out of a file far larger than RAM. On a
25 GiB interleaved spiral measurement the line table costs ~32 ms and one shot ~4 ms, where
the reference readers must assemble the whole array first.

The numbers, the methodology and where the speed comes from are in
[`docs/implementation.md`](docs/implementation.md).

## Known limitations

- A measurement whose line offsets are not 8-byte aligned raises `UnsupportedLayoutError`.
- Lines of differing `(ncha, ncol)` are tabled together but cannot be read in one call;
  select a single-shaped subset.
- PMU/SYNCDATA blocks are skipped, not decoded, so they do not appear in the line table.
- No ramp-sampling regridding or slice-geometry parsing.
- No oversampling removal: it is signal processing (an FFT round-trip), and along a
  non-Cartesian readout it is not meaningful.

## Documentation

- [`docs/twix-format.md`](docs/twix-format.md) — what a `.dat` file contains, byte by byte.
- [`docs/implementation.md`](docs/implementation.md) — how this reader works and why.

## Development

```
uv sync                    # numpy + dev tools only
uv run pytest
uv run ruff check .
uv run ty check src
uv sync --group parity     # to also run tests/test_parity.py against real references
```
