# turbotwix

A from-scratch, minimal-dependency (numpy only) reader for Siemens MRI raw data
(`.dat` / TWIX) files, built to read very large files (tens of GB) as fast as the
underlying storage allows.

Functional scope matches [pymapvbvd](https://github.com/wtclarke/pymapVBVD): VB and
VD/VE files, the standard scan categories (`image`, `noise`, `refscan`, `refscanPC`,
`phasecor`, `phasestab` (+ `ref0`/`ref1` variants), `rtfeedback`, `vop`), and the
`remove_os` / `squeeze` read flags. It does not implement PMU or geometry parsing
(neither does pymapvbvd) or `regrid` (ramp-sampling regridding) yet.

## Usage

```python
import turbotwix as tw

scan = tw.read_twix("meas.dat")
print(scan.category_names())  # e.g. ['image', 'phasecor', 'rtfeedback']
print(scan.hdr.MeasYaps["alTR"])  # parsed ascconv/XProtocol header

# default order: (Lin, Par, Sli, Ave, Phs, Eco, Rep, Set, Seg, Ida, Idb, Idc, Idd, Ide, Cha, Col)
image = scan.image[:]

# pymapvbvd-compatible (Col, Cha, Lin, ...) axis order instead (costs one extra
# full-array transpose+copy -- see "Axis order" below):
scan = tw.read_twix("meas.dat", legacy_layout=True)
image = scan.image[
    :
]  # (Col, Cha, Lin, Par, Sli, Ave, Phs, Eco, Rep, Set, Seg, Ida, Idb, Idc, Idd, Ide)
```

### Axis order

The default array order is `(Lin, Par, Sli, Ave, Phs, Eco, Rep, Set, Seg, Ida, Idb,
Idc, Idd, Ide, Cha, Col)` — loop counters outermost, `Cha`/`Col` innermost. This is
**not** pymapvbvd's order. It's deliberately the same *shape* of convention twixtools
itself uses internally (`_dim_order` in `map_twix.py`, loop-counters first, `Cha`/`Col`
last): scattering per-line data into the *outermost* axis of a numpy array is
dramatically faster (measured ~40x for this workload) than scattering into the
innermost one, and pymapvbvd's `(Col, Cha, ...)` convention forces exactly that slow
case. Pass `legacy_layout=True` (on `read_twix` or per `TwixArray`) to get pymapvbvd's
axis order anyway, paying one explicit full-array transpose+copy for it — useful for
drop-in comparison, not recommended for the hot path on large files.

## Why this exists

`twixtools` and `pymapvbvd` both bottleneck on a per-line Python loop
(`seek()`+`read()`/`fromfile()` once per ADC line) when assembling the final k-space
array; `twixtools` additionally reads every sample byte twice (once discarded while
building its line list, again on access). turbotwix instead:

1. Memory-maps the whole file once (`numpy.memmap`) instead of repeated syscalls.
2. Walks line headers with a cheap structured-dtype view (no per-line Python objects,
   no manual bit-twiddling of a raw byte blob).
3. Classifies every line into its scan category with one vectorized bitwise pass over
   all headers at once.
4. Extracts sample data via a **batched vectorized gather** (one `numpy` fancy-index
   read per memory-bounded batch of lines) instead of a per-line read, and
   **scatter-writes** into the output array using vectorized advanced indexing instead
   of a per-line assembly loop.

All three example categories on the two bundled sample files reproduce pymapvbvd's
output **bit-exactly** (with `legacy_layout=True`), including the `remove_os` FFT path;
the default axis order is separately verified bit-exact against twixtools' own native
(untransposed) output too (see `tests/test_parity.py`).

## Benchmarks

`benchmarks/make_synthetic_dat.py` generates a structurally-valid synthetic `.dat` at
any size (no proprietary scan data needed); `benchmarks/bench_read.py` times reading
the full `image` array with turbotwix, pymapvbvd, and twixtools, each in its own
subprocess for isolated peak-RSS measurement:

```
uv sync --group parity   # installs pymapvbvd + twixtools for comparison only
python benchmarks/make_synthetic_dat.py /tmp/test.dat --size 1GB
python benchmarks/bench_read.py /tmp/test.dat
```

Indicative results on this machine (files fully page-cache-resident, so this measures
algorithmic/CPU overhead rather than raw disk throughput), using the default axis order:

| file size | turbotwix | pymapvbvd | twixtools |
|---|---|---|---|
| 100 MB | **0.13 s** / 238 MB | 0.81 s / 199 MB | 0.59 s / 240 MB |
| 2 GB | **1.42 s** / 4128 MB | 6.00 s / 1123 MB | 4.10 s / 2255 MB |

turbotwix wins clearly at both sizes. An earlier version of this benchmark (with
turbotwix defaulting to pymapvbvd's `(Col, Cha, ...)` axis order) showed turbotwix only
tying pymapvbvd at 2 GB — investigating that gap is *why* the axis order changed: both
pymapvbvd and that earlier version pay for scattering into the (slow) innermost axis
implied by `(Col, Cha, ...)`; twixtools avoids it by construction (see "Axis order"
above), which is what let it win despite doing a completely unvectorized per-line
Python loop. Switching turbotwix's default to the same *kind* of axis order twixtools
uses removed that cost entirely rather than trying to make the transpose itself faster.
(Storing the `(Col, Cha, ...)` array in Fortran/column-major order instead of
transposing was also tested as a middle ground — it does help, about 2x on the same
micro-benchmark — but numpy's advanced-indexing assignment still has a real special
case for indexing an array's *leading* axis specifically, regardless of memory order,
so it's not competitive with just choosing the leading axis to be the scattered one.)

**Peak memory** is still higher than the references (~4.1 GB vs 1.1–2.3 GB at 2 GB),
for three additive reasons: (1) the output accumulator itself, ~2 GB — unavoidable,
it's the size of the returned array; (2) per-batch gather/index buffers, controlled by
a `batch_bytes` budget; (3) mmap-resident file pages picked up while gathering, roughly
the size of the file. (2) turned out to be a real, free win: profiling showed the
initial 512 MB default batch budget bought *no* extra speed over a ~1–2 MB budget once
the axis-order fix landed (both plateau at the same time), so shrinking the default
batch budget to 2 MB cut peak RSS by about 1 GB *and* got faster (less allocator
churn) — that's what produced the 4.1 GB/1.42 s numbers above, down from an earlier
5.1 GB/2.48 s at the old 512 MB default. (3) is harder to fix directly and is partly a
measurement artifact of this machine having abundant free RAM: mmap'd, read-only,
file-backed pages are trivially evictable and the kernel will reclaim them under real
memory pressure without our involvement, so on a genuinely memory-constrained machine
reading a real 50 GB file, resident memory should stay closer to accumulator size +
working set rather than growing with file size the way it appears to here. Proactively
hinting eviction (`madvise(MADV_DONTNEED)` on already-consumed byte ranges) could pull
that number down further but wasn't implemented — it's the next thing worth trying if
real-file memory pressure turns out to matter in practice.

The real-world win this architecture is ultimately aimed at — avoiding thousands of
small `seek()+read()` syscalls on genuinely disk-bound 50 GB files rather than
page-cache-hot ones — is also not directly exercised by this in-memory benchmark and
should be validated on real large files.

## Known limitations (v1)

- No `regrid` (ramp-sampling) support yet.
- No PMU or slice-geometry parsing (out of scope, matching pymapvbvd).
- Assumes homogeneous (NCol, NCha) within a category for the fast path; heterogeneous
  shapes (e.g. partial Fourier mixed with full lines) fall back to per-shape bucketing,
  which is less tested.
- Indexing (`array[...]`) always reads the whole category first, then slices in
  memory — there is no partial/streaming read path.

## Development

```
uv sync                    # numpy + dev tools only
uv run pytest
uv run ruff check .
uv run ty check src
uv sync --group parity     # to also run tests/test_parity.py against real references
```
