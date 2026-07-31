# How turbotwix is implemented

Why the reader is shaped the way it is, and where the speed comes from. For the on-disk
format itself see [`twix-format.md`](twix-format.md); for the API see the README.

The code is split by the order a file is read: `dtypes.py` (binary layout, the eval-info
mask), `header.py` (the text protocol), `data.py` (the line table, sample extraction, and
the object model). `__init__.py` just re-exports.

## The invariant, and the one requirement that is not one

**8-byte-aligned line offsets** are required rather than guessed, and a measurement that
breaks it raises `UnsupportedLayoutError`: every sample offset is
`line_offset + prefix + (c+1)*chan_hdr + c*8*ncol`, and every term but the line offset is
a multiple of 8. Keeping line offsets aligned is what lets extraction view the whole
mapped file as one `complex64` array and address samples in complex units — a fixed stride
within a line, no per-line bookkeeping.

**One `(ncha, ncol)`** is *not* required of a measurement, only of a read. Mixed shapes are
common and legitimate: an embedded parallel-imaging reference scan is Cartesian and short
where the imaging lines are spiral and long, and a coil-sensitivity adjustment stores
body-coil (`ncha=2`) and array-coil (`ncha=44`) lines together. Refusing the file over that
would refuse data that is perfectly readable line by line, so `LINE_DTYPE` carries each
line's own shape and the table stays whole. `LineTable.read` is where the requirement actually
bites, because its result is one `(n_lines, ncha, ncol)` array; it calls `common_shape`,
which raises `UnsupportedLayoutError` naming the shapes present. A selection
(`.image`, `.refscan`, `.noise`) is the fix, and is what the caller wanted anyway.

## 1. One mmap

The file is mapped once, read-only, with a best-effort `MADV_SEQUENTIAL` hint
(`open_mmap`). There are no per-line `seek()`+`read()` syscalls anywhere; every read path
is a view of that mapping.

## 2. Two line-table paths, one per regime

The stream carries no index and no line count — each line's length is computed from its own
header — so building the table looks inherently sequential. `build_table` splits it into
two regimes and tries the cheap one first.

**A single uniform run** (`_uniform_run`) is what Cartesian scans produce: identically
shaped lines ending in ACQEND, often millions of small ones. There the stride computed from
the first header holds throughout, so there is no per-line work at all — the count is a
division, the offsets an `arange`, and the flags and counters one strided `as_strided` read
of all N headers. About 0.09 ms on the bundled sample.

The hypothesis is verified exactly before being used, over every line, and testing it costs
one strided read of the headers a walk would have read anyway:

- same `(ncha, ncol)` everywhere;
- no ACQEND/SYNCDATA bit mid-run;
- `ScanCounter` advancing by a constant positive step.

That last check is the one a walk never needs. A walk is correct because it never guesses a
stride; this path guesses once, so it needs independent evidence that what it read at that
stride really were headers and not sample data. `ScanCounter` increments once per block, so
a constant positive step across the run provides it.

What is left over past `n * stride` must be exactly the ACQEND block — shorter than a line,
so it falls outside the division — or a further run of another shape is hiding there and
the path declines. Less than a header left means the file was cut short: still a uniform
run, just an unterminated one.

**Anything else** (`_walk`) — interleaved SYNCDATA/PMU blocks, several runs — is what
non-Cartesian sequences produce, and is walked block by block. Deliberately plain: one
Python iteration per *block*, which is affordable exactly where this path is needed, since
interleaved acquisitions are interleaved because their lines are large. A 25 GiB spiral
measurement is ~7.6k blocks of ~5 MiB, so the walk costs ~32 ms.

The fast path is tried first and declines when its hypothesis fails; the walk is the
general answer, not a penalty box.

## 3. 48-byte line rows

`LINE_DTYPE` keeps only what selection and extraction need: offset, flags, `ncol`, `ncha`,
and the 14 loop counters. The full 192-byte scan header — slice position and orientation,
ICE program parameters, timestamps, centre indices — is re-read on demand by
`LineTable.headers()`.

So the hot table is 48 bytes per line instead of 204. On a multi-million-line acquisition
that is tens of MB rather than a GB, and every boolean selection over it gets
proportionally cheaper (~4x).

`_varying_counters` takes the counter block as one contiguous `(n, 14)` uint16 array rather
than field by field, for the same reason: 14 passes strided across 48-byte rows cost twice
what one compacted pass does — 50 ms vs 25 ms on a million lines.

## 4. Strided copies, no index arrays, no intermediate buffer

`(n_lines, ncha, ncol)` *is* the file's own layout — one contiguous `(ncha, ncol)` block per
line, in file order — so reading is a strided copy rather than a gather.

An index-array gather would need an int64 index per complex64 sample: as much index traffic
as data traffic, where a strided view is a plain memcpy. Evenly spaced lines (the norm,
since a measurement is usually one uniform run) are a single `as_strided` view for the whole
selection and one `copyto`; an irregular selection is one view per line, each still at a
fixed within-line stride.

Nothing is batched, because nothing needs a temporary. That is what makes `out=` a genuine
mmap-to-mmap copy, with peak memory equal to the output — the mechanism behind reading a
selection out of a file far larger than RAM.

The one exception is un-reflecting: reversing the REFLECT-flagged lines is a fancy-index
assignment, so it needs a temporary the size of the lines it touches. `_FLIP_BUDGET_BYTES`
(2 MiB) bounds that temporary without bounding anything else.

## 5. Lazy protocol

A measurement's text header is ~800 KiB across six buffers, and regex-parsing all of it
costs ~40 ms — most of the time spent reading a small file, and pure waste for callers that
only want k-space. Splitting it into named buffers costs 0.04 ms.

So `Protocol` stores raw bytes and replaces each value with its parsed `AttrDict` on first
lookup. Every read path (`p["Meas"]`, `p.Meas`, `.get`, `.values`, `.items`, `dict(p)`)
parses on the way through, so the laziness is not observable apart from where the time is
spent. `__iter__` is defined rather than inherited on purpose: `dict(p)` and `{**p}` take a
fast path that reads the underlying table directly and would hand back raw bytes;
overriding `tp_iter` forces the generic path.

## 6. Typed protocol values

Types come from the value's own syntax — quoting, `0x`, a decimal point, a sign — and from
XProtocol's declared tags (`<ParamLong."x">`), which actually state the type.

Not from the key's Hungarian prefix, which is what the reference readers do. That guesses
(`b` → bool, `l` → long, …) and misfires whenever a sequence author named a variable
freely, returning `"10000"` as a string for an int field or `True` for a flag holding
`"0"`. Where the text says nothing, the text is the value.

The numeric match is an explicit regex rather than a bare `try: float(value)`, because
Python's `int()`/`float()` accept `_` as a digit separator: the scanner ID
`6_0_66327775_20210622_151635_650` would otherwise parse as `6.07e+26`.

## Grid folding

`read(dims=...)` is never implicit. `_varying_counters` gives the counters that are not constant
over a selection (min ≠ max — two reductions, not a sort), and is never empty: with nothing
varying there is no grid at all, and `("Lin",)` gives a degenerate size-1 axis instead of a
rank-0 one.

`minimal_dims` goes further and drops the counters the others already determine. Varying
counters never collide, but take the *product* of the counter ranges even when correlated:
an EPI `Seg` that merely tracks `Lin` parity doubles the grid and leaves half of it empty.
A counter can be dropped exactly when the remaining ones still identify every line — one
uniqueness test per candidate, tried largest-first, stopped as soon as the grid is
perfectly packed. (If the packed key would overflow int64, the reduction is skipped rather
than done with a wider one.)

Nothing is *lost* by dropping a determined counter: it is recoverable from what is kept, by
definition. What is lost is an axis to slice on — folding EPI on `Lin` alone leaves no way
to address the two readout polarities separately — which is why it is opt-in.

## Benchmark methodology

`scripts/bench_read.py` times reading the image data of a `.dat` file with turbotwix,
pymapvbvd and twixtools, each in its own subprocess so peak RSS is measured in isolation.

```
uv sync --group parity
python scripts/bench_read.py YOUR_FILE.dat --libs turbotwix pymapvbvd twixtools
```

Measured on a 1 GB Cartesian file, page-cache-resident, so the numbers reflect CPU overhead
rather than storage. The file was generated by a synthetic-`.dat` writer that has since been
removed — turbotwix reads, it does not write — so reproduce with a file of your own. The
turbotwix rows predate the uniform-run fast path and the removal of read batching, so they
are pessimistic:

| library | time (s) | peak RSS (MB) |
|---|---|---|
| turbotwix (lines) | **0.52** | 2071 |
| turbotwix (+ `read(dims=...)`) | 0.80 | 3087 |
| pymapvbvd | 3.13 | 602 |
| twixtools | 2.21 | 1181 |

Peak RSS is higher than the references mostly because it counts mmap-resident file pages —
read-only, file-backed, trivially evictable under pressure — on top of the output array.

The `read(dims=...)` row shows what the dense hypercube costs when you actually want it: 1 GB more
memory and 50% more time, on data that is *densely* sampled. For undersampled or
non-Cartesian data the gap widens with the sampling ratio.

What this benchmark does not exercise is the case the design is really for: a selection out
of a file far larger than RAM. On a synthetic 25 GiB interleaved spiral measurement
(5.04 MiB lines, a PMU block after every second one) the line table costs ~32 ms and one
shot ~4 ms, where the reference readers must assemble the whole array first.
