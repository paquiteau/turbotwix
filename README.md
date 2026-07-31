# turbotwix

A fast, and opiniated, Siemens TWIX (`.dat`, VB and VD/VE) reader. 

- 6x faster than pymapvbvd on a full Cartesian read, and able to pull a single shot out of a
24 GiB file in a couple of milliseconds without ever loading the rest.
- Use modern python (type annotations) and numpy structured dtype array.
- NB: It does not implement slice-geometry parsing or ramp-sampling regridding.


### Disclosure
This library is not affiliated with Siemens, and the TWIX format is not documented by them. The format has been reverse-engineered from the files and the existing implementations: 

- [pymapvbvd](https://github.com/pehses/mapVBVD)
- [twixtools](https://github.com/pehses/twixtools),  

Without them, turbotwix would not exists. 

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

`f.lines`, `f.hdr` and `f.read` act on **the last measurement** (`f.scan`),
which is the scan itself, the  measurements before it are usually calibration data you don't necessary need. They are still there, by index or by iteration:

```python
len(f)  # 2
f[0].protocol_name  # 'AdjCoilSens'
noise = f[0].lines.noise  # a calibration measurement, explicitly
```

Selections are boolean queries and compose, so a partial read is just a smaller
selection.  Nothing else is touched on disk:

```python
img = f.lines.image
rep0 = img[img.counter("Rep") == 0]  # one volume's shots
vol = f.read(rep0)  # (40, 44, 15000), 201 MiB
shot = f.read(img[5:6])  # one shot, whatever the file's size

lines.has(tw.Flag.REFLECT)  # any of the 64 eval-info bits, by name
lines.image.headers()  # full scan headers on demand: SliceData,
# IceProgramPara, timestamps, centre indices
```

Read into your own buffer to bound memory on files that do not fit in RAM. The copy goes straight from the mapped file into it, with no intermediate buffer:

```python
buf = np.empty((len(rep0), 44, 15000), dtype=np.complex64)
for r in np.unique(img.counter("Rep")):
    sel = img[img.counter("Rep") == r]
    f.read(sel, out=buf[: len(sel)])
    ...
```

The header/text protocol is parsed per buffer on first access, by attribute or by key:

```python
f.hdr.Meas.alTR[0]
f.scan.protocol_name, f.scan.patient_name
```

A tuple indexes as a nested path in one step, `.get` is path-aware too, and
`search_header_for_val` finds a key wherever it sits in the tree (an exact, structural
walk -- not pymapvbvd's regex/substring scan over a flattened namespace):

```python
f.hdr.Phoenix["sSliceArray", "asSlice", 0, "dThickness"]
f.hdr.Phoenix.get(("sKSpace", "lPartitions"), default=1)
tw.search_header_for_val(f.hdr.Phoenix, "sFastImaging", "lTurboFactor")
```

Grid-size counters (`NLin`, `NPar`, ...) are available on any line selection, computed
and cached on first access:

```python
f.lines.image.NLin, f.lines.image.NCha  # 44, 4
```

You can also use a context manager if you want:

```python
with tw.open_twix("meas.dat") as f:
    samples = f.read(f.lines.image)
```

### The data model

A TWIX measurement is a **list of acquisition lines**, each with metadata and a
`(ncha, ncol)` block of samples. `turbotwix` hands you exactly that: a queryable line table, and reads that return `(ncha, n_lines, ncol)` — channel first, matching its on-disk order, contiguous.

You can query and filter the lines, and when you want it get a numpy array of the data you need

### PMU

Physiological (ECG/pulse/respiration) data interleaved with the acquisition, if any:

```python
pmu = f.scan.pmu  # empty if this measurement carries no PMU data
pmu.signal["ECG1"]  # normalized waveform
pmu.trigger["PULS"]  # matching boolean trigger channel
pmu.timestamp["ECG1"]  # per-sample clock, 2.5 ms ticks since midnight
```

### Cartesian data

```python
dense = f.read(dims=("Lin", "Par"))  # (Lin, Par, Cha, Col)
```

`read(dims=...)` raises if several lines land on the same grid position — that normally
means a counter is missing from `dims`, not that the data wants averaging — and names the
counters responsible. `dims="minimal"` picks the axes for you: the counters that vary,
minus those the others already determine.

## Correctness

turbotwix is checked against two existing readers:
- [pymapvbvd](https://github.com/pehses/mapVBVD)
- [twixtools](https://github.com/pehses/twixtools),  

they are the prior art for this format, and the ground truth `tests/test_parity.py`
verifies against.

## turbotwix vs. twixtools vs. pymapvbvd

| | turbotwix | twixtools | pymapvbvd |
|---|---|---|---|
| Language | Python (numpy, mmap) | Python | Python/MATLAB |
| Full Cartesian read | fastest (~6x pymapvbvd) | slower | slower |
| Partial / out-of-core read | yes, mmap-backed, no full load | no, reads the whole file | no, reads the whole file |
| Line selection / query API | boolean queries over a line table | list of dicts | struct/index based |
| Slice-geometry parsing | no | yes | no |
| Ramp-sampling regridding | no | yes | yes |
| Oversampling removal | no | yes | yes |
| PMU (ECG/pulse/resp) parsing | yes | yes | no |

turbotwix trades the signal-processing conveniences (regridding, oversampling removal,
geometry) for raw read speed and the ability to pull a small selection out of a file that
doesn't fit in RAM. twixtools and pymapvbvd remain the more complete choice when you need
those conveniences and can afford to load the full measurement.

## Performance

Full read of a 234 MB Cartesian measurement, cold page cache, each library in its own
process (`scripts/bench_read.py`):

| library | time (s) | peak anon (MB) | mmap (MB) | maxrss (MB) |
|---|---|---|---|---|
| turbotwix | 0.35 | 140 | 159 | 296 |
| pymapvbvd | 0.82 | 192 | 38 | 228 |
| twixtools | 0.69 | 206 | 41 | 242 |

The case this doesn't show is the one the design is really for: a selection out of a file
far larger than RAM, where the reference readers have no choice but to assemble the whole
array first. On a 24 GiB interleaved spiral measurement, turbotwix builds the line table in
~65 ms and pulls a single shot out of it in ~3 ms — pymapvbvd and twixtools have no
equivalent operation, since without a k-space grid there is nothing partial to fold onto.

The methodology and where the speed comes from are in
[`docs/implementation.md`](docs/implementation.md).

## Known limitations

- A measurement whose line offsets are not 8-byte aligned raises `UnsupportedLayoutError`.
- Lines of differing `(ncha, ncol)` are tabled together but cannot be read in one call;
  select a single-shaped subset.
- SYNCDATA blocks are not ADC data and never appear in the line table; PMU ones decode
  separately via `Measurement.pmu`.
- No ramp-sampling regridding or slice-geometry parsing.
- No oversampling removal: it is signal processing (an FFT round-trip), and along a
  non-Cartesian readout it is not meaningful.

## Documentation

- [`docs/twix-format.md`](docs/twix-format.md): A candidate explanation of what a `.dat` file contains, byte by byte.
- [`docs/implementation.md`](docs/implementation.md): The big picture on  how this reader works and why.

## Development

```
uv sync                    # numpy + dev tools only
uv run pytest
uv run ruff check .
uv run ty check src
uv sync --group parity     # to also run tests/test_parity.py against real references
```
