# The Siemens TWIX (`.dat`) raw-data format

What a Siemens MRI raw-data file actually contains, at the byte level, as implemented in
this repository (mainly `src/turbotwix/dtypes.py` and `src/turbotwix/header.py`).
Everything below is little-endian; all structures are unpadded/packed.

The format is undocumented by the vendor. This description was re-derived from the
reference readers (pymapvbvd, twixtools — see `NOTICE`) and validated against real VD/VE
files; field names follow the Siemens ICE `sMDH` / `sScanHeader` naming so they can be
cross-referenced with those implementations.

## 1. Two generations

There are two on-disk layouts, distinguished only by sniffing the first 8 bytes:

| generation | software baselines | file layout |
|---|---|---|
| **VB** | VA / VB (e.g. VB17) | one measurement per file, no directory |
| **VD/VE** | VD11 … VE11, XA (early) | multi-RAID container, several measurements per file |

`detect_version` reads the first two `uint32`:

- VD/VE: first word is small (unused/`hdSize`), second is the measurement count `<= 64`.
- VB: first word is the byte length of the ASCII header, so it is large.

The heuristic is `first < 10_000 and second <= 64` → VD/VE, else VB. It is the same test
pymapvbvd uses. For VB, the header length is additionally range-checked against the file
size so non-TWIX input fails immediately rather than deep inside the header walk.

## 2. File-level layout

### VB

```
+0                       ASCII/XProtocol text header, length = *(u32)file
+hdr_len                 line 1: MDH(128 B) + samples, MDH + samples, ...  (per channel)
...                      line N ...
                         ACQEND line
```

A VB file is a single measurement; turbotwix models it as a one-entry RAID directory
(`RaidEntry(0, 0, filesize, "", "")`) so the rest of the code has one path.

### VD/VE — the multi-RAID container

```
+0    MrParcRaidFileHeader   hdSize:u4, count:u4                       (8 B)
+8    MrParcRaidFileEntry[64]                                          (64 * 152 B)
      measId:u4, fileId:u4, off:u8, len:u8, patName:S64, protName:S64
...   measurement 0 : text header, then MDH data
...   measurement 1 : ...
```

The entry array is always 64 slots long regardless of `count`; unused slots are zeroed.
`off` is an absolute file offset, in practice aligned up to a 512-byte boundary (the
directory ends at 9736, and the first measurement starts at 10240 in both sample files).
`patName`/`protName` are NUL-terminated `latin1`.

Zero-length entries occur:

- one single entry with `len == 0` means "to end of file" (`parse_raid_directory`);
- trailing zero-length entries are measurements that were announced but never written
  (aborted scans). turbotwix drops those and keeps the complete ones (`TwixFile.__init__`).

A clinical file typically holds several measurements in acquisition order: coil-sensitivity
and noise adjustments first, the actual protocol last. Nothing in the format marks which
one is "the" scan.

## 3. The text header (per measurement)

Each measurement starts with its own text header:

```
+0        hdr_len : u4      total byte length of the text header (data starts at +hdr_len)
+4        n_buffer: u4      number of named buffers
then, n_buffer times:
          name    : NUL-terminated ASCII (>= 4 chars)
          buf_len : u4
          buffer  : buf_len bytes of text
```

`parse_protocol` locates each name/length pair with the regex
`(\w{4,})\x00(.{4})` inside the next 48 bytes, which tolerates the small amount of
padding real files carry. Typical buffer names: `Config`, `Dicom`, `Meas`, `MeasYaps`,
`Phoenix`, `Spice` (all six are present in both sample files; ~800 KiB total).

Each buffer mixes two text syntaxes:

**ascconv** — a flat, `key = value` dump of the MrProt structure, delimited by

```
### ASCCONV BEGIN ... ###
sKSpace.lBaseResolution  = 320
sSliceArray.asSlice[0].dThickness = 5.0
### ASCCONV END ###
```

Keys are dotted paths with `[i]` indices and are rebuilt into nested dicts/lists
(`_parse_ascconv`). Values are typed from their own syntax — quoting, a `0x` prefix, a
decimal point, a sign — rather than from the Hungarian prefix of the key (`l` = long,
`b` = bool, …). Prefix-based typing is what the reference readers do and it misfires
whenever a sequence author named a variable freely: it returns `"10000"` as a string for
an int field, `True` for a flag holding `"0"`, and floats for scanner IDs such as
`6_0_66327775_20210622_151635_650`.

**XProtocol** — a nested, brace-delimited tree with declared types:

```
<ParamLong."NoOfFourierColumns">  { 320 }
<ParamDouble."ReadoutOSFactor">   { <Precision> 6  2.000000 }
<ParamString."tPatientName">      { "Doe^John" }
<ParamBool."SwapReadPhase">       { "true" }
```

turbotwix parses the scalar leaves only (`ParamBool/Long/String/Double`), flattened into
one dict per buffer, taking the type from the tag. Multi-token bodies and names starting
with `a` (array convention) become lists.

Because parsing all buffers costs ~40 ms while splitting them costs ~0.04 ms, `Protocol`
stores raw bytes and parses each buffer on first lookup.

## 4. The MDH stream

After the text header comes the acquisition data: a flat, self-describing sequence of
**lines** (one ADC readout for all channels), each prefixed by a *measurement data
header* (MDH). There is no index, no line count, and no length field for the stream — the
only way to find line *k* is to walk lines 0…*k*−1, because each line's total length is
computed from its own header.

### VB line

```
sMDH (128 B) + ncol complex64 samples          <- channel 0
sMDH (128 B) + ncol complex64 samples          <- channel 1
...                                            <- ncha times
```

The full 128-byte header repeats per channel; `ChannelId` distinguishes them.

### VD/VE line

```
sScanHeader (192 B)                            <- once per line
  sChannelHeader (32 B) + ncol complex64        <- channel 0
  sChannelHeader (32 B) + ncol complex64        <- channel 1
  ...                                           <- ncha times
```

So, uniformly (`header_sizes`):

```
line_length = scan_prefix + ncha * (channel_header + 8 * ncol)
```

with `(scan_prefix, channel_header) = (192, 32)` for VD/VE and `(0, 128)` for VB.
Samples are interleaved real/imag `float32`, i.e. `complex64`, `ncol` per channel,
contiguous — which is why extraction is a strided copy and not a gather.

### Header fields

`VD_SCAN_HEADER` (192 B) and `VB_HEADER` (128 B) share
most fields; the VD header adds patient-table position, an `ApplicationCounter`/`Mask`
pair, a CRC, and 24 rather than 4 `IceProgramPara` words. The ones that matter:

| field | meaning |
|---|---|
| `FlagsAndDMALength` u4 | low 25 bits = encoded block length (`dma_len`), high bits = flags |
| `MeasUID` i4, `ScanCounter` u4 | measurement id, 1-based line counter |
| `TimeStamp`, `PMUTimeStamp` u4 | 2.5 ms ticks since midnight; physiological clock |
| `EvalInfoMask` u8 | 64 flag bits, see §5 |
| `SamplesInScan` u2 (`ncol`) | complex samples per channel |
| `UsedChannels` u2 (`ncha`) | channels in this line |
| `Counter` (14 × u2) | the loop counters, see §6 |
| `CutOff` (Pre, Post) u2 | samples to discard at either end of the readout |
| `CenterCol`, `CenterLin`, `CenterPar` u2 | k-space centre indices |
| `ReadOutOffcentre` f4, `TimeSinceLastRF` u4 | |
| `SliceData` | `SlicePos` (Sag, Cor, Tra f4) + rotation `Quaternion` (4 × f4) |
| `IceProgramPara` u2[24] (VD) / [4] (VB) | free per-line words; custom non-Cartesian sequences commonly stash interleaf/trajectory indices here |

The VD channel header (32 B) carries `TypeAndChannelLength`, `MeasUID`, `ScanCounter`,
a packed `SequenceTime` bitfield, `ChannelId` and a CRC. Nothing in it is needed to
locate samples, so extraction skips over it by stride.

**`FlagsAndDMALength` is not trustworthy as a length.** Siemens' PackBit compression (seen
in EPI) makes the encoded length disagree with the real one, so the walk always recomputes
`line_length` from `ncol`/`ncha`. Both reference readers do the same
(`loop_mdh_read`, `Mdb._get_dma_len`). The two exceptions are ACQEND and SYNCDATA blocks,
which have no `ncol`/`ncha`-derived shape; for those the raw 25-bit field *is* the length.

### Special blocks

- **ACQEND** (`EvalInfoMask` bit 0) — the last block of a measurement. Its body is
  meaningless; its presence is the only reliable end-of-data marker. A file whose walk
  hits EOF first was aborted mid-scan, and turbotwix raises `TruncatedFileError` unless
  `allow_truncated=True`.
- **SYNCDATA** (bit 5) — physiological/PMU and other sideband blocks interleaved with
  imaging lines. Not ADC data; length from the DMA field. A spiral file in testing
  alternated 1–2 image lines of 5.04 MiB with one 2208-byte SYNCDATA block.
- **Shape-less lines** show up in the table as `(ncha, ncol) = (0, 0)` — both sample
  files' tables report shapes `[(0, 0), (2, 320)]`, the first being the ACQEND row.

## 5. `EvalInfoMask` — the 64 flag bits

Bit *n* of the 64-bit mask, in the order Siemens defines them (`_MASK_ID`;
exposed as the `Flag` IntFlag). Undocumented positions keep their slot
as `RESERVED_n` rather than being dropped.

| bit | name | bit | name |
|---|---|---|---|
| 0 | ACQEND | 32 | REF_POSITION |
| 1 | RTFEEDBACK | 33 | SLC_AVERAGED |
| 2 | HPFEEDBACK | 34 | TAGFLAG1 |
| 3 | ONLINE | 35 | CT_NORMALIZE |
| 4 | OFFLINE | 36 | SCAN_FIRST |
| 5 | SYNCDATA | 37 | SCAN_LAST |
| 8 | LASTSCANINCONCAT | 38 | SLICE_ACCEL_REFSCAN |
| 10 | RAWDATACORRECTION | 39 | SLICE_ACCEL_PHASCOR |
| 11 | LASTSCANINMEAS | 40 | FIRST_SCAN_IN_BLADE |
| 12 | SCANSCALEFACTOR | 41 | LAST_SCAN_IN_BLADE |
| 13 | 2NDHADAMARPULSE | 42 | LAST_BLADE_IN_TR |
| 14 | REFPHASESTABSCAN | 44 | PACE |
| 15 | PHASESTABSCAN | 45 | RETRO_LASTPHASE |
| 16 | D3FFT | 46 | RETRO_ENDOFMEAS |
| 17 | SIGNREV | 47 | RETRO_REPEATTHISHEARTBEAT |
| 18 | PHASEFFT | 48 | RETRO_REPEATPREVHEARTBEAT |
| 19 | SWAPPED | 49 | RETRO_ABORTSCANNOW |
| 20 | POSTSHAREDLINE | 50 | RETRO_LASTHEARTBEAT |
| 21 | PHASCOR | 51 | RETRO_DUMMYSCAN |
| 22 | PATREFSCAN | 52 | RETRO_ARRDETDISABLED |
| 23 | PATREFANDIMASCAN | 53 | B1_CONTROLLOOP |
| 24 | REFLECT | 54 | SKIP_ONLINE_PHASCOR |
| 25 | NOISEADJSCAN | 55 | SKIP_REGRIDDING |
| 26 | SHARENOW | 56 | MDH_VOP |
| 27 | LASTMEASUREDLINE | 61 | WIP_1 |
| 28 | FIRSTSCANINSLICE | 62 | WIP_2 |
| 29 | LASTSCANINSLICE | 63 | WIP_3 |
| 30 | TREFFECTIVEBEGIN | | |
| 31 | TREFFECTIVEEND | | |

(Bits 6, 7, 9, 43, 57–60 are unnamed → `RESERVED_n`.)

The ones a reader must act on:

- **ACQEND**, **SYNCDATA** — structural, see above.
- **REFLECT** — the readout was acquired on the opposite gradient polarity and is stored
  reversed; consumers must flip it along `ncol` (turbotwix does, unless `reflect=False`).
- **NOISEADJSCAN** — noise-only lines, for pre-whitening. Not image data.
- **PHASCOR** — phase-correction navigators (EPI).
- **PATREFSCAN** / **PATREFANDIMASCAN** — parallel-imaging reference lines. `PATREFSCAN`
  alone is reference-only and must be excluded from image data; with `PATREFANDIMASCAN`
  the line is both.
- **RTFEEDBACK**, **HPFEEDBACK**, **PHASESTABSCAN**, **REFPHASESTABSCAN** — feedback and
  stabilization lines, not image data.

Flags are the *primary* classification but not a complete one: they do not always
separate distinct acquisitions inside one measurement. A coil-sensitivity adjustment
stores its body-coil (`ncha=2`) and array-coil (`ncha=44`) images both as plain ONLINE
image lines, told apart only by shape. Hence `LineTable.by_shape()`.

## 6. The loop counters

Every header carries 14 `uint16` counters (`LINE_COUNTER`, 28 B), in this order:

`Lin`, `Ave`, `Sli`, `Par`, `Eco`, `Phs`, `Rep`, `Set`, `Seg`, `Ida`, `Idb`, `Idc`, `Idd`, `Ide`

Conventionally: phase-encode line, average, slice, partition (3D second phase encode),
echo, cardiac phase, repetition, set, segment (EPI shot / bipolar polarity), then five
free `Id*` counters a sequence may use as it likes.

These are *labels the sequence wrote*, not k-space coordinates. On a Cartesian scan
`(Lin, Par)` do index the grid, and folding the lines onto a dense array indexed by
counters is meaningful (`read(dims=...)`). On a spiral or radial acquisition the same counters
index shots, interleaves or spokes and there is no grid to fold onto — which is why
turbotwix keeps the line list as the primary model and treats gridding as opt-in. Note
also that `Lin` values are the *nominal* matrix indices: an undersampled or
partial-Fourier scan leaves gaps, and reference scans cover only part of the range.

## 7. Reading a file, end to end

1. mmap the whole file; sniff VB vs VD/VE from the first 8 bytes.
2. VD/VE: parse the RAID directory → a list of `(measId, off, len, patName, protName)`.
   VB: synthesize a single entry covering the file.
3. Per measurement: read `hdr_len` at `off`, split the text header into named buffers.
   MDH data starts at `off + hdr_len`.
4. Walk the MDH stream from there to `off + len` or until an ACQEND block, recording per
   ADC line: absolute offset, `EvalInfoMask`, `ncol`, `ncha`, the 14 counters. Step by
   `scan_prefix + ncha * (chan_hdr + 8 * ncol)`, except for ACQEND/SYNCDATA where the
   step is the raw 25-bit DMA length.
5. Select lines by flags/counters, then read samples: channel *c* of the line at `off_l`
   begins at `off_l + scan_prefix + (c+1) * chan_hdr + c * 8 * ncol` and runs for
   `8 * ncol` bytes.
6. Flip lines flagged REFLECT along the readout.

Step 4 looks inherently sequential, and in general it is — turbotwix walks it block by
block, skipping ACQEND and SYNCDATA by their raw DMA length and recording the ADC lines.
The walk costs one Python iteration per block, which is affordable exactly where it is
needed: an interleaved acquisition is interleaved because its lines are large, so a 25 GiB
spiral measurement is ~7.6k blocks (~32 ms), not millions.

Millions of blocks happen in the opposite regime — small Cartesian lines — and there the
layout is invariably **one uniform run of identically shaped ADC lines terminated by
ACQEND**, which needs no walk at all:

- the stride computed from the *first* header holds for the whole region, so
  `n = (scan_end - data_start) // stride` and `offsets = data_start + arange(n)*stride`;
- the flags and counters of all *n* lines are one strided read (`as_strided` over the
  headers at `stride` spacing), not *n* reads;
- sample offsets are then guaranteed 8-byte aligned, so extraction can view the whole
  file as a `complex64` array and copy with strides — no index arrays, no bounds checks.

The hypothesis is *verified*, not trusted (`_uniform_run`), and declined — falling
back to the walk — if any of these fails:

| check | what it rules out |
|---|---|
| every line reports the same `(ncol, ncha)` | a shape change mid-measurement |
| no line carries ACQEND or SYNCDATA | interleaved sideband blocks; a second run |
| `ScanCounter` advances by a constant positive step | the headers not being consecutive headers at all — i.e. a stride so wrong that sample data was read as a header |
| the bytes left over after the run are an ACQEND block | a further run of a different shape |

Both bundled sample files satisfy all of them (single stride, `ScanCounter` `1..N` dense,
exactly `sizeof(ACQEND)` bytes left over), so their tables are built in under 0.1 ms.

One thing *is* required rather than hypothesized, and raises `UnsupportedLayoutError` with
a message pointing at pymapvbvd or twixtools: ADC line offsets must be 8-byte aligned,
which is what lets extraction view the file as one `complex64` array and copy with
strides. A measurement mixing `(ncha, ncol)` is tabled normally — only a read, whose
result is one rectangular array, needs a single-shaped selection.

The `ScanCounter` check is the one that does not correspond to anything the walk tests.
A walk gets its correctness from never guessing a stride; the fast path guesses once, so
it needs independent evidence that what it read at that stride were really headers. A
counter that increments once per block provides it.

## 8. Practical pitfalls

- **The DMA length lies** (PackBit). Recompute from `ncol`/`ncha` except for
  ACQEND/SYNCDATA.
- **No line count anywhere.** The table must be built by walking; a truncated file simply
  runs out of bytes before ACQEND.
- **`ncol` includes readout oversampling** (usually 2×) and the `CutOff` samples. Removing
  oversampling is an FFT round-trip (crop the central half in image space), not a slice —
  and along a non-Cartesian readout it is not a meaningful operation.
- **`ncha` can change between lines** inside one measurement, as can `ncol`. Any code
  assuming one shape per measurement will silently mis-read coil-adjust scans — which is
  why turbotwix makes that assumption *loudly*, verifying it and raising.
- **Counters are `uint16`** and are nominal indices, not offsets into acquired data.
- **The first measurement is usually not the one you want** — adjustments come first.
- **Not covered by this reader:** PMU/SYNCDATA payload decoding, ramp-sampling
  regridding, slice-geometry (quaternion → orientation) interpretation.
