"""Binary layout of Siemens TWIX (.dat) files: structured dtypes, eval-info-mask flag
bits, VB/VD version detection, and multi-raid directory parsing.

Byte layouts are cross-checked against the reference implementations of pymapvbvd and
twixtools (see NOTICE) but re-derived from scratch as plain little-endian numpy
structured dtypes so header decoding is a single vectorized struct-array view rather
than per-object construction or manual bit-slicing of a raw byte blob.
"""

from __future__ import annotations

import enum
from typing import NamedTuple

import numpy as np


class TwixParseError(Exception):
    """Raised when the binary structure of a .dat file cannot be parsed."""


class TruncatedFileError(TwixParseError):
    """Raised when the file ends before an ACQEND block is reached."""


class UnsupportedVersionError(TwixParseError):
    """Raised when the file does not look like a supported VB or VD/VE twix file."""


# ---------------------------------------------------------------------------
# Sub-structures shared by VB and VD/VE headers
# ---------------------------------------------------------------------------

LINE_COUNTER = np.dtype(
    [
        ("Lin", "<u2"),
        ("Ave", "<u2"),
        ("Sli", "<u2"),
        ("Par", "<u2"),
        ("Eco", "<u2"),
        ("Phs", "<u2"),
        ("Rep", "<u2"),
        ("Set", "<u2"),
        ("Seg", "<u2"),
        ("Ida", "<u2"),
        ("Idb", "<u2"),
        ("Idc", "<u2"),
        ("Idd", "<u2"),
        ("Ide", "<u2"),
    ]
)  # 28 bytes

CUTOFF = np.dtype([("Pre", "<u2"), ("Post", "<u2")])  # 4 bytes

SLICE_POS = np.dtype([("Sag", "<f4"), ("Cor", "<f4"), ("Tra", "<f4")])  # 12 bytes

SLICE_DATA = np.dtype([("SlicePos", SLICE_POS), ("Quaternion", "<f4", (4,))])  # 28 bytes

# ---------------------------------------------------------------------------
# VB: single 128-byte header, repeated once per channel (no separate channel header)
# ---------------------------------------------------------------------------

VB_HEADER = np.dtype(
    [
        ("FlagsAndDMALength", "<u4"),
        ("MeasUID", "<i4"),
        ("ScanCounter", "<u4"),
        ("TimeStamp", "<u4"),
        ("PMUTimeStamp", "<u4"),
        ("EvalInfoMask", "<u8"),
        ("SamplesInScan", "<u2"),
        ("UsedChannels", "<u2"),
        ("Counter", LINE_COUNTER),
        ("CutOff", CUTOFF),
        ("CenterCol", "<u2"),
        ("CoilSelect", "<u2"),
        ("ReadOutOffcentre", "<f4"),
        ("TimeSinceLastRF", "<u4"),
        ("CenterLin", "<u2"),
        ("CenterPar", "<u2"),
        ("IceProgramPara", "<u2", (4,)),
        ("FreePara", "<u2", (4,)),
        ("SliceData", SLICE_DATA),
        ("ChannelId", "<u2"),
        ("PTABPosNeg", "<u2"),
    ]
)
assert VB_HEADER.itemsize == 128

# ---------------------------------------------------------------------------
# VD/VE: 192-byte scan header (once per line) + 32-byte channel header (per channel)
# ---------------------------------------------------------------------------

VD_SCAN_HEADER = np.dtype(
    [
        ("FlagsAndDMALength", "<u4"),
        ("MeasUID", "<i4"),
        ("ScanCounter", "<u4"),
        ("TimeStamp", "<u4"),
        ("PMUTimeStamp", "<u4"),
        ("SystemType", "<u2"),
        ("PTABPosDelay", "<u2"),
        ("PTABPosX", "<i4"),
        ("PTABPosY", "<i4"),
        ("PTABPosZ", "<i4"),
        ("Reserved1", "<i4"),
        ("EvalInfoMask", "<u8"),
        ("SamplesInScan", "<u2"),
        ("UsedChannels", "<u2"),
        ("Counter", LINE_COUNTER),
        ("CutOff", CUTOFF),
        ("CenterCol", "<u2"),
        ("CoilSelect", "<u2"),
        ("ReadOutOffcentre", "<f4"),
        ("TimeSinceLastRF", "<u4"),
        ("CenterLin", "<u2"),
        ("CenterPar", "<u2"),
        ("SliceData", SLICE_DATA),
        ("IceProgramPara", "<u2", (24,)),
        ("ReservedPara", "<u2", (4,)),
        ("ApplicationCounter", "<u2"),
        ("ApplicationMask", "<u2"),
        ("CRC", "<u4"),
    ]
)
assert VD_SCAN_HEADER.itemsize == 192

VD_CHANNEL_HEADER = np.dtype(
    [
        ("TypeAndChannelLength", "<u4"),
        ("MeasUID", "<i4"),
        ("ScanCounter", "<u4"),
        ("Reserved1", "<i4"),
        ("SequenceTime", "<u4"),  # packed bitfield, opaque here (unused for k-space extraction)
        ("Unused2", "<u4"),
        ("ChannelId", "<u2"),
        ("Unused3", "<u2"),
        ("CRC", "<u4"),
    ]
)
assert VD_CHANNEL_HEADER.itemsize == 32

# ---------------------------------------------------------------------------
# Multi-raid file directory (VD/VE only)
# ---------------------------------------------------------------------------

MR_PARC_RAID_FILE_HEADER = np.dtype([("hdSize", "<u4"), ("count", "<u4")])  # 8 bytes

MR_PARC_RAID_FILE_ENTRY = np.dtype(
    [
        ("measId", "<u4"),
        ("fileId", "<u4"),
        ("off", "<u8"),
        ("len", "<u8"),
        ("patName", "S64"),
        ("protName", "S64"),
    ]
)  # 152 bytes
assert MR_PARC_RAID_FILE_ENTRY.itemsize == 152

MAX_RAID_ENTRIES = 64

MULTI_RAID_FILE_HEADER = np.dtype(
    [("hdr", MR_PARC_RAID_FILE_HEADER), ("entry", MR_PARC_RAID_FILE_ENTRY, (MAX_RAID_ENTRIES,))]
)
assert MULTI_RAID_FILE_HEADER.itemsize == 8 + MAX_RAID_ENTRIES * 152


class TwixVersion(enum.Enum):
    VB = "vb"
    VD = "vd"


def detect_version(mm: np.memmap) -> TwixVersion:
    """Sniff VB vs VD/VE from the first 8 bytes, replicating pymapvbvd's heuristic:
    for VD/VE the first u32 is 0 (unused) and the second u32 is the scan count
    (<= MAX_RAID_ENTRIES); anything else is treated as VB (first u32 = ASCII header length).
    """
    if mm.size < 8:
        raise UnsupportedVersionError(f"file is only {mm.size} bytes; not a twix file")
    first, second = (int(v) for v in np.frombuffer(mm, dtype="<u4", count=2, offset=0))
    if first < 10_000 and second <= MAX_RAID_ENTRIES:
        return TwixVersion.VD
    # VB: `first` is the text header length. Sanity-check it before committing to the VB
    # path, so a non-twix file fails here rather than deep inside the header walk.
    if not 8 <= first <= mm.size:
        raise UnsupportedVersionError(
            f"implausible VB header length {first} for a {mm.size}-byte file; "
            "not a supported VB or VD/VE twix file"
        )
    return TwixVersion.VB


def header_sizes(version: TwixVersion) -> tuple[int, int]:
    """`(scan_header_prefix, per_channel_header)` byte sizes for one acquisition line.

    VD/VE has a 192-byte per-line scan header followed by a 32-byte header per channel;
    VB has no per-line prefix and repeats its full 128-byte header per channel.
    """
    if version is TwixVersion.VD:
        return VD_SCAN_HEADER.itemsize, VD_CHANNEL_HEADER.itemsize
    return 0, VB_HEADER.itemsize


class RaidEntry(NamedTuple):
    """One measurement's entry in the multi-raid directory."""

    meas_id: int  # the MID that names the file on the scanner
    offset: int
    length: int
    patient_name: str
    protocol_name: str


def parse_raid_directory(mm: np.memmap, version: TwixVersion) -> list[RaidEntry]:
    """Return the list of scans stored in the file, in file order."""
    if version is TwixVersion.VB:
        return [RaidEntry(0, 0, mm.size, "", "")]

    header = np.frombuffer(mm, dtype=MULTI_RAID_FILE_HEADER, count=1, offset=0)[0]
    count = int(header["hdr"]["count"])
    if count <= 0 or count > MAX_RAID_ENTRIES:
        raise TwixParseError(f"invalid raid directory scan count: {count}")

    def name(field: str) -> str:
        return bytes(raw[field]).split(b"\x00", 1)[0].decode("latin1")

    entries = []
    for k in range(count):
        raw = header["entry"][k]
        length = int(raw["len"])
        # A single zero-length entry means "to end of file"; several mean the trailing
        # ones were never written, and the caller skips those.
        if length == 0 and count == 1:
            length = mm.size - int(raw["off"])
        entries.append(
            RaidEntry(
                int(raw["measId"]), int(raw["off"]), length, name("patName"), name("protName")
            )
        )
    return entries


# ---------------------------------------------------------------------------
# Per-line table produced by the header walk
# ---------------------------------------------------------------------------

# Only what selecting and gathering actually need. The full 192-byte scan header is
# available on demand (`LineTable.headers()`), re-read from the file by offset, so the
# hot table stays 48 bytes per line instead of 204 — on a multi-million-line
# acquisition that is the difference between tens of MB and a GB, and every boolean
# selection over it gets proportionally cheaper.
LINE_DTYPE = np.dtype(
    [
        ("offset", "<i8"),  # absolute byte offset of this line's scan header
        ("flags", "<u8"),  # EvalInfoMask
        ("ncol", "<u2"),  # SamplesInScan
        ("ncha", "<u2"),  # UsedChannels
        ("counters", LINE_COUNTER),
    ]
)
assert LINE_DTYPE.itemsize == 48


# ---------------------------------------------------------------------------
# Eval-info-mask flag bits (64-bit mask, bit position == index in this tuple)
# ---------------------------------------------------------------------------

_MASK_ID: tuple[str, ...] = (
    "ACQEND",
    "RTFEEDBACK",
    "HPFEEDBACK",
    "ONLINE",
    "OFFLINE",
    "SYNCDATA",
    "noname6",
    "noname7",
    "LASTSCANINCONCAT",
    "noname9",
    "RAWDATACORRECTION",
    "LASTSCANINMEAS",
    "SCANSCALEFACTOR",
    "2NDHADAMARPULSE",
    "REFPHASESTABSCAN",
    "PHASESTABSCAN",
    "D3FFT",
    "SIGNREV",
    "PHASEFFT",
    "SWAPPED",
    "POSTSHAREDLINE",
    "PHASCOR",
    "PATREFSCAN",
    "PATREFANDIMASCAN",
    "REFLECT",
    "NOISEADJSCAN",
    "SHARENOW",
    "LASTMEASUREDLINE",
    "FIRSTSCANINSLICE",
    "LASTSCANINSLICE",
    "TREFFECTIVEBEGIN",
    "TREFFECTIVEEND",
    "REF_POSITION",
    "SLC_AVERAGED",
    "TAGFLAG1",
    "CT_NORMALIZE",
    "SCAN_FIRST",
    "SCAN_LAST",
    "SLICE_ACCEL_REFSCAN",
    "SLICE_ACCEL_PHASCOR",
    "FIRST_SCAN_IN_BLADE",
    "LAST_SCAN_IN_BLADE",
    "LAST_BLADE_IN_TR",
    "noname43",
    "PACE",
    "RETRO_LASTPHASE",
    "RETRO_ENDOFMEAS",
    "RETRO_REPEATTHISHEARTBEAT",
    "RETRO_REPEATPREVHEARTBEAT",
    "RETRO_ABORTSCANNOW",
    "RETRO_LASTHEARTBEAT",
    "RETRO_DUMMYSCAN",
    "RETRO_ARRDETDISABLED",
    "B1_CONTROLLOOP",
    "SKIP_ONLINE_PHASCOR",
    "SKIP_REGRIDDING",
    "MDH_VOP",
    "noname57",
    "noname58",
    "noname59",
    "noname60",
    "WIP_1",
    "WIP_2",
    "WIP_3",
)
# The mask as a real type, so selections read as `lines.has(Flag.PHASCOR)` rather than
# as bit arithmetic against a string-keyed lookup table. Bits Siemens never documented
# keep their position under a RESERVED_n name instead of being silently dropped.
Flag = enum.IntFlag(
    "Flag",
    {
        (name.upper() if not name.startswith("noname") else f"RESERVED_{bit}"): 1 << bit
        for bit, name in enumerate(_MASK_ID)
    },
)
Flag.__doc__ = "One bit of a line's 64-bit EvalInfoMask."


def has_flag(flags: np.ndarray, flag: int) -> np.ndarray:
    """Vectorized "carries *all* bits of `flag`" over a (N,) uint64 array of masks."""
    wanted = np.uint64(int(flag))
    return (np.asarray(flags) & wanted) == wanted


def has_any_flag(flags: np.ndarray, flag: int) -> np.ndarray:
    """Vectorized "carries *any* bit of `flag`" — one test for a whole set of bits."""
    return (np.asarray(flags) & np.uint64(int(flag))) != 0


DMA_LEN_MASK = np.uint32(2**25 - 1)


def dma_len(flags_and_dma_length: np.ndarray) -> np.ndarray:
    """Vectorized extraction of the DMA block length (low 25 bits)."""
    return np.bitwise_and(flags_and_dma_length, DMA_LEN_MASK)
