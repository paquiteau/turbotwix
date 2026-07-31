#!/usr/bin/env python3
"""Binary layout: structured dtypes for the VB/VD headers and the raid directory, and
the `Flag` EvalInfoMask bits.

The byte layouts were re-derived from the reference readers (pymapvbvd, twixtools — see
NOTICE) and validated against real VD/VE files; field names follow the Siemens ICE
`sMDH` / `sScanHeader` naming. `docs/twix-format.md` describes the format itself.
"""

from __future__ import annotations

import enum
from typing import NamedTuple

import numpy as np

__all__ = [
    "COUNTERS",
    "Flag",
    "HEADER_SIZES",
    "LINE_COUNTER",
    "LINE_DTYPE",
    "MAX_RAID_ENTRIES",
    "RAID_DIRECTORY",
    "RaidEntry",
    "SCAN_HEADER_DTYPES",
    "SEQDATA_HEADER",
    "SYNC_DTYPE",
    "TwixParseError",
    "TwixVersion",
    "UnsupportedLayoutError",
    "UnsupportedVersionError",
    "VB_HEADER",
    "VD_CHANNEL_HEADER",
    "VD_SCAN_HEADER",
    "detect_version",
    "parse_raid_directory",
]


class TwixParseError(Exception):
    """Raised when the binary structure of a .dat file cannot be parsed."""


class UnsupportedLayoutError(TwixParseError):
    """Raised when a measurement is not laid out the way turbotwix requires.

    One thing is required rather than guessed: ADC line offsets are 8-byte aligned,
    which is what lets extraction view the file as one `complex64` array and copy by
    strides. Use pymapvbvd or twixtools for a file that breaks it.

    Also raised when a *read* is asked for lines of differing `(ncha, ncol)` — an
    embedded reference scan, or a coil-sensitivity adjustment holding body-coil and
    array-coil lines. That is not a limit on the file, only on one rectangular result:
    the table holds those lines, and selecting a single-shaped subset reads them.
    """


class UnsupportedVersionError(TwixParseError):
    """Raised when the file does not look like a supported VB or VD/VE twix file."""


_USE_A_REFERENCE_READER = (
    "turbotwix reads lines at 8-byte-aligned offsets. "
    "Use pymapvbvd or twixtools for this file."
)


# ---------------------------------------------------------------------------
# 1. Binary layout
# ---------------------------------------------------------------------------

#: The 14 loop counters every header carries, in header order.
COUNTERS = [
    "Lin",
    "Ave",
    "Sli",
    "Par",
    "Eco",
    "Phs",
    "Rep",
    "Set",
    "Seg",
    "Ida",
    "Idb",
    "Idc",
    "Idd",
    "Ide",
]
LINE_COUNTER = np.dtype([(c, "<u2") for c in COUNTERS])  # 28 bytes

_CUTOFF = np.dtype([("Pre", "<u2"), ("Post", "<u2")])
_SLICE_POS = np.dtype([("Sag", "<f4"), ("Cor", "<f4"), ("Tra", "<f4")])
_SLICE_DATA = np.dtype([("SlicePos", _SLICE_POS), ("Quaternion", "<f4", (4,))])

#: VB: one 128-byte header, repeated per channel (there is no separate channel header).
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
        ("CutOff", _CUTOFF),
        ("CenterCol", "<u2"),
        ("CoilSelect", "<u2"),
        ("ReadOutOffcentre", "<f4"),
        ("TimeSinceLastRF", "<u4"),
        ("CenterLin", "<u2"),
        ("CenterPar", "<u2"),
        ("IceProgramPara", "<u2", (4,)),
        ("FreePara", "<u2", (4,)),
        ("SliceData", _SLICE_DATA),
        ("ChannelId", "<u2"),
        ("PTABPosNeg", "<u2"),
    ]
)
# assert VB_HEADER.itemsize == 128

#: VD/VE: a 192-byte scan header once per line, then a 32-byte header per channel.
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
        ("CutOff", _CUTOFF),
        ("CenterCol", "<u2"),
        ("CoilSelect", "<u2"),
        ("ReadOutOffcentre", "<f4"),
        ("TimeSinceLastRF", "<u4"),
        ("CenterLin", "<u2"),
        ("CenterPar", "<u2"),
        ("SliceData", _SLICE_DATA),
        ("IceProgramPara", "<u2", (24,)),
        ("ReservedPara", "<u2", (4,)),
        ("ApplicationCounter", "<u2"),
        ("ApplicationMask", "<u2"),
        ("CRC", "<u4"),
    ]
)
# assert VD_SCAN_HEADER.itemsize == 192

VD_CHANNEL_HEADER = np.dtype(
    [
        ("TypeAndChannelLength", "<u4"),
        ("MeasUID", "<i4"),
        ("ScanCounter", "<u4"),
        ("Reserved1", "<i4"),
        (
            "SequenceTime",
            "<u4",
        ),  # packed bitfield, opaque here: not needed to find samples
        ("Unused2", "<u4"),
        ("ChannelId", "<u2"),
        ("Unused3", "<u2"),
        ("CRC", "<u4"),
    ]
)
# assert VD_CHANNEL_HEADER.itemsize == 32

MAX_RAID_ENTRIES = 64

_RAID_ENTRY = np.dtype(
    [
        ("measId", "<u4"),
        ("fileId", "<u4"),
        ("off", "<u8"),
        ("len", "<u8"),
        ("patName", "S64"),
        ("protName", "S64"),
    ]
)  # 152 bytes

#: The VD/VE container directory: a count, then always 64 slots (unused ones zeroed).
RAID_DIRECTORY = np.dtype(
    [("hdSize", "<u4"), ("count", "<u4"), ("entry", _RAID_ENTRY, (MAX_RAID_ENTRIES,))]
)
# assert RAID_DIRECTORY.itemsize == 8 + MAX_RAID_ENTRIES * 152

# One row per ADC line: only what selection and extraction need. The full 192-byte scan
# header is re-read on demand (`LineTable.headers()`), so the hot table stays 48 bytes
# per line instead of 204 — on a multi-million-line acquisition that is tens of MB
# rather than a GB, and every boolean selection over it gets proportionally cheaper.
LINE_DTYPE = np.dtype(
    [
        ("offset", "<i8"),  # absolute byte offset of this line's scan header
        ("flags", "<u8"),  # EvalInfoMask
        ("ncol", "<u2"),  # SamplesInScan
        ("ncha", "<u2"),  # UsedChannels
        ("counters", LINE_COUNTER),
    ]
)
# assert LINE_DTYPE.itemsize == 48

#: One row per SYNCDATA/PMU block: enough to re-read and decode it on demand.
SYNC_DTYPE = np.dtype([("offset", "<i8"), ("length", "<i8")])

#: The header a SYNCDATA block's payload starts with, right after the scan header.
#: `id` names the sideband stream; only ids starting with ``b"PMU"`` are physiological
#: data (`turbotwix.pmu`). Re-derived from twixtools' `seqdata.SeqDataHeader`.
SEQDATA_HEADER = np.dtype(
    [
        ("packet_size", "<u4"),
        ("id", "S52"),
        ("swapped", "<u4"),
    ]
)
# assert SEQDATA_HEADER.itemsize == 60

#: Names of the 14 loop counters, in header order.


class TwixVersion(enum.StrEnum):
    VB = "vb"  # VA / VB baselines: one measurement per file, no directory
    VD = "vd"  # VD11 ... VE11, early XA: a multi-raid container


HEADER_SIZES = {
    TwixVersion.VB: (0, VB_HEADER.itemsize),
    TwixVersion.VD: (VD_SCAN_HEADER.itemsize, VD_CHANNEL_HEADER.itemsize),
}
SCAN_HEADER_DTYPES = {TwixVersion.VB: VB_HEADER, TwixVersion.VD: VD_SCAN_HEADER}


def detect_version(mm: np.ndarray) -> TwixVersion:
    """Sniff VB vs VD/VE from the first 8 bytes.

    For VD/VE the first `uint32` is small (unused) and the second is the measurement
    count; for VB the first is the byte length of the ASCII header, so it is large.

    Uses the same heuristic as pymapvbvd.

    Parameters
    ----------
    mm : numpy.ndarray
        The whole file as a ``uint8`` array (typically a `numpy.memmap`).

    Returns
    -------
    TwixVersion
        `TwixVersion.VB` or `TwixVersion.VD`.

    Raises
    ------
    UnsupportedVersionError
        If the file is shorter than 8 bytes, or if neither layout is plausible.
    """
    if mm.size < 8:
        raise UnsupportedVersionError(f"file is only {mm.size} bytes; not a twix file")
    first, second = (int(v) for v in np.frombuffer(mm, dtype="<u4", count=2))
    if first < 10_000 and second <= MAX_RAID_ENTRIES:
        return TwixVersion.VD
    # VB: `first` is the text header length. Range-check it here, so a non-twix file
    # fails immediately rather than deep inside the header walk.
    if not 8 <= first <= mm.size:
        raise UnsupportedVersionError(
            f"implausible VB header length {first} for a {mm.size}-byte file; "
            "not a supported VB or VD/VE twix file"
        )
    return TwixVersion.VB


class RaidEntry(NamedTuple):
    """One measurement's entry in the container directory."""

    meas_id: int  # the MID that names the file on the scanner
    offset: int
    length: int
    patient_name: str
    protocol_name: str


def parse_raid_directory(mm: np.ndarray, version: TwixVersion) -> list[RaidEntry]:
    """The measurements stored in the file, in acquisition order.

    A VB file is a single measurement, modelled as a one-entry directory covering the
    whole file so the rest of the code has one path.

    Parameters
    ----------
    mm : numpy.ndarray
        The whole file as a ``uint8`` array.
    version : TwixVersion
        The layout detected by `detect_version`.

    Returns
    -------
    list of RaidEntry
        One entry per announced measurement, in file order.

    Raises
    ------
    TwixParseError
        If the VD/VE directory announces an implausible measurement count.
    """
    if version is TwixVersion.VB:
        return [RaidEntry(0, 0, mm.size, "", "")]

    directory = np.frombuffer(mm, dtype=RAID_DIRECTORY, count=1)[0]
    count = int(directory["count"])
    if count <= 0 or count > MAX_RAID_ENTRIES:
        raise TwixParseError(f"invalid raid directory measurement count: {count}")

    entries = []
    for raw in directory["entry"][:count]:
        length = int(raw["len"])
        # A single zero-length entry means "to end of file"; several mean the trailing
        # ones were announced but never written (an aborted scan), and are dropped by
        # TwixFile.
        if length == 0 and count == 1:
            length = mm.size - int(raw["off"])
        entries.append(
            RaidEntry(
                int(raw["measId"]),
                int(raw["off"]),
                length,
                _cstr(raw["patName"]),
                _cstr(raw["protName"]),
            )
        )
    return entries


def _cstr(raw) -> str:
    """Decode a fixed-width, NUL-padded byte field as text.

    Parameters
    ----------
    raw : bytes-like
        The raw field, e.g. one ``S64`` entry of the raid directory.

    Returns
    -------
    str
        Everything before the first NUL, decoded as latin-1.
    """
    return bytes(raw).split(b"\x00", 1)[0].decode("latin1")


# ---------------------------------------------------------------------------
# 2. The eval-info mask
# ---------------------------------------------------------------------------

# Bit position == index in this tuple. Positions Siemens never documented keep their
# slot under a RESERVED_n name rather than being silently dropped.
_MASK_ID: tuple[str, ...] = (
    "ACQEND",
    "RTFEEDBACK",
    "HPFEEDBACK",
    "ONLINE",
    "OFFLINE",
    "SYNCDATA",
    "",
    "",
    "LASTSCANINCONCAT",
    "",
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
    "",
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
    "",
    "",
    "",
    "",
    "WIP_1",
    "WIP_2",
    "WIP_3",
)

#: The mask as a real type, so selections read as `lines.has(Flag.PHASCOR)`.
Flag = enum.IntFlag(
    "Flag", {name or f"RESERVED_{bit}": 1 << bit for bit, name in enumerate(_MASK_ID)}
)
Flag.__doc__ = "One bit of a line's 64-bit EvalInfoMask."

# Only ACQEND and SYNCDATA blocks take their length from the low 25 bits of the header's
# first word; a normal line's length is recomputed from `(ncol, ncha)`, because Siemens'
# PackBit compression can make that field disagree with the real one.
_DMA_LEN_MASK = 2**25 - 1
