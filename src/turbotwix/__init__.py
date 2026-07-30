"""turbotwix: a reader for Siemens MRI raw data (`.dat` / TWIX) files.

Everything lives in this one module, in the order a file is read:

1. binary layout — structured dtypes for the VB/VD headers and the raid directory;
2. `Flag` — the 64 `EvalInfoMask` bits;
3. the text protocol — ascconv / XProtocol parsing, lazy per buffer;
4. the line table — one 48-byte row per ADC line;
5. extraction — a strided copy of the selected lines' samples;
6. the object model — `open_twix` -> `TwixFile` -> `Measurement` -> `LineTable`.

The data model is the file's own: a measurement is a *list of acquisition lines*, each
with its metadata and its `(ncha, ncol)` block of samples. Selecting lines is a boolean
query over that table; reading returns `(n_lines, ncha, ncol)`. Folding onto a Cartesian
grid is available (`to_dense`) but never implicit — for a spiral or radial acquisition
the loop counters index shots, interleaves or spokes, and there is no grid to fold onto.

The byte layouts were re-derived from the reference readers (pymapvbvd, twixtools — see
NOTICE) and validated against real VD/VE files; field names follow the Siemens ICE
`sMDH` / `sScanHeader` naming. `docs/twix-format.md` describes the format itself.
"""

from __future__ import annotations

import enum
import functools
import mmap as _mmap
import re
import struct
import warnings
from typing import Literal, NamedTuple

import numpy as np
from numpy.lib.stride_tricks import as_strided

__all__ = [
    "COUNTERS",
    "Flag",
    "LineTable",
    "Measurement",
    "Protocol",
    "TruncatedFileError",
    "TwixFile",
    "TwixParseError",
    "TwixVersion",
    "UnsupportedLayoutError",
    "UnsupportedVersionError",
    "open_twix",
    "minimal_dims",
    "to_dense",
]


class TwixParseError(Exception):
    """Raised when the binary structure of a .dat file cannot be parsed."""


class TruncatedFileError(TwixParseError):
    """Raised when a measurement ends before an ACQEND block is reached."""


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


def has_flag(flags: np.ndarray, flag: int) -> np.ndarray:
    """Vectorized "carries *all* bits of `flag`" over an array of masks.

    Parameters
    ----------
    flags : numpy.ndarray
        ``uint64`` EvalInfoMask values, one per line.
    flag : int
        The bit or combination of bits to test for, typically a `Flag`.

    Returns
    -------
    numpy.ndarray
        Boolean array, True where every bit of `flag` is set.
    """
    wanted = np.uint64(int(flag))
    return (np.asarray(flags) & wanted) == wanted


def has_any_flag(flags: np.ndarray, flag: int) -> np.ndarray:
    """Vectorized "carries *any* bit of `flag`" — one test for a whole set of bits.

    Parameters
    ----------
    flags : numpy.ndarray
        ``uint64`` EvalInfoMask values, one per line.
    flag : int
        The bits to test for, typically a union of `Flag` members.

    Returns
    -------
    numpy.ndarray
        Boolean array, True where at least one bit of `flag` is set.
    """
    return (np.asarray(flags) & np.uint64(int(flag))) != 0


# ---------------------------------------------------------------------------
# 3. The text protocol
# ---------------------------------------------------------------------------

_BUFFER_NAME = re.compile(rb"(\w{4,})\x00(.{4})", re.DOTALL)
_ASCCONV = re.compile(r"### ASCCONV BEGIN[^\n]*\n(.*)\s### ASCCONV END ###", re.DOTALL)
_ASCCONV_LINE = re.compile(r"(?P<name>\S*)\s*=\s*(?P<value>\S*)\n")
_ASCCONV_KEY = re.compile(r"(?P<name>\w+)(\[(?P<ix>[0-9]+)\])?")
_XPROT_SCALAR = re.compile(r'<Param(Bool|Long|String)\."(\w+)">\s*{([^}]*)')
_XPROT_DOUBLE = re.compile(
    r'<ParamDouble\."(\w+)">\s*{\s*(<Precision>\s*[0-9]*)?\s*([^}]*)'
)

_INT = re.compile(r"[+-]?\d+$")
_HEX = re.compile(r"[+-]?0[xX][0-9a-fA-F]+$")
# Explicit, because Python's int()/float() accept `_` as a digit separator: a bare
# `try: float(value)` turns the scanner ID "6_0_66327775_20210622_151635_650" into
# 6.07e+26 rather than failing.
_FLOAT = re.compile(r"[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")


class AttrDict(dict):
    """A dict that also allows attribute-style access, recursively."""

    def __getattr__(self, name: str):
        try:
            value = self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
        if isinstance(value, dict) and not isinstance(value, AttrDict):
            value = AttrDict(value)
            self[name] = value
        return value

    __setattr__ = dict.__setitem__


class Protocol(AttrDict):
    """Buffer name -> parsed contents, with each buffer parsed on first access.

    A measurement's text header is ~800 KiB across six buffers and regex-parsing all of
    it costs ~40 ms — most of the time spent reading a small file, and pure waste for
    callers that only want k-space. So each value starts out as raw bytes and is
    replaced by its parsed `AttrDict` the first time it is looked up. Every read path
    (`p["Meas"]`, `p.Meas`, `.get`, `.values`, `.items`, `dict(p)`) parses on the way
    through, so the laziness is not observable apart from where the time is spent.
    """

    def __getitem__(self, name: str):
        value = dict.__getitem__(self, name)
        if isinstance(value, bytes):  # raw buffer, not parsed yet
            value = AttrDict(_parse_buffer(value.decode("latin1")))
            dict.__setitem__(self, name, value)
        return value

    def get(self, name, default=None):
        return self[name] if name in self else default

    def values(self):
        return [self[name] for name in self]

    def items(self):
        return [(name, self[name]) for name in self]

    def __iter__(self):
        # Defined (rather than inherited) on purpose: `dict(p)` and `{**p}` take a fast
        # path that reads the underlying table directly and would hand back raw bytes.
        # Overriding tp_iter forces the generic path, which goes through __getitem__.
        return dict.__iter__(self)


def parse_protocol(mm: np.ndarray, offset: int) -> tuple[Protocol, int]:
    """Split the text header of the measurement at `offset` into its named buffers.

    Each buffer is located by its `name\\0` + `uint32` length pair, searched for within
    the next 48 bytes because real files carry a little padding there.

    Parameters
    ----------
    mm : numpy.ndarray
        The whole file as a ``uint8`` array.
    offset : int
        Byte offset of the measurement, from its `RaidEntry`.

    Returns
    -------
    protocol : Protocol
        Buffer name -> raw bytes, parsed on first access.
    hdr_len : int
        Byte length of the text header; the MDH stream starts at ``offset + hdr_len``.
    """
    hdr_len, n_buffer = (
        int(v) for v in np.frombuffer(mm, dtype="<u4", count=2, offset=offset)
    )
    protocol = Protocol()
    pos = offset + 8
    for _ in range(n_buffer):
        match = _BUFFER_NAME.search(bytes(mm[pos : pos + 48]))
        if match is None:
            break
        (buf_len,) = struct.unpack("<I", match.group(2))
        pos += match.end()
        dict.__setitem__(
            protocol, match.group(1).decode("latin1"), bytes(mm[pos : pos + buf_len])
        )
        pos += buf_len
    return protocol, hdr_len


def _as_number(text: str, integer: bool = False):
    """`text` as an int/float if it is syntactically a number, else None.

    Parameters
    ----------
    text : str
        The token to type, already stripped.
    integer : bool, default False
        Whether a value written as a float should be truncated to an int, which is what
        a `ParamLong` declaration asks for.

    Returns
    -------
    int or float or None
        The parsed number, or None if `text` is not numeric syntax.
    """
    if _HEX.match(text):
        return int(text, 16)
    if _INT.match(text):
        return int(text)
    if _FLOAT.match(text):
        return int(float(text)) if integer else float(text)
    return None


def _cast_value(value: str):
    """Type an ascconv value from its own syntax.

    Not from the key's Hungarian prefix, which is what the reference readers do: that
    guesses (`b` -> bool, `l` -> long, ...) and misfires whenever a sequence author
    named a variable freely, returning `"10000"` as a string for an int field or True
    for a flag holding `"0"`. Quoting, `0x`, a decimal point and a sign are unambiguous;
    where the text says nothing, the text is the value.

    Parameters
    ----------
    value : str
        The right-hand side of one ascconv `key = value` line.

    Returns
    -------
    str or int or float
        The typed value; the unquoted text itself when the syntax says nothing.
    """
    text = value.strip()
    if text.startswith('"'):
        return text.strip('"')
    number = _as_number(text)
    return text if number is None else number


def _cast_xprot(value: str, kind: str):
    """Type an XProtocol value from its declared tag.

    Unlike ascconv, a `<ParamLong."x">` tag actually states the type, so nothing has to
    be inferred from the syntax.

    Parameters
    ----------
    value : str
        The token as it appears in the buffer.
    kind : {'Bool', 'Long', 'String', 'Double'}
        The type named by the enclosing tag.

    Returns
    -------
    str or bool or int or float or None
        The typed value; None for an empty non-string body.
    """
    text = value.strip().strip('"').strip()
    if kind == "String":
        return value.strip().strip('"')
    if text == "":
        return None
    if kind == "Bool":
        return text.lower() == "true"
    number = _as_number(text, integer=kind == "Long")
    return text if number is None else number


def _update_ascconv(prot, key: list, value: str) -> None:
    """Insert `value` at the dotted/indexed `key` path, growing lists as needed.

    Parameters
    ----------
    prot : dict or list
        The container to insert into, modified in place.
    key : list
        The path, as alternating names and integer indices (``sSlice[2].dThickness``
        becomes ``["sSlice", 2, "dThickness"]``).
    value : str
        The raw text to type and store at that path.

    Returns
    -------
    None
    """
    if "__attribute__" in key:
        return
    head = key[0]
    if len(key) > 1:
        if isinstance(head, int):
            while len(prot) < head + 1:
                prot.append([] if isinstance(key[1], int) else {})
        elif head not in prot:
            prot[head] = [] if isinstance(key[1], int) else {}
        _update_ascconv(prot[head], key[1:], value)
        return
    if isinstance(head, int):
        while len(prot) < head + 1:
            prot.append([])
    prot[head] = _cast_value(value)


def _parse_ascconv(buffer: str) -> dict:
    """The flat `key = value` MrProt dump, rebuilt into nested dicts/lists.

    Parameters
    ----------
    buffer : str
        The ascconv section, delimiters included.

    Returns
    -------
    dict
        The MrProt tree, with indexed keys as lists.
    """
    mrprot: dict = {}
    for line in _ASCCONV_LINE.finditer(buffer):
        key: list = []
        for part in _ASCCONV_KEY.finditer(line.group("name")):
            key.append(part.group("name"))
            if part.group("ix") is not None:
                key.append(int(part.group("ix")))
        if key:
            _update_ascconv(mrprot, key, line.group("value"))
    return mrprot


def _parse_xprot(buffer: str) -> dict:
    """The scalar leaves of the XProtocol tree, flattened into one dict.

    Multi-token bodies and names starting with `a` (the array convention) become lists;
    the nesting itself is not modelled.

    Parameters
    ----------
    buffer : str
        The buffer text, with the ascconv section already removed.

    Returns
    -------
    dict
        Parameter name -> typed value.
    """
    xprot: dict = {}
    tokens = [(m, m.group(1)) for m in _XPROT_SCALAR.finditer(buffer)]
    tokens += [(m, "Double") for m in _XPROT_DOUBLE.finditer(buffer)]
    for match, kind in tokens:
        name = match.group(1) if kind == "Double" else match.group(2)
        body = re.sub(r" *<\w*> *[^\n]*", "", match.groups()[-1])
        body = re.sub(r"[\t\n\r\f\v]+", " ", body).strip()
        if kind == "String" and not name.startswith("a"):
            xprot[name] = body.strip('"')
        elif len(body.split()) > 1 or name.startswith("a"):
            xprot[name] = [_cast_xprot(part, kind) for part in body.split()]
        else:
            xprot[name] = _cast_xprot(body, kind)
    return xprot


def _parse_buffer(buffer: str) -> dict:
    """One header buffer, which mixes an ascconv section with XProtocol text.

    Parameters
    ----------
    buffer : str
        The decoded buffer contents.

    Returns
    -------
    dict
        The ascconv tree, updated with the flattened XProtocol scalars.
    """
    match = _ASCCONV.search(buffer)
    prot = _parse_ascconv(match.group(0)) if match else {}
    # `.sub()` rather than `.split()`: the pattern has a capture group, so splitting
    # would re-insert the ascconv body into the text scanned as XProtocol.
    prot.update(_parse_xprot(_ASCCONV.sub("", buffer)))
    return prot


# ---------------------------------------------------------------------------
# 4. The line table
# ---------------------------------------------------------------------------


def open_mmap(path: str) -> np.memmap:
    """Memory-map the whole file read-only, with a best-effort sequential-access hint.

    Parameters
    ----------
    path : str
        Path to the `.dat` file.

    Returns
    -------
    numpy.memmap
        The file as a read-only ``uint8`` array.
    """
    mm = np.memmap(path, dtype=np.uint8, mode="r")
    raw = getattr(mm, "_mmap", None)
    if raw is not None:
        try:
            raw.madvise(_mmap.MADV_SEQUENTIAL)
        except (AttributeError, OSError):
            pass  # best-effort only; not every platform supports it
    return mm


def build_table(
    mm: np.ndarray, data_start: int, scan_end: int, version: TwixVersion
) -> tuple[np.ndarray, bool]:
    """Build the line table of one measurement.

    Blocks that hold no samples (ACQEND, SYNCDATA/PMU) are stepped over, not recorded:
    nothing here can decode them.

    A measurement may mix `(ncha, ncol)` — an embedded parallel-imaging reference scan
    is Cartesian and short where the imaging lines are spiral and long, and a
    coil-sensitivity adjustment stores body-coil and array-coil lines together. The
    table records each line's own shape and stays whole; the one-shape rule belongs to
    `read_lines`, which needs a rectangular result, and is met by selecting (`.image`,
    `.refscan`) before reading.

    The stream carries no index and no line count — each line's length is computed from
    its own header — so this looks inherently sequential. Two regimes, each with its own
    path:

    * a **single uniform run** of identically shaped lines ending in ACQEND, which is
      what Cartesian scans produce, often with millions of small lines. There the
      stride from the first header holds throughout, so `_uniform_run` needs no
      per-line work at all: the count is a division, the offsets an `arange`, the flags
      and counters one strided read of all N headers.
    * **anything else** — interleaved SYNCDATA/PMU blocks, several runs — which is what
      non-Cartesian sequences produce. There the block count is small precisely because
      the lines are big (a 25 GiB spiral measurement is ~7.6k blocks of ~5 MiB), so
      `_walk` steps block by block and costs tens of milliseconds on such a file.

    The fast path is tried first and declines when its hypothesis fails; the walk is the
    general answer, not a penalty box.

    Parameters
    ----------
    mm : numpy.ndarray
        The whole file as a ``uint8`` array.
    data_start : int
        Byte offset of the first block, past the text header. Must be 8-byte aligned.
    scan_end : int
        Byte offset one past the end of this measurement.
    version : TwixVersion
        The layout detected by `detect_version`.

    Returns
    -------
    table : numpy.ndarray
        A `LINE_DTYPE` array with one row per ADC line.
    truncated : bool
        Whether the measurement ran out of bytes before its ACQEND block.

    Raises
    ------
    UnsupportedLayoutError
        If `data_start` or any line offset is not 8-byte aligned.
    """
    if data_start % 8:
        # Every sample offset is `line_offset + prefix + (c+1)*chan_hdr + c*8*ncol`, and
        # every term but the line offset is a multiple of 8. Keeping line offsets
        # aligned is what lets extraction view the whole file as a complex64 array.
        raise UnsupportedLayoutError(
            f"measurement data starts at unaligned offset {data_start}. "
            f"{_USE_A_REFERENCE_READER}"
        )
    header_dtype = SCAN_HEADER_DTYPES[version]
    if data_start + header_dtype.itemsize > scan_end:
        return np.empty(0, dtype=LINE_DTYPE), True

    uniform = _uniform_run(mm, data_start, scan_end, version, header_dtype)
    if uniform is not None:
        return uniform
    return _walk(mm, data_start, scan_end, version, header_dtype)


def _uniform_run(
    mm: np.ndarray,
    data_start: int,
    scan_end: int,
    version: TwixVersion,
    header_dtype: np.dtype,
) -> tuple[np.ndarray, bool] | None:
    """The whole table by arithmetic, or None if this is not one uniform run.

    The hypothesis — the stride computed from the first header holds to the end — is
    verified exactly over every line before being used, and testing it costs one strided
    read of the headers the walk would have read anyway.

    Parameters
    ----------
    mm : numpy.ndarray
        The whole file as a ``uint8`` array.
    data_start : int
        Byte offset of the first block.
    scan_end : int
        Byte offset one past the end of this measurement.
    version : TwixVersion
        The layout detected by `detect_version`.
    header_dtype : numpy.dtype
        The scan-header dtype for `version`.

    Returns
    -------
    tuple of (numpy.ndarray, bool) or None
        The `(table, truncated)` pair as for `build_table`, or None if the measurement
        is not a single uniform run and the caller must fall back to `_walk`.
    """
    hdr_size = header_dtype.itemsize
    prefix, chan_hdr = HEADER_SIZES[version]
    stop_bits = np.uint64(Flag.ACQEND | Flag.SYNCDATA)

    first = mm[data_start : data_start + hdr_size].view(header_dtype)[0]
    ncol, ncha = int(first["SamplesInScan"]), int(first["UsedChannels"])
    if ncol == 0 or ncha == 0 or first["EvalInfoMask"] & stop_bits:
        return None

    stride = prefix + ncha * (chan_hdr + 8 * ncol)
    n = (scan_end - data_start) // stride
    if n == 0:
        return np.empty(0, dtype=LINE_DTYPE), True

    # A zero-copy strided view of the N candidate headers. `as_strided` does no bounds
    # checking; the division above is what keeps every one of them inside the region.
    headers = as_strided(
        mm[data_start : data_start + hdr_size].view(header_dtype),
        shape=(n,),
        strides=(stride,),
    )
    if np.any(headers["SamplesInScan"] != ncol) or np.any(
        headers["UsedChannels"] != ncha
    ):
        return None  # a shape change mid-measurement
    if np.any(headers["EvalInfoMask"] & stop_bits):
        return None  # interleaved sideband blocks, or a second run
    # The one check the walk never needs: a walk is correct because it never guesses a
    # stride, while this path guesses once, so it needs independent evidence that what
    # it read at that stride really were headers. ScanCounter increments once per block,
    # so a constant positive step across the run provides it.
    counter = headers["ScanCounter"].astype(np.int64)
    if n > 1:
        step = int(counter[1] - counter[0])
        if step <= 0 or not np.all(np.diff(counter) == step):
            return None

    # ACQEND is shorter than a line, so it falls outside the division above; what is
    # left over must be exactly it, or a further run of another shape is hiding there.
    # Less than a header left means the file was cut short — still a uniform run, just
    # an unterminated one.
    tail = data_start + n * stride
    truncated = scan_end - tail < hdr_size
    if not truncated:
        terminator = mm[tail : tail + hdr_size].view(header_dtype)[0]
        if not terminator["EvalInfoMask"] & np.uint64(int(Flag.ACQEND)):
            return None

    table = np.empty(n, dtype=LINE_DTYPE)
    table["offset"] = data_start + np.arange(n, dtype=np.int64) * stride
    table["flags"] = headers["EvalInfoMask"]
    table["ncol"] = ncol
    table["ncha"] = ncha
    table["counters"] = headers["Counter"]
    return table, truncated


def _walk(
    mm: np.ndarray,
    data_start: int,
    scan_end: int,
    version: TwixVersion,
    header_dtype: np.dtype,
) -> tuple[np.ndarray, bool]:
    """Step block by block, recording the ADC lines and skipping the rest.

    Deliberately plain: one Python iteration per *block*, which is affordable exactly
    where this path is needed, since interleaved acquisitions are interleaved because
    their lines are large.

    Parameters
    ----------
    mm : numpy.ndarray
        The whole file as a ``uint8`` array.
    data_start : int
        Byte offset of the first block.
    scan_end : int
        Byte offset one past the end of this measurement.
    version : TwixVersion
        The layout detected by `detect_version`.
    header_dtype : numpy.dtype
        The scan-header dtype for `version`.

    Returns
    -------
    table : numpy.ndarray
        A `LINE_DTYPE` array with one row per ADC line.
    truncated : bool
        Whether the walk ran out of bytes before reaching ACQEND.

    Raises
    ------
    UnsupportedLayoutError
        If a line starts at an offset that is not 8-byte aligned.
    """
    hdr_size = header_dtype.itemsize
    prefix, chan_hdr = HEADER_SIZES[version]
    acqend_bit = np.uint64(Flag.ACQEND)
    sync_bit = np.uint64(Flag.SYNCDATA)

    rows: list[tuple] = []
    pos = data_start
    truncated = False

    while True:
        if pos + hdr_size > scan_end:
            truncated = True
            break
        header = mm[pos : pos + hdr_size].view(header_dtype)[0]
        flags = header["EvalInfoMask"]

        if flags & acqend_bit:
            break
        if flags & sync_bit:
            # No (ncol, ncha)-derived shape, so the raw 25-bit length is all there is.
            length = int(header["FlagsAndDMALength"]) & _DMA_LEN_MASK
            if length <= 0 or pos + length > scan_end:
                truncated = True
                break
            pos += length
            continue

        line_ncol, line_ncha = int(header["SamplesInScan"]), int(header["UsedChannels"])
        if pos % 8:
            raise UnsupportedLayoutError(
                f"line {len(rows)} starts at unaligned offset {pos}. "
                f"{_USE_A_REFERENCE_READER}"
            )
        length = prefix + line_ncha * (chan_hdr + 8 * line_ncol)
        if length <= 0 or pos + length > scan_end:
            truncated = True
            break

        rows.append((pos, flags, line_ncol, line_ncha, header["Counter"]))
        pos += length

    return np.array(rows, dtype=LINE_DTYPE), truncated


def common_shape(table: np.ndarray) -> tuple[int, int]:
    """The one `(ncha, ncol)` of every line in `table`.

    Parameters
    ----------
    table : numpy.ndarray
        A `LINE_DTYPE` array, as in `LineTable.rows`.

    Returns
    -------
    tuple of (int, int)
        The common `(ncha, ncol)`; `(0, 0)` for an empty table.

    Raises
    ------
    UnsupportedLayoutError
        If the lines do not all share one shape. Select a single-shaped subset first —
        `.image`, `.refscan`, `.noise` usually separate them.
    """
    if len(table) == 0:
        return (0, 0)
    ncha, ncol = int(table["ncha"][0]), int(table["ncol"][0])
    if len(table) == 1:
        return (ncha, ncol)
    shapes, counts = np.unique(
        np.stack([table["ncha"], table["ncol"]], axis=1), axis=0, return_counts=True
    )
    if len(shapes) > 1:
        listed = ", ".join(
            f"(ncha={int(a)}, ncol={int(c)}) on {int(n)} lines"
            for (a, c), n in zip(shapes, counts)
        )
        raise UnsupportedLayoutError(
            f"these lines mix shapes: {listed}. A read returns one "
            f"(n_lines, ncha, ncol) array, so select a single-shaped subset first — "
            f"e.g. .image, .refscan or .noise."
        )
    return (ncha, ncol)


def read_headers(
    mm: np.ndarray, offsets: np.ndarray, version: TwixVersion
) -> np.ndarray:
    """Materialize the full VB/VD scan headers of the lines at `offsets`.

    Everything the compact table leaves out lives here: slice position and orientation,
    ICE program parameters (where custom non-Cartesian sequences usually stash
    trajectory or interleaf indices), timestamps, centre indices.

    Parameters
    ----------
    mm : numpy.ndarray
        The whole file as a ``uint8`` array.
    offsets : numpy.ndarray
        Absolute byte offsets of the scan headers to read, as in `LineTable.offset`.
    version : TwixVersion
        The layout detected by `detect_version`.

    Returns
    -------
    numpy.ndarray
        A contiguous `(len(offsets),)` array of `VB_HEADER` or `VD_SCAN_HEADER` records.
    """
    dtype = SCAN_HEADER_DTYPES[version]
    idx = (
        np.asarray(offsets, dtype=np.int64)[:, None]
        + np.arange(dtype.itemsize)[None, :]
    )
    return np.ascontiguousarray(mm[idx]).view(dtype).reshape(len(idx))


# ---------------------------------------------------------------------------
# 5. Sample extraction
# ---------------------------------------------------------------------------

# Un-reflecting is a fancy-index assignment, so it needs a temporary the size of the
# lines it touches; this bounds that temporary without bounding anything else.
_FLIP_BUDGET_BYTES = 2 * 1024 * 1024


def read_lines(
    mm: np.ndarray,
    table: np.ndarray,
    version: TwixVersion,
    *,
    out: np.ndarray | None = None,
    reflect: bool = True,
) -> np.ndarray:
    """Read every line in `table` into a `(len(table), ncha, ncol)` array.

    That shape *is* the file's own layout — one contiguous `(ncha, ncol)` block per
    line, in file order — so reading is a strided copy: no index arithmetic, no gather
    and no
    intermediate buffer, which is what makes an `out=` memmap enough to read files far
    larger than RAM.

    Parameters
    ----------
    mm : numpy.ndarray
        The whole file as a ``uint8`` array.
    table : numpy.ndarray
        A `LINE_DTYPE` array selecting the lines to read, as in `LineTable.rows`.
    version : TwixVersion
        The layout detected by `detect_version`.
    out : numpy.ndarray, optional
        Preallocated `(len(table), ncha, ncol)` complex64 destination — a
        `numpy.memmap`, a slice of a bigger array. Allocated if not given.
    reflect : bool, default True
        Un-reverse the lines flagged REFLECT (bipolar readouts store them backwards).
        Pass False for the samples exactly as laid down on disk.

    Returns
    -------
    numpy.ndarray
        The `(len(table), ncha, ncol)` complex64 samples; `out` itself when given.

    Raises
    ------
    ValueError
        If `out` does not have the required shape and dtype.
    UnsupportedLayoutError
        If `table` mixes `(ncha, ncol)`: the result would not be rectangular.
    """
    n = len(table)
    ncha, ncol = common_shape(table)
    shape = (n, ncha, ncol)
    if out is None:
        out = np.empty(shape, dtype=np.complex64)
    elif out.shape != shape or out.dtype != np.complex64:
        raise ValueError(f"out must be {shape} complex64, got {out.shape} {out.dtype}")
    if n == 0:
        return out

    prefix, chan_hdr = HEADER_SIZES[version]
    block = chan_hdr + 8 * ncol  # one channel: its header, then its samples
    # Index in complex64 units, not bytes: `build_table` has verified that every offset
    # involved is a multiple of 8.
    c8 = np.asarray(mm[: (mm.size // 8) * 8].view("<c8"))
    starts = (table["offset"].astype(np.int64) + prefix + chan_hdr) // 8

    # An index-array gather would need an int64 index per complex64 sample — as much
    # index traffic as data traffic — where a strided view is a plain memcpy. Evenly
    # spaced lines (the norm, since a measurement is usually one uniform run) are a
    # single view for the whole selection; an irregular selection is one view per line,
    # each still at a fixed within-line stride.
    step = int(starts[1] - starts[0]) if n > 1 else 0
    if step > 0 and np.all(np.diff(starts) == step):
        view = as_strided(
            c8[int(starts[0]) :], shape=shape, strides=(8 * step, block, 8)
        )
        np.copyto(out, view)
    else:
        for i, start in enumerate(starts):
            out[i] = as_strided(
                c8[int(start) :], shape=(ncha, ncol), strides=(block, 8)
            )

    if reflect:
        idx = np.flatnonzero(has_flag(table["flags"], Flag.REFLECT))
        chunk = max(1, _FLIP_BUDGET_BYTES // max(8 * ncha * ncol, 1))
        for at in range(0, idx.size, chunk):
            picked = idx[at : at + chunk]
            out[picked] = out[picked][:, :, ::-1]
    return out


# ---------------------------------------------------------------------------
# 6. The object model
# ---------------------------------------------------------------------------

# Anything carrying one of these is calibration, feedback or navigator data rather than
# imaging data. One combined mask instead of one boolean array per flag.
_NOT_IMAGING = (
    Flag.RTFEEDBACK
    | Flag.HPFEEDBACK
    | Flag.PHASCOR
    | Flag.NOISEADJSCAN
    | Flag.PHASESTABSCAN
    | Flag.REFPHASESTABSCAN
)
_NOT_PLAIN_REFERENCE = (
    Flag.PHASCOR
    | Flag.PHASESTABSCAN
    | Flag.REFPHASESTABSCAN
    | Flag.RTFEEDBACK
    | Flag.HPFEEDBACK
)


class LineTable:
    """The acquisition lines of one measurement, as a queryable table.

    Indexing returns another `LineTable`, so selections compose:
    `m.lines.image[m.lines.image.counter("Rep") == 0]`.

    Parameters
    ----------
    rows : numpy.ndarray
        A `LINE_DTYPE` array, one row per ADC line.
    mm : numpy.ndarray
        The whole file as a ``uint8`` array, kept so `headers` can re-read from it.
    version : TwixVersion
        The layout detected by `detect_version`.
    """

    def __init__(self, rows: np.ndarray, mm: np.ndarray, version: TwixVersion) -> None:
        self._rows = rows
        self._mm = mm
        self._version = version

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, key) -> LineTable:
        return LineTable(np.atleast_1d(self._rows[key]), self._mm, self._version)

    def __repr__(self) -> str:
        if len(self) == 0:
            return "LineTable(0 lines)"
        try:
            return f"LineTable({len(self)} lines, shape={self.shape})"
        except UnsupportedLayoutError:
            shapes = sorted(
                {
                    (int(a), int(c))
                    for a, c in zip(self._rows["ncha"], self._rows["ncol"])
                }
            )
            return f"LineTable({len(self)} lines, mixed shapes {shapes})"

    @property
    def rows(self) -> np.ndarray:
        """The raw `LINE_DTYPE` array (offset, flags, ncol, ncha, counters)."""
        return self._rows

    @property
    def offset(self) -> np.ndarray:
        """Absolute byte offset of each line's scan header."""
        return self._rows["offset"]

    @property
    def flags(self) -> np.ndarray:
        """Each line's 64-bit `EvalInfoMask`. Query it with `has` / `has_any`."""
        return self._rows["flags"]

    @property
    def counters(self) -> np.ndarray:
        """Structured `(N,)` array with one field per loop counter (see `COUNTERS`)."""
        return self._rows["counters"]

    def counter(self, name: str) -> np.ndarray:
        """One loop counter's value for every line.

        Parameters
        ----------
        name : str
            A counter name from `COUNTERS`, e.g. ``"Lin"``.

        Returns
        -------
        numpy.ndarray
            The `(N,)` int64 counter values.
        """
        return self._rows["counters"][name].astype(np.int64)

    @property
    def shape(self) -> tuple[int, int]:
        """The `(ncha, ncol)` shape shared by these lines.

        Raises
        ------
        UnsupportedLayoutError
            If the selection mixes shapes, as a measurement with an embedded reference
            scan does. Select a single-shaped subset (`.image`, `.refscan`, `.noise`).
        """
        return common_shape(self._rows)

    def headers(self) -> np.ndarray:
        """The full VB/VD scan headers of these lines, re-read from the file.

        Returns
        -------
        numpy.ndarray
            A `(len(self),)` array of `VB_HEADER` or `VD_SCAN_HEADER` records.
        """
        return read_headers(self._mm, self.offset, self._version)

    # -- selection ---------------------------------------------------------
    def has(self, flag: Flag) -> np.ndarray:
        """Boolean mask of lines carrying *all* bits of `flag`.

        Parameters
        ----------
        flag : Flag
            The bit or combination of bits to test for.

        Returns
        -------
        numpy.ndarray
            Boolean `(len(self),)` mask.
        """
        return has_flag(self.flags, flag)

    def has_any(self, flag: Flag) -> np.ndarray:
        """Boolean mask of lines carrying *any* bit of `flag`.

        Parameters
        ----------
        flag : Flag
            The bits to test for.

        Returns
        -------
        numpy.ndarray
            Boolean `(len(self),)` mask.
        """
        return has_any_flag(self.flags, flag)

    def select(self, flag: Flag) -> LineTable:
        """The lines carrying all bits of `flag`.

        Parameters
        ----------
        flag : Flag
            The bit or combination of bits to select on.

        Returns
        -------
        LineTable
            The matching lines.
        """
        return self[self.has(flag)]

    @property
    def image(self) -> LineTable:
        """Imaging lines: all but calibration, feedback and reference-only lines."""
        return self[~self.has_any(_NOT_IMAGING) & ~self._reference_only]

    @property
    def noise(self) -> LineTable:
        """Noise-calibration lines (for pre-whitening)."""
        return self.select(Flag.NOISEADJSCAN)

    @property
    def refscan(self) -> LineTable:
        """Parallel-imaging reference lines."""
        is_ref = self.has_any(Flag.PATREFSCAN | Flag.PATREFANDIMASCAN)
        return self[is_ref & ~self.has_any(_NOT_PLAIN_REFERENCE)]

    @property
    def phasecor(self) -> LineTable:
        """Phase-correction (navigator) lines."""
        return self[self.has(Flag.PHASCOR) & ~self._reference_only]

    @property
    def _reference_only(self) -> np.ndarray:
        """PATREFSCAN without PATREFANDIMASCAN: reference that is not also image."""
        return self.has(Flag.PATREFSCAN) & ~self.has(Flag.PATREFANDIMASCAN)


def _varying_counters(lines: LineTable) -> tuple[str, ...]:
    """The loop counters that are not constant over `lines`, in header order.

    A counter varies iff its min differs from its max — two reductions, not a sort.
    Taken over the counter block as one contiguous `(n, 14)` uint16 array rather than
    field by field, since 14 passes strided across 48-byte rows cost twice what one
    compacted pass does (50 ms vs 25 ms on a million lines).

    Never empty: with nothing varying there is no grid at all, and `("Lin",)` gives the
    degenerate size-1 axis instead of a rank-0 one.

    Parameters
    ----------
    lines : LineTable
        The lines to inspect.

    Returns
    -------
    tuple of str
        The names of the varying counters, in header order; ``("Lin",)`` if none vary.
    """
    if len(lines) == 0:
        return ("Lin",)
    flat = np.ascontiguousarray(lines.counters).view("<u2")
    counters = np.reshape(flat, (len(lines), len(COUNTERS)))
    lo, hi = counters.min(0), counters.max(0)
    varying = tuple(name for name, a, b in zip(COUNTERS, lo, hi, strict=True) if a != b)
    return varying or ("Lin",)


def minimal_dims(lines: LineTable) -> tuple[str, ...]:
    """The varying counters, minus those the others already determine.

    `varying_counters` never collides, but takes the *product* of the counter ranges
    even when they are correlated: an EPI `Seg` that merely tracks `Lin` parity doubles
    the grid and leaves half of it empty. A counter can be dropped exactly when the
    remaining ones still identify every line — i.e. when it is a function of them —
    which is one uniqueness test per candidate, tried largest-first and stopped as soon
    as the grid is perfectly packed.

    Nothing is *lost* by dropping a determined counter: it is recoverable from what is
    kept, by definition. What is lost is an axis to slice on — folding EPI on `Lin`
    alone leaves no way to address the two readout polarities separately — so this is
    opt-in, not
    the meaning of `dims=None`.

    Parameters
    ----------
    lines : LineTable
        The lines the grid is to be built from.

    Returns
    -------
    tuple of str
        The counter names to fold on, a subset of `_varying_counters`.
    """
    dims = list(_varying_counters(lines))
    n = len(lines)
    columns = {d: lines.counter(d) - int(lines.counter(d).min()) for d in dims}
    sizes = {d: int(columns[d].max()) + 1 for d in dims}
    slots = int(np.prod([sizes[d] for d in dims]))
    if slots >= 2**62:
        return tuple(dims)  # the packed key would overflow int64; not worth a wider one

    for candidate in sorted(dims, key=lambda d: -sizes[d]):
        if slots == n or len(dims) == 1:
            break  # perfectly packed, or down to the last axis: nothing left to gain
        trial = [d for d in dims if d != candidate]
        key = np.zeros(n, dtype=np.int64)
        for d in trial:
            key = key * sizes[d] + columns[d]
        if len(np.unique(key)) == n:
            dims, slots = trial, slots // sizes[candidate]
    return tuple(dims)


def to_dense(
    samples: np.ndarray,
    lines: LineTable,
    dims: tuple[str, ...] | str = "minimal",
) -> np.ndarray:
    """Fold `(n_lines, ncha, ncol)` samples onto a grid indexed by loop counters.

    Parameters
    ----------
    samples : numpy.ndarray
        The `(n_lines, ncha, ncol)` samples, as returned by `read_lines`.
    lines : LineTable
        The lines `samples` was read from, in the same order.
    dims : tuple of str or {'minimal'}, default 'minimal'
        The counter names to use as grid axes. ``"minimal"`` drops the counters the
        others already determine (`minimal_dims`), which packs the grid but costs an
        axis you may have wanted to slice on. Naming the dims keeps both the rank and
        the memory predictable.

    Returns
    -------
    numpy.ndarray
        An array shaped `(*dim_sizes, ncha, ncol)`, zero where no line was acquired.

    Raises
    ------
    ValueError
        If `samples` and `lines` differ in length, or if explicit `dims` leave several
        lines landing on one grid position.
    """
    if len(samples) != len(lines):
        raise ValueError(f"samples has {len(samples)} lines, table has {len(lines)}")
    minimal = False
    if dims == "minimal":
        dims = minimal_dims(lines)
        minimal = True

    idx = [lines.counter(name) for name in dims]
    # shift the grid so the acquired range starts at 0.
    idx = [values - int(values.min()) for values in idx]
    sizes = [int(values.max()) + 1 for values in idx]
    flat = np.ravel_multi_index(tuple(idx), sizes)

    n_flat = int(np.prod(sizes))
    dense = np.zeros((n_flat, *samples.shape[1:]), dtype=samples.dtype)
    counts = np.bincount(flat, minlength=n_flat)

    if counts.max(initial=0) > 1 and not minimal:
        # probably going to delete all of this
        varying = [
            name
            for name in COUNTERS
            if name not in dims and len(np.unique(lines.counter(name))) > 1
        ]
        raise ValueError(
            f"{int((counts > 1).sum())} grid positions receive more than one line. "
            + (f"Counters varying but not in dims: {varying}.")
        )
    else:
        dense[flat] = samples

    return dense.reshape(*sizes, *samples.shape[1:])


class Measurement:
    """One measurement (raid entry) inside a `.dat` file.

    Parameters
    ----------
    file : TwixFile
        The open file this measurement belongs to.
    entry : RaidEntry
        Its directory entry: id, offset, length and names.
    index : int
        Its position in the file, in acquisition order.
    """

    def __init__(self, file: TwixFile, entry: RaidEntry, index: int) -> None:
        self._file = file
        self.index = index
        (
            self.meas_id,
            self.offset,
            self.length,
            self.patient_name,
            self.protocol_name,
        ) = entry

    def __repr__(self) -> str:
        return (
            f"Measurement(index={self.index}, protocol_name={self.protocol_name!r}, "
            f"{self.length / 2**20:.1f} MiB)"
        )

    @functools.cached_property
    def _header(self) -> tuple[Protocol, int]:
        return parse_protocol(self._file.mm, self.offset)

    @property
    def hdr(self) -> Protocol:
        """The parsed text protocol; each buffer is parsed on first access."""
        return self._header[0]

    @functools.cached_property
    def lines(self) -> LineTable:
        """The acquisition lines of this measurement.

        Raises
        ------
        TruncatedFileError
            If the measurement ends before ACQEND and the file was not opened with
            ``allow_truncated=True``.

        Warns
        -----
        UserWarning
            When a truncated measurement is read anyway, giving the line count kept.
        """
        data_start = self.offset + self._header[1]
        rows, truncated = build_table(
            self._file.mm, data_start, self.offset + self.length, self._file.version
        )
        if truncated:
            message = (
                f"measurement {self.index} ended before ACQEND after {len(rows)} lines"
            )
            if not self._file.allow_truncated:
                raise TruncatedFileError(
                    f"{message}; pass allow_truncated=True to read anyway"
                )
            warnings.warn(f"{message}; using the acquired lines only", stacklevel=2)
        return LineTable(rows, self._file.mm, self._file.version)

    def read(
        self,
        lines: LineTable | None = None,
        *,
        out: np.ndarray | None = None,
        reflect: bool = True,
    ) -> np.ndarray:
        """Read the samples of `lines` (default: all of them) into `(n, ncha, ncol)`.

        Parameters
        ----------
        lines : LineTable, optional
            The lines to read; every line of the measurement by default.
        out : numpy.ndarray, optional
            Preallocated `(n, ncha, ncol)` complex64 destination — a `numpy.memmap`, a
            slice of a bigger array — which is what makes files larger than RAM
            readable, since the copy goes straight from the mapped file into it with
            nothing in between.
        reflect : bool, default True
            Un-reverse the lines flagged REFLECT.

        Returns
        -------
        numpy.ndarray
            The `(n, ncha, ncol)` complex64 samples; `out` itself when given.
        """
        table = self.lines if lines is None else lines
        return read_lines(
            self._file.mm, table.rows, self._file.version, out=out, reflect=reflect
        )

    def to_dense(
        self,
        lines: LineTable | None = None,
        dims: tuple[str, ...] | Literal["minimal"] = "minimal",
        *,
        reflect: bool = True,
    ) -> np.ndarray:
        """Read the k-space data and return a dense array of it.

        Parameters
        ----------
        lines : LineTable, optional
            The lines to fold; the imaging lines of the measurement by default.
        dims : tuple of str or str, optional
            The counter names to use as grid axes, as for `to_dense`.
        reflect : bool, default True
            Un-reverse the lines flagged REFLECT.

        Returns
        -------
        numpy.ndarray
            An array shaped `(*dim_sizes, ncha, ncol)`.
        """
        table = self.lines.image if lines is None else lines
        samples = self.read(table, reflect=reflect)
        return to_dense(samples, table, dims)


class TwixFile:
    """A memory-mapped `.dat` file and the measurements it contains.

    Indexable and iterable over its `Measurement` objects, in acquisition order. `scan`
    is the last of them — the actual scan, rather than the adjustments before it — and
    `hdr`, `lines`, `read` and `to_dense` are shorthands for its members, so the common
    case needs no indexing at all.

    Parameters
    ----------
    path : str
        Path to the `.dat` file.
    allow_truncated : bool, default False
        Read a measurement that ends before ACQEND, warning instead of raising.

    Raises
    ------
    TwixParseError
        If the file holds no non-empty measurement.
    UnsupportedVersionError
        If the file does not look like a VB or VD/VE twix file.

    Examples
    --------
    >>> f = open_twix("meas.dat")
    >>> samples = f.read(f.lines.image)  # the last measurement
    >>> noise = f[0].lines.noise  # an earlier one, explicitly
    """

    def __init__(self, path: str, *, allow_truncated: bool = False) -> None:
        self.path = path
        self.allow_truncated = allow_truncated
        self._mm = open_mmap(path)
        self.version = detect_version(self.mm)
        # Zero-length entries are aborted measurements with no data written; the
        # complete ones around them stay readable.
        self.measurements = [
            Measurement(self, entry, index)
            for index, entry in enumerate(parse_raid_directory(self.mm, self.version))
            if entry.length > 0
        ]
        if not self.measurements:
            raise TwixParseError(f"{path}: no non-empty measurement")

    @property
    def mm(self) -> np.ndarray:
        """The mapped file. Raises `ValueError` once `close` has been called."""
        if self._mm is None:
            raise ValueError(f"{self.path} is closed")
        return self._mm

    def close(self) -> None:
        """Drop this object's reference to the mapping.

        The mapping itself goes away once nothing else holds it — a `LineTable` keeps
        its own reference, so tables built before `close` stay readable. Nothing
        turbotwix returns is a view of the file (samples, headers and protocol buffers
        are all copies), so closing can never invalidate data you already hold.

        Optional: the mapping is released on garbage collection anyway. Call it, or use
        the file as a context manager, when you want the descriptor released at a
        definite point — looping over many files, or on Windows, where an open mapping
        blocks renaming and deleting the `.dat`.
        """
        self._mm = None

    def __enter__(self) -> TwixFile:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    @property
    def scan(self) -> Measurement:
        """The last measurement — the one you almost always want.

        A `.dat` file is written in acquisition order, and a scan is commonly preceded
        by the adjustments it needed (coil sensitivity, noise). The interesting one is
        therefore the last, and `hdr`, `lines`, `read` and `to_dense` on the file are
        shorthands for the same members of this measurement. Index the file (`f[0]`,
        `f[-2]`) or iterate it to reach the others.
        """
        return self.measurements[-1]

    @property
    def hdr(self) -> Protocol:
        """`scan.hdr` — the parsed text protocol of the last measurement."""
        return self.scan.hdr

    @property
    def lines(self) -> LineTable:
        """`scan.lines` — the acquisition lines of the last measurement."""
        return self.scan.lines

    def read(
        self,
        lines: LineTable | None = None,
        *,
        out: np.ndarray | None = None,
        reflect: bool = True,
    ) -> np.ndarray:
        """`scan.read(...)` — read the last measurement's samples.

        See `Measurement.read`.
        """
        return self.scan.read(lines, out=out, reflect=reflect)

    def to_dense(
        self,
        lines: LineTable | None = None,
        dims: tuple[str, ...] | Literal["minimal"] = "minimal",
        *,
        reflect: bool = True,
    ) -> np.ndarray:
        """`scan.to_dense(...)` — fold the last measurement onto a grid.

        See `Measurement.to_dense`.
        """
        return self.scan.to_dense(lines, dims, reflect=reflect)

    def __len__(self) -> int:
        return len(self.measurements)

    def __getitem__(self, index: int) -> Measurement:
        return self.measurements[index]

    def __iter__(self):
        return iter(self.measurements)

    def __repr__(self) -> str:
        names = [m.protocol_name for m in self.measurements]
        return (
            f"TwixFile({self.path!r}, version={self.version.name}, "
            f"measurements={names})"
        )


def open_twix(path: str, *, allow_truncated: bool = False) -> TwixFile:
    """Open a Siemens `.dat` (TWIX) file.

    Nothing is read beyond the raid directory: protocols and line tables are parsed per
    measurement on first access.

    Every measurement in the file is kept, in acquisition order. Reads default to the
    last one (`TwixFile.scan`), which is the scan itself — the measurements before it
    are the adjustments it needed — while `f[0]`, `f[-2]` and iteration reach the rest.

    Closing is optional; see `TwixFile.close`.

    Parameters
    ----------
    path : str
        Path to the `.dat` file.
    allow_truncated : bool, default False
        Read a measurement that ends before ACQEND, warning instead of raising.

    Returns
    -------
    TwixFile
        The open file, indexable and iterable over its `Measurement` objects.
    """
    return TwixFile(path, allow_truncated=allow_truncated)
