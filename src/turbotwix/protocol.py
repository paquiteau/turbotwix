"""Parser for the Siemens ascconv/XProtocol text header embedded in each TWIX scan.

Adapted from twixtools' `twixprot.py`, which in turn credits pymapvbvd for most of the
string-parsing approach (see NOTICE). Pure text parsing, no file I/O here — the caller
slices the raw header bytes out of the memory-mapped file.
"""

from __future__ import annotations

import re
import struct

import numpy as np

_BUFFER_NAME_PATTERN = re.compile(rb"(\w{4,})\x00(.{4})", re.DOTALL)
_ASCCONV_PATTERN = re.compile(r"### ASCCONV BEGIN[^\n]*\n(.*)\s### ASCCONV END ###", re.DOTALL)
_ASCCONV_LINE_PATTERN = re.compile(r"(?P<name>\S*)\s*=\s*(?P<value>\S*)\n")
_ASCCONV_KEY_PATTERN = re.compile(r"(?P<name>\w+)(\[(?P<ix>[0-9]+)\])?")
_XPROT_SCALAR_PATTERN = re.compile(r'<Param(Bool|Long|String)\."(\w+)">\s*{([^}]*)')
_XPROT_DOUBLE_PATTERN = re.compile(
    r'<ParamDouble\."(\w+)">\s*{\s*(<Precision>\s*[0-9]*)?\s*([^}]*)'
)


class AttrDict(dict):
    """Minimal dict that also allows attribute-style access, recursively."""

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
    it costs ~40 ms — which is most of the time spent reading a small file, and pure
    waste for callers that only want k-space. So each value starts out as the raw bytes
    and is replaced by its parsed `AttrDict` the first time it is looked up. Every read
    path (`p["Meas"]`, `p.Meas`, `.get`, `.values`, `.items`, `dict(p)`) parses on the
    way through, so the laziness is not observable apart from where the time is spent.
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
        # path that reads the underlying table directly, and would hand back raw bytes.
        # Overriding tp_iter forces the generic path, which goes through __getitem__.
        return dict.__iter__(self)


_INT_PATTERN = re.compile(r"[+-]?\d+$")
_HEX_PATTERN = re.compile(r"[+-]?0[xX][0-9a-fA-F]+$")
# Explicit, because Python's own int()/float() accept `_` as a digit separator: the
# scanner ID "6_0_66327775_20210622_151635_650" parses as 6.07e+26 rather than failing,
# so a bare `try: float(value)` silently turns identifiers into numbers.
_FLOAT_PATTERN = re.compile(r"[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")


def _as_number(text: str, integer: bool = False):
    """`text` as int/float if it is syntactically a number, else None."""
    if _HEX_PATTERN.match(text):
        return int(text, 16)
    if _INT_PATTERN.match(text):
        return int(text)
    if _FLOAT_PATTERN.match(text):
        return int(float(text)) if integer else float(text)
    return None


def _cast_value(value: str):
    """Type an ascconv value from its own syntax.

    Not from the key's Hungarian prefix, which is what pymapvbvd and twixtools do: that
    guesses (`b` -> bool, `l` -> long, ...) and is wrong whenever a sequence author
    named a variable freely, silently returning the raw string for an int field or True
    for a flag holding "0". Quoting, `0x`, a decimal point and a sign are unambiguous;
    where the text says nothing, the text is the value.
    """
    text = value.strip()
    if text.startswith('"'):
        return text.strip('"')
    number = _as_number(text)
    return text if number is None else number


def _cast_xprot(value: str, kind: str):
    """Type an XProtocol value from its declared tag (`<ParamLong."x">`), which — unlike
    ascconv — actually states the type.
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
    if "__attribute__" in key:
        return
    if len(key) > 1:
        if isinstance(key[0], int):
            while len(prot) < key[0] + 1:
                prot.append(list() if isinstance(key[1], int) else dict())
            _update_ascconv(prot[key[0]], key[1:], value)
        else:
            if key[0] not in prot:
                prot[key[0]] = list() if isinstance(key[1], int) else dict()
            _update_ascconv(prot[key[0]], key[1:], value)
        return

    last_key = key[0]
    if isinstance(last_key, int):
        while len(prot) < last_key + 1:
            prot.append(list())
        prot[last_key] = _cast_value(value)
        return

    prot[last_key] = _cast_value(value)


def _parse_ascconv(buffer: str) -> dict:
    mrprot: dict = {}
    for line in _ASCCONV_LINE_PATTERN.finditer(buffer):
        key: list = []
        for part in _ASCCONV_KEY_PATTERN.finditer(line.group("name")):
            key.append(part.group("name"))
            if part.group("ix") is not None:
                key.append(int(part.group("ix")))
        if key:
            _update_ascconv(mrprot, key, line.group("value"))
    return mrprot


def _parse_xprot(buffer: str) -> dict:
    xprot: dict = {}
    tokens = [(m, m.group(1)) for m in _XPROT_SCALAR_PATTERN.finditer(buffer)]
    tokens += [(m, "Double") for m in _XPROT_DOUBLE_PATTERN.finditer(buffer)]
    for match, kind in tokens:
        name = match.group(2) if kind != "Double" else match.group(1)
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
    ascconv_match = _ASCCONV_PATTERN.search(buffer)
    prot = _parse_ascconv(ascconv_match.group(0)) if ascconv_match else {}
    # `.split()` would re-insert the body (the pattern has a capture group); `.sub()`
    # actually drops the ascconv section, so _parse_xprot only scans XProtocol text.
    remainder = _ASCCONV_PATTERN.sub("", buffer)
    prot.update(_parse_xprot(remainder))
    return prot


def parse_protocol(mm: np.memmap, scan_offset: int) -> tuple[Protocol, int]:
    """Split the text header of the scan starting at `scan_offset` into its buffers.

    Returns `(protocol, hdr_len)`: `protocol` maps each buffer name (Config, Dicom,
    Meas, MeasYaps, Phoenix, Spice, ...) to its parsed dict, parsed on first access
    (see `Protocol`); `hdr_len` is the total byte length of the text header (MDH data
    for this scan starts at `scan_offset + hdr_len`).
    """
    hdr_len, n_buffer = (
        int(v) for v in np.frombuffer(mm, dtype="<u4", count=2, offset=scan_offset)
    )
    pos = scan_offset + 8
    protocol = Protocol()
    for _ in range(n_buffer):
        chunk = bytes(mm[pos : pos + 48])
        match = _BUFFER_NAME_PATTERN.search(chunk)
        if match is None:
            break
        name = match.group(1).decode("latin1")
        (buf_len,) = struct.unpack("<I", match.group(2))
        pos += match.end()
        dict.__setitem__(protocol, name, bytes(mm[pos : pos + buf_len]))
        pos += buf_len
    return protocol, hdr_len
