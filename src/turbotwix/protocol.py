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
_XPROT_SCALAR_PATTERN = re.compile(r'<Param(?:Bool|Long|String)\."(\w+)">\s*{([^}]*)')
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


def _try_cast(value: str, key: str):
    if key.startswith("t"):
        return value.strip('"')
    if key.startswith("b"):
        return bool(value)
    if key.startswith("l") or key.startswith("ul"):
        try:
            return int(value)
        except ValueError:
            return value
    if key.startswith("uc"):
        try:
            return int(value, 16) if value.startswith("0x") else int(value)
        except ValueError:
            return value
    if key == "PatientID":
        return value
    try:
        return float(value)
    except ValueError:
        return value


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
        prot[last_key] = value
        return

    name = last_key[1:] if last_key.startswith("a") else last_key
    prot[last_key] = _try_cast(value, name)


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
    tokens = list(_XPROT_SCALAR_PATTERN.finditer(buffer)) + list(
        _XPROT_DOUBLE_PATTERN.finditer(buffer)
    )
    for match in tokens:
        name = match.group(1)
        value = re.sub(r'("*)|( *<\w*> *[^\n]*)', "", match.groups()[-1])
        value = re.sub(r"[\t\n\r\f\v]*", "", value.strip())
        if name.startswith("a"):
            xprot[name] = [_try_cast(v, name[1:]) for v in value.split()]
        else:
            xprot[name] = _try_cast(value, name)
    return xprot


def _parse_buffer(buffer: str) -> dict:
    ascconv_match = _ASCCONV_PATTERN.search(buffer)
    prot = _parse_ascconv(ascconv_match.group(0)) if ascconv_match else {}
    remainder = "".join(_ASCCONV_PATTERN.split(buffer))
    prot.update(_parse_xprot(remainder))
    return prot


def parse_protocol(mm: np.memmap, scan_offset: int) -> tuple[AttrDict, int]:
    """Parse the text header of the scan starting at `scan_offset`.

    Returns `(protocol, hdr_len)`: `protocol` maps each buffer name (Config, Dicom,
    Meas, MeasYaps, Phoenix, Spice, ...) to its parsed dict; `hdr_len` is the total
    byte length of the text header (MDH data for this scan starts at
    `scan_offset + hdr_len`).
    """
    hdr_len, n_buffer = (
        int(v) for v in np.frombuffer(mm, dtype="<u4", count=2, offset=scan_offset)
    )
    pos = scan_offset + 8
    protocol = AttrDict()
    for _ in range(n_buffer):
        chunk = bytes(mm[pos : pos + 48])
        match = _BUFFER_NAME_PATTERN.search(chunk)
        if match is None:
            break
        name = match.group(1).decode("latin1")
        (buf_len,) = struct.unpack("<I", match.group(2))
        pos += len(match.group(0))
        buf = bytes(mm[pos : pos + buf_len]).decode("latin1")
        protocol[name] = _parse_buffer(buf)
        pos += buf_len
    return protocol, hdr_len
