#!/usr/bin/env python3
"""The text protocol: ascconv / XProtocol parsing, lazy per buffer.

`docs/twix-format.md` describes the format itself.
"""

from __future__ import annotations

import re
import struct

import numpy as np

__all__ = ["AttrDict", "Protocol", "parse_protocol"]

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
