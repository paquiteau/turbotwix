"""Public object model: `read_twix`, `TwixScan`, `TwixArray`."""

from __future__ import annotations

import numpy as np

from turbotwix import _extract, _format, _read, protocol
from turbotwix._format import (
    TruncatedFileError,
    TwixParseError,
    TwixVersion,
    UnsupportedVersionError,
)

__all__ = [
    "TwixArray",
    "TwixScan",
    "TwixParseError",
    "TruncatedFileError",
    "UnsupportedVersionError",
    "read_twix",
]

# Categories that keep the full nominal Lin/Par matrix size; every other category is
# shrunk to its acquired range (pymapvbvd's `flagSkipToFirstLine` default).
_NO_SKIP_CATEGORIES = frozenset({"image", "phasestab"})


class TwixArray:
    """Lazy accessor for one scan category's k-space data.

    Nothing is read from disk until `.data`/`[...]` is accessed. Unlike pymapvbvd's
    array object, indexing here always reads the whole category first (vectorized) and
    then slices in memory — simpler, and the read itself is the fast part; there is no
    partial/streaming read path in this v1.
    """

    def __init__(
        self,
        mm: np.memmap,
        table: np.ndarray,
        mask: np.ndarray,
        version: TwixVersion,
        name: str,
        *,
        remove_os: bool = False,
        squeeze: bool = False,
        legacy_layout: bool = False,
    ) -> None:
        self._mm = mm
        self._table = table
        self._mask = mask
        self._version = version
        self.name = name
        self.remove_os = remove_os
        self.squeeze = squeeze
        self.legacy_layout = legacy_layout
        self._skip_to_first_line = name not in _NO_SKIP_CATEGORIES
        self.n_acq = int(mask.sum())

    @property
    def shape(self) -> tuple[int, ...]:
        shape = _extract.category_shape(
            self._table, self._mask, self._skip_to_first_line, self.remove_os, self.legacy_layout
        )
        return tuple(s for s in shape if s > 1) if self.squeeze else shape

    @property
    def dims(self) -> tuple[str, ...]:
        names = (
            ("Col", "Cha", *_extract.DIM_NAMES)
            if self.legacy_layout
            else (*_extract.DIM_NAMES, "Cha", "Col")
        )
        if not self.squeeze:
            return names
        full = _extract.category_shape(
            self._table, self._mask, self._skip_to_first_line, self.remove_os, self.legacy_layout
        )
        return tuple(n for n, s in zip(names, full) if s > 1)

    @property
    def data(self) -> np.ndarray:
        arr = _extract.read_category(
            self._mm,
            self._table,
            self._mask,
            self._version,
            remove_os=self.remove_os,
            skip_to_first_line=self._skip_to_first_line,
            legacy_layout=self.legacy_layout,
        )
        return np.squeeze(arr) if self.squeeze else arr

    def __getitem__(self, key) -> np.ndarray:
        return self.data[key]

    def __array__(self, dtype=None) -> np.ndarray:
        arr = self.data
        return arr.astype(dtype, copy=False) if dtype is not None else arr

    def __repr__(self) -> str:
        return f"TwixArray(name={self.name!r}, shape={self.shape}, n_acq={self.n_acq})"


class TwixScan:
    """One measurement: parsed protocol header plus one `TwixArray` per scan category
    that was actually present in the file (matching pymapvbvd, empty categories are
    omitted).
    """

    def __init__(self, hdr: protocol.AttrDict, arrays: dict[str, TwixArray]) -> None:
        self.hdr = hdr
        self._arrays = arrays
        for name, arr in arrays.items():
            setattr(self, name, arr)

    def __getitem__(self, name: str):
        if name == "hdr":
            return self.hdr
        return self._arrays[name]

    def category_names(self) -> list[str]:
        return list(self._arrays)

    def __repr__(self) -> str:
        return f"TwixScan(categories={self.category_names()!r})"


def read_twix(
    path: str,
    scans: str = "last",
    *,
    remove_os: bool = False,
    squeeze: bool = False,
    legacy_layout: bool = False,
):
    """Read a Siemens .dat (TWIX) file.

    Parameters
    ----------
    path: path to the .dat file.
    scans: "last" (default, matches pymapvbvd) to return only the final measurement,
        or "all" to return a list of `TwixScan`, one per scan stored in the file.
    remove_os: crop out 2x readout oversampling (FFT-based), applied per category.
    squeeze: drop size-1 dimensions from each category's array/shape.
    legacy_layout: return arrays in pymapvbvd's (Col, Cha, Lin, Par, ...) axis order
        instead of the default (Lin, Par, ..., Cha, Col). The default order is the one
        the fast gather/scatter path produces natively (see `_extract.py`); this flag
        pays one extra full-array transpose+copy to match pymapvbvd's convention.

    Returns
    -------
    A single `TwixScan`, or a list of them if `scans="all"`.
    """
    mm = _read.open_mmap(path)
    version = _format.detect_version(mm)
    raid_entries = _format.parse_raid_directory(mm, version)

    if scans == "last":
        selected = [raid_entries[-1]]
    elif scans == "all":
        selected = raid_entries
    else:
        raise ValueError("scans must be 'last' or 'all'")

    results = []
    for entry in selected:
        hdr, hdr_len = protocol.parse_protocol(mm, entry.offset)
        data_start = entry.offset + hdr_len
        data_end = entry.offset + entry.length
        table, truncated = _read.walk_headers(mm, data_start, data_end, version)
        _read.require_complete(table, truncated)
        categories = _read.classify(table)

        arrays = {}
        for name, mask in categories.items():
            if not mask.any():
                continue
            arrays[name] = TwixArray(
                mm,
                table,
                mask,
                version,
                name,
                remove_os=remove_os,
                squeeze=squeeze,
                legacy_layout=legacy_layout,
            )
        results.append(TwixScan(hdr, arrays))

    return results[0] if len(results) == 1 else results
