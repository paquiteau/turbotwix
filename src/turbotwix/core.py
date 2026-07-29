"""Public object model: `open_twix` -> `TwixFile` -> `Measurement` -> `LineTable`.

The data model is the file's own: a measurement is a *list of acquisition lines*, each
with its metadata and its `(ncha, ncol)` block of samples. Selecting lines is a boolean
query over that table; reading returns `(n_lines, ncha, ncol)`.

Folding lines onto a Cartesian grid is available (`Measurement.to_dense`) but is never
implicit, because for non-Cartesian acquisitions there is no grid to fold onto — the
loop counters index shots, interleaves or spokes. Making the dense hypercube the
default (as pymapvbvd and twixtools do) costs memory proportional to the *nominal*
matrix rather than to the acquired data, forces a choice about duplicate indices, and
makes partial reads impossible.
"""

from __future__ import annotations

import functools
import warnings

import numpy as np

from turbotwix import _extract, _format, _read, protocol
from turbotwix._format import (
    Flag,
    TruncatedFileError,
    TwixParseError,
    TwixVersion,
    UnsupportedVersionError,
)

__all__ = [
    "Flag",
    "LineTable",
    "Measurement",
    "TruncatedFileError",
    "TwixFile",
    "TwixParseError",
    "UnsupportedVersionError",
    "open_twix",
    "remove_oversampling",
    "to_dense",
]

remove_oversampling = _extract.remove_oversampling

#: Names of the 14 loop counters carried by every line header, in header order.
COUNTERS: tuple[str, ...] = tuple(_format.LINE_COUNTER.names or ())

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
    Flag.PHASCOR | Flag.PHASESTABSCAN | Flag.REFPHASESTABSCAN | Flag.RTFEEDBACK | Flag.HPFEEDBACK
)


class LineTable:
    """The acquisition lines of one measurement, as a queryable table.

    Indexing returns another `LineTable`, so selections compose:
    `m.lines.image[m.lines.image.counter("Rep") == 0]`.
    """

    def __init__(self, rows: np.ndarray, mm: np.ndarray, version: TwixVersion) -> None:
        self._rows = rows
        self._mm = mm
        self._version = version

    # -- structure ---------------------------------------------------------
    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, key) -> LineTable:
        rows = self._rows[key]
        return LineTable(np.atleast_1d(rows), self._mm, self._version)

    def __repr__(self) -> str:
        if len(self) == 0:
            return "LineTable(0 lines)"
        return f"LineTable({len(self)} lines, shapes={sorted(self.shapes)})"

    @property
    def rows(self) -> np.ndarray:
        """The raw `_format.LINE_DTYPE` array (offset, flags, ncol, ncha, counters)."""
        return self._rows

    # -- per-line fields ---------------------------------------------------
    @property
    def offset(self) -> np.ndarray:
        return self._rows["offset"]

    @property
    def flags(self) -> np.ndarray:
        return self._rows["flags"]

    @property
    def ncol(self) -> np.ndarray:
        return self._rows["ncol"]

    @property
    def ncha(self) -> np.ndarray:
        return self._rows["ncha"]

    @property
    def counters(self) -> np.ndarray:
        """Structured (N,) array with one field per loop counter (see `COUNTERS`)."""
        return self._rows["counters"]

    def counter(self, name: str) -> np.ndarray:
        return self._rows["counters"][name].astype(np.int64)

    @property
    def shapes(self) -> set[tuple[int, int]]:
        """The distinct `(ncha, ncol)` shapes present."""
        return {(int(a), int(c)) for a, c in zip(self.ncha, self.ncol)}

    def by_shape(self) -> dict[tuple[int, int], LineTable]:
        """Split into one `LineTable` per `(ncha, ncol)`, so each can be read.

        Flags do not always separate distinct acquisitions: a coil-sensitivity
        adjustment stores its body-coil (ncha=2) and array-coil (ncha=44) images both
        as plain ONLINE image lines, told apart only by shape.
        """
        return {
            shape: self[(self.ncha == shape[0]) & (self.ncol == shape[1])]
            for shape in sorted(self.shapes)
        }

    def headers(self) -> np.ndarray:
        """The full VB/VD scan headers of these lines, re-read from the file.

        Everything the compact table leaves out lives here: slice position and
        orientation, ICE program parameters (where custom non-Cartesian sequences
        usually stash trajectory or interleaf indices), timestamps, centre indices.
        """
        return _read.read_headers(self._mm, self.offset, self._version)

    # -- selection ---------------------------------------------------------
    def has(self, flag: Flag) -> np.ndarray:
        """Boolean mask of lines carrying *all* bits of `flag`."""
        return _format.has_flag(self.flags, flag)

    def has_any(self, flag: Flag) -> np.ndarray:
        """Boolean mask of lines carrying *any* bit of `flag`."""
        return _format.has_any_flag(self.flags, flag)

    def select(self, flag: Flag) -> LineTable:
        """The lines carrying all bits of `flag`."""
        return self[self.has(flag)]

    @property
    def data(self) -> LineTable:
        """Everything except the end-of-acquisition marker and PMU/sync blocks."""
        return self[~self.has_any(Flag.ACQEND | Flag.SYNCDATA)]

    @property
    def image(self) -> LineTable:
        """Imaging lines: data, minus calibration, feedback and reference-only lines."""
        rows = self.data
        return rows[~rows.has_any(_NOT_IMAGING) & ~rows._reference_only]

    @property
    def noise(self) -> LineTable:
        """Noise-calibration lines (for pre-whitening)."""
        return self.data.select(Flag.NOISEADJSCAN)

    @property
    def refscan(self) -> LineTable:
        """Parallel-imaging reference lines."""
        rows = self.data
        is_ref = rows.has_any(Flag.PATREFSCAN | Flag.PATREFANDIMASCAN)
        return rows[is_ref & ~rows.has_any(_NOT_PLAIN_REFERENCE)]

    @property
    def phasecor(self) -> LineTable:
        """Phase-correction (navigator) lines."""
        rows = self.data
        return rows[rows.has(Flag.PHASCOR) & ~rows._reference_only]

    @property
    def _reference_only(self) -> np.ndarray:
        """PATREFSCAN without PATREFANDIMASCAN: reference data that is not also image."""
        return self.has(Flag.PATREFSCAN) & ~self.has(Flag.PATREFANDIMASCAN)


def to_dense(
    samples: np.ndarray,
    lines: LineTable,
    dims: tuple[str, ...] = ("Lin", "Par"),
    *,
    reduce: str = "error",
    origin: str = "min",
) -> np.ndarray:
    """Fold `(n_lines, ncha, ncol)` samples onto a grid indexed by loop counters.

    Returns an array shaped `(*dim_sizes, ncha, ncol)`.

    `reduce` decides what happens when several lines land on the same index — which
    normally means a counter is missing from `dims`, not that the data wants averaging.
    So the default is to raise and say which; pass "mean", "sum" or "last" to opt in.
    `origin="min"` shifts each axis so the acquired range starts at 0 (the useful
    default for reference and navigator scans, which cover part of k-space);
    `origin="zero"` keeps the raw counter values, i.e. the nominal matrix.
    """
    if len(samples) != len(lines):
        raise ValueError(f"samples has {len(samples)} lines, table has {len(lines)}")
    if reduce not in ("error", "mean", "sum", "last"):
        raise ValueError("reduce must be 'error', 'mean', 'sum' or 'last'")
    if origin not in ("min", "zero"):
        raise ValueError("origin must be 'min' or 'zero'")

    idx = [lines.counter(name) for name in dims]
    if origin == "min":
        idx = [values - int(values.min()) for values in idx]
    sizes = [int(values.max()) + 1 for values in idx]
    flat = np.ravel_multi_index(tuple(idx), sizes)

    n_flat = int(np.prod(sizes))
    dense = np.zeros((n_flat, *samples.shape[1:]), dtype=samples.dtype)
    counts = np.bincount(flat, minlength=n_flat)

    if counts.max(initial=0) > 1:
        if reduce == "error":
            varying = [
                name
                for name in COUNTERS
                if name not in dims and len(np.unique(lines.counter(name))) > 1
            ]
            raise ValueError(
                f"{int((counts > 1).sum())} grid positions receive more than one line. "
                + (
                    f"Counters varying but not in dims: {varying}."
                    if varying
                    else "The same counter tuple is acquired repeatedly."
                )
                + " Add them to dims, or pass reduce='mean'/'sum'/'last'."
            )
        if reduce == "last":
            dense[flat] = samples
        else:
            np.add.at(dense, flat, samples)
            if reduce == "mean":
                dense /= np.maximum(counts, 1).reshape(-1, *(1,) * (dense.ndim - 1))
    else:
        # No collisions: a plain vectorized scatter, far faster than np.add.at.
        dense[flat] = samples

    return dense.reshape(*sizes, *samples.shape[1:])


class Measurement:
    """One measurement (raid entry) inside a `.dat` file."""

    def __init__(self, file: TwixFile, entry: _format.RaidEntry, index: int) -> None:
        self._file = file
        self.index = index
        self.meas_id, self.offset, self.length, self.patient_name, self.protocol_name = entry

    def __repr__(self) -> str:
        return (
            f"Measurement(index={self.index}, protocol_name={self.protocol_name!r}, "
            f"{self.length / 2**20:.1f} MiB)"
        )

    @functools.cached_property
    def _header(self) -> tuple[protocol.Protocol, int]:
        return protocol.parse_protocol(self._file.mm, self.offset)

    @property
    def hdr(self) -> protocol.Protocol:
        """Parsed text protocol; each buffer is parsed on first access."""
        return self._header[0]

    @functools.cached_property
    def lines(self) -> LineTable:
        """The acquisition lines, from one pass over the line headers."""
        data_start = self.offset + self._header[1]
        rows, truncated = _read.walk_headers(
            self._file.mm, data_start, self.offset + self.length, self._file.version
        )
        if truncated:
            message = f"measurement {self.index} ended before ACQEND after {len(rows)} lines"
            if not self._file.allow_truncated:
                raise TruncatedFileError(f"{message}; pass allow_truncated=True to read anyway")
            warnings.warn(f"{message}; using the acquired lines only", stacklevel=2)
        return LineTable(rows, self._file.mm, self._file.version)

    def read(
        self,
        lines: LineTable | None = None,
        *,
        out: np.ndarray | None = None,
        reflect: bool = True,
        batch_bytes: int = _extract._DEFAULT_BATCH_BYTES,
    ) -> np.ndarray:
        """Read the samples of `lines` (default: all of them) into `(n, ncha, ncol)`.

        Pass `out=` to read into a preallocated buffer — a `np.memmap`, a slice of a
        bigger array — which is what makes files larger than RAM readable, since the
        data is filled in one batch at a time.
        """
        table = self.lines if lines is None else lines
        return _extract.read_lines(
            self._file.mm,
            table.rows,
            self._file.version,
            out=out,
            reflect=reflect,
            batch_bytes=batch_bytes,
        )

    def to_dense(
        self,
        lines: LineTable | None = None,
        dims: tuple[str, ...] = ("Lin", "Par"),
        *,
        reduce: str = "error",
        origin: str = "min",
        reflect: bool = True,
    ) -> np.ndarray:
        """`read` followed by `to_dense` — convenience for Cartesian acquisitions."""
        table = self.lines.image if lines is None else lines
        samples = self.read(table, reflect=reflect)
        return to_dense(samples, table, dims, reduce=reduce, origin=origin)


class TwixFile:
    """A memory-mapped `.dat` file and the measurements it contains."""

    def __init__(self, path: str, *, allow_truncated: bool = False) -> None:
        self.path = path
        self.allow_truncated = allow_truncated
        self.mm = _read.open_mmap(path)
        self.version = _format.detect_version(self.mm)
        entries = _format.parse_raid_directory(self.mm, self.version)
        # Zero-length entries are aborted measurements with no data written; the
        # complete ones around them stay readable.
        self.measurements = [
            Measurement(self, entry, index)
            for index, entry in enumerate(entries)
            if entry.length > 0
        ]
        if not self.measurements:
            raise TwixParseError(f"{path}: no non-empty measurement")

    def __len__(self) -> int:
        return len(self.measurements)

    def __getitem__(self, index: int) -> Measurement:
        return self.measurements[index]

    def __iter__(self):
        return iter(self.measurements)

    def __repr__(self) -> str:
        names = [m.protocol_name for m in self.measurements]
        return f"TwixFile({self.path!r}, version={self.version.name}, measurements={names})"


def open_twix(path: str, *, allow_truncated: bool = False) -> TwixFile:
    """Open a Siemens `.dat` (TWIX) file.

    Nothing is read beyond the raid directory: protocols and line tables are parsed per
    measurement on first access.

    Every measurement in the file is returned, in file order — a scan is commonly
    preceded by calibration measurements (coil sensitivity, noise), and which one you
    want is your choice to make, not a default to inherit.
    """
    return TwixFile(path, allow_truncated=allow_truncated)
