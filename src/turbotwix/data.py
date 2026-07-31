#!/usr/bin/env python3
"""The line table and sample extraction: one 48-byte row per ADC line, and a strided
copy of the selected lines' samples.

`docs/twix-format.md` describes the format itself; `docs/implementation.md` describes
the uniform-run / walk split and the extraction strategy in more depth.
"""

from __future__ import annotations

import functools
import logging
import mmap as _mmap
from typing import Literal, NamedTuple, TypedDict

import numpy as np
from numpy.lib.stride_tricks import as_strided
from numpy.typing import NDArray

try:  # optional dependency, only used for the _walk progress bar
    from tqdm import tqdm
except ImportError:
    tqdm: type | None = None  # type: ignore[no-redef]

from .dtypes import (
    _DMA_LEN_MASK,
    _USE_A_REFERENCE_READER,
    COUNTERS,
    HEADER_SIZES,
    LINE_DTYPE,
    SCAN_HEADER_DTYPES,
    SYNC_DTYPE,
    Flag,
    RaidEntry,
    TwixParseError,
    TwixVersion,
    UnsupportedLayoutError,
    detect_version,
    parse_raid_directory,
)
from .header import Protocol, parse_protocol
from .pmu import Pmu

logger = logging.getLogger("turbotwix")

__all__ = [
    "LineTable",
    "Measurement",
    "TwixFile",
    "build_table",
    "common_shape",
    "minimal_dims",
    "open_mmap",
    "open_twix",
    "read_headers",
]

# Un-reflecting is a fancy-index assignment, so it needs a temporary the size of the
# lines it touches; this bounds that temporary without bounding anything else.
_FLIP_BUDGET_BYTES = 2 * 1024 * 1024

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


class RowShape(NamedTuple):
    """The `(ncha, ncol)` shape of one line."""

    ncha: int
    ncol: int


class RowStruct(TypedDict):
    """One row of the line table."""

    offset: NDArray[np.int64]
    flags: NDArray[np.uint64]
    ncol: NDArray[np.uint16]
    ncha: NDArray[np.uint16]
    counters: NDArray[np.void]


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
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Build the line table of one measurement.

    ACQEND blocks hold no samples and are stepped over, not recorded. SYNCDATA/PMU
    blocks are also stepped over here, but their `(offset, length)` is recorded in
    `sync` so `turbotwix.pmu` can decode them on demand.

    A measurement may mix `(ncha, ncol)` — an embedded parallel-imaging reference scan
    is Cartesian and short where the imaging lines are spiral and long, and a
    coil-sensitivity adjustment stores body-coil and array-coil lines together. The
    table records each line's own shape and stays whole; the one-shape rule belongs to
    `LineTable.read`, which needs a rectangular result, and is met by selecting
    (`.image`, `.refscan`) before reading.

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
    sync : numpy.ndarray
        A `SYNC_DTYPE` array with one `(offset, length)` row per SYNCDATA/PMU block.
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
        return np.empty(0, dtype=LINE_DTYPE), np.empty(0, dtype=SYNC_DTYPE), True

    uniform = _uniform_run(mm, data_start, scan_end, version, header_dtype)
    if uniform is not None:
        table, sync, truncated = uniform
        logger.debug(
            "build_table: uniform run, %d lines%s",
            len(table),
            " (truncated)" if truncated else "",
        )
        return uniform
    logger.debug(
        "build_table: non-uniform layout, walking %d bytes", scan_end - data_start
    )
    return _walk(mm, data_start, scan_end, version, header_dtype)


def _uniform_run(
    mm: np.ndarray,
    data_start: int,
    scan_end: int,
    version: TwixVersion,
    header_dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray, bool] | None:
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
    tuple of (numpy.ndarray, numpy.ndarray, bool) or None
        The `(table, sync, truncated)` triplet as for `build_table` — `sync` is always
        empty here, since a uniform run by construction never contains a SYNCDATA
        block — or None if the measurement is not a single uniform run and the caller
        must fall back to `_walk`.
    """
    hdr_size = header_dtype.itemsize
    prefix, chan_hdr = HEADER_SIZES[version]
    stop_bits = np.uint64(Flag.ACQEND | Flag.SYNCDATA)
    empty_sync = np.empty(0, dtype=SYNC_DTYPE)

    first = mm[data_start : data_start + hdr_size].view(header_dtype)[0]
    ncol, ncha = int(first["SamplesInScan"]), int(first["UsedChannels"])
    if ncol == 0 or ncha == 0 or first["EvalInfoMask"] & stop_bits:
        return None

    stride = prefix + ncha * (chan_hdr + 8 * ncol)
    n = (scan_end - data_start) // stride
    if n == 0:
        return np.empty(0, dtype=LINE_DTYPE), empty_sync, True

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
    return table, empty_sync, truncated


def _walk(
    mm: np.ndarray,
    data_start: int,
    scan_end: int,
    version: TwixVersion,
    header_dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray, bool]:
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
    sync : numpy.ndarray
        A `SYNC_DTYPE` array with one `(offset, length)` row per SYNCDATA/PMU block.
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
    sync_rows: list[tuple[int, int]] = []
    pos = data_start
    truncated = False

    # Block count is small on this path (see the docstring), so the bar is cosmetic
    # more often than not — but a spiral/radial file with many shots can still take
    # long enough for feedback to matter, and tqdm is a no-op when not installed.
    progress = (
        tqdm(
            total=scan_end - data_start,
            unit="B",
            unit_scale=True,
            desc="turbotwix: walking blocks",
            leave=False,
        )
        if tqdm is not None
        else None
    )
    try:
        while True:
            if pos + hdr_size > scan_end:
                truncated = True
                break
            header = mm[pos : pos + hdr_size].view(header_dtype)[0]
            flags = header["EvalInfoMask"]

            if flags & acqend_bit:
                break
            if flags & sync_bit:
                # No (ncol, ncha)-derived shape, so the raw 25-bit length is all
                # there is.
                length = int(header["FlagsAndDMALength"]) & _DMA_LEN_MASK
                if length <= 0 or pos + length > scan_end:
                    truncated = True
                    break
                sync_rows.append((pos, length))
                pos += length
                if progress is not None:
                    progress.update(length)
                continue

            line_ncol, line_ncha = (
                int(header["SamplesInScan"]),
                int(header["UsedChannels"]),
            )
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
            if progress is not None:
                progress.update(length)
    finally:
        if progress is not None:
            progress.close()

    logger.debug("_walk: %d lines%s", len(rows), " (truncated)" if truncated else "")
    return (
        np.array(rows, dtype=LINE_DTYPE),
        np.array(sync_rows, dtype=SYNC_DTYPE),
        truncated,
    )


def common_shape(table: np.ndarray) -> RowShape:
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
        return RowShape(0, 0)
    ncha, ncol = int(table["ncha"][0]), int(table["ncol"][0])
    if len(table) == 1:
        return RowShape(ncha, ncol)
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
            f"(ncha, n_lines, ncol) array, so select a single-shaped subset first — "
            f"e.g. .image, .refscan or .noise."
        )
    return RowShape(ncha, ncol)


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

    def __init__(
        self, rows: NDArray[np.void], mm: NDArray, version: TwixVersion
    ) -> None:
        self.rows = rows
        self._mm = mm
        self._version = version

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, key) -> LineTable:
        return LineTable(np.atleast_1d(self.rows[key]), self._mm, self._version)

    def __repr__(self) -> str:
        if len(self) == 0:
            return "LineTable(0 lines)"

        shapes = np.stack([self.rows["ncha"], self.rows["ncol"]], axis=1)
        unique_shapes, counts = np.unique(shapes, axis=0, return_counts=True)
        return (
            f"LineTable({len(self)} lines, shapes"
            + " ".join(f"{n}x{tuple(s)}" for s, n in zip(unique_shapes, counts))
            + ")"
        )

    def __getattr__(self, name: str):
        if name in ["offset", "flags", "ncol", "ncha", "counters"]:
            return self.rows[name]
        else:
            raise AttributeError(
                f"{type(self).__name__!r} object has no attribute {name!r}"
            )

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
        return self.rows["counters"][name].astype(np.int64)

    @functools.cached_property
    def row_shape(self) -> RowShape:
        """The `(ncha, ncol)` shape shared by these lines.

        Raises
        ------
        UnsupportedLayoutError
            If the selection mixes shapes, as a measurement with an embedded reference
            scan does. Select a single-shaped subset (`.image`, `.refscan`, `.noise`).
        """
        return common_shape(self.rows)

    def headers(self) -> np.ndarray:
        """The full VB/VD scan headers of these lines, re-read from the file.

        Returns
        -------
        numpy.ndarray
            A `(len(self),)` array of `VB_HEADER` or `VD_SCAN_HEADER` records.
        """
        return read_headers(self._mm, self.offset, self._version)

    def read(
        self,
        *,
        out: np.ndarray | None = None,
        reflect: bool = True,
        dest: np.ndarray | None = None,
        dims: tuple[str, ...] | Literal["minimal"] | None = None,
    ) -> np.ndarray:
        """Read every line into a `(ncha, len(self), ncol)` array, compact or folded.

        The channel axis is first, and `ncol` stays last: each line is stored on disk
        channel-major (all `ncol` samples of one channel contiguous, then the next
        channel), so keeping that same relative order — channel outermost, column
        innermost — lets every channel's run be copied as one contiguous block. Only
        the line/grid axes move, out from between them to the front. A channel-last
        array avoids putting channel in the middle too, but forces a strided gather
        instead of a contiguous per-channel copy, which is markedly slower once
        `ncha` is more than a couple of channels.

        Requires that every line has the same `(ncha, ncol)`.

        Parameters
        ----------
        out : numpy.ndarray, optional
            Preallocated destination.
            Without `dims`: `(ncha, len(self), ncol)` complex64 when `dest` is not
            given (allocated if not given either); `(ncha, *, ncol)` complex64,
            required, when `dest` is given. With `dims`: `(ncha, *dim_sizes, ncol)` or
            the equivalent flattened `(ncha, prod(dim_sizes), ncol)`, complex64
            (allocated, and zero-filled, if not given) — only its shape is checked
            against `dims`, so an `out` with unfilled cells is the caller's own doing.
        reflect : bool, default True
            Un-reverse the lines flagged REFLECT (bipolar readouts store them
            backwards). Pass False for the samples exactly as laid down on disk.
        dest : numpy.ndarray, optional
            Index into `out`'s line axis for each line of `self`, in order, so a line
            is written straight to where it belongs instead of to a compact
            `(ncha, len(self), ncol)` buffer that then has to be copied again.
            Requires `out`. Defaults to file order (position `i` for line `i`). Not
            valid together with `dims`, which computes its own.
        dims : tuple of str or str, optional
            The counter names to use as grid axes; folds onto a grid when given,
            instead of returning the compact `(ncha, len(self), ncol)` array.

        Returns
        -------
        numpy.ndarray
            The samples without `dims` (`out` itself when given); the
            `(ncha, *dim_sizes, ncol)` grid with it (`out` itself, reshaped, when
            given).

        Raises
        ------
        ValueError
            If `out` does not have the required shape and dtype, if `dest` is given
            without `out`, or if `dest` is given together with `dims`.
        UnsupportedLayoutError
            If `self` mixes `(ncha, ncol)`: the result would not be rectangular.
        """
        n = len(self.rows)
        if n == 0:
            raise ValueError("Empty Table, nothing to read.")
        ncha, ncol = self.row_shape
        shape = (ncha, n, ncol)
        if out is not None and out.dtype != np.complex64:
            raise ValueError(f"out must be complex64, got {out.dtype}")
        if dims is not None and dest is not None:
            raise ValueError("dest is not supported together with dims")

        elif dims is not None and dest is None:
            _, flat, sizes = _fold_index(self, dims)
            grid_shape = (ncha, *sizes, ncol)
            if out is None:
                out = np.zeros(grid_shape, dtype=np.complex64)

        elif dims is None and dest is not None:
            if out is None:
                raise ValueError("out is required when dest is given")
            if out.ndim != 3 or out.shape[0] != ncha or out.shape[2] != ncol:
                raise ValueError(
                    f"out must be ({ncha}, *, {ncol}) complex64, got {out.shape}"
                )
            flat = dest
        else:  # dims and dest are both None
            if out is None:
                out = np.empty(shape, dtype=np.complex64)
            elif out.shape != shape:
                raise ValueError(f"out must be {shape} complex64, got {out.shape}")
            flat = None
        self._read_into(out.reshape(ncha, -1, ncol), flat, reflect)
        return out

    def _read_into(
        self,
        out: np.ndarray,
        dest: np.ndarray | None,
        reflect: bool,
    ) -> None:
        """Copy `table`'s samples into `out` (row `i`, or `dest[i]` if given), in place.

        The core shared by both of `read`'s branches (compact/`dest` and folded/`dims`):
        by the time this runs, `out`'s shape and `dest`'s meaning are already the
        caller's responsibility, validated.
        """
        n = len(self.rows)
        ncha, ncol = self.row_shape
        logger.debug("%d lines, (%d, %d) samples each", n, ncha, ncol)
        prefix, chan_hdr = HEADER_SIZES[self._version]
        # size for a single channel: its header, then its samples
        chan_block_size = chan_hdr + 8 * ncol
        # The whole file as a complex64 array, ignoring any trailing bytes.
        c8 = np.asarray(self._mm[: (self._mm.size // 8) * 8].view("<c8"))
        # The byte offset of each line's first sample, as an index into `c8`.

        starts = (self.rows["offset"].astype(np.int64) + prefix + chan_hdr) // 8

        # `out` is (ncha, ..., ncol): channel first, column last, same relative order
        # as on disk (channel-major, ncol contiguous per channel) — so every channel's
        # run copies as one contiguous block instead of a strided gather.
        # Try to do a strided copy of the whole selection at once when the
        # offsets are evenly spaced.
        # Otherwise, fall back to copy each line one by one.
        step = int(starts[1] - starts[0]) if n > 1 else 0
        if step > 0 and np.all(np.diff(starts) == step):
            view = as_strided(
                c8[int(starts[0]) :],
                shape=(ncha, n, ncol),
                strides=(chan_block_size, 8 * step, 8),
            )
            if dest is None:
                np.copyto(out, view)
            else:
                out[:, dest, :] = view
        else:
            logger.debug("%d irregularly-spaced lines, reading one by one", n)
            iterator = enumerate(starts)
            if dest is None:
                dest = np.arange(n, dtype=np.int64)
            if tqdm is not None:
                iterator = tqdm(iterator, unit="line", desc="turbotwix: reading lines")
            for i, start in iterator:
                out[:, dest[i], :] = as_strided(
                    c8[int(start) :], shape=(ncha, ncol), strides=(chan_block_size, 8)
                )
        # Apply reflect Flag: the samples are stored backwards on disk, so flip
        # them to the right order. Data is chunked to avoid creating a temporary
        # array that is too large.
        if reflect:
            idx = np.flatnonzero(self.has_flag(Flag.REFLECT))
            if dest is not None:
                idx = dest[idx]
            chunk = max(1, _FLIP_BUDGET_BYTES // (8 * ncha * ncol))
            for at in range(0, idx.size, chunk):
                picked = idx[at : at + chunk]
                out[:, picked, :] = out[:, picked, :][:, :, ::-1]

    def has_flag(self, flag: Flag) -> np.ndarray:
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
        return (self.rows["flags"] & np.uint64(flag)) == np.uint64(flag)

    def has_any_flag(self, flag: Flag) -> np.ndarray:
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
        return (self.rows["flags"] & np.uint64(flag)) != 0

    @property
    def image(self) -> LineTable:
        """Imaging lines: all but calibration, feedback and reference-only lines."""
        return self[~self.has_any_flag(_NOT_IMAGING) & ~self._reference_only]

    @property
    def noise(self) -> LineTable:
        """Noise-calibration lines (for pre-whitening)."""
        return self[self.has_flag(Flag.NOISEADJSCAN)]

    @property
    def refscan(self) -> LineTable:
        """Parallel-imaging reference lines."""
        is_ref = self.has_any_flag(Flag.PATREFSCAN | Flag.PATREFANDIMASCAN)
        return self[is_ref & ~self.has_any_flag(_NOT_PLAIN_REFERENCE)]

    @property
    def phasecor(self) -> LineTable:
        """Phase-correction (navigator) lines."""
        return self[self.has_flag(Flag.PHASCOR) & ~self._reference_only]

    @property
    def _reference_only(self) -> np.ndarray:
        """PATREFSCAN without PATREFANDIMASCAN: reference that is not also image."""
        return self.has_flag(Flag.PATREFSCAN) & ~self.has_flag(Flag.PATREFANDIMASCAN)


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


def _fold_index(
    lines: LineTable, dims: tuple[str, ...] | str
) -> tuple[tuple[str, ...], np.ndarray, list[int]]:
    """Each line's flat grid position for `read(dims=...)`, from counters alone.

    Metadata-scale (one int per line), so it is cheap to compute before any sample data
    is read — which is what lets `Measurement.read` write each line straight to its
    grid cell instead of into a compact buffer that then has to be folded separately.

    Parameters
    ----------
    lines : LineTable
        The lines to place on a grid.
    dims : tuple of str or {'minimal'}
        The counter names to use as grid axes, as for `read`.

    Returns
    -------
    dims : tuple of str
        The dims actually used (`minimal_dims(lines)` when `dims` was ``"minimal"``).
    flat : numpy.ndarray
        The `(len(lines),)` flat grid index of each line.
    sizes : list of int
        The size of each axis in `dims`.

    Raises
    ------
    ValueError
        If two lines land on the same grid cell under explicit (non-``"minimal"``)
        `dims` — `minimal_dims` guarantees a collision-free grid, so only explicit
        `dims` need the check.
    """
    minimal = dims == "minimal"
    resolved_dims = minimal_dims(lines) if minimal else dims
    assert not isinstance(resolved_dims, str)

    idx = [lines.counter(name) for name in resolved_dims]
    # shift the grid so the acquired range starts at 0.
    idx = [values - int(values.min()) for values in idx]
    sizes = [int(values.max()) + 1 for values in idx]
    flat = np.ravel_multi_index(tuple(idx), sizes)

    if not minimal:
        counts = np.bincount(flat, minlength=int(np.prod(sizes)))
        if counts.max(initial=0) > 1:
            varying = [
                name
                for name in COUNTERS
                if name not in resolved_dims and len(np.unique(lines.counter(name))) > 1
            ]
            raise ValueError(
                f"{int((counts > 1).sum())} grid positions receive more than one line. "
                f"Counters varying but not in dims: {varying}."
            )
    return resolved_dims, flat, sizes


# ---------------------------------------------------------------------------
# The object model
# ---------------------------------------------------------------------------


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
    def _table(self) -> tuple[np.ndarray, np.ndarray, bool]:
        data_start = self.offset + self._header[1]
        rows, sync, truncated = build_table(
            self._file.mm, data_start, self.offset + self.length, self._file.version
        )
        if truncated:
            logger.warning(
                "measurement %i ended before ACQEND after %i lines,"
                " using the acquired lines only",
                self.index,
                len(rows),
            )
        return rows, sync, truncated

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
        rows, _, _ = self._table
        return LineTable(rows, self._file.mm, self._file.version)

    @functools.cached_property
    def pmu(self) -> Pmu:
        """The physiological (PMU) data interleaved with this measurement's lines.

        Empty (no channels) when the measurement carries no SYNCDATA/PMU blocks, which
        is the common case.
        """
        _, sync, _ = self._table
        syngo_version = self.hdr["Dicom"]["SoftwareVersions"]
        prefix, _ = HEADER_SIZES[self._file.version]
        return Pmu.decode(self._file.mm, sync, prefix, syngo_version)

    def read(
        self,
        lines: LineTable | None = None,
        dims: tuple[str, ...] | Literal["minimal"] | None = None,
        *,
        out: np.ndarray | None = None,
        reflect: bool = True,
    ) -> np.ndarray:
        """Read the samples of `lines`, compact or folded onto a grid.

        Parameters
        ----------
        lines : LineTable, optional
            The lines to read: every line of the measurement without `dims`, the
            imaging lines with it.
        dims : tuple of str or str, optional
            The counter names to use as grid axes, as for `LineTable.read`.
        out : numpy.ndarray, optional
            Preallocated destination, as for `LineTable.read`.
        reflect : bool, default True
            Un-reverse the lines flagged REFLECT.

        Returns
        -------
        numpy.ndarray
            See `LineTable.read`.
        """
        if dims is None:
            table = self.lines if lines is None else lines
        else:
            table = self.lines.image if lines is None else lines
        return table.read(out=out, reflect=reflect, dims=dims)


class TwixFile:
    """A memory-mapped `.dat` file and the measurements it contains.

    Indexable and iterable over its `Measurement` objects, in acquisition order. `scan`
    is the last of them, and very likely the one that contains the data you are after.
    ``hdr``, ``lines`` and ``read`` are shorthands for ``.scan.hdr``,
    ``scan.lines`` and ``.scan.lines.read(...)``


    Parameters
    ----------
    path : str
        Path to the `.dat` file.

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

    def __init__(self, path: str) -> None:
        self.path = path
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
        logger.debug(
            "%s: %s, %d measurement(s), %.1f MiB",
            path,
            self.version.name,
            len(self.measurements),
            self.mm.size / 2**20,
        )

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

    def __exit__(self, *_) -> None:
        self.close()

    @property
    def scan(self) -> Measurement:
        """Shorthand for last measurement in file, the one you almost always want."""
        return self.measurements[-1]

    @property
    def hdr(self) -> Protocol:
        """Header of the scan measurement."""
        return self.scan.hdr

    @property
    def lines(self) -> LineTable:
        """Raw acquisition lines of the scan measurement, as a queryable table."""
        return self.scan.lines

    def read(
        self,
        lines: LineTable | None = None,
        dims: tuple[str, ...] | Literal["minimal"] | None = None,
        *,
        out: np.ndarray | None = None,
        reflect: bool = True,
    ) -> np.ndarray:
        """`scan.read(...)` — read the last measurement's samples, compact or folded.

        See `Measurement.read`.
        """
        return self.scan.read(lines, dims, out=out, reflect=reflect)

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


def open_twix(path) -> TwixFile:
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

    Returns
    -------
    TwixFile
        The open file, indexable and iterable over its `Measurement` objects.
    """
    return TwixFile(path)
