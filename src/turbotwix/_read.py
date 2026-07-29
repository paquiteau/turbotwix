"""Memory-mapping the file and walking its line headers (run-vectorized, see
`walk_headers`).
"""

from __future__ import annotations

import mmap as _mmap_module

import numpy as np
from numpy.lib.stride_tricks import as_strided

from turbotwix import _format
from turbotwix._format import TwixVersion

_INITIAL_CAPACITY = 4096

# Verification window growth for the run detector (see `_run_length`).
#
# Probing `w` lines ahead touches one page every `stride` bytes, so its real cost is
# `w * stride` of page faults, not `w` — and every probe that overshoots the end of a
# run pays that for nothing. Two guards, both learned from real files:
#   * grow geometrically from a small start, so the overshoot is at most the length of
#     the run just verified rather than a fixed large block. A 25 GB spiral file whose
#     first line is a shape-less SYNCDATA block otherwise verifies a bogus 192-byte
#     stride across the whole measurement: 44 s of pure page faulting.
#   * clamp each probe by a byte budget, because interleaved sequences make runs short
#     while strides stay huge. That same file alternates 1-2 spiral lines of 5.04 MiB
#     with one 2208-byte line (median run length: 1), where a flat 64-line probe reads
#     320 MB per run boundary. At this budget such a run probes exactly one line ahead
#     — the header the walk reads next anyway, so overshoot costs nothing.
_MIN_VERIFY_WINDOW = 4
_MAX_VERIFY_WINDOW = 8192
_PROBE_BUDGET_BYTES = 8 * 1024 * 1024

# Run detection is not free: a probe that finds no run costs ~37 us against ~3.6 us to
# just step to the next header, so it only pays on runs of ~16 lines or more. Sequences
# exist that never provide them — the spiral file above interleaves image and feedback
# lines with a median run length of 1. So after this many consecutive short runs the
# walk stops probing and finishes line by line, re-arming every `_RUN_RETRY_LINES` in
# case the acquisition switches to a long uniform block later on.
_RUN_PAYOFF = 16
_RUN_STRIKES = 8
_RUN_RETRY_LINES = 4096


def open_mmap(path: str) -> np.memmap:
    """Memory-map the whole file read-only, with a best-effort sequential-access hint."""
    mm = np.memmap(path, dtype=np.uint8, mode="r")
    raw_mmap = getattr(mm, "_mmap", None)
    if raw_mmap is not None:
        try:
            raw_mmap.madvise(_mmap_module.MADV_SEQUENTIAL)
        except Exception:
            pass  # best-effort only; not all platforms/numpy versions support this
    return mm


def read_headers(mm: np.ndarray, offsets: np.ndarray, version: TwixVersion) -> np.ndarray:
    """Materialize the full VB/VD scan headers for the given line offsets.

    The walk keeps only the 48 bytes it needs per line; everything else (slice
    geometry, ICE program parameters, timestamps, centre indices) lives here and is
    re-read on demand, so metadata-heavy work stays possible without every read paying
    for it.
    """
    header_dtype = _format.VD_SCAN_HEADER if version is TwixVersion.VD else _format.VB_HEADER
    size = header_dtype.itemsize
    idx = np.asarray(offsets, dtype=np.int64)[:, None] + np.arange(size, dtype=np.int64)[None, :]
    return np.ascontiguousarray(mm[idx]).view(header_dtype).reshape(len(idx))


def _header_view(mm: np.ndarray, pos: int, header_dtype: np.dtype, count: int, stride: int):
    """Zero-copy strided view of `count` line headers spaced `stride` bytes apart.

    The caller is responsible for bounds: `as_strided` performs no checking, so every
    header must lie fully inside the walked region.
    """
    first = mm[pos : pos + header_dtype.itemsize].view(header_dtype)
    return as_strided(first, shape=(count,), strides=(stride,))


def _run_length(
    mm: np.ndarray,
    pos: int,
    scan_end: int,
    header_dtype: np.dtype,
    stride: int,
    ncol: int,
    ncha: int,
    stop_bits: np.uint64,
) -> int:
    """Number of consecutive lines from `pos` that the sequential walk would step over
    with this exact `stride`.

    The sequential rule is `stride = prefix + ncha * (chan_hdr + 8 * ncol)`, so a line
    is part of the run iff it reports the same (ncol, ncha) and carries neither ACQEND
    nor SYNCDATA (both of which take their length from the raw DMA field instead). That
    makes this check exactly equivalent to walking the lines one at a time — it just
    tests a whole window of headers per numpy call rather than one per Python iteration.
    """
    n_max = (scan_end - pos) // stride
    if n_max <= 1:
        return 1

    n = 1  # the caller already read and matched the line at `pos`
    window = _MIN_VERIFY_WINDOW
    budget = max(1, _PROBE_BUDGET_BYTES // stride)
    while n < n_max:
        count = min(window, budget, n_max - n)
        headers = _header_view(mm, pos + n * stride, header_dtype, count, stride)
        ok = (
            (headers["SamplesInScan"] == ncol)
            & (headers["UsedChannels"] == ncha)
            & ((headers["EvalInfoMask"] & stop_bits) == 0)
        )
        if not ok.all():
            return n + int(np.argmin(ok))
        n += count
        window = min(window * 2, _MAX_VERIFY_WINDOW)
    return n_max


def walk_headers(
    mm: np.ndarray, scan_offset: int, scan_end: int, version: TwixVersion
) -> tuple[np.ndarray, bool]:
    """Walk every line header from `scan_offset` to `scan_end` (or until ACQEND).

    Returns `(table, truncated)` where `table` is a `_format.LINE_DTYPE` array — one
    48-byte row per line, carrying only what selection and gathering need. `truncated`
    is True if the walk hit EOF before an ACQEND block.

    The stream is self-describing (each line's stride comes from its own header), so
    this looks inherently sequential — but real acquisitions are made of long *runs* of
    identically-shaped lines. So each iteration hypothesizes the stride from one header
    and then verifies vectorized how far that stride holds (`_run_length`), copying the
    whole run out of a strided view in one go. Both sample files are 1-3 runs end to
    end; a 245 MB TW1 volume is 3 runs for 641 lines. Any line that breaks the pattern
    just ends the run and starts the next one, so the result is identical to stepping
    line by line.
    """
    is_vd = version is TwixVersion.VD
    header_dtype = _format.VD_SCAN_HEADER if is_vd else _format.VB_HEADER
    hdr_size = header_dtype.itemsize
    scan_hdr_prefix, channel_hdr_size = _format.header_sizes(version)

    acqend_bit = np.uint64(int(_format.Flag.ACQEND))
    sync_bit = np.uint64(int(_format.Flag.SYNCDATA))

    capacity = _INITIAL_CAPACITY
    table = np.empty(capacity, dtype=_format.LINE_DTYPE)
    n = 0
    pos = scan_offset
    truncated = False
    probing = True
    short_runs = 0
    lines_since_retry = 0

    while True:
        if pos + hdr_size > scan_end:
            truncated = True
            break

        header = mm[pos : pos + hdr_size].view(header_dtype)[0]
        eval_mask = header["EvalInfoMask"]
        is_acqend = bool(eval_mask & acqend_bit)
        is_sync = bool(eval_mask & sync_bit)

        if is_acqend or is_sync:
            # These blocks don't have the usual NCol/NCha-derived shape; trust the
            # raw encoded length (matches twixtools' mdh_def.get_dma_len).
            length = int(_format.dma_len(header["FlagsAndDMALength"]))
            run = 1
        else:
            # Recompute from NCol/NCha rather than trusting the raw field: Siemens'
            # PackBit (common in EPI) can make the raw field wrong (same approach
            # as both pymapvbvd's loop_mdh_read and twixtools' Mdb._get_dma_len).
            ncol = int(header["SamplesInScan"])
            ncha = int(header["UsedChannels"])
            length = scan_hdr_prefix + ncha * (channel_hdr_size + 8 * ncol)
            run = 1
            if probing and length > 0:
                run = _run_length(
                    mm, pos, scan_end, header_dtype, length, ncol, ncha, acqend_bit | sync_bit
                )
                short_runs = 0 if run >= _RUN_PAYOFF else short_runs + 1
                if short_runs >= _RUN_STRIKES:
                    probing = False
                    lines_since_retry = 0
            elif not probing:
                lines_since_retry += 1
                if lines_since_retry >= _RUN_RETRY_LINES:
                    probing = True
                    short_runs = 0

        if n + run > capacity:
            while n + run > capacity:
                capacity *= 2
            grown = np.empty(capacity, dtype=_format.LINE_DTYPE)
            grown[:n] = table[:n]
            table = grown

        rows = table[n : n + run]
        if run == 1:
            # One structured store, not five field stores: interleaved sequences make
            # this the per-line path, where five numpy calls per line show up directly
            # in the walk time.
            table[n] = (
                pos,
                eval_mask,
                header["SamplesInScan"],
                header["UsedChannels"],
                header["Counter"],
            )
        else:
            headers = _header_view(mm, pos, header_dtype, run, length)
            rows["offset"] = pos + np.arange(run, dtype=np.int64) * length
            rows["flags"] = headers["EvalInfoMask"]
            rows["ncol"] = headers["SamplesInScan"]
            rows["ncha"] = headers["UsedChannels"]
            rows["counters"] = headers["Counter"]
        n += run

        if is_acqend:
            break
        if length <= 0 or pos + run * length > scan_end:
            truncated = True
            break
        pos += run * length

    return table[:n], truncated
