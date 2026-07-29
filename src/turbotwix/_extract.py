"""Sample-data extraction: read the ADC samples of a set of selected lines into a
`(n_lines, ncha, ncol)` complex64 array.

That shape *is* the file's own layout — one contiguous (ncha, ncol) block per line, in
file order — so reading is a batched strided copy with no index arithmetic and no
scatter. Mapping lines onto a Cartesian grid is a separate, optional step
(`core.to_dense`), because for non-Cartesian acquisitions there is no such grid: the
loop counters index shots, interleaves or spokes, not k-space positions.

Reads go batch by batch into the destination buffer, so peak memory is the output plus
one batch, and callers can pass their own `out=` (a memmap, a preallocated buffer) to
read files far larger than RAM.
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import as_strided

from turbotwix._format import Flag, TwixVersion, has_flag, header_sizes

# Deliberately small: a batch only needs to be big enough to amortize per-call numpy
# overhead — a few hundred KB to a few MB of samples already gets the full vectorization
# benefit (measured no improvement past ~1-2MB for this workload), while a large budget
# costs real peak memory for the transient buffers a batch holds.
_DEFAULT_BATCH_BYTES = 2 * 1024 * 1024

# A per-line strided copy costs ~3 us of Python/numpy call overhead per line, so it
# only beats the index-array gather once a line is big enough to dominate that. Lines
# below this size fall back to the index gather (which is insensitive to spacing).
_PER_LINE_MIN_BYTES = 32 * 1024


def _batch_size(ncol: int, ncha: int, budget_bytes: int = _DEFAULT_BATCH_BYTES) -> int:
    bytes_per_line = max(ncha * ncol * 8, 1)
    return max(1, budget_bytes // bytes_per_line)


def _strided_copy(
    mm: np.ndarray, sample0: np.ndarray, ncol: int, ncha: int, block_len: int, out: np.ndarray
) -> bool:
    """Fill `out` (m, ncha, ncol) from strided views, with no index array at all.

    The index-array gather below needs an int64 index per complex64 sample — as much
    index traffic as data traffic — where a strided view is a plain memcpy. Two shapes
    of that apply, both bit-identical to the gather:

    * evenly spaced lines (the norm inside a run of identically-shaped lines): the whole
      batch is one view. Measured ~7x faster.
    * irregularly spaced lines: one view per line. Real sequences interleave line kinds
      — a spiral acquisition puts a PMU block between every second image line, so image
      offsets alternate between two gaps and are never evenly spaced — and each kind
      still has a fixed stride *within* the line. Measured ~6x faster on such a batch.

    Returns False when the layout doesn't fit, so the caller can fall back.
    """
    m = sample0.size
    if block_len % 8 or not bool(np.all(sample0 % 8 == 0)):
        return False

    # as_strided is unchecked, so bound the last byte it can reach before building it.
    if int(sample0.max()) + (ncha - 1) * block_len + 8 * ncol > mm.size:
        return False

    c8: np.ndarray = np.asarray(mm[: (mm.size // 8) * 8].view("<c8"))
    line_stride = int(sample0[1] - sample0[0]) if m > 1 else 0
    evenly_spaced = m < 2 or (line_stride > 0 and bool(np.all(np.diff(sample0) == line_stride)))
    if evenly_spaced:
        view = as_strided(
            c8[int(sample0[0]) // 8 :], shape=(m, ncha, ncol), strides=(line_stride, block_len, 8)
        )
        np.copyto(out, view)
        return True

    if 8 * ncha * ncol < _PER_LINE_MIN_BYTES:
        return False
    for i, start in enumerate(sample0):
        out[i] = as_strided(c8[int(start) // 8 :], shape=(ncha, ncol), strides=(block_len, 8))
    return True


def _gather_batch(
    mm: np.ndarray,
    offsets: np.ndarray,
    ncol: int,
    ncha: int,
    version: TwixVersion,
    out: np.ndarray,
) -> None:
    """Read one batch of lines (given by scan-header `offsets`) into `out`."""
    scan_prefix, chan_hdr = header_sizes(version)
    block_len = chan_hdr + 8 * ncol
    sample0 = offsets.astype(np.int64) + scan_prefix + chan_hdr  # byte offset of channel-0 samples

    if _strided_copy(mm, sample0, ncol, ncha, block_len, out):
        return

    # Gather at complex64 granularity (one index per *sample*, not per byte) — an 8x
    # smaller index array than a byte-level gather, since every offset involved
    # (scan/channel header sizes, 8-byte samples) is itself a multiple of 8.
    if block_len % 8 == 0 and bool(np.all(sample0 % 8 == 0)):
        c8: np.ndarray = np.asarray(mm[: (mm.size // 8) * 8].view("<c8"))
        elem_idx = (sample0 // 8)[:, None] + np.arange(ncha)[None, :] * (block_len // 8)
        np.take(c8, elem_idx[:, :, None] + np.arange(ncol)[None, None, :], out=out)
        return

    # Last resort: byte-granular gather (works regardless of alignment, costlier).
    chan_start = (
        offsets[:, None].astype(np.int64) + scan_prefix + np.arange(ncha)[None, :] * block_len
    )
    byte_idx = chan_start[:, :, None] + np.arange(block_len)[None, None, :]
    raw = mm[byte_idx]  # fresh contiguous (m, ncha, block_len) uint8 copy
    out[:] = raw[:, :, chan_hdr:].view("<c8")


def read_lines(
    mm: np.ndarray,
    table: np.ndarray,
    version: TwixVersion,
    *,
    out: np.ndarray | None = None,
    reflect: bool = True,
    batch_bytes: int = _DEFAULT_BATCH_BYTES,
) -> np.ndarray:
    """Read the samples of every line in `table` into a `(len(table), ncha, ncol)` array.

    `reflect` un-reverses lines flagged REFLECT (bipolar readouts store them backwards);
    pass False to get the samples exactly as laid down on disk.

    A selection mixing `(ncha, ncol)` shapes raises: silently padding them into one array
    (what pymapvbvd does) hides partial-Fourier and mixed-acquisition selections behind
    plausible-looking zeros.
    """
    n = len(table)
    ncol_all, ncha_all = np.unique(table["ncol"]), np.unique(table["ncha"])
    if ncol_all.size > 1 or ncha_all.size > 1:
        raise ValueError(
            f"selection mixes line shapes: ncol={ncol_all.tolist()}, ncha={ncha_all.tolist()}. "
            "Split it first, e.g. with LineTable.by_shape()."
        )
    ncha, ncol = (int(ncha_all[0]), int(ncol_all[0])) if n else (0, 0)
    shape = (n, ncha, ncol)
    if out is None:
        out = np.empty(shape, dtype=np.complex64)
    elif out.shape != shape or out.dtype != np.complex64:
        raise ValueError(f"out must be {shape} complex64, got {out.shape} {out.dtype}")
    if n == 0:
        return out

    offsets = table["offset"]
    reflected = has_flag(table["flags"], Flag.REFLECT) if reflect else None
    batch = _batch_size(ncol, ncha, batch_bytes)
    for start in range(0, n, batch):
        stop = min(start + batch, n)
        chunk = out[start:stop]
        _gather_batch(mm, offsets[start:stop], ncol, ncha, version, chunk)
        if reflected is not None:
            flip = reflected[start:stop]
            if flip.any():
                chunk[flip] = chunk[flip][:, :, ::-1]
    return out


def remove_oversampling(samples: np.ndarray, axis: int = -1) -> np.ndarray:
    """Crop a readout axis to its central half via an inverse FFT (2x oversampling
    removal).

    Kept out of the read path on purpose: it is signal processing, not I/O, it costs a
    full complex round-trip over the data, and along a non-Cartesian readout (spiral,
    radial) it is not a meaningful operation at all.
    """
    n = samples.shape[axis]
    keep = np.concatenate([np.arange(n // 4), np.arange(3 * n // 4, n)])
    return np.fft.fft(np.fft.ifft(samples, axis=axis).take(keep, axis=axis), axis=axis)
