"""Batched vectorized sample data extraction: one `numpy` fancy-index gather per
memory-bounded batch instead of a per-line `seek()`+`read()` Python loop, followed by
vectorized postprocessing and a vectorized scatter-write into the output k-space array.

Default dimension order is (Lin, Par, Sli, Ave, Phs, Eco, Rep, Set, Seg, Ida, Idb, Idc,
Idd, Ide, Cha, Col) — loop counters outermost, Cha/Col innermost. This is *not*
pymapvbvd's order; it's the order twixtools itself uses internally
(`map_twix.py`'s `_dim_order`), because scattering per-line data into the *outermost*
axis of a numpy array is dramatically faster than scattering into the innermost one
(confirmed empirically: ~40x for this workload). Building the array in pymapvbvd's
(Col, Cha, ...) order instead requires either a slow per-batch scatter into the fast
axis, or one big transpose+copy at the end — both cost real time on large arrays.
Pass `legacy_layout=True` to get the pymapvbvd-compatible (Col, Cha, ...) order anyway
(e.g. for drop-in comparison/testing); that path pays the transpose cost explicitly.
"""

from __future__ import annotations

import numpy as np

from turbotwix._format import MASK_BIT, TwixVersion, unpack_flags

DIM_NAMES = (
    "Lin",
    "Par",
    "Sli",
    "Ave",
    "Phs",
    "Eco",
    "Rep",
    "Set",
    "Seg",
    "Ida",
    "Idb",
    "Idc",
    "Idd",
    "Ide",
)

# Deliberately small: once the scatter target is the outermost axis (see above), a
# batch only needs to be big enough to amortize per-call numpy overhead — a few
# hundred KB to a few MB of samples per batch already gets full vectorization
# benefit. A large budget here doesn't buy more speed (measured no improvement past
# ~1-2MB for this workload) but does cost real peak memory: each batch transiently
# holds both a gathered-samples buffer and an index buffer of matching size.
_DEFAULT_BATCH_BYTES = 2 * 1024 * 1024


def _batch_size(ncol: int, ncha: int, budget_bytes: int = _DEFAULT_BATCH_BYTES) -> int:
    bytes_per_line = max(ncha * ncol * 8, 1)
    return max(1, budget_bytes // bytes_per_line)


def _gather_batch(
    mm: np.ndarray, offsets: np.ndarray, ncol: int, ncha: int, is_vd: bool
) -> np.ndarray:
    """One vectorized fancy-index gather for a batch of `offsets` (scan-header start
    positions), all sharing the same (ncol, ncha) shape. Returns complex64 samples of
    shape (len(offsets), ncha, ncol).
    """
    scan_prefix = 192 if is_vd else 0
    chan_hdr = 32 if is_vd else 128
    block_len = chan_hdr + 8 * ncol
    sample0 = offsets.astype(np.int64) + scan_prefix + chan_hdr  # byte offset of channel-0 samples

    # Fast path: gather at complex64 granularity (one index per *sample*, not per
    # byte) — an 8x smaller index array than a byte-level gather, since every offset
    # involved (scan/channel header sizes, 8-byte samples) is itself a multiple of 8.
    if block_len % 8 == 0 and bool(np.all(sample0 % 8 == 0)):
        n_c8 = (mm.size // 8) * 8
        c8: np.ndarray = np.asarray(mm[:n_c8].view("<c8"))
        elem0 = sample0 // 8
        chan_stride = block_len // 8
        elem_idx = elem0[:, None] + np.arange(ncha)[None, :] * chan_stride
        sample_idx = elem_idx[:, :, None] + np.arange(ncol)[None, None, :]
        return np.take(c8, sample_idx)

    # Fallback: byte-granular gather (works regardless of alignment, costlier).
    chan_start = (
        offsets[:, None].astype(np.int64) + scan_prefix + np.arange(ncha)[None, :] * block_len
    )
    byte_idx = chan_start[:, :, None] + np.arange(block_len)[None, None, :]
    raw = mm[byte_idx]  # fresh contiguous (M, ncha, block_len) uint8 copy
    return raw[:, :, chan_hdr:].view("<c8")  # (M, ncha, block_len - chan_hdr) == (M, ncha, ncol)


def remove_oversampling(arr: np.ndarray, axis: int = 0) -> np.ndarray:
    """Crop the readout axis to its central half after an inverse FFT (standard 2x
    oversampling removal), matching pymapvbvd's `readData` behaviour.
    """
    n = arr.shape[axis]
    keep = np.concatenate([np.arange(n // 4), np.arange(3 * n // 4, n)])
    return np.fft.fft(np.fft.ifft(arr, axis=axis).take(keep, axis=axis), axis=axis)


def _dim_values(hdr: np.ndarray, skip_to_first_line: bool) -> dict[str, np.ndarray]:
    counter = hdr["Counter"]
    dim_values = {name: counter[name].astype(np.int64) for name in DIM_NAMES}
    if skip_to_first_line:
        for name in ("Lin", "Par"):
            dim_values[name] = dim_values[name] - int(dim_values[name].min())
    return dim_values


def category_shape(
    table: np.ndarray,
    mask: np.ndarray,
    skip_to_first_line: bool = True,
    remove_os: bool = False,
    legacy_layout: bool = False,
) -> tuple[int, ...]:
    """Cheaply compute a category's full array shape from headers only (no data read)."""
    sub = table[mask]
    if len(sub) == 0:
        empty = (0, 0) + (1,) * len(DIM_NAMES)
        return empty if legacy_layout else empty[2:] + empty[:2]
    hdr = sub["hdr"]
    dim_values = _dim_values(hdr, skip_to_first_line)
    dim_sizes = tuple(int(dim_values[name].max()) + 1 for name in DIM_NAMES)
    ncol = int(hdr["SamplesInScan"][0])
    if remove_os:
        ncol -= ncol // 2
    ncha = int(hdr["UsedChannels"][0])
    return (ncol, ncha) + dim_sizes if legacy_layout else dim_sizes + (ncha, ncol)


def read_category(
    mm: np.ndarray,
    table: np.ndarray,
    mask: np.ndarray,
    version: TwixVersion,
    *,
    remove_os: bool = False,
    skip_to_first_line: bool = True,
    legacy_layout: bool = False,
    batch_bytes: int = _DEFAULT_BATCH_BYTES,
) -> np.ndarray:
    """Extract and assemble the full k-space array for one scan category.

    By default returns a complex64 array shaped (NLin, NPar, NSli, NAve, NPhs, NEco,
    NRep, NSet, NSeg, NIda, NIdb, NIdc, NIdd, NIde, NCha, NCol) — see the module
    docstring for why. Pass `legacy_layout=True` for pymapvbvd's (NCol, NCha, NLin,
    ...) order instead (costs one extra full-array transpose+copy).

    Lines that share the same target index (e.g. repeated averages) are accumulated
    and averaged, matching both pymapvbvd's and twixtools' default behaviour.

    `skip_to_first_line` matches pymapvbvd's `flagSkipToFirstLine` (on by default for
    every category except 'image'/'phasestab'): the Lin/Par axes are shrunk to the
    acquired range (offset by their minimum) instead of the full nominal matrix size,
    which matters for scans like refscan/phasecor that only cover part of k-space.
    """
    sub = table[mask]
    n = len(sub)
    if n == 0:
        empty_shape = (0, 0) + (1,) * len(DIM_NAMES)
        if not legacy_layout:
            empty_shape = empty_shape[2:] + empty_shape[:2]
        return np.zeros(empty_shape, dtype=np.complex64)

    is_vd = version is TwixVersion.VD
    hdr = sub["hdr"]
    dim_values = _dim_values(hdr, skip_to_first_line)
    dim_sizes = tuple(int(dim_values[name].max()) + 1 for name in DIM_NAMES)

    ncol_all = hdr["SamplesInScan"].astype(np.int64)
    ncha_all = hdr["UsedChannels"].astype(np.int64)
    # Canonical shape: pymapvbvd itself just takes the first line's shape (no partial
    # Fourier column re-centering in its current implementation either), so this is
    # exact behavioural parity, not a shortcut.
    ncol = int(ncol_all[0])
    ncha = int(ncha_all[0])

    offsets = sub["file_offset"].astype(np.int64)
    is_reflected = unpack_flags(hdr["EvalInfoMask"])[:, MASK_BIT["REFLECT"]]

    # Real acquisitions almost always map each line to a distinct target index (no
    # repeated averages) — detect that once, globally, and use plain vectorized
    # assignment in that (common) case instead of `np.add.at`, which is dramatically
    # slower. Only fall back to add.at (+ count normalization) when duplicates exist.
    full_ravel = np.ravel_multi_index(tuple(dim_values[name] for name in DIM_NAMES), dim_sizes)
    has_duplicates = np.unique(full_ravel).size != n

    # Accumulate in (flat_counters, Cha, Col) order — matching the natural layout the
    # gather produces — and scatter along the *outermost* axis. Scattering into the
    # innermost (fastest-varying, contiguous) axis instead, as the public (Col, Cha,
    # ...) layout would require, is dramatically slower in numpy (indexed writes into
    # a contiguous axis touch scattered cache lines per element rather than copying
    # whole contiguous (Cha, Col) planes). The public array is produced by a single
    # bulk transpose+copy at the end instead of per-batch scattered writes.
    n_flat = 1
    for s in dim_sizes:
        n_flat *= s
    acc = np.zeros((n_flat, ncha, ncol), dtype=np.complex64)
    count_flat = np.zeros(n_flat, dtype=np.int32)

    homogeneous = bool(np.all(ncol_all == ncol) and np.all(ncha_all == ncha))
    if homogeneous:
        buckets = [(np.arange(n), ncol, ncha)]
    else:
        shapes = np.stack([ncol_all, ncha_all], axis=1)
        uniq, inverse = np.unique(shapes, axis=0, return_inverse=True)
        buckets = [
            (np.flatnonzero(inverse == k), int(uniq[k, 0]), int(uniq[k, 1]))
            for k in range(len(uniq))
        ]

    for idx, b_ncol, b_ncha in buckets:
        batch_lines = _batch_size(b_ncol, b_ncha, batch_bytes)
        for start in range(0, idx.size, batch_lines):
            chunk = idx[start : start + batch_lines]
            samples = _gather_batch(mm, offsets[chunk], b_ncol, b_ncha, is_vd)  # (m, ncha, ncol)

            refl = is_reflected[chunk]
            if refl.any():
                samples[refl] = samples[refl, :, ::-1]

            if b_ncol != ncol or b_ncha != ncha:
                fitted = np.zeros((samples.shape[0], ncha, ncol), dtype=np.complex64)
                m_col, m_cha = min(ncol, b_ncol), min(ncha, b_ncha)
                fitted[:, :m_cha, :m_col] = samples[:, :m_cha, :m_col]
                samples = fitted

            target = full_ravel[chunk]
            if has_duplicates:
                np.add.at(acc, target, samples)
                np.add.at(count_flat, target, 1)
            else:
                acc[target] = samples

    if has_duplicates:
        np.maximum(count_flat, 1, out=count_flat)
        acc /= count_flat[:, np.newaxis, np.newaxis]

    if legacy_layout:
        # (flat, Cha, Col) -> (Col, Cha, *dim_sizes): the one full transpose+copy this
        # module exists to avoid by default (see module docstring).
        out = np.ascontiguousarray(
            acc.reshape(dim_sizes + (ncha, ncol)).transpose(
                len(dim_sizes) + 1, len(dim_sizes), *range(len(dim_sizes))
            )
        )
        col_axis = 0
    else:
        # Free reshape (acc is already contiguous in exactly this order) -> no copy.
        out = acc.reshape(dim_sizes + (ncha, ncol))
        col_axis = -1

    if remove_os:
        out = remove_oversampling(out, axis=col_axis)

    return out
