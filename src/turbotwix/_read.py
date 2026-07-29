"""Sequential header walk (the one pass that can't be vectorized away) and fully
vectorized category classification.
"""

from __future__ import annotations

import mmap as _mmap_module

import numpy as np

from turbotwix import _format
from turbotwix._format import TruncatedFileError, TwixVersion

_INITIAL_CAPACITY = 4096

# Category boolean formulas replicate pymapvbvd's `evalMDH`/`readMDH` predicates
# (mapvbvd/mapVBVD.py) exactly, for drop-in behavioural parity.
CATEGORY_NAMES = (
    "image",
    "noise",
    "refscan",
    "refscanPC",
    "phasecor",
    "phasestab",
    "refscan_phasestab",
    "phasestab_ref0",
    "refscan_phasestab_ref0",
    "phasestab_ref1",
    "refscan_phasestab_ref1",
    "rtfeedback",
    "vop",
)


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


def _extended_dtype(header_dtype: np.dtype) -> np.dtype:
    return np.dtype([("hdr", header_dtype), ("file_offset", "<u8"), ("dma_len", "<u4")])


def walk_headers(
    mm: np.ndarray, scan_offset: int, scan_end: int, version: TwixVersion
) -> tuple[np.ndarray, bool]:
    """Walk every line header from `scan_offset` to `scan_end` (or until ACQEND).

    Returns `(table, truncated)` where `table` is a structured array with fields
    `hdr` (the raw VB/VD scan header), `file_offset` (absolute byte offset of the
    header) and `dma_len` (byte length of this line's full DMA block, i.e. the
    stride to the next header). `truncated` is True if the walk hit EOF before an
    ACQEND block.
    """
    is_vd = version is TwixVersion.VD
    header_dtype = _format.VD_SCAN_HEADER if is_vd else _format.VB_HEADER
    hdr_size = header_dtype.itemsize
    channel_hdr_size = 32 if is_vd else 128
    scan_hdr_prefix = 192 if is_vd else 0

    acqend_bit = np.uint64(1) << np.uint64(_format.MASK_BIT["ACQEND"])
    sync_bit = np.uint64(1) << np.uint64(_format.MASK_BIT["SYNCDATA"])

    ext_dtype = _extended_dtype(header_dtype)
    capacity = _INITIAL_CAPACITY
    table = np.empty(capacity, dtype=ext_dtype)
    n = 0
    pos = scan_offset
    truncated = False

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
            length = int(header["FlagsAndDMALength"]) & 0x01FFFFFF
        else:
            # Recompute from NCol/NCha rather than trusting the raw field: Siemens'
            # PackBit (common in EPI) can make the raw field wrong (same approach
            # as both pymapvbvd's loop_mdh_read and twixtools' Mdb._get_dma_len).
            ncol = int(header["SamplesInScan"])
            ncha = int(header["UsedChannels"])
            length = scan_hdr_prefix + ncha * (channel_hdr_size + 8 * ncol)

        if n == capacity:
            capacity *= 2
            grown = np.empty(capacity, dtype=ext_dtype)
            grown[:n] = table[:n]
            table = grown

        table[n]["hdr"] = header
        table[n]["file_offset"] = pos
        table[n]["dma_len"] = length
        n += 1

        if is_acqend:
            break
        if length <= 0 or pos + length > scan_end:
            truncated = True
            break
        pos += length

    return table[:n], truncated


def require_complete(table: np.ndarray, truncated: bool) -> None:
    if truncated:
        raise TruncatedFileError(
            f"scan ended before ACQEND after {len(table)} lines; file may be incomplete"
        )


def classify(table: np.ndarray) -> dict[str, np.ndarray]:
    """Vectorized per-category boolean masks, one array of shape (N,) per category."""
    bits = _format.unpack_flags(table["hdr"]["EvalInfoMask"])

    def flag(name: str) -> np.ndarray:
        return bits[:, _format.MASK_BIT[name]]

    acqend = flag("ACQEND")
    syncdata = flag("SYNCDATA")
    rtfb = flag("RTFEEDBACK")
    hpfb = flag("HPFEEDBACK")
    phascor = flag("PHASCOR")
    noiseadj = flag("NOISEADJSCAN")
    phasestab = flag("PHASESTABSCAN")
    refphasestab = flag("REFPHASESTABSCAN")
    patref = flag("PATREFSCAN")
    patrefandima = flag("PATREFANDIMASCAN")
    vop = flag("MDH_VOP")

    active = ~acqend & ~syncdata
    not_pure_patref = ~(patref & ~patrefandima)
    is_patref_any = patref | patrefandima
    is_phasestab_only = phasestab & ~refphasestab
    is_ref0 = refphasestab & ~phasestab
    is_ref1 = refphasestab & phasestab

    return {
        "image": ~(
            acqend
            | rtfb
            | hpfb
            | phascor
            | noiseadj
            | phasestab
            | refphasestab
            | syncdata
            | (patref & ~patrefandima)
        ),
        "noise": noiseadj & active,
        "refscan": is_patref_any & ~(phascor | phasestab | refphasestab | rtfb | hpfb) & active,
        "refscanPC": is_patref_any & phascor & active,
        "phasecor": phascor & not_pure_patref & active,
        "phasestab": is_phasestab_only & not_pure_patref & active,
        "refscan_phasestab": is_phasestab_only & is_patref_any & active,
        "phasestab_ref0": is_ref0 & not_pure_patref & active,
        "refscan_phasestab_ref0": is_ref0 & is_patref_any & active,
        "phasestab_ref1": is_ref1 & not_pure_patref & active,
        "refscan_phasestab_ref1": is_ref1 & is_patref_any & active,
        "rtfeedback": (rtfb | hpfb) & ~vop & active,
        "vop": rtfb & vop & active,
    }
