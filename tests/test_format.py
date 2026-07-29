import struct

import numpy as np
import pytest

from turbotwix import _format, _read
from turbotwix._format import Flag


def test_dtype_sizes():
    assert _format.VB_HEADER.itemsize == 128
    assert _format.VD_SCAN_HEADER.itemsize == 192
    assert _format.VD_CHANNEL_HEADER.itemsize == 32
    assert _format.LINE_COUNTER.itemsize == 28
    assert _format.LINE_DTYPE.itemsize == 48
    assert _format.MR_PARC_RAID_FILE_ENTRY.itemsize == 152
    assert _format.MULTI_RAID_FILE_HEADER.itemsize == 8 + 64 * 152


def test_flags_cover_all_64_bits():
    assert Flag.ACQEND == 1
    assert Flag.SYNCDATA == 1 << 5
    assert Flag.REFLECT == 1 << 24
    assert max(int(f) for f in Flag) == 1 << 63


def test_has_flag_requires_every_named_bit():
    flags = np.array([0, int(Flag.PHASCOR), int(Flag.PHASCOR | Flag.REFLECT)], dtype=np.uint64)
    assert _format.has_flag(flags, Flag.PHASCOR).tolist() == [False, True, True]
    assert _format.has_flag(flags, Flag.PHASCOR | Flag.REFLECT).tolist() == [False, False, True]


def test_header_sizes_match_dtypes():
    assert _format.header_sizes(_format.TwixVersion.VD) == (192, 32)
    assert _format.header_sizes(_format.TwixVersion.VB) == (0, 128)


def test_detect_version_and_raid_directory(gre_path, epi_path):
    for path in (gre_path, epi_path):
        mm = _read.open_mmap(path)
        version = _format.detect_version(mm)
        assert version is _format.TwixVersion.VD
        entries = _format.parse_raid_directory(mm, version)
        assert len(entries) == 1
        assert entries[0].length > 0


def test_detect_version_rejects_non_twix_files():
    with pytest.raises(_format.UnsupportedVersionError):
        _format.detect_version(np.zeros(4, dtype=np.uint8))

    # Not VD (second u32 > MAX_RAID_ENTRIES) and an implausible VB header length.
    garbage = np.frombuffer(struct.pack("<II", 0xDEADBEEF, 0xFFFF) + bytes(64), dtype=np.uint8)
    with pytest.raises(_format.UnsupportedVersionError):
        _format.detect_version(garbage)


# --- header walk ----------------------------------------------------------


def line(ncol: int, ncha: int, eval_mask: int = 0, lin: int = 0, dma_len: int = 0) -> bytes:
    """One complete VD line: scan header + per-channel (header + samples)."""
    h = np.zeros(1, dtype=_format.VD_SCAN_HEADER)[0]
    h["SamplesInScan"] = ncol
    h["UsedChannels"] = ncha
    h["EvalInfoMask"] = eval_mask
    h["FlagsAndDMALength"] = dma_len
    h["Counter"]["Lin"] = lin
    return h.tobytes() + bytes(ncha * (32 + 8 * ncol))


def acqend() -> bytes:
    return line(0, 0, eval_mask=int(Flag.ACQEND), dma_len=192)


def _sequential_walk(mm: np.ndarray, start: int, end: int) -> tuple[list, bool]:
    """Reference: the line-at-a-time walk that `walk_headers` vectorizes into runs."""
    pos, out = start, []
    while pos + 192 <= end:
        h = mm[pos : pos + 192].view(_format.VD_SCAN_HEADER)[0]
        ev = int(h["EvalInfoMask"])
        if ev & 1 or ev & (1 << 5):
            length = int(h["FlagsAndDMALength"]) & (2**25 - 1)
        else:
            length = 192 + int(h["UsedChannels"]) * (32 + 8 * int(h["SamplesInScan"]))
        out.append((pos, length))
        if ev & 1:
            return out, False
        if length <= 0 or pos + length > end:
            return out, True
        pos += length
    return out, True


def assert_matches_sequential(raw: bytes) -> np.ndarray:
    mm = np.frombuffer(raw, dtype=np.uint8)
    table, truncated = _read.walk_headers(mm, 0, mm.size, _format.TwixVersion.VD)
    expected, exp_truncated = _sequential_walk(mm, 0, mm.size)
    assert truncated == exp_truncated
    assert table["offset"].tolist() == [off for off, _ in expected]
    return table


def test_walk_run_detection_matches_sequential():
    # Long runs either side of a single odd line, then a differently-shaped tail.
    raw = b"".join(line(4, 2, lin=i) for i in range(50))
    raw += line(6, 1, lin=99)
    raw += b"".join(line(4, 2, lin=i) for i in range(30))
    sync_len = 192 + 64
    raw += line(0, 0, eval_mask=int(Flag.SYNCDATA), dma_len=sync_len) + bytes(sync_len - 192)
    raw += b"".join(line(8, 3, lin=i) for i in range(20))
    raw += acqend()

    table = assert_matches_sequential(raw)
    assert len(table) == 50 + 1 + 30 + 1 + 20 + 1
    assert table["counters"]["Lin"][:50].tolist() == list(range(50))
    assert table["ncol"][:50].tolist() == [4] * 50


def test_walk_interleaved_shapes_match_sequential():
    # Median run length 1 (as in real spiral/feedback-interleaved sequences): exercises
    # the adaptive give-up path, which must not change the result.
    raw = b"".join(line(4, 2, lin=i) + line(1, 1, lin=i) for i in range(40))
    raw += acqend()

    table = assert_matches_sequential(raw)
    assert len(table) == 81


def test_walk_reports_truncation():
    # A header claiming more channels/samples than the buffer holds: the walk must stop
    # and report truncation, not read out of bounds.
    mm = np.frombuffer(line(4, 1)[:192], dtype=np.uint8)
    table, truncated = _read.walk_headers(mm, 0, mm.size, _format.TwixVersion.VD)
    assert truncated
    assert len(table) == 1


def test_read_headers_recovers_full_scan_headers():
    raw = b"".join(line(4, 2, lin=i) for i in range(5)) + acqend()
    mm = np.frombuffer(raw, dtype=np.uint8)
    table, _ = _read.walk_headers(mm, 0, mm.size, _format.TwixVersion.VD)

    headers = _read.read_headers(mm, table["offset"], _format.TwixVersion.VD)
    assert headers.dtype == _format.VD_SCAN_HEADER
    assert headers["Counter"]["Lin"][:5].tolist() == list(range(5))
    assert headers["SamplesInScan"][:5].tolist() == [4] * 5
