"""Binary layout: dtype sizes, flag bits, version sniffing, the raid directory."""

import struct

import numpy as np
import pytest

import turbotwix as tw
from turbotwix import Flag


def test_dtype_sizes():
    assert tw.dtypes.VB_HEADER.itemsize == 128
    assert tw.dtypes.VD_SCAN_HEADER.itemsize == 192
    assert tw.dtypes.VD_CHANNEL_HEADER.itemsize == 32
    assert tw.dtypes.LINE_COUNTER.itemsize == 28
    assert tw.dtypes.LINE_DTYPE.itemsize == 48
    assert tw.dtypes.RAID_DIRECTORY.itemsize == 8 + 64 * 152


def test_header_sizes_match_dtypes():
    assert tw.dtypes.HEADER_SIZES[tw.TwixVersion.VD] == (192, 32)
    assert tw.dtypes.HEADER_SIZES[tw.TwixVersion.VB] == (0, 128)


def test_flags_cover_all_64_bits():
    assert Flag.ACQEND == 1
    assert Flag.SYNCDATA == 1 << 5
    assert Flag.REFLECT == 1 << 24
    assert Flag.RESERVED_6 == 1 << 6  # undocumented bits keep their position
    assert max(int(f) for f in Flag) == 1 << 63


def _lines_with_flags(*flags: int) -> tw.LineTable:
    rows = np.zeros(len(flags), dtype=tw.dtypes.LINE_DTYPE)
    rows["flags"] = flags
    return tw.LineTable(rows, np.zeros(0, dtype=np.uint8), tw.TwixVersion.VD)


def test_has_flag_requires_every_named_bit():
    lines = _lines_with_flags(0, int(Flag.PHASCOR), int(Flag.PHASCOR | Flag.REFLECT))
    assert lines.has_flag(Flag.PHASCOR).tolist() == [False, True, True]
    assert lines.has_flag(Flag.PHASCOR | Flag.REFLECT).tolist() == [False, False, True]
    assert lines.has_any_flag(Flag.PHASCOR | Flag.REFLECT).tolist() == [
        False,
        True,
        True,
    ]


def test_counters_are_the_14_loop_counters():
    assert len(tw.COUNTERS) == 14
    assert tw.COUNTERS[:4] == ["Lin", "Ave", "Sli", "Par"]


def test_detect_version_and_raid_directory(gre_path, epi_path):
    for path in (gre_path, epi_path):
        mm = tw.data.open_mmap(path)
        version = tw.dtypes.detect_version(mm)
        assert version is tw.TwixVersion.VD
        entries = tw.dtypes.parse_raid_directory(mm, version)
        assert len(entries) == 1
        assert entries[0].length > 0


def test_detect_version_rejects_non_twix_files():
    with pytest.raises(tw.UnsupportedVersionError):
        tw.dtypes.detect_version(np.zeros(4, dtype=np.uint8))

    # Not VD (second u32 > MAX_RAID_ENTRIES) and an implausible VB header length.
    garbage = np.frombuffer(
        struct.pack("<II", 0xDEADBEEF, 0xFFFF) + bytes(64), dtype=np.uint8
    )
    with pytest.raises(tw.UnsupportedVersionError):
        tw.dtypes.detect_version(garbage)


def test_vb_files_are_one_synthetic_raid_entry():
    # A VB file starts with its text header length, which is large — that is the whole
    # difference from a VD container, whose first word is small.
    mm = np.frombuffer(struct.pack("<I", 20_000) + bytes(30_000), dtype=np.uint8)
    assert tw.dtypes.detect_version(mm) is tw.TwixVersion.VB
    [entry] = tw.dtypes.parse_raid_directory(mm, tw.TwixVersion.VB)
    assert (entry.offset, entry.length) == (0, mm.size)
