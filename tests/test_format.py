import numpy as np

from turbotwix import _format, _read


def test_dtype_sizes():
    assert _format.VB_HEADER.itemsize == 128
    assert _format.VD_SCAN_HEADER.itemsize == 192
    assert _format.VD_CHANNEL_HEADER.itemsize == 32
    assert _format.LINE_COUNTER.itemsize == 28
    assert _format.SLICE_DATA.itemsize == 28
    assert _format.MR_PARC_RAID_FILE_ENTRY.itemsize == 152
    assert _format.MULTI_RAID_FILE_HEADER.itemsize == 8 + 64 * 152
    assert len(_format.MASK_ID) == 64
    assert _format.MASK_BIT["ACQEND"] == 0
    assert _format.MASK_BIT["SYNCDATA"] == 5


def test_detect_version_and_raid_directory(gre_path, epi_path):
    for path in (gre_path, epi_path):
        mm = _read.open_mmap(path)
        version = _format.detect_version(mm)
        assert version is _format.TwixVersion.VD
        entries = _format.parse_raid_directory(mm, version)
        assert len(entries) == 1
        assert entries[0].length > 0


def test_walk_headers_and_classify_gre(gre_path):
    mm = _read.open_mmap(gre_path)
    version = _format.detect_version(mm)
    entry = _format.parse_raid_directory(mm, version)[-1]
    from turbotwix import protocol

    _, hdr_len = protocol.parse_protocol(mm, entry.offset)
    table, truncated = _read.walk_headers(
        mm, entry.offset + hdr_len, entry.offset + entry.length, version
    )
    assert not truncated
    assert len(table) == 161  # 160 image lines + 1 ACQEND
    categories = _read.classify(table)
    assert int(categories["image"].sum()) == 160


def test_walk_headers_and_classify_epi(epi_path):
    mm = _read.open_mmap(epi_path)
    version = _format.detect_version(mm)
    entry = _format.parse_raid_directory(mm, version)[-1]
    from turbotwix import protocol

    _, hdr_len = protocol.parse_protocol(mm, entry.offset)
    table, truncated = _read.walk_headers(
        mm, entry.offset + hdr_len, entry.offset + entry.length, version
    )
    assert not truncated
    categories = _read.classify(table)
    assert int(categories["image"].sum()) == 80
    assert int(categories["phasecor"].sum()) == 3
    assert int(categories["rtfeedback"].sum()) == 3


def _pack_header(dma_len: int, eval_mask: int, ncol: int = 0, ncha: int = 0) -> bytes:
    h = np.zeros(1, dtype=_format.VD_SCAN_HEADER)[0]
    h["FlagsAndDMALength"] = dma_len
    h["EvalInfoMask"] = eval_mask
    h["SamplesInScan"] = ncol
    h["UsedChannels"] = ncha
    return h.tobytes()


def test_walk_headers_handles_syncdata_and_acqend():
    # One image line (ncol=4, ncha=1): 192 (scan header) + 1*(32+8*4) = 256 bytes.
    image_line = _pack_header(dma_len=0, eval_mask=0, ncol=4, ncha=1)
    image_line += bytes(32 + 8 * 4)  # one channel header + 4 complex64 samples

    # A SYNCDATA block: header trusts the raw dma length field directly (bit 5 set).
    sync_len = 192 + 16
    sync_line = _pack_header(dma_len=sync_len, eval_mask=1 << 5)
    sync_line += bytes(16)

    # ACQEND: bit 0 set, raw dma length == header size only.
    acqend_line = _pack_header(dma_len=192, eval_mask=1)

    buf = np.frombuffer(image_line + sync_line + acqend_line, dtype=np.uint8)

    table, truncated = _read.walk_headers(buf, 0, buf.size, _format.TwixVersion.VD)
    assert not truncated
    assert len(table) == 3
    assert table["file_offset"].tolist() == [0, 256, 256 + sync_len]
    assert table["dma_len"].tolist() == [256, sync_len, 192]

    categories = _read.classify(table)
    assert categories["image"].tolist() == [True, False, False]


def test_walk_headers_reports_truncation():
    # A single image-shaped header claiming more channels/samples than are actually
    # present in the buffer -> the walk must stop and report truncation, not read OOB.
    image_line = _pack_header(dma_len=0, eval_mask=0, ncol=4, ncha=1)
    buf = np.frombuffer(image_line, dtype=np.uint8)  # missing the channel+data payload

    table, truncated = _read.walk_headers(buf, 0, buf.size, _format.TwixVersion.VD)
    assert truncated
    assert len(table) == 1
