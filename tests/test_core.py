import pathlib
import struct

import numpy as np
import pytest
from conftest import build

import turbotwix as tw
from turbotwix import _format, _read, protocol


def test_open_twix_exposes_every_measurement(gre_path):
    f = tw.open_twix(gre_path)
    assert len(f) == 1
    assert f.version is _format.TwixVersion.VD
    assert isinstance(f[-1], tw.Measurement)
    assert f[0].length > 0
    assert list(f) == f.measurements


def test_line_selection(gre_path, epi_path):
    m = tw.open_twix(gre_path)[-1]
    assert len(m.lines) == 161  # 160 image lines + ACQEND
    assert len(m.lines.data) == 160
    assert len(m.lines.image) == 160
    assert len(m.lines.noise) == 0

    m = tw.open_twix(epi_path)[-1]
    assert len(m.lines.image) == 80
    assert len(m.lines.phasecor) == 3
    # Selections compose and stay LineTables.
    first_rep = m.lines.image[m.lines.image.counter("Seg") == 0]
    assert isinstance(first_rep, tw.LineTable)
    assert len(first_rep) <= len(m.lines.image)


def test_flags_and_counters(gre_path):
    m = tw.open_twix(gre_path)[-1]
    lines = m.lines
    assert lines.has(tw.Flag.ACQEND).sum() == 1
    assert lines.image.counter("Lin").tolist() == list(range(160))
    assert set(tw.COUNTERS) >= {"Lin", "Par", "Sli", "Rep", "Seg"}
    assert lines.image.shapes == {(2, 320)}


def test_read_returns_lines_not_a_hypercube(gre_path):
    m = tw.open_twix(gre_path)[-1]
    lines = m.lines.image
    samples = m.read(lines)
    assert samples.shape == (len(lines), 2, 320)
    assert samples.dtype == np.complex64

    # Partial read: only the requested lines are touched.
    subset = lines[10:20]
    np.testing.assert_array_equal(m.read(subset), samples[10:20])


def test_read_into_preallocated_buffer(gre_path, tmp_path):
    m = tw.open_twix(gre_path)[-1]
    lines = m.lines.image
    dst = np.memmap(tmp_path / "out.npy", dtype=np.complex64, mode="w+", shape=(len(lines), 2, 320))
    m.read(lines, out=dst)
    np.testing.assert_array_equal(np.asarray(dst), m.read(lines))


def test_headers_available_on_demand(gre_path):
    m = tw.open_twix(gre_path)[-1]
    headers = m.lines.image.headers()
    assert len(headers) == 160
    assert headers.dtype == _format.VD_SCAN_HEADER
    assert headers["Counter"]["Lin"].tolist() == list(range(160))
    # Fields the compact table drops are still reachable.
    assert "IceProgramPara" in headers.dtype.names
    assert "SliceData" in headers.dtype.names


def test_to_dense_folds_onto_chosen_counters(gre_path):
    m = tw.open_twix(gre_path)[-1]
    lines = m.lines.image
    samples = m.read(lines)
    dense = tw.to_dense(samples, lines, ("Lin",))
    assert dense.shape == (160, 2, 320)
    np.testing.assert_array_equal(dense[5], samples[5])
    np.testing.assert_array_equal(m.to_dense(dims=("Lin",)), dense)


def test_to_dense_refuses_to_average_silently():
    # Two lines with the same Lin but different Rep: folding on Lin alone collides.
    mm, table, expected = build([(4, 2, 100, 0), (4, 2, 200, 0)])
    table["counters"]["Lin"] = [0, 0]
    table["counters"]["Rep"] = [0, 1]
    lines = tw.LineTable(table, mm, _format.TwixVersion.VD)
    samples = np.stack(expected)

    with pytest.raises(ValueError, match="Rep"):
        tw.to_dense(samples, lines, ("Lin",))

    np.testing.assert_allclose(
        tw.to_dense(samples, lines, ("Lin",), reduce="mean")[0], (expected[0] + expected[1]) / 2
    )
    np.testing.assert_array_equal(
        tw.to_dense(samples, lines, ("Lin",), reduce="last")[0], expected[1]
    )
    assert tw.to_dense(samples, lines, ("Lin", "Rep")).shape == (1, 2, 2, 4)


def test_to_dense_origin(gre_path):
    mm, table, expected = build([(4, 2, 100, 0), (4, 2, 200, 0)])
    table["counters"]["Lin"] = [10, 11]
    lines = tw.LineTable(table, mm, _format.TwixVersion.VD)
    samples = np.stack(expected)

    assert tw.to_dense(samples, lines, ("Lin",), origin="min").shape[0] == 2
    assert tw.to_dense(samples, lines, ("Lin",), origin="zero").shape[0] == 12


def _truncated_copy(src: str, dst: pathlib.Path) -> str:
    """Copy `src` cut off just before its ACQEND line, rewriting the raid entry length
    to 0 so `parse_raid_directory` recomputes it from the new file size.
    """
    mm = _read.open_mmap(src)
    version = _format.detect_version(mm)
    entry = _format.parse_raid_directory(mm, version)[-1]
    _, hdr_len = protocol.parse_protocol(mm, entry.offset)
    table, _ = _read.walk_headers(mm, entry.offset + hdr_len, entry.offset + entry.length, version)
    keep = int(table["offset"][-1])  # offset of the ACQEND header

    raw = bytearray(pathlib.Path(src).read_bytes()[:keep])
    len_offset = _format.MR_PARC_RAID_FILE_HEADER.itemsize + 16  # measId, fileId, off
    raw[len_offset : len_offset + 8] = struct.pack("<Q", 0)
    dst.write_bytes(raw)
    return str(dst)


def test_truncated_file_raises_unless_allowed(gre_path, tmp_path):
    path = _truncated_copy(gre_path, tmp_path / "truncated.dat")

    with pytest.raises(tw.TruncatedFileError):
        tw.open_twix(path)[-1].lines

    with pytest.warns(UserWarning, match="before ACQEND"):
        m = tw.open_twix(path, allow_truncated=True)[-1]
        lines = m.lines
    assert len(lines.image) == 160
    np.testing.assert_array_equal(
        m.read(lines.image), tw.open_twix(gre_path)[-1].read(tw.open_twix(gre_path)[-1].lines.image)
    )
