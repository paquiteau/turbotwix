"""Sample extraction, line selection and the object model, on synthetic and real files."""

import pathlib
import struct

import numpy as np
import pytest
from conftest import build

import turbotwix as tw
from turbotwix import Flag

VD = tw.TwixVersion.VD


# --- extraction -----------------------------------------------------------


def test_read_lines_returns_file_order_samples():
    mm, table, expected = build([(4, 2, 100, 0), (4, 2, 200, 0), (4, 2, 300, 0)])
    got = tw.read_lines(mm, table, VD)
    assert got.shape == (3, 2, 4)
    assert got.dtype == np.complex64
    np.testing.assert_array_equal(got, np.stack(expected))


def test_read_lines_unreflects_by_default():
    mm, table, expected = build([(4, 2, 100, 0), (4, 2, 200, int(Flag.REFLECT))])
    np.testing.assert_array_equal(
        tw.read_lines(mm, table, VD), np.stack([expected[0], expected[1][:, ::-1]])
    )
    np.testing.assert_array_equal(
        tw.read_lines(mm, table, VD, reflect=False), np.stack(expected)
    )


def test_read_lines_unreflects_in_bounded_chunks(monkeypatch):
    # The flip is chunked to bound its temporary; the chunking must not be observable.
    mm, table, expected = build(
        [(4, 2, 100 * k, int(Flag.REFLECT) * (k % 2)) for k in range(9)]
    )
    monkeypatch.setattr(tw, "_FLIP_BUDGET_BYTES", 1)
    np.testing.assert_array_equal(
        tw.read_lines(mm, table, VD),
        np.stack([d[:, ::-1] if k % 2 else d for k, d in enumerate(expected)]),
    )


def test_read_lines_into_caller_buffer():
    mm, table, expected = build([(4, 2, 100, 0), (4, 2, 200, 0)])
    out = np.zeros((2, 2, 4), dtype=np.complex64)
    assert tw.read_lines(mm, table, VD, out=out) is out
    np.testing.assert_array_equal(out, np.stack(expected))

    with pytest.raises(ValueError, match="out must be"):
        tw.read_lines(mm, table, VD, out=np.zeros((2, 2, 4), dtype=np.complex128))


def test_read_lines_empty_selection():
    mm, table, _ = build([(4, 2, 100, 0)])
    assert tw.read_lines(mm, table[:0], VD).shape == (0, 0, 0)


def test_read_lines_irregularly_spaced_selection():
    # A selection with uneven gaps cannot be one strided view, so it takes the per-line
    # path; the samples must be identical either way.
    mm, table, expected = build([(4, 2, 100 * k, 0) for k in range(6)])
    for picked in ([0, 1, 2], [0, 1, 3], [5, 0], [2]):
        np.testing.assert_array_equal(
            tw.read_lines(mm, table[picked], VD),
            np.stack([expected[i] for i in picked]),
        )


# --- selection ------------------------------------------------------------


def test_selection_presets():
    kinds = [
        ("image", 0),
        ("image", int(Flag.PATREFANDIMASCAN)),  # reference *and* image: counts as both
        ("noise", int(Flag.NOISEADJSCAN)),
        ("refscan", int(Flag.PATREFSCAN)),
        ("phasecor", int(Flag.PHASCOR)),
        (
            "refscanPC",
            int(Flag.PATREFSCAN | Flag.PHASCOR),
        ),  # reference-only, not phasecor
        ("feedback", int(Flag.RTFEEDBACK)),
    ]
    mm, table, _ = build([(4, 2, 0, mask) for _, mask in kinds])
    lines = tw.LineTable(table, mm, VD)

    def picked(sel):
        return sorted(int(np.flatnonzero(table["offset"] == o)[0]) for o in sel.offset)

    assert picked(lines.image) == [0, 1]  # plain image, and reference-and-image
    assert picked(lines.noise) == [2]
    assert picked(lines.refscan) == [1, 3]  # reference-and-image counts here too
    assert picked(lines.phasecor) == [4]  # not 5: that one is reference-only
    assert picked(lines.select(Flag.RTFEEDBACK)) == [6]


def test_selections_compose_and_stay_line_tables(epi_path):
    m = tw.open_twix(epi_path)[-1]
    assert len(m.lines.image) == 80
    assert len(m.lines.phasecor) == 3
    first_seg = m.lines.image[m.lines.image.counter("Seg") == 0]
    assert isinstance(first_seg, tw.LineTable)
    assert 0 < len(first_seg) <= len(m.lines.image)
    assert repr(m.lines.image).startswith("LineTable(80 lines")
    assert repr(m.lines[:0]) == "LineTable(0 lines)"
    assert m.lines[:0].shape == (0, 0)


# --- to_dense -------------------------------------------------------------


def test_to_dense_folds_onto_chosen_counters(gre_path):
    m = tw.open_twix(gre_path)[-1]
    lines = m.lines.image
    samples = m.read(lines)
    dense = tw.to_dense(samples, lines, ("Lin",))
    assert dense.shape == (160, 2, 320)
    np.testing.assert_array_equal(dense[5], samples[5])
    np.testing.assert_array_equal(m.to_dense(dims=("Lin",)), dense)


def test_minimal_dims_keeps_genuinely_independent_counters():
    # Ave and Seg each take both values and neither determines the other, so nothing may
    # be dropped — the grid stays 2x2 for 3 lines.
    mm, table, expected = build([(4, 2, 100 * k, 0) for k in range(3)])
    table["counters"]["Lin"] = 40
    table["counters"]["Seg"] = [1, 0, 1]
    table["counters"]["Ave"] = [0, 0, 1]
    lines = tw.LineTable(table, mm, VD)

    assert tw._varying_counters(lines) == ("Ave", "Seg")
    assert tw.minimal_dims(lines) == ("Ave", "Seg")
    assert tw.to_dense(np.stack(expected), lines, "minimal").shape == (2, 2, 2, 4)


# --- the object model on real files ---------------------------------------


def test_open_twix_exposes_every_measurement(gre_path):
    f = tw.open_twix(gre_path)
    assert len(f) == 1
    assert f.version is tw.TwixVersion.VD
    assert isinstance(f[-1], tw.Measurement)
    assert f[0].length > 0
    assert list(f) == f.measurements
    assert "gre.dat" in repr(f) and "MiB" in repr(f[0])


def test_flags_and_counters(gre_path):
    lines = tw.open_twix(gre_path)[-1].lines
    assert len(lines) == 160  # the ACQEND terminator is not a line
    assert len(lines.image) == 160
    assert len(lines.noise) == 0
    assert lines.has(Flag.ACQEND).sum() == 0  # not part of the table
    assert lines.image.counter("Lin").tolist() == list(range(160))
    assert lines.image.shape == (2, 320)
    assert lines.counters["Lin"].tolist() == lines.counter("Lin").tolist()


def test_read_returns_lines_not_a_hypercube(gre_path):
    m = tw.open_twix(gre_path)[-1]
    lines = m.lines.image
    samples = m.read(lines)
    assert samples.shape == (len(lines), 2, 320)
    assert samples.dtype == np.complex64

    # Partial read: only the requested lines are touched.
    np.testing.assert_array_equal(m.read(lines[10:20]), samples[10:20])
    # No selection means every line of the measurement.
    np.testing.assert_array_equal(m.read(), samples)


def test_read_into_preallocated_buffer(gre_path, tmp_path):
    m = tw.open_twix(gre_path)[-1]
    lines = m.lines.image
    dst = np.memmap(
        tmp_path / "out.dat", dtype=np.complex64, mode="w+", shape=(len(lines), 2, 320)
    )
    m.read(lines, out=dst)
    np.testing.assert_array_equal(np.asarray(dst), m.read(lines))


def test_headers_available_on_demand(gre_path):
    headers = tw.open_twix(gre_path)[-1].lines.image.headers()
    assert len(headers) == 160
    assert headers.dtype == tw.VD_SCAN_HEADER
    assert headers["Counter"]["Lin"].tolist() == list(range(160))


def _truncated_copy(src: str, dst: pathlib.Path) -> str:
    """Copy `src` cut off just before its ACQEND line, rewriting the raid entry length to
    0 so `parse_raid_directory` recomputes it from the new file size.
    """
    mm = tw.open_mmap(src)
    version = tw.detect_version(mm)
    entry = tw.parse_raid_directory(mm, version)[-1]
    _, hdr_len = tw.parse_protocol(mm, entry.offset)
    table, _ = tw.build_table(
        mm, entry.offset + hdr_len, entry.offset + entry.length, version
    )
    stride = int(table["offset"][1] - table["offset"][0])
    keep = int(table["offset"][-1]) + stride  # first byte of the ACQEND header

    raw = bytearray(pathlib.Path(src).read_bytes()[:keep])
    len_offset = 8 + 16  # raid header, then measId, fileId, off
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

    reference = tw.open_twix(gre_path)[-1]
    np.testing.assert_array_equal(
        m.read(lines.image), reference.read(reference.lines.image)
    )
