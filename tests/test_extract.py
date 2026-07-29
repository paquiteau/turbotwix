import numpy as np
import pytest
from conftest import build, make_line

from turbotwix import _extract, _format
from turbotwix._format import Flag

VD = _format.TwixVersion.VD


def test_read_lines_returns_file_order_samples():
    mm, table, expected = build([(4, 2, 100, 0), (4, 2, 200, 0), (4, 2, 300, 0)])
    got = _extract.read_lines(mm, table, VD)
    assert got.shape == (3, 2, 4)
    assert got.dtype == np.complex64
    np.testing.assert_array_equal(got, np.stack(expected))


def test_read_lines_unreflects_by_default():
    mm, table, expected = build([(4, 2, 100, 0), (4, 2, 200, int(Flag.REFLECT))])
    np.testing.assert_array_equal(
        _extract.read_lines(mm, table, VD), np.stack([expected[0], expected[1][:, ::-1]])
    )
    np.testing.assert_array_equal(
        _extract.read_lines(mm, table, VD, reflect=False), np.stack(expected)
    )


def test_read_lines_into_caller_buffer():
    mm, table, expected = build([(4, 2, 100, 0), (4, 2, 200, 0)])
    out = np.zeros((2, 2, 4), dtype=np.complex64)
    got = _extract.read_lines(mm, table, VD, out=out)
    assert got is out
    np.testing.assert_array_equal(out, np.stack(expected))

    with pytest.raises(ValueError, match="out must be"):
        _extract.read_lines(mm, table, VD, out=np.zeros((2, 2, 4), dtype=np.complex128))


def test_read_lines_rejects_mixed_shapes():
    mm, table, _ = build([(4, 2, 100, 0), (8, 2, 200, 0)])
    with pytest.raises(ValueError, match="mixes line shapes"):
        _extract.read_lines(mm, table, VD)
    # ... and reading each shape separately works.
    assert _extract.read_lines(mm, table[table["ncol"] == 4], VD).shape == (1, 2, 4)


def test_read_lines_empty_selection():
    mm, table, _ = build([(4, 2, 100, 0)])
    assert _extract.read_lines(mm, table[:0], VD).shape == (0, 0, 0)


@pytest.mark.parametrize("batch_bytes", [1, 64, 1 << 20])
def test_read_lines_batching_is_transparent(batch_bytes):
    mm, table, expected = build([(4, 2, 100 * k, 0) for k in range(7)])
    np.testing.assert_array_equal(
        _extract.read_lines(mm, table, VD, batch_bytes=batch_bytes), np.stack(expected)
    )


def test_gather_paths_agree_for_evenly_and_irregularly_spaced_lines():
    # Big lines (>= _PER_LINE_MIN_BYTES) so the irregular selection takes the per-line
    # strided path; a short line between them makes the spacing irregular.
    entries, expected = [], []
    raw = b""
    offsets = []
    for k in range(3):
        offsets.append(len(raw))
        buf, data = make_line(2048, 2, base=k * 10_000)
        raw += buf
        expected.append(data)
        raw += make_line(8, 1, base=0)[0]
        entries.append(None)
    mm = np.frombuffer(raw, dtype=np.uint8)

    scan_prefix, chan_hdr = _format.header_sizes(VD)
    block_len = chan_hdr + 8 * 2048
    for picked in ([0, 1, 2], [0, 2]):
        sample0 = np.array(offsets, dtype=np.int64)[picked] + scan_prefix + chan_hdr
        strided = np.empty((len(picked), 2, 2048), dtype=np.complex64)
        assert _extract._strided_copy(mm, sample0, 2048, 2, block_len, strided)
        np.testing.assert_array_equal(strided, np.stack([expected[i] for i in picked]))


def test_small_irregular_batch_falls_back_to_index_gather():
    mm, table, expected = build([(4, 2, 100 * k, 0) for k in range(4)])
    picked = table[[0, 1, 3]]  # gaps of 1 then 2 lines
    scan_prefix, chan_hdr = _format.header_sizes(VD)
    out = np.empty((3, 2, 4), dtype=np.complex64)
    # Too small for the per-line loop to pay for itself; the index gather handles it.
    assert not _extract._strided_copy(
        mm, picked["offset"] + scan_prefix + chan_hdr, 4, 2, chan_hdr + 8 * 4, out
    )
    np.testing.assert_array_equal(
        _extract.read_lines(mm, picked, VD), np.stack([expected[0], expected[1], expected[3]])
    )


def test_remove_oversampling_crops_to_half():
    rng = np.random.default_rng(0)
    arr = rng.standard_normal((3, 2, 8)).astype(np.complex64)
    assert _extract.remove_oversampling(arr).shape == (3, 2, 4)
    for n in (7, 8, 11, 15, 16):
        arr = rng.standard_normal((1, 1, n)).astype(np.complex64)
        assert _extract.remove_oversampling(arr).shape[-1] == _extract.removed_os_len(n)
