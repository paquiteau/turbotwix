import numpy as np

from turbotwix import _extract, _format, _read

NCOL, NCHA = 2, 2


def _make_line(lin: int, base: int, eval_mask: int = 0) -> tuple[bytes, np.ndarray]:
    h = np.zeros(1, dtype=_format.VD_SCAN_HEADER)[0]
    h["SamplesInScan"] = NCOL
    h["UsedChannels"] = NCHA
    h["EvalInfoMask"] = eval_mask
    h["Counter"]["Lin"] = lin
    buf = h.tobytes()
    data = np.zeros((NCHA, NCOL), dtype="<c8")
    for c in range(NCHA):
        for s in range(NCOL):
            data[c, s] = complex(base + c * 10 + s, 0)
        ch = np.zeros(1, dtype=_format.VD_CHANNEL_HEADER)[0]
        buf += ch.tobytes()
        buf += data[c].tobytes()
    return buf, data


def _build_test_table():
    buf0, d0 = _make_line(lin=0, base=100)
    buf1, d1 = _make_line(lin=1, base=200)
    buf2, d2 = _make_line(lin=1, base=300)  # duplicate Lin -> must be averaged with d1
    reflect_bit = 1 << _format.MASK_BIT["REFLECT"]
    buf3, d3 = _make_line(lin=2, base=400, eval_mask=reflect_bit)

    raw = buf0 + buf1 + buf2 + buf3
    mm = np.frombuffer(raw, dtype=np.uint8)
    table, truncated = _read.walk_headers(mm, 0, mm.size, _format.TwixVersion.VD)
    assert len(table) == 4
    return mm, table, (d0, d1, d2, d3)


def test_read_category_gather_average_and_reflect_legacy_layout():
    mm, table, (d0, d1, d2, d3) = _build_test_table()
    mask = np.ones(4, dtype=bool)
    arr = _extract.read_category(
        mm,
        table,
        mask,
        _format.TwixVersion.VD,
        remove_os=False,
        skip_to_first_line=False,
        legacy_layout=True,
    )
    assert arr.shape[:3] == (NCOL, NCHA, 3)

    lin0 = arr[:, :, 0, ...].squeeze()
    lin1 = arr[:, :, 1, ...].squeeze()
    lin2 = arr[:, :, 2, ...].squeeze()

    np.testing.assert_allclose(lin0, d0.T)
    np.testing.assert_allclose(lin1, ((d1 + d2) / 2).T)  # duplicate Lin averaged
    np.testing.assert_allclose(lin2, d3[:, ::-1].T)  # REFLECT flips the readout axis


def test_read_category_default_layout_matches_legacy_transposed():
    mm, table, _ = _build_test_table()
    mask = np.ones(4, dtype=bool)
    kwargs = dict(remove_os=False, skip_to_first_line=False)

    legacy = _extract.read_category(
        mm, table, mask, _format.TwixVersion.VD, legacy_layout=True, **kwargs
    )
    default = _extract.read_category(
        mm, table, mask, _format.TwixVersion.VD, legacy_layout=False, **kwargs
    )

    assert legacy.shape == (NCOL, NCHA, 3) + (1,) * (len(_extract.DIM_NAMES) - 1)
    assert default.shape == (3,) + (1,) * (len(_extract.DIM_NAMES) - 1) + (NCHA, NCOL)

    # Only Lin varies (size 3); every other axis is a singleton. Same data, different
    # (documented) axis order once squeezed down to (Lin, Cha, Col) vs (Col, Cha, Lin).
    np.testing.assert_array_equal(default.squeeze(), legacy.squeeze().transpose(2, 1, 0))


def test_read_category_empty_mask_returns_empty_array():
    buf0, _ = _make_line(lin=0, base=1)
    mm = np.frombuffer(buf0, dtype=np.uint8)
    table, _ = _read.walk_headers(mm, 0, mm.size, _format.TwixVersion.VD)

    arr = _extract.read_category(mm, table, np.zeros(1, dtype=bool), _format.TwixVersion.VD)
    assert arr.shape[-2:] == (0, 0)

    arr_legacy = _extract.read_category(
        mm, table, np.zeros(1, dtype=bool), _format.TwixVersion.VD, legacy_layout=True
    )
    assert arr_legacy.shape[:2] == (0, 0)


def test_remove_oversampling_crops_to_half():
    arr = np.random.default_rng(0).standard_normal((8, 1, 1)).astype(np.complex64)
    out = _extract.remove_oversampling(arr, axis=0)
    assert out.shape[0] == 4
