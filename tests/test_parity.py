"""End-to-end parity against pymapvbvd and twixtools. Skipped automatically if either
reference package is not installed (`uv sync --group parity`).
"""

import importlib.util

import numpy as np
import pytest

import turbotwix as tw

HAS_PYMAPVBVD = importlib.util.find_spec("mapvbvd") is not None
HAS_TWIXTOOLS = importlib.util.find_spec("twixtools") is not None


@pytest.mark.skipif(not HAS_PYMAPVBVD, reason="pymapvbvd not installed")
@pytest.mark.parametrize("dataset", ["gre_path", "epi_path"])
def test_parity_pymapvbvd_raw_image(dataset, request):
    path = request.getfixturevalue(dataset)
    import mapvbvd

    ref_scan = mapvbvd.mapVBVD(path, quiet=True)
    if isinstance(ref_scan, list):
        ref_scan = ref_scan[-1]
    ref_scan.image.flagRemoveOS = False
    ref_scan.image.flagRampSampRegrid = False
    ref = ref_scan.image[tuple([slice(None)] * 16)]

    scan = tw.read_twix(path, legacy_layout=True)
    got = scan.image[:]

    assert got.shape == ref.shape
    np.testing.assert_array_equal(got, ref)


@pytest.mark.skipif(not HAS_PYMAPVBVD, reason="pymapvbvd not installed")
def test_parity_pymapvbvd_remove_os(gre_path):
    import mapvbvd

    ref_scan = mapvbvd.mapVBVD(gre_path, quiet=True)
    if isinstance(ref_scan, list):
        ref_scan = ref_scan[-1]
    ref_scan.image.flagRemoveOS = True
    ref_scan.image.flagRampSampRegrid = False
    ref = ref_scan.image[tuple([slice(None)] * 16)]

    scan = tw.read_twix(gre_path, remove_os=True, legacy_layout=True)
    got = scan.image[:]

    assert got.shape == ref.shape
    np.testing.assert_allclose(got, ref, atol=1e-4)


@pytest.mark.skipif(not HAS_PYMAPVBVD, reason="pymapvbvd not installed")
def test_parity_pymapvbvd_default_layout_matches_legacy_transposed(gre_path):
    """The default (fast) axis order should carry the exact same data as
    legacy_layout=True, just permuted -- this is the whole point of the layout split.
    """
    scan_default = tw.read_twix(gre_path)
    scan_legacy = tw.read_twix(gre_path, legacy_layout=True)

    default = scan_default.image[:]
    legacy = scan_legacy.image[:]

    ndim = legacy.ndim
    # legacy order: (Col, Cha, *loop_dims) -> default order: (*loop_dims, Cha, Col)
    perm = (*range(2, ndim), 1, 0)
    np.testing.assert_array_equal(default, legacy.transpose(perm))


@pytest.mark.skipif(not HAS_TWIXTOOLS, reason="twixtools not installed")
def test_parity_twixtools_default_layout_no_transpose_needed(gre_path):
    """twixtools itself keeps Cha/Col innermost (map_twix.py's `_dim_order`) for the
    same reason turbotwix's default layout does; confirm the two line up (after
    permuting twixtools' reversed loop-dim order to ours) with no transpose on our
    side, and that the values agree bit-exactly.
    """
    import twixtools
    from twixtools.map_twix import map_twix

    [scan] = twixtools.read_twix(gre_path, include_scans=[-1], parse_geometry=False, verbose=False)
    ref_array = map_twix(scan)["image"]
    ref_data = ref_array[:]
    ref_dims = tuple(ref_array.dim_order)

    got_scan = tw.read_twix(gre_path)
    got_data = got_scan.image[:]
    got_dims = got_scan.image.dims

    assert set(ref_dims) == set(got_dims)
    perm = tuple(ref_dims.index(name) for name in got_dims)
    np.testing.assert_array_equal(got_data, ref_data.transpose(perm))


@pytest.mark.skipif(not HAS_TWIXTOOLS, reason="twixtools not installed")
def test_parity_twixtools_line_counts(gre_path, epi_path):
    import twixtools

    for path, expected in ((gre_path, {"image": 160}), (epi_path, {"image": 80})):
        [scan] = twixtools.read_twix(path, include_scans=[-1])
        image_count = sum(1 for mdb in scan["mdb"] if mdb.is_image_scan())
        assert image_count == expected["image"]

        got = tw.read_twix(path)
        assert got.image.n_acq == expected["image"]
