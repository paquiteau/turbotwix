"""Sample-level parity against pymapvbvd and twixtools. Skipped automatically if either
reference package is not installed (`uv sync --group parity`).

These are validation only — turbotwix's data model deliberately differs — so the checks
compare the *samples*, per line where possible, which is the strongest statement that
can be made independently of how each library chooses to arrange them.
"""

import importlib.util

import numpy as np
import pytest

import turbotwix as tw

HAS_PYMAPVBVD = importlib.util.find_spec("mapvbvd") is not None
HAS_TWIXTOOLS = importlib.util.find_spec("twixtools") is not None


@pytest.mark.skipif(not HAS_TWIXTOOLS, reason="twixtools not installed")
@pytest.mark.parametrize("dataset", ["gre_path", "epi_path"])
def test_parity_twixtools_per_line_samples(dataset, request):
    """Line for line, sample for sample, as stored on disk."""
    import twixtools

    path = request.getfixturevalue(dataset)
    [scan] = twixtools.read_twix(path, include_scans=[-1], parse_geometry=False, verbose=False)
    ref = np.stack([mdb.data for mdb in scan["mdb"] if mdb.is_image_scan()])

    m = tw.open_twix(path)[-1]
    got = m.read(m.lines.image, reflect=False)

    assert got.shape == ref.shape
    np.testing.assert_array_equal(got, ref)


@pytest.mark.skipif(not HAS_TWIXTOOLS, reason="twixtools not installed")
def test_parity_twixtools_image_line_counts(gre_path, epi_path):
    import twixtools

    for path, expected in ((gre_path, 160), (epi_path, 80)):
        [scan] = twixtools.read_twix(path, include_scans=[-1])
        assert sum(1 for mdb in scan["mdb"] if mdb.is_image_scan()) == expected
        assert len(tw.open_twix(path)[-1].lines.image) == expected


@pytest.mark.skipif(not HAS_PYMAPVBVD, reason="pymapvbvd not installed")
def test_parity_pymapvbvd_dense_image(gre_path):
    """`to_dense` must reproduce pymapvbvd's k-space array (modulo its axis order)."""
    import mapvbvd

    ref_scan = mapvbvd.mapVBVD(gre_path, quiet=True)
    if isinstance(ref_scan, list):
        ref_scan = ref_scan[-1]
    ref_scan.image.flagRemoveOS = False
    ref_scan.image.flagRampSampRegrid = False
    ref = np.squeeze(ref_scan.image[tuple([slice(None)] * 16)])  # (Col, Cha, Lin)

    m = tw.open_twix(gre_path)[-1]
    lines = m.lines.image
    got = tw.to_dense(m.read(lines), lines, ("Lin",))  # (Lin, Cha, Col)

    np.testing.assert_array_equal(got.transpose(2, 1, 0), ref)


@pytest.mark.skipif(not HAS_PYMAPVBVD, reason="pymapvbvd not installed")
def test_parity_pymapvbvd_remove_os(gre_path):
    import mapvbvd

    ref_scan = mapvbvd.mapVBVD(gre_path, quiet=True)
    if isinstance(ref_scan, list):
        ref_scan = ref_scan[-1]
    ref_scan.image.flagRemoveOS = True
    ref_scan.image.flagRampSampRegrid = False
    ref = np.squeeze(ref_scan.image[tuple([slice(None)] * 16)])

    m = tw.open_twix(gre_path)[-1]
    lines = m.lines.image
    got = tw.to_dense(tw.remove_oversampling(m.read(lines)), lines, ("Lin",))

    np.testing.assert_allclose(got.transpose(2, 1, 0), ref, atol=1e-4)
