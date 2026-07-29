import pathlib

import numpy as np
import pytest

from turbotwix import _format, _read

DATA_DIR = pathlib.Path(__file__).parent / "data"

# The two bundled samples, for reference: both VD, single measurement.
GRE = {"lines": 161, "image": 160, "shape": (2, 320)}
EPI = {"lines": 84, "image": 80, "phasecor": 3, "shape": (2, 160)}


@pytest.fixture
def gre_path() -> str:
    return str(DATA_DIR / "gre.dat")


@pytest.fixture
def epi_path() -> str:
    return str(DATA_DIR / "epi.dat")


def make_line(ncol: int, ncha: int, base: float, eval_mask: int = 0) -> tuple[bytes, np.ndarray]:
    """One synthetic VD line whose samples carry recognisable values."""
    h = np.zeros(1, dtype=_format.VD_SCAN_HEADER)[0]
    h["SamplesInScan"] = ncol
    h["UsedChannels"] = ncha
    h["EvalInfoMask"] = eval_mask
    buf = h.tobytes()
    data = np.zeros((ncha, ncol), dtype="<c8")
    for c in range(ncha):
        data[c] = np.arange(ncol) + base + 1000 * c
        buf += np.zeros(1, dtype=_format.VD_CHANNEL_HEADER)[0].tobytes() + data[c].tobytes()
    return buf, data


def build(lines: list[tuple[int, int, float, int]]):
    """`(mm, table, expected_samples)` for a synthetic stream of (ncol, ncha, base, flags)."""
    raw, expected = b"", []
    for ncol, ncha, base, mask in lines:
        buf, data = make_line(ncol, ncha, base, mask)
        raw += buf
        expected.append(data)
    mm = np.frombuffer(raw, dtype=np.uint8)
    table, _ = _read.walk_headers(mm, 0, mm.size, _format.TwixVersion.VD)
    return mm, table, expected


@pytest.fixture
def build_lines():
    return build
