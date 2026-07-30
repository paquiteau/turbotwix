import pathlib

import numpy as np
import pytest

import turbotwix as tw

DATA_DIR = pathlib.Path(__file__).parent / "data"


@pytest.fixture
def gre_path() -> str:
    return str(DATA_DIR / "gre.dat")


@pytest.fixture
def epi_path() -> str:
    return str(DATA_DIR / "epi.dat")


def line(
    ncol: int, ncha: int, flags: int = 0, counter: int = 1, lin: int = 0, base: float | None = None
) -> tuple[bytes, np.ndarray]:
    """One synthetic VD line; its samples carry recognisable values if `base` is given."""
    header = np.zeros(1, dtype=tw.VD_SCAN_HEADER)[0]
    header["SamplesInScan"] = ncol
    header["UsedChannels"] = ncha
    header["EvalInfoMask"] = flags
    header["ScanCounter"] = counter
    header["Counter"]["Lin"] = lin

    raw = header.tobytes()
    samples = np.zeros((ncha, ncol), dtype="<c8")
    for c in range(ncha):
        if base is not None:
            samples[c] = np.arange(ncol) + base + 1000 * c
        raw += np.zeros(1, dtype=tw.VD_CHANNEL_HEADER)[0].tobytes() + samples[c].tobytes()
    return raw, samples


def acqend(counter: int = 1) -> bytes:
    """The terminator block: a bare scan header carrying ACQEND, no samples."""
    header = np.zeros(1, dtype=tw.VD_SCAN_HEADER)[0]
    header["EvalInfoMask"] = int(tw.Flag.ACQEND)
    header["ScanCounter"] = counter
    return header.tobytes()


def sync(length: int, counter: int = 1) -> bytes:
    """A shape-less PMU/sideband block, whose length lives in the raw DMA field."""
    header = np.zeros(1, dtype=tw.VD_SCAN_HEADER)[0]
    header["EvalInfoMask"] = int(tw.Flag.SYNCDATA)
    header["FlagsAndDMALength"] = length
    header["ScanCounter"] = counter
    return header.tobytes() + bytes(length - tw.VD_SCAN_HEADER.itemsize)


def table_of(raw: bytes) -> tuple[np.ndarray, np.ndarray, bool]:
    """`(mm, table, truncated)` for a raw synthetic VD stream."""
    mm = np.frombuffer(raw, dtype=np.uint8)
    rows, truncated = tw.build_table(mm, 0, mm.size, tw.TwixVersion.VD)
    return mm, rows, truncated


def build(lines: list[tuple[int, int, float, int]]):
    """`(mm, table, expected)` for a stream of `(ncol, ncha, base, flags)` lines.

    An ACQEND terminator is appended, so the result is a complete measurement in the only
    layout turbotwix accepts.
    """
    raw, expected = b"", []
    for k, (ncol, ncha, base, flags) in enumerate(lines):
        chunk, samples = line(ncol, ncha, flags, counter=k + 1, lin=k, base=base)
        raw += chunk
        expected.append(samples)
    mm, rows, _ = table_of(raw + acqend(len(lines) + 1))
    return mm, rows, expected
