import pathlib

import pytest

DATA_DIR = pathlib.Path(__file__).parent / "data"


@pytest.fixture
def gre_path() -> str:
    return str(DATA_DIR / "gre.dat")


@pytest.fixture
def epi_path() -> str:
    return str(DATA_DIR / "epi.dat")
