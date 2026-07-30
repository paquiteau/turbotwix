"""`build_table`: the uniform-run fast path, the walk, and truncation."""

import numpy as np
import pytest
from conftest import acqend, line, sync, table_of

import turbotwix as tw

STRIDE = 192 + 2 * (32 + 8 * 4)  # one (ncha=2, ncol=4) VD line


def lines(count: int, first_counter: int = 1) -> bytes:
    return b"".join(line(4, 2, counter=first_counter + i, lin=i)[0] for i in range(count))


def test_uniform_run_is_the_lines_without_their_terminator():
    mm, table, truncated = table_of(lines(50) + acqend(51))
    assert not truncated
    assert len(table) == 50  # the ACQEND block is verified, not returned as a row
    assert table["offset"].tolist() == [i * STRIDE for i in range(50)]
    assert table["counters"]["Lin"].tolist() == list(range(50))
    assert table["ncol"].tolist() == [4] * 50
    assert table["ncha"].tolist() == [2] * 50


def test_uniform_run_declines_on_a_broken_scan_counter():
    # Right shape and flags everywhere, but the counters do not advance by a constant step,
    # so these cannot be N consecutive headers: the hypothesis is not accepted.
    raw = b"".join(line(4, 2, counter=c)[0] for c in (1, 2, 7, 8)) + acqend(9)
    mm = np.frombuffer(raw, dtype=np.uint8)
    assert tw._uniform_run(mm, 0, mm.size, tw.TwixVersion.VD, tw.VD_SCAN_HEADER) is None
    # The walk reads it anyway: a strange counter is not a reason to refuse samples.
    _, table, _ = table_of(raw)
    assert len(table) == 4


def test_walk_skips_interleaved_syncdata():
    # The non-Cartesian layout: image lines with a sideband block between them. The
    # uniform-run hypothesis fails, the walk handles it, and the sync blocks are stepped
    # over rather than recorded.
    raw, counter, offsets = b"", 1, []
    for i in range(6):
        offsets.append(len(raw))
        raw += line(4, 2, counter=counter, lin=i)[0]
        counter += 1
        if i % 2:
            raw += sync(256, counter)
            counter += 1

    _, table, truncated = table_of(raw + acqend(counter))
    assert not truncated
    assert table["offset"].tolist() == offsets
    assert table["counters"]["Lin"].tolist() == list(range(6))
    assert sorted(set(np.diff(table["offset"]).tolist())) == [STRIDE, STRIDE + 256]


def test_walk_handles_a_non_adc_first_block():
    _, table, truncated = table_of(sync(256, 1) + line(4, 2, counter=2, lin=7)[0] + acqend(3))
    assert not truncated
    assert table["counters"]["Lin"].tolist() == [7]
    assert table["offset"].tolist() == [256]


def test_walk_handles_several_runs_of_one_shape():
    # Same shape either side of a sync block, but the run is interrupted, so the arithmetic
    # path declines and the walk produces the lines.
    raw = lines(4) + sync(320, 5) + lines(4, first_counter=6) + acqend(10)
    _, table, truncated = table_of(raw)
    assert not truncated
    assert table["counters"]["Lin"].tolist() == [0, 1, 2, 3, 0, 1, 2, 3]


@pytest.mark.parametrize("interrupted", [False, True])
def test_a_change_of_shape_is_refused(interrupted):
    # Both paths refuse it: reaching the walk (a sync block breaks the run) does not make
    # mixed shapes readable. One measurement, one shape, or an error.
    raw = lines(3)
    if interrupted:
        raw += sync(256, 4)
    raw += line(6, 1, counter=5)[0] + acqend(6)

    with pytest.raises(tw.UnsupportedLayoutError, match="one shape per measurement"):
        table_of(raw)


def test_unaligned_data_start_is_refused():
    mm = np.frombuffer(lines(2) + acqend(3), dtype=np.uint8)
    with pytest.raises(tw.UnsupportedLayoutError, match="unaligned"):
        tw.build_table(mm, 4, mm.size, tw.TwixVersion.VD)


def test_truncation_is_reported():
    # No ACQEND at all: the complete lines are returned and truncation is reported.
    _, table, truncated = table_of(lines(3))
    assert truncated
    assert len(table) == 3

    # A header claiming more channels/samples than the buffer holds: no complete line.
    _, table, truncated = table_of(line(4, 1)[0][:192])
    assert truncated
    assert len(table) == 0

    # Not even one header.
    _, table, truncated = table_of(bytes(64))
    assert truncated
    assert len(table) == 0


def test_read_headers_recovers_the_full_scan_headers():
    mm, table, _ = table_of(lines(5) + acqend(6))
    headers = tw.read_headers(mm, table["offset"], tw.TwixVersion.VD)
    assert headers.dtype == tw.VD_SCAN_HEADER
    assert headers["Counter"]["Lin"].tolist() == list(range(5))
    assert headers["SamplesInScan"].tolist() == [4] * 5
    # Fields the compact table drops are still reachable.
    assert {"IceProgramPara", "SliceData"} <= set(headers.dtype.names)
