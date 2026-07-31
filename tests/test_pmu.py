"""PMU decoding: SYNCDATA capture in the walk, and the three payload variants.

Neither bundled `.dat` file (`gre.dat`, `epi.dat`) contains real SYNCDATA/PMU blocks
(checked directly: the literal string "PMU" only appears in protocol text), so these
build synthetic blocks the same way `conftest.py`'s `sync()` helper builds synthetic
ACQEND/SYNCDATA blocks for `test_table.py` -- but with real PMU payload bytes.
"""

import importlib.util
import struct

import numpy as np
import pytest
from conftest import acqend

import turbotwix as tw

HAS_TWIXTOOLS = importlib.util.find_spec("twixtools") is not None


def seqdata_block(ident: bytes, payload: bytes) -> bytes:
    """A `SEQDATA_HEADER` + payload, right after a SYNCDATA block's scan header."""
    header = np.zeros(1, dtype=tw.dtypes.SEQDATA_HEADER)[0]
    header["packet_size"] = len(payload)
    header["id"] = ident
    return header.tobytes() + payload


def sync_block(body: bytes, counter: int = 1) -> bytes:
    """A full SYNCDATA block: VD scan header (flagged SYNCDATA) + `body`."""
    header = np.zeros(1, dtype=tw.dtypes.VD_SCAN_HEADER)[0]
    header["EvalInfoMask"] = int(tw.Flag.SYNCDATA)
    header["ScanCounter"] = counter
    total = tw.dtypes.VD_SCAN_HEADER.itemsize + len(body)
    header["FlagsAndDMALength"] = total
    return header.tobytes() + body


def classic_payload(
    timestamp: int, duration: int, channels: dict[str, tuple[np.ndarray, np.ndarray]]
) -> bytes:
    """The classic (non-XA) PMU payload: u16 (value, trigger) pairs per channel."""
    magic = {
        "ECG1": 0x01010000,
        "PULS": 0x01050000,
        "RESP": 0x01060000,
    }
    out = struct.pack("<IIII", 0, timestamp, 1, duration)
    for name, (values, triggers) in channels.items():
        n_pts = len(values)
        period = duration // n_pts
        out += struct.pack("<II", magic[name], period)
        interleaved = np.empty(2 * n_pts, dtype="<u2")
        interleaved[0::2] = values
        interleaved[1::2] = triggers.astype("<u2")
        out += interleaved.tobytes()
    out += struct.pack("<II", 0x01FF0000, 1)  # END
    return out


def xa_pre61_payload(
    timestamp: int, duration: int, channels: dict[str, np.ndarray]
) -> bytes:
    """The XA, syngo < 61 PMU payload: u32 samples per channel."""
    magic = {"ECG1": 0, "PULS": 4, "RESP": 5}
    out = struct.pack("<IIII", 0, timestamp, 1, duration)
    for name, values in channels.items():
        n_pts = len(values)
        period = duration // n_pts
        out += struct.pack("<II", magic[name], period)
        out += values.astype("<u4").tobytes()
    out += struct.pack("<II", 0xFFFFFFFF, 1)  # END
    return out


def xa61_payload(
    timestamp: int,
    duration: int,
    channels: dict[str, tuple[np.ndarray, float, float]],
) -> bytes:
    """The 61+ XA PMU payload: f32 samples per channel with an affine scale/offset."""
    magic = {"ECG1": 1, "PULS": 5, "RESP CUSH": 8}
    out = struct.pack("<HBBIQ", 0, 16, 0, 0, timestamp)
    out += struct.pack("<QIIQ", 0, duration, 0, 0)
    for name, (values, divisor, offset) in channels.items():
        n_pts = len(values)
        period = duration // n_pts if n_pts else 1
        out += struct.pack("<BBHHH", magic[name], 0, 4, 4 * n_pts, n_pts)
        out += struct.pack("<II", period, 0)
        out += struct.pack("<ddd", 0.0, divisor, offset)
        out += values.astype("<f4").tobytes()
    return out


def test_sync_blocks_are_captured_and_decoded_classic():
    values = np.array([100, 200, 300, 4095], dtype="<u2")
    triggers = np.array([0, 0, 1, 0], dtype="<u2")
    payload = classic_payload(1000, 40, {"ECG1": (values, triggers)})
    body = seqdata_block(b"PMU1", payload)
    raw = sync_block(body) + acqend(2)

    mm = np.frombuffer(raw, dtype=np.uint8)
    table, sync, truncated = tw.data.build_table(mm, 0, mm.size, tw.TwixVersion.VD)
    assert not truncated
    assert len(table) == 0  # no ADC lines, only the sync block and its terminator
    assert sync["offset"].tolist() == [0]
    assert sync["length"].tolist() == [tw.dtypes.VD_SCAN_HEADER.itemsize + len(body)]

    pmu = tw.Pmu.decode(mm, sync, 192, "syngo MR E11")
    assert list(pmu.signal) == ["ECG1"]
    np.testing.assert_allclose(pmu.signal["ECG1"], values / 4096.0)
    assert pmu.trigger["ECG1"].tolist() == [False, False, True, False]
    assert pmu.timestamp["ECG1"].shape == (4,)


def test_learning_phase_blocks_get_a_learn_prefix():
    values = np.array([1, 2], dtype="<u2")
    triggers = np.array([0, 0], dtype="<u2")
    payload = classic_payload(0, 20, {"PULS": (values, triggers)})
    body = seqdata_block(b"PMULearnPhase", payload)
    raw = sync_block(body) + acqend(2)

    mm = np.frombuffer(raw, dtype=np.uint8)
    _, sync, _ = tw.data.build_table(mm, 0, mm.size, tw.TwixVersion.VD)
    pmu = tw.Pmu.decode(mm, sync, 192, "syngo MR E11")
    assert list(pmu.signal) == ["LEARN_PULS"]


def test_non_pmu_syncdata_is_ignored():
    body = seqdata_block(b"TRAJECTORY", b"\x00" * 16)
    raw = sync_block(body) + acqend(2)
    mm = np.frombuffer(raw, dtype=np.uint8)
    _, sync, _ = tw.data.build_table(mm, 0, mm.size, tw.TwixVersion.VD)
    pmu = tw.Pmu.decode(mm, sync, 192, "syngo MR E11")
    assert pmu.signal == {}


def test_no_sync_blocks_gives_an_empty_pmu():
    mm = np.frombuffer(acqend(1), dtype=np.uint8)
    _, sync, _ = tw.data.build_table(mm, 0, mm.size, tw.TwixVersion.VD)
    assert len(sync) == 0
    pmu = tw.Pmu.decode(mm, sync, 192, "syngo MR E11")
    assert pmu.signal == {} and pmu.trigger == {}


def test_variant_selection():
    assert tw.pmu.select_decoder("syngo MR E11") is tw.pmu._decode_classic
    assert tw.pmu.select_decoder("syngo MR XA30") is tw.pmu._decode_xa_pre61
    assert tw.pmu.select_decoder("syngo MR XA61") is tw.pmu._decode_xa61
    assert tw.pmu.select_decoder("syngo MR XA61A") is tw.pmu._decode_xa61


def test_xa_pre61_variant_decodes():
    values = np.array([0, 1000, 4095], dtype="<u4")
    payload = xa_pre61_payload(500, 30, {"ECG1": values})
    block = tw.pmu._decode_xa_pre61(payload)
    np.testing.assert_allclose(block.signal["ECG1"], values / 4095.0)
    assert block.timestamp == 500
    assert block.variant == "xa_pre61"


def test_xa61_variant_decodes():
    values = np.array([1.0, 2.0, 3.0], dtype="<f4")
    payload = xa61_payload(700, 30, {"ECG1": (values, 2.0, 1.0)})
    block = tw.pmu._decode_xa61(payload)
    np.testing.assert_allclose(block.signal["ECG1"], values * 2.0 + 1.0)
    assert block.timestamp == 700
    assert block.variant == "xa61"


@pytest.mark.skipif(not HAS_TWIXTOOLS, reason="twixtools not installed")
def test_parity_twixtools_classic_pmu_block():
    """turbotwix's classic decoder must agree with twixtools' `PMUblock`."""
    from twixtools.pmu import PMUblock

    values = np.array([10, 20, 30, 40, 50], dtype="<u2")
    triggers = np.array([0, 1, 0, 0, 1], dtype="<u2")
    payload = classic_payload(12345, 50, {"ECG1": (values, triggers)})

    ref = PMUblock(payload)
    got = tw.pmu._decode_classic(payload)

    np.testing.assert_allclose(got.signal["ECG1"], ref.signal["ECG1"])
    np.testing.assert_array_equal(got.trigger["ECG1"], ref.trigger["ECG1"])
    assert got.timestamp == ref.timestamp
    assert got.duration == ref.duration
