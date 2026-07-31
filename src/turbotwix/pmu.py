#!/usr/bin/env python3
"""Siemens PMU (physiological monitor) decoding: the SYNCDATA sideband payload.

`docs/twix-format.md` describes the SYNCDATA block itself; `turbotwix.data`'s line-table
walk records only each block's `(offset, length)` (`SYNC_DTYPE`), since nothing there can
decode it. This module does: it turns those blocks into per-channel ECG/pulse/respiration
signal, trigger and timestamp arrays (`Measurement.pmu`).

The byte layout is undocumented by Siemens; re-derived from twixtools' `pmu.py` /
`seqdata.py` (see NOTICE), which is itself the product of independent reverse-engineering.
There are three on-disk variants, selected by the scanner's software version
(`hdr['Dicom']['SoftwareVersions']`): classic (VB/VD/VE, e.g. "syngo MR E11"), and two XA
variants split at syngo version 61.
"""

from __future__ import annotations

import re
from typing import NamedTuple

import numpy as np

from .dtypes import SEQDATA_HEADER

__all__ = ["Pmu", "PmuBlock", "select_decoder"]


class _SeqData(NamedTuple):
    """One SYNCDATA block's body, split into its id and payload."""

    id: str
    payload: bytes


def _decode_seqdata(raw: bytes) -> _SeqData | None:
    """The `(id, payload)` of one SYNCDATA block's body, past its scan header.

    Parameters
    ----------
    raw : bytes
        The block's body, i.e. the DMA-length block with its scan header already
        stripped off.

    Returns
    -------
    _SeqData or None
        None if `raw` is too short to hold a `SEQDATA_HEADER`.
    """
    if len(raw) < SEQDATA_HEADER.itemsize:
        return None
    hdr = np.frombuffer(raw, dtype=SEQDATA_HEADER, count=1)[0]
    packet_size = int(hdr["packet_size"])
    ident = bytes(hdr["id"]).split(b"\x00", 1)[0].decode("latin1")
    start = SEQDATA_HEADER.itemsize
    return _SeqData(ident, raw[start : start + packet_size])


# ---------------------------------------------------------------------------
# Magic tables: channel id -> name, per variant.
# ---------------------------------------------------------------------------

_MAGIC_CLASSIC = {
    0x01FF0000: "END",
    0x01010000: "ECG1",
    0x01020000: "ECG2",
    0x01030000: "ECG3",
    0x01040000: "ECG4",
    0x01050000: "PULS",
    0x01060000: "RESP",
    0x01070000: "EXT1",
    0x01080000: "EXT2",
}

_MAGIC_XA_PRE61 = {
    0xFFFFFFFF: "END",
    0: "ECG1",
    1: "ECG2",
    2: "ECG3",
    3: "ECG4",
    4: "PULS",
    5: "RESP",
    6: "EXT1",
    7: "EXT2",
    8: "EVENTS",
    9: "CardPT",
    10: "RespPT",
}
#: `EVENTS` channel values, for the pre-61 XA variant: event code -> trigger channel.
_EVENTS_XA_PRE61 = {
    "Patient Table": 0x1,
    "ECG1": 0x2,
    "ECG2": 0x2,
    "ECG3": 0x2,
    "ECG4": 0x2,
    "PULS": 0x4,
    "RESP": 0x8,
    "EXT1": 0x10,
    "EXT2": 0x20,
    "CardPT": 0x40,
    "RespPT": 0x80,
}

_MAGIC_XA61 = {
    1: "ECG1",
    2: "ECG2",
    3: "ECG3",
    4: "ECG4",
    5: "PULS",
    6: "RSR1",
    7: "RSR2",
    8: "RESP CUSH",
    9: "EXT1",
    10: "EXT2",
    35: "CardPT",
    36: "RespPT",
    49: "EVENTS",
}
#: `EVENTS` channel values, for the 61+ XA variant.
_EVENTS_XA61 = {
    "ECG1": 0x2,
    "ECG2": 0x2,
    "ECG3": 0x2,
    "ECG4": 0x2,
    "PULS": 0x4,
    "RSR1": 0x4,
    "RSR2": 0x4,
    "RESP CUSH": 0x4,
    "EXT1": 0x10,
    "EXT2": 0x20,
    "CardPT": 0x40,
    "RespPT": 0x80,
}


class PmuBlock(NamedTuple):
    """One decoded SYNCDATA/PMU packet: a handful of channels over one short interval."""

    timestamp: int  #: 2.5 ms ticks since midnight, same unit as `PMUTimeStamp`.
    duration: int
    signal: dict[str, np.ndarray]
    trigger: dict[str, np.ndarray]
    variant: str  #: "classic", "xa_pre61" or "xa61" -- which timestamp formula applies


def _decode_classic(payload: bytes) -> PmuBlock:
    """The VB/VD/VE (non-XA) packet layout: u16 (value, trigger) pairs per channel."""
    timestamp0, timestamp, packet_no, duration = (
        int(v) for v in np.frombuffer(payload, dtype="<u4", count=4)
    )
    del timestamp0, packet_no  # not needed to place samples
    signal: dict[str, np.ndarray] = {}
    trigger: dict[str, np.ndarray] = {}
    i = 16
    while i < len(payload):
        magic = int(np.frombuffer(payload, dtype="<u4", count=1, offset=i)[0])
        key = _MAGIC_CLASSIC.get(magic)
        if key == "END":
            break
        i += 4
        period = int(np.frombuffer(payload, dtype="<u4", count=1, offset=i)[0])
        i += 4
        n_pts = duration // period
        if key is not None:
            block = np.frombuffer(
                payload, dtype="<u2", count=2 * n_pts, offset=i
            ).reshape(n_pts, 2)
            signal[key] = block[:, 0].astype(np.float64) / 4096.0
            trigger[key] = block[:, 1].astype(bool)
        i += 4 * n_pts
    return PmuBlock(timestamp, duration, signal, trigger, "classic")


def _decode_xa_pre61(payload: bytes) -> PmuBlock:
    """The XA, syngo < 61 packet layout: u32 samples per channel, one EVENTS channel."""
    timestamp0, timestamp, packet_no, duration_version = (
        int(v) for v in np.frombuffer(payload, dtype="<u4", count=4)
    )
    del timestamp0, packet_no
    duration = duration_version & 0xFFFFFF
    signal: dict[str, np.ndarray] = {}
    trigger: dict[str, np.ndarray] = {}
    i = 16
    while i < len(payload):
        magic = int(np.frombuffer(payload, dtype="<u4", count=1, offset=i)[0])
        key = _MAGIC_XA_PRE61.get(magic)
        if key == "END":
            break
        i += 4
        period = int(np.frombuffer(payload, dtype="<u4", count=1, offset=i)[0])
        i += 4
        n_pts = duration // period
        if key == "EVENTS":
            block = np.frombuffer(payload, dtype="<u4", count=n_pts, offset=i)
            event_type = block & ~np.uint32(0xF00)
            for name, code in _EVENTS_XA_PRE61.items():
                trigger[name] = event_type == code
        elif key is not None:
            block = np.frombuffer(payload, dtype="<u4", count=n_pts, offset=i)
            signal[key] = block.astype(np.float64) / 4095.0
        i += 4 * n_pts
    return PmuBlock(timestamp, duration, signal, trigger, "xa_pre61")


def _decode_xa61(payload: bytes) -> PmuBlock:
    """The 61+ XA packet layout: f32 samples per channel, with an affine scale/offset."""
    # packet header (16 B): version u2, size_header u1, _ u1, size_full_packet u4,
    # timestamp u8.
    timestamp = int(np.frombuffer(payload, dtype="<u8", count=1, offset=8)[0])
    # extended packet header (24 B): packet_id u8, duration u4, _ u4, active_signal u8.
    duration = int(np.frombuffer(payload, dtype="<u4", count=1, offset=24)[0])
    signal: dict[str, np.ndarray] = {}
    trigger: dict[str, np.ndarray] = {}
    i = 40
    while i < len(payload):
        magic = payload[i]
        added_bytes = payload[i + 1]
        nb_samples = int(np.frombuffer(payload, dtype="<u2", count=1, offset=i + 6)[0])
        key = _MAGIC_XA61.get(magic)
        i += 8  # magic, added_bytes, size_one, size_data, nb_samples
        i += 8  # period, delta_timestamp: not needed to place samples
        _ref_value, divisor, offset = (
            float(v) for v in np.frombuffer(payload, dtype="<f8", count=3, offset=i)
        )
        i += 24
        block = np.frombuffer(payload, dtype="<f4", count=nb_samples, offset=i)
        if magic == 49:
            for name, code in _EVENTS_XA61.items():
                trigger[name] = block == code
        elif key is not None:
            signal[key] = block.astype(np.float64) * divisor + offset
        i += 4 * nb_samples + added_bytes
    return PmuBlock(timestamp, duration, signal, trigger, "xa61")


def select_decoder(syngo_version: str):
    """The payload decoder for `syngo_version` (`hdr['Dicom']['SoftwareVersions']`).

    Parameters
    ----------
    syngo_version : str
        The scanner software version string, e.g. ``"syngo MR E11"`` or
        ``"syngo MR XA30"``.

    Returns
    -------
    callable
        `_decode_classic`, `_decode_xa_pre61` or `_decode_xa61`.
    """
    tag = syngo_version.split(" ")[-1]
    if tag.startswith("XA"):
        digits = re.match(r"\d+", tag[2:])
        number = int(digits.group()) if digits else 0
        return _decode_xa61 if number >= 61 else _decode_xa_pre61
    return _decode_classic


def _block_timestamps(block: PmuBlock, n_pts: int) -> np.ndarray:
    """Per-sample timestamps for `n_pts` samples of `block`, in `PMUTimeStamp` ticks."""
    if block.variant == "xa61":
        return (
            block.timestamp + np.linspace(0, block.duration, n_pts, endpoint=False)
        ) / 2.5e3
    return block.timestamp + np.linspace(
        0, block.duration / 10.0, n_pts, endpoint=False
    ) / 2.5


class Pmu:
    """Physiological (PMU) channels decoded from a measurement's SYNCDATA blocks.

    Built once per measurement (`Measurement.pmu`); empty if it carries none. Each
    channel's `signal`/`trigger` is the concatenation, in block order, of every
    SYNCDATA/PMU packet that carries it. Learning-phase blocks are kept under a
    ``LEARN_`` prefixed channel name rather than merged with the real acquisition.
    """

    def __init__(self) -> None:
        self.signal: dict[str, np.ndarray] = {}
        self.trigger: dict[str, np.ndarray] = {}
        self.timestamp: dict[str, np.ndarray] = {}
        self.timestamp_trigger: dict[str, np.ndarray] = {}

    def __repr__(self) -> str:
        return f"Pmu(channels={sorted(self.signal)})"

    @classmethod
    def decode(
        cls,
        mm: np.ndarray,
        sync: np.ndarray,
        header_size: int,
        syngo_version: str,
    ) -> Pmu:
        """Decode every SYNCDATA/PMU block recorded for one measurement.

        Parameters
        ----------
        mm : numpy.ndarray
            The whole file as a ``uint8`` array.
        sync : numpy.ndarray
            `SYNC_DTYPE` rows -- `(offset, length)` per SYNCDATA block, as recorded by
            `build_table`.
        header_size : int
            Bytes to skip past each block's scan header before its SYNCDATA payload
            starts (`HEADER_SIZES[version][0]`: 192 for VD/VE, 0 for VB).
        syngo_version : str
            `hdr['Dicom']['SoftwareVersions']`, used to pick the payload variant.

        Returns
        -------
        Pmu
            Empty (no channels) if `sync` is empty, or none of its blocks are PMU data.
        """
        decode_payload = select_decoder(syngo_version)
        signal: dict[str, list[np.ndarray]] = {}
        trigger: dict[str, list[np.ndarray]] = {}
        timestamp: dict[str, list[np.ndarray]] = {}
        timestamp_trigger: dict[str, list[np.ndarray]] = {}

        for offset, length in sync:
            body = bytes(mm[int(offset) + header_size : int(offset) + int(length)])
            seqdata = _decode_seqdata(body)
            if seqdata is None or not seqdata.id.startswith("PMU"):
                continue
            prefix = "LEARN_" if seqdata.id.startswith("PMULearnPhase") else ""
            block = decode_payload(seqdata.payload)
            for key, values in block.signal.items():
                name = prefix + key
                signal.setdefault(name, []).append(values)
                timestamp.setdefault(name, []).append(
                    _block_timestamps(block, len(values))
                )
            for key, values in block.trigger.items():
                name = prefix + key
                trigger.setdefault(name, []).append(values)
                timestamp_trigger.setdefault(name, []).append(
                    _block_timestamps(block, len(values))
                )

        pmu = cls()
        for name, chunks in signal.items():
            pmu.signal[name] = np.concatenate(chunks)
            pmu.timestamp[name] = np.concatenate(timestamp[name])
        for name, chunks in trigger.items():
            pmu.trigger[name] = np.concatenate(chunks)
            pmu.timestamp_trigger[name] = np.concatenate(timestamp_trigger[name])
        return pmu

    def get_time(self, key: str) -> np.ndarray:
        """`timestamp[key]`, converted from 2.5 ms ticks to seconds."""
        return self.timestamp[key] * 2.5e-3
