"""Generate a structurally-valid synthetic VD/VE .dat file at any size, so read-path
scaling can be benchmarked without needing real (and often proprietary) scan data.

Usage: python make_synthetic_dat.py OUT.dat --size 1GB [--ncol 512] [--ncha 32]
"""

from __future__ import annotations

import argparse
import re
import struct
import sys

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "src"))
from turbotwix import _format  # noqa: E402

_SIZE_RE = re.compile(r"^(\d+(?:\.\d+)?)\s*([KMGT]?B)?$", re.IGNORECASE)
_UNITS = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3, "TB": 1024**4}

# Minimal protocol buffers so reference readers (pymapvbvd/twixtools), which expect
# these standard buffer names to exist, don't choke on the synthetic file.
_PROTOCOL_BUFFERS = {
    "Config": "",
    "Dicom": '<ParamString."SoftwareVersions"> { "syngo_synthetic" }\n',
    "Meas": "### ASCCONV BEGIN ###\n### ASCCONV END ###\n",
    "MeasYaps": "### ASCCONV BEGIN ###\nalTR[0]\t = \t10000\n### ASCCONV END ###\n",
    "Phoenix": "### ASCCONV BEGIN ###\n### ASCCONV END ###\n",
    "Spice": "",
}


def _pack_protocol_header() -> bytes:
    out = b""
    for name, content in _PROTOCOL_BUFFERS.items():
        payload = content.encode("latin1")
        out += name.encode("latin1") + b"\x00" + struct.pack("<I", len(payload)) + payload
    return out


def parse_size(text: str) -> int:
    match = _SIZE_RE.match(text.strip())
    if not match:
        raise ValueError(f"unrecognized size: {text!r}")
    value, unit = match.groups()
    return int(float(value) * _UNITS[(unit or "B").upper()])


def make_synthetic_dat(
    path: str, total_bytes: int, ncol: int = 512, ncha: int = 32, n_lin: int = 128
) -> int:
    raid = np.zeros(1, dtype=_format.MULTI_RAID_FILE_HEADER)[0]
    raid["hdr"]["count"] = 1
    scan_offset = _format.MULTI_RAID_FILE_HEADER.itemsize

    protocol_bytes = _pack_protocol_header()
    init = np.zeros(1, dtype=_format.SINGLE_MEAS_INIT)[0]
    init["hdr_len"] = _format.SINGLE_MEAS_INIT.itemsize + len(protocol_bytes)
    init["unknown"] = len(_PROTOCOL_BUFFERS)
    data_start = scan_offset + int(init["hdr_len"])

    line_len = 192 + ncha * (32 + 8 * ncol)
    acqend_len = 192
    remaining = max(total_bytes - data_start - acqend_len, line_len)
    n_lines = max(1, remaining // line_len)

    raid["entry"][0]["off"] = scan_offset
    raid["entry"][0]["len"] = int(init["hdr_len"]) + n_lines * line_len + acqend_len

    chan_block = np.zeros(1, dtype=_format.VD_CHANNEL_HEADER)[0].tobytes() + bytes(8 * ncol)
    line_body = chan_block * ncha

    with open(path, "wb") as f:
        f.write(raid.tobytes())
        f.write(init.tobytes())
        f.write(protocol_bytes)

        header = np.zeros(1, dtype=_format.VD_SCAN_HEADER)[0]
        header["SamplesInScan"] = ncol
        header["UsedChannels"] = ncha
        # Reference readers (pymapvbvd/twixtools) also use the raw DMA-length field as
        # a first-pass sanity/continuation check even though the authoritative length
        # for regular lines is recomputed from NCol/NCha; keep it consistent.
        header["FlagsAndDMALength"] = line_len
        for i in range(n_lines):
            # Cycle Lin within [0, n_lin) and increment Rep every n_lin lines, so
            # (Lin, Rep) stays unique per line like a real multi-repetition
            # acquisition -- a single Lin range repeating with no other varying
            # counter would make every line a duplicate-average target, which is
            # not representative of typical scans.
            header["Counter"]["Lin"] = i % n_lin
            header["Counter"]["Rep"] = i // n_lin
            header["ScanCounter"] = i + 1
            f.write(header.tobytes())
            f.write(line_body)

        acqend = np.zeros(1, dtype=_format.VD_SCAN_HEADER)[0]
        acqend["EvalInfoMask"] = int(_format.Flag.ACQEND)
        acqend["FlagsAndDMALength"] = acqend_len
        f.write(acqend.tobytes())

    return n_lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", help="output .dat path")
    parser.add_argument(
        "--size", default="1GB", help="approximate total file size, e.g. 100MB, 1GB, 10GB"
    )
    parser.add_argument("--ncol", type=int, default=512)
    parser.add_argument("--ncha", type=int, default=32)
    parser.add_argument("--nlin", type=int, default=128)
    args = parser.parse_args()

    total_bytes = parse_size(args.size)
    n_lines = make_synthetic_dat(args.out, total_bytes, args.ncol, args.ncha, args.nlin)
    print(f"wrote {args.out}: {n_lines} lines, ncol={args.ncol}, ncha={args.ncha}")


if __name__ == "__main__":
    main()
