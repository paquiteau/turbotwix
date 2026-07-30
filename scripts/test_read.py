#!/usr/bin/env python3

import os
import sys
import logging
import time

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s")

from turbotwix import open_twix

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <path_to_twix_file>")
        sys.exit(1)

    tic = time.perf_counter()
    path = sys.argv[1]
    f = open_twix(path)
    lines = f.lines  # one strided read of the line headers
    lines.image  # LineTable: imaging lines only
    lines.noise  # noise-calibration lines, for pre-whitening
    len(lines.image), lines.shape  # 4800, (44, 15000)

    samples = f.read(lines.image)  # (4800, 44, 15000) complex64
    toc = time.perf_counter()

    print(f"Read {samples.shape} samples from {path} in {toc-tic:.2f} seconds")
