#!/usr/bin/env python3

import logging
import sys
import time

from turbotwix import open_twix

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <path_to_twix_file>")
        sys.exit(1)
    path = sys.argv[1]

    f = open_twix(path)
    img = f.scan.lines.image
    print(img[img.counter("Rep") == 0])  # one volume's shots

    tic = time.perf_counter()
    f = open_twix(path)
    samples = f.read(dims="minimal")
    toc = time.perf_counter()

    print(f"Read {samples.shape} samples from {path} in {toc - tic:.2f} seconds")
