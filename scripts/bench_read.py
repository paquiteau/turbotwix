"""Wall-clock + peak-memory comparison of turbotwix vs pymapvbvd vs twixtools reading
the full `image` array of a .dat file.

Each library is run in its own fresh subprocess so memory is measured in isolation, then
the parent process collects and prints a comparison table.

The headline number is peak *anonymous* RSS (`RssAnon` in /proc/self/status, sampled
from a background thread). Plain `ru_maxrss` is not comparable across these libraries:
turbotwix mmaps the file, so its page-cache pages are charged to RSS, whereas the
buffered `read()`s of the other two land in page cache without ever showing up in RSS.
Anonymous RSS counts only memory the process actually has to own. Peak file-backed RSS
is reported alongside as `mmap`, and raw `ru_maxrss` as `maxrss`.

Usage:
    python bench_read.py FILE.dat [--libs turbotwix pymapvbvd twixtools]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

_READ_CODE = {
    # turbotwix returns lines, the references return a dense counter-indexed array, so
    # the comparable operation is read + fold onto the grid.
    "turbotwix": (
        "import logging\n"
        "logging.getLogger('turbotwix').setLevel(logging.DEBUG)\n"
        "import turbotwix as tw\n"
        "m = tw.open_twix(path)[-1]\n"
        "m.to_dense(m.lines.image, ('Lin', 'Rep'))\n"
    ),
    "pymapvbvd": (
        "import mapvbvd\n"
        "t = mapvbvd.mapVBVD(path, quiet=True)\n"
        "t = t[-1] if isinstance(t, list) else t\n"
        "t.image[tuple([slice(None)] * 16)]\n"
    ),
    "twixtools": (
        "import twixtools\n"
        "from twixtools.map_twix import map_twix\n"
        "scan = twixtools.read_twix(\n"
        "    path, include_scans=[-1], parse_geometry=False, verbose=False\n"
        ")[0]\n"
        "map_twix(scan)['image'][:]\n"
    ),
}

_SINGLE_SCRIPT = """
import json, resource, threading, time

_FIELDS = ("RssAnon:", "RssFile:", "RssShmem:")
peak = dict.fromkeys(_FIELDS, 0)
_stop = threading.Event()


def _sample():
    with open("/proc/self/status") as fh:
        while True:
            fh.seek(0)
            for line in fh:
                key = line.split(maxsplit=1)[0]
                if key in peak:
                    kb = int(line.split()[1])
                    if kb > peak[key]:
                        peak[key] = kb
            if _stop.wait(0.001):
                return


sampler = threading.Thread(target=_sample, daemon=True)
sampler.start()

path = {path!r}
t0 = time.perf_counter()
{code}
elapsed = time.perf_counter() - t0

_stop.set()
sampler.join()
print(json.dumps({{
    "elapsed_s": elapsed,
    "peak_anon_kb": peak["RssAnon:"],
    "peak_file_kb": peak["RssFile:"] + peak["RssShmem:"],
    "maxrss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
}}))
"""


def run_single(lib: str, path: str) -> dict:
    script = _SINGLE_SCRIPT.format(path=path, code=_READ_CODE[lib])
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        return {
            "error": (
                result.stderr.strip().splitlines()[-1] if result.stderr else "failed"
            )
        }
    line = result.stdout.strip().splitlines()[-1]
    return json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help=".dat file to read")
    parser.add_argument(
        "--libs", nargs="+", default=["turbotwix", "pymapvbvd", "twixtools"]
    )
    args = parser.parse_args()

    rows = []
    for lib in args.libs:
        print(f"running {lib}...", file=sys.stderr)
        rows.append((lib, run_single(lib, args.path)))

    print("\n| library | time (s) | peak anon (MB) | mmap (MB) | maxrss (MB) |")
    print("|---|---|---|---|---|")
    for lib, r in rows:
        if "error" in r:
            print(f"| {lib} | ERROR | {r['error']} | | |")
        else:
            print(
                f"| {lib} | {r['elapsed_s']:.3f} | {r['peak_anon_kb'] / 1024:.1f} "
                f"| {r['peak_file_kb'] / 1024:.1f} | {r['maxrss_kb'] / 1024:.1f} |"
            )


if __name__ == "__main__":
    main()
