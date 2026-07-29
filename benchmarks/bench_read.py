"""Wall-clock + peak-RSS comparison of turbotwix vs pymapvbvd vs twixtools reading the
full `image` array of a .dat file.

Each library is run in its own fresh subprocess (`--single`) so peak RSS
(`resource.getrusage(RUSAGE_SELF).ru_maxrss`) is measured in isolation, then the parent
process collects and prints a comparison table.

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
        "import turbotwix as tw\n"
        "m = tw.open_twix(path)[-1]\n"
        "lines = m.lines.image\n"
        "tw.to_dense(m.read(lines), lines, ('Lin', 'Rep'))\n"
    ),
    "turbotwix-lines": (
        "import turbotwix as tw\nm = tw.open_twix(path)[-1]\nm.read(m.lines.image)\n"
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
import json, resource, sys, time
path = {path!r}
t0 = time.perf_counter()
{code}
elapsed = time.perf_counter() - t0
peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
print(json.dumps({{"elapsed_s": elapsed, "peak_rss_kb": peak_kb}}))
"""


def run_single(lib: str, path: str) -> dict:
    script = _SINGLE_SCRIPT.format(path=path, code=_READ_CODE[lib])
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        return {"error": result.stderr.strip().splitlines()[-1] if result.stderr else "failed"}
    line = result.stdout.strip().splitlines()[-1]
    return json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help=".dat file to read")
    parser.add_argument("--libs", nargs="+", default=["turbotwix", "pymapvbvd", "twixtools"])
    args = parser.parse_args()

    rows = []
    for lib in args.libs:
        print(f"running {lib}...", file=sys.stderr)
        rows.append((lib, run_single(lib, args.path)))

    print("\n| library | time (s) | peak RSS (MB) |")
    print("|---|---|---|")
    for lib, r in rows:
        if "error" in r:
            print(f"| {lib} | ERROR | {r['error']} |")
        else:
            print(f"| {lib} | {r['elapsed_s']:.3f} | {r['peak_rss_kb'] / 1024:.1f} |")


if __name__ == "__main__":
    main()
