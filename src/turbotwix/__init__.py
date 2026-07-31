"""turbotwix: a reader for Siemens MRI raw data (`.dat` / TWIX) files.

A file is read in three layers, one module each:

* `turbotwix.dtypes` — the binary layout: structured dtypes for the VB/VD headers and
  the raid directory, version sniffing, and `Flag`, the 64 `EvalInfoMask` bits.
* `turbotwix.header` — the text protocol: ascconv / XProtocol parsing, lazy per buffer.
* `turbotwix.data` — the line table, sample extraction, and the object model built on
  top of them: `open_twix` -> `TwixFile` -> `Measurement` -> `LineTable`.
* `turbotwix.pmu` — the SYNCDATA sideband payload: physiological (PMU) channels,
  exposed as `Measurement.pmu`.

The data model is the file's own: a measurement is a *list of acquisition lines*, each
with its metadata and its `(ncha, ncol)` block of samples. Selecting lines is a boolean
query over that table; reading returns `(ncha, n_lines, ncol)` — channel first, matching
its on-disk order, so it is contiguous with no extra transpose. Folding onto a Cartesian
grid is available (`read(dims=...)`) but never implicit — for a spiral or radial
acquisition the loop counters index shots, interleaves or spokes, and there is no grid
to fold onto.

The byte layouts were re-derived from the reference readers (pymapvbvd, twixtools — see
NOTICE) and validated against real VD/VE files; field names follow the Siemens ICE
`sMDH` / `sScanHeader` naming. `docs/twix-format.md` describes the format itself.
"""

from __future__ import annotations

from .data import (
    LineTable,
    Measurement,
    TwixFile,
    open_twix,
)
from .dtypes import (
    COUNTERS,
    Flag,
    TwixParseError,
    TwixVersion,
    UnsupportedLayoutError,
    UnsupportedVersionError,
)
from .header import (
    AttrDict,
    Protocol,
)
from .pmu import Pmu

__all__ = [
    "COUNTERS",
    "Flag",
    "LineTable",
    "Measurement",
    "Pmu",
    "Protocol",
    "TwixFile",
    "TwixParseError",
    "TwixVersion",
    "UnsupportedLayoutError",
    "UnsupportedVersionError",
    "open_twix",
]
