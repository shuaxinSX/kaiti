"""Repo-local Python path bootstrap for script entrypoints.

Append optional user-site and vendored dependency roots after the active
environment's own site-packages, so SX keeps its native core packages while
still seeing local extras such as matplotlib.
"""

from __future__ import annotations

import site
import sys
from pathlib import Path


def _append_if_exists(raw_path: str) -> None:
    if not raw_path:
        return
    path = Path(raw_path)
    if path.exists() and raw_path not in sys.path:
        sys.path.append(raw_path)


try:
    _append_if_exists(site.getusersitepackages())
except Exception:
    pass

_append_if_exists(str(Path(__file__).resolve().parents[1] / ".vendor_sx"))
