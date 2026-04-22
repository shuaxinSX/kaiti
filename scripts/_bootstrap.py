"""Local path bootstrap for standalone script entrypoints."""

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


REPO_ROOT = Path(__file__).resolve().parents[1]

try:
    _append_if_exists(site.getusersitepackages())
except Exception:
    pass

_append_if_exists(str(REPO_ROOT / ".vendor_sx"))
_append_if_exists(str(REPO_ROOT))
