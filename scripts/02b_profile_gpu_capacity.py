"""Run bounded, serialized GPU capacity profiling for TabPFN and TabICL."""

# ruff: noqa: E402, I001

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from schemaguard.compatibility.gpu_profiles import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
