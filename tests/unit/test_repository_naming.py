from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from validate_repository_naming import audit_repository


def test_tracked_repository_has_no_execution_order_names() -> None:
    assert audit_repository(Path(__file__).resolve().parents[2]) == []
