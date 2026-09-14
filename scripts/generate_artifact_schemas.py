"""Generate the closed-world artifact schemas from Pydantic contracts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from schemaguard.artifact_contracts import schema_documents  # noqa: E402


def main() -> int:
    destination = ROOT / "schemas"
    destination.mkdir(parents=True, exist_ok=True)
    for filename, document in schema_documents().items():
        rendered = json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        (destination / filename).write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
