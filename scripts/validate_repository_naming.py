"""Audit tracked repository names and references for semantic naming compliance."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

_FORBIDDEN_REFERENCE = re.compile(
    r"(?:scripts/[0-9]{1,2}_[A-Za-z0-9_]+\.py|phase_[0-9]+[A-Za-z]*|phase[0-9]+|"
    r"phase-[0-9]+[A-Za-z-]+|PHASE[0-9]+|FAIL_PHASE[0-9]+|phase" + "_numbering)"
)
_FORBIDDEN_COMPONENT = re.compile(r"^(?:phase[_-]?\d+|step[_-]?\d+|\d{1,2}_.+)$", re.I)


def tracked_paths(root: Path = ROOT) -> list[Path]:
    completed = subprocess.run(
        ["git", "ls-files"], cwd=root, capture_output=True, text=True, check=True
    )
    paths = {root / line for line in completed.stdout.splitlines() if line}
    for directory in ("configs", "schemas", "scripts", "src", "tests", "artifacts/handoff"):
        base = root / directory
        if base.exists():
            paths.update(
                path
                for path in base.rglob("*")
                if path.is_file() and "__pycache__" not in path.parts
            )
    return sorted(path for path in paths if "__pycache__" not in path.parts)


def audit_repository(root: Path = ROOT) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    paths = tracked_paths(root)
    for path in paths:
        relative = path.relative_to(root).as_posix()
        for component in Path(relative).parts:
            if component.lower().startswith("fig") and component[3:].split("_", 1)[0].isdigit():
                continue  # Scientific figure identifiers are not execution stages.
            if _FORBIDDEN_COMPONENT.match(component):
                violations.append(
                    {
                        "file": relative,
                        "identifier": component,
                        "line": 0,
                        "replacement": "semantic responsibility-based name",
                    }
                )
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (UnicodeDecodeError, OSError):
            continue
        in_migration_table = False
        for number, line in enumerate(lines, 1):
            if relative == "artifacts/handoff/repository_repair_review.md":
                if line.startswith("## Naming migration"):
                    in_migration_table = True
                elif in_migration_table and line.startswith("## "):
                    in_migration_table = False
                if in_migration_table and line.startswith("|"):
                    continue  # Historical old->new map, not a live reference.
            match = _FORBIDDEN_REFERENCE.search(line)
            if match:
                identifier = match.group(0)
                violations.append(
                    {
                        "file": relative,
                        "identifier": identifier,
                        "line": number,
                        "replacement": "semantic responsibility-based path or identifier",
                    }
                )
    return violations


def main() -> int:
    violations = audit_repository()
    payload = {
        "schema_version": 1,
        "status": "PASS" if not violations else "FAIL",
        "violations": violations,
    }
    destination = ROOT / "artifacts" / "repository_repair" / "naming_audit.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not violations else 2


if __name__ == "__main__":
    raise SystemExit(main())
