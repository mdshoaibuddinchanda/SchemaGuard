"""Freeze the metadata-only SchemaGuard representation-sensitivity pilot plan."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from schemaguard.experiments.pilot_planning import (  # noqa: E402
    PlanningError,
    build_pilot_plan,
    build_staged_schedule,
    load_pilot_config,
)
from schemaguard.utils.io import atomic_write_json  # noqa: E402

EXPECTED_BASE = "9a59e5a5dda320ec00f1915c80fb82d2531cb442"


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments], cwd=root, capture_output=True, text=True, check=False
    )
    if result.returncode:
        raise PlanningError(
            f"required git identity command failed ({' '.join(arguments)}): {result.stderr.strip()}"
        )
    return result.stdout.strip()


def freeze(config_path: Path, output_directory: Path) -> dict[str, Any]:
    """Verify the authorized source identity and atomically write the frozen plan."""
    config = load_pilot_config(config_path)
    head = _git(ROOT, "rev-parse", "HEAD")
    branch = _git(ROOT, "branch", "--show-current")
    remote_head = _git(ROOT, "rev-parse", "origin/main")
    if head != EXPECTED_BASE or branch != "main" or remote_head != EXPECTED_BASE:
        raise PlanningError(
            "BLOCKED_UNEXPECTED_START_STATE: expected main and origin/main at "
            f"{EXPECTED_BASE}; found HEAD={head}, branch={branch}, origin/main={remote_head}"
        )
    if config.protocol.source_commit != EXPECTED_BASE:
        raise PlanningError("protocol source commit does not match the authorized base")
    if sys.version_info[:2] != (3, 12):
        raise PlanningError(f"P12 Python 3.12 is required; observed {sys.version.split()[0]}")
    if Path(sys.executable).resolve().as_posix().lower() != "d:/conda/p12/python.exe":
        raise PlanningError("the only permitted interpreter is D:\\Conda\\P12\\python.exe")

    protocol, inventory = build_pilot_plan(ROOT, config)
    destination = output_directory.resolve()
    if ROOT.resolve() not in destination.parents:
        raise PlanningError("protocol output directory must remain inside the repository")
    protocol_path = destination / "pilot_protocol.json"
    inventory_path = destination / "pilot_condition_inventory.json"
    schedule_path = destination / "pilot_staged_schedule.json"
    schedule = build_staged_schedule(protocol, inventory)
    atomic_write_json(protocol_path, protocol.model_dump(mode="json"))
    atomic_write_json(inventory_path, inventory.model_dump(mode="json"))
    atomic_write_json(schedule_path, schedule.model_dump(mode="json"))
    return {
        "status": "FROZEN_PLAN_ONLY",
        "protocol_sha256": protocol.protocol_sha256,
        "condition_inventory_sha256": inventory.inventory_sha256,
        "staged_schedule_sha256": schedule.schedule_sha256,
        "selected_dataset_ids": [item.dataset_id for item in protocol.selected_datasets],
        "selected_seeds": list(protocol.seed_selection.seeds),
        "applicable_view_tuples": inventory.applicable_view_tuple_count,
        "not_applicable_view_tuples": inventory.not_applicable_view_count,
        "condition_count": inventory.actual_condition_count,
        "pilot_execution_performed": False,
        "test_labels_accessed": False,
        "stages": [
            {
                "stage_id": stage.stage_id,
                "dataset_ids": list(stage.dataset_ids),
                "condition_count": len(stage.condition_ids),
                "conservative_wall_seconds": stage.conservative_wall_seconds,
            }
            for stage in schedule.stages
        ],
        "outputs": [
            str(protocol_path.relative_to(ROOT)),
            str(inventory_path.relative_to(ROOT)),
            str(schedule_path.relative_to(ROOT)),
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/runtime/pilot_protocol.yaml"))
    parser.add_argument("--output-directory", type=Path, default=Path("artifacts/handoff"))
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    output_directory = (
        args.output_directory
        if args.output_directory.is_absolute()
        else ROOT / args.output_directory
    )
    try:
        result = freeze(config_path, output_directory)
    except (PlanningError, OSError, ValueError) as exc:
        print(json.dumps({"status": "BLOCKED", "detail": str(exc)}, indent=2))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
