"""Validate generated split artifacts and write a local inventory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from schemaguard.splits.contracts import SplitGenerationConfig  # noqa: E402
from schemaguard.splits.inventory import (  # noqa: E402
    compare_snapshots,
    protected_paths,
    snapshot_paths,
    write_inventory,
)
from schemaguard.splits.validation import validate_all, validate_split  # noqa: E402
from schemaguard.utils.io import atomic_write_json  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/runtime/split_generation.yaml"
    )
    parser.add_argument("--dataset-id", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument(
        "--inventory-output",
        type=Path,
        default=ROOT / "artifacts/validation/split_generation_inventory.json",
    )
    args = parser.parse_args()
    del args.offline  # Validation only reads local artifacts and never opens a network client.
    try:
        with args.config.open("r", encoding="utf-8") as handle:
            config = SplitGenerationConfig.model_validate(yaml.safe_load(handle))
        if args.all or (args.dataset_id is None and args.seed is None):
            results = validate_all(ROOT, config)
        elif args.dataset_id is not None and args.seed is not None:
            results = [validate_split(ROOT, config, args.dataset_id, args.seed)]
        else:
            raise ValueError("--dataset-id and --seed must be provided together")
        inventory = write_inventory(ROOT, config, args.inventory_output)
        before_path = ROOT / "artifacts" / "validation" / "split_generation_hashes_before.json"
        after_path = ROOT / "artifacts" / "validation" / "split_generation_hashes_after.json"
        comparison_path = ROOT / "artifacts" / "validation" / (
            "split_generation_hash_comparison.json"
        )
        if before_path.is_file():
            before = json.loads(before_path.read_text(encoding="utf-8"))
            after = snapshot_paths(ROOT, protected_paths(ROOT, config))
            atomic_write_json(after_path, after)
            atomic_write_json(comparison_path, compare_snapshots(before, after))
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "validated": len(results),
                    "inventory": inventory.model_dump(mode="json"),
                },
                sort_keys=True,
            )
        )
        return 0
    except Exception as exc:
        print(
            json.dumps({"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}, sort_keys=True)
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
