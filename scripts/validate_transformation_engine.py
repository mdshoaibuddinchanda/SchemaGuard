"""Validate the certified transformation engine and its complete inventory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from schemaguard.splits.contracts import SplitGenerationInventoryContract  # noqa: E402
from schemaguard.transformations.inventory import (  # noqa: E402
    DATASET_IDS,
    SEEDS,
    VIEW_IDS,
    compare_protected_snapshot,
    validate_dataset_seed,
    write_inventory,
)
from schemaguard.transformations.registry import (  # noqa: E402
    get_view,
    load_transformation_config,
    registry_hash,
)
from schemaguard.utils.hashing import sha256_file  # noqa: E402
from schemaguard.utils.io import atomic_write_json  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/runtime/transformation_engine.yaml"
    )
    parser.add_argument(
        "--all", action="store_true", help="validate all datasets, seeds, and views"
    )
    parser.add_argument("--dataset-id", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--view")
    parser.add_argument("--offline", action="store_true", help="prohibit network access")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--inventory-output", type=Path)
    return parser


def _validate_split_inventory() -> str:
    path = ROOT / "artifacts/handoff/split_generation_inventory.json"
    inventory = SplitGenerationInventoryContract.model_validate(
        json.loads(path.read_text(encoding="utf-8"))
    )
    expected = {(dataset_id, seed) for dataset_id in DATASET_IDS for seed in SEEDS}
    observed = {(record.dataset_id, record.seed) for record in inventory.records}
    if observed != expected or any(record.status != "PASS" for record in inventory.records):
        raise ValueError("split inventory does not cover all 70 passing dataset-seed records")
    if any(record.cross_partition_group_count != 0 for record in inventory.records):
        raise ValueError("a predictor group crosses split partitions")
    if inventory.expected_dataset_ids != DATASET_IDS or inventory.expected_seeds != SEEDS:
        raise ValueError("split inventory dataset or seed registry changed")
    return sha256_file(path)


def main() -> int:
    args = _parser().parse_args()
    if not args.all and args.dataset_id is None:
        raise SystemExit("one of --all or --dataset-id is required")
    if args.offline:
        # The implementation has no network-capable code path.  Keep this
        # explicit so an offline invocation is auditable in command logs.
        print("network_access=disabled")
    config = load_transformation_config(args.config)
    split_hash = _validate_split_inventory()
    before = ROOT / "artifacts/transformation_engine/review/protected_hashes_before.json"
    if not before.is_file():
        raise SystemExit("BLOCKED: protected pre-implementation snapshot is missing")
    selected_datasets = DATASET_IDS if args.all else [args.dataset_id]
    selected_seeds = SEEDS if args.seed is None else [args.seed]
    selected_views = VIEW_IDS if args.view is None else [args.view]
    for view_id in selected_views:
        get_view(view_id)
    if any(dataset_id not in DATASET_IDS for dataset_id in selected_datasets):
        raise SystemExit("unknown frozen dataset id")
    if any(seed not in SEEDS for seed in selected_seeds):
        raise SystemExit("unknown frozen seed")
    if args.all and args.view is None and args.seed is None:
        inventory = write_inventory(
            ROOT,
            config,
            before,
            args.inventory_output or ROOT / "artifacts/handoff/transformation_inventory.json",
        )
        print(json.dumps(inventory.canonical_dict(), indent=2, sort_keys=True))
        return 0 if inventory.status == "PASS_PENDING_REVIEW" else 1
    records = []
    for dataset_id in selected_datasets:
        for seed in selected_seeds:
            records.extend(validate_dataset_seed(ROOT, dataset_id, seed, config))
    if args.view is not None:
        records = [record for record in records if record.view_id == args.view]
    result = {
        "schema_version": 1,
        "status": "PASS" if all(record.status in {"PASS", "N/A"} for record in records) else "FAIL",
        "record_count": len(records),
        "registry_hash": registry_hash(),
        "split_inventory_hash": split_hash,
        "records": [record.canonical_dict() for record in records],
        "protected_hash_comparison": compare_protected_snapshot(ROOT, before),
    }
    if args.inventory_output and not args.validate_only:
        atomic_write_json(args.inventory_output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return (
        0
        if result["status"] == "PASS" and result["protected_hash_comparison"]["status"] == "PASS"
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
