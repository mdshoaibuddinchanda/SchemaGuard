"""Validate the certified transformation engine and its complete inventory."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path, PurePosixPath, PureWindowsPath

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

ALLOWED_REPAIR_SCHEMA_CHANGES = {
    "artifacts/handoff/transformation_engine_review.md",
    "artifacts/handoff/transformation_inventory.json",
    "schemas/transformation_certificate.schema.json",
    "schemas/transformation_inventory.schema.json",
    "schemas/transformation_manifest.schema.json",
}
CACHE_SCHEDULER_SCHEMA_ADDITIONS = {
    "schemas/scheduler_run_plan.schema.json",
    "schemas/scheduler_run_manifest.schema.json",
    "schemas/scheduler_resource_record.schema.json",
    "schemas/cache_scheduler_protected_hash_record.schema.json",
    "schemas/scheduler_failure_record.schema.json",
    "schemas/cache_scheduler_protected_hash_comparison.schema.json",
    "schemas/scheduler_task_attempt.schema.json",
    "schemas/cache_scheduler_probe_run.schema.json",
    "schemas/cache_scheduler_probe_resource.schema.json",
    "schemas/scheduler_state.schema.json",
    "schemas/cache_scheduler_probe_evidence.schema.json",
    "schemas/scheduler_task_record.schema.json",
    "schemas/cache_scheduler_inventory.schema.json",
    "schemas/scheduler_task_result.schema.json",
    "schemas/cache_scheduler_fault_record.schema.json",
    "schemas/cache_scheduler_fault_evidence.schema.json",
    "schemas/cache_scheduler_config.schema.json",
    "schemas/cache_identity.schema.json",
    "schemas/cache_artifact_manifest.schema.json",
    "schemas/cache_completion_marker.schema.json",
    "schemas/scheduler_task_transition.schema.json",
    "schemas/scheduler_task_spec.schema.json",
}
ALLOWED_REPAIR_SCHEMA_ADDITIONS = {
    "schemas/smoke_condition.schema.json",
    "schemas/smoke_condition_resource.schema.json",
    "schemas/smoke_config.schema.json",
    "schemas/smoke_evidence_inventory.schema.json",
    "schemas/smoke_metric.schema.json",
    "schemas/smoke_paired_metric.schema.json",
    "schemas/smoke_plan.schema.json",
    "schemas/smoke_prediction_file.schema.json",
    "schemas/smoke_protected_foundation_hash_comparison.schema.json",
    "schemas/smoke_protected_split_validation.schema.json",
    "schemas/smoke_resume_verification.schema.json",
    "schemas/smoke_run_report.schema.json",
    "schemas/smoke_runtime_estimate.schema.json",
    "schemas/smoke_validation_report.schema.json",
    "schemas/model_adapter_inventory.schema.json",
    "schemas/model_adapter_leakage_evidence.schema.json",
    "schemas/model_adapter_result.schema.json",
    "schemas/transformation_cache_manifest.schema.json",
    "schemas/transformation_property_evidence.schema.json",
} | CACHE_SCHEDULER_SCHEMA_ADDITIONS
FROZEN_TRANSFORMATION_BASELINE_COMMIT = "5a8d59b53baaee8de26314c3803f8260dab3df16"
FROZEN_TRANSFORMATION_PATHS = (
    "src/schemaguard/transformations",
    "src/schemaguard/utils/hashing.py",
    "scripts/run_transformation_properties.py",
    "schemas/transformation_cache_manifest.schema.json",
    "schemas/transformation_certificate.schema.json",
    "schemas/transformation_inventory.schema.json",
    "schemas/transformation_manifest.schema.json",
    "schemas/transformation_property_evidence.schema.json",
    "schemas/transformation_validation.schema.json",
)


def validate_schema_addition_paths(paths: Iterable[str]) -> frozenset[str]:
    """Accept only exact, portable schema-file paths in an explicit allowlist."""

    values = tuple(paths)
    normalized: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            raise ValueError("schema additions must be repository-relative strings")
        posix = PurePosixPath(value)
        windows = PureWindowsPath(value)
        if (
            posix.is_absolute()
            or windows.is_absolute()
            or "\\" in value
            or ".." in posix.parts
            or any(character in value for character in "*?[]")
            or posix.as_posix() != value
            or len(posix.parts) != 2
            or posix.parts[0] != "schemas"
            or not posix.name.endswith(".schema.json")
        ):
            raise ValueError(f"invalid exact schema addition path: {value!r}")
        normalized.add(value)
    if len(normalized) != len(values):
        raise ValueError("schema addition paths must be unique")
    return frozenset(normalized)


def assert_frozen_transformation_tree(
    root: str | Path = ROOT,
    *,
    baseline_commit: str = FROZEN_TRANSFORMATION_BASELINE_COMMIT,
    paths: Iterable[str] = FROZEN_TRANSFORMATION_PATHS,
) -> None:
    """Reject transformation code or protected schema changes since accepted evidence."""

    project_root = Path(root)
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", baseline_commit, "HEAD"],
        cwd=project_root,
        check=False,
        timeout=30,
    )
    if ancestor.returncode != 0:
        raise ValueError("accepted transformation baseline is missing or not an ancestor")
    differences = subprocess.run(
        ["git", "diff", "--quiet", baseline_commit, "--", *paths],
        cwd=project_root,
        check=False,
        timeout=30,
    )
    if differences.returncode == 1:
        raise ValueError("transformation implementation or protected schema changed")
    if differences.returncode != 0:
        raise ValueError("could not compare the frozen transformation tree")


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
    assert_frozen_transformation_tree()
    allowed_schema_additions = validate_schema_addition_paths(
        ALLOWED_REPAIR_SCHEMA_ADDITIONS
    )
    config = load_transformation_config(args.config)
    split_hash = _validate_split_inventory()
    repair_before = ROOT / "artifacts/transformation_engine/review/repair_hashes_before.json"
    before = (
        repair_before
        if repair_before.is_file()
        else ROOT / "artifacts/transformation_engine/review/protected_hashes_before.json"
    )
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
        inventory_output = args.inventory_output or ROOT / (
            "artifacts/handoff/transformation_inventory.json"
        )
        inventory = write_inventory(
            ROOT,
            config,
            before,
            inventory_output,
        )
        if inventory.protected_hash_comparison["status"] == "FAIL":
            protected = compare_protected_snapshot(
                ROOT,
                before,
                allowed_changed_paths=ALLOWED_REPAIR_SCHEMA_CHANGES,
                allowed_added_paths=set(allowed_schema_additions),
            )
            if protected["status"] == "PASS":
                inventory = inventory.model_copy(
                    update={
                        "status": "PASS_PENDING_REVIEW",
                        "protected_hash_comparison": protected,
                    }
                )
                atomic_write_json(inventory_output, inventory.canonical_dict())
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
        "protected_hash_comparison": compare_protected_snapshot(
            ROOT,
            before,
            allowed_changed_paths=ALLOWED_REPAIR_SCHEMA_CHANGES,
            allowed_added_paths=set(allowed_schema_additions),
        ),
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
