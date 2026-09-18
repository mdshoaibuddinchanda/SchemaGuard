from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from schemaguard.transformations.inventory import _protected_snapshot, compare_protected_snapshot
from scripts import validate_transformation_engine as validator


def _write_baseline(root: Path) -> Path:
    for relative in (
        "configs/baselines/config.json",
        "data/raw/openml/source.bin",
        "data/processed/openml/features.parquet",
        "data/splits/openml/assignments.parquet",
        "schemas/transformation_inventory.schema.json",
        "artifacts/handoff/split_generation_inventory.json",
        "artifacts/handoff/transformation_inventory.json",
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"approved baseline")
    before = root / "protected-before.json"
    before.write_text(json.dumps(_protected_snapshot(root)), encoding="utf-8")
    return before


def _compare(root: Path, before: Path, additions: set[str]) -> dict[str, object]:
    return compare_protected_snapshot(
        root,
        before,
        allowed_changed_paths=validator.ALLOWED_REPAIR_SCHEMA_CHANGES,
        allowed_added_paths=set(validator.validate_schema_addition_paths(additions)),
    )


def test_all_exact_cache_scheduler_schema_additions_are_permitted(tmp_path: Path) -> None:
    assert len(validator.CACHE_SCHEDULER_SCHEMA_ADDITIONS) == 22
    assert validator.CACHE_SCHEDULER_SCHEMA_ADDITIONS.issubset(
        validator.ALLOWED_REPAIR_SCHEMA_ADDITIONS
    )
    before = _write_baseline(tmp_path)
    for relative in validator.CACHE_SCHEDULER_SCHEMA_ADDITIONS:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("downstream schema", encoding="utf-8")

    result = _compare(tmp_path, before, validator.ALLOWED_REPAIR_SCHEMA_ADDITIONS)
    assert result["status"] == "PASS"
    assert set(result["allowed_added_files"]) == validator.CACHE_SCHEDULER_SCHEMA_ADDITIONS
    assert result["unexpected_added_files"] == []


def test_exact_smoke_contract_schemas_are_permitted_without_weakening_unknown_paths(
    tmp_path: Path,
) -> None:
    expected = {
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
    }
    assert expected.issubset(validator.ALLOWED_REPAIR_SCHEMA_ADDITIONS)
    before = _write_baseline(tmp_path)
    for relative in expected:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("smoke contract schema", encoding="utf-8")

    result = _compare(tmp_path, before, validator.ALLOWED_REPAIR_SCHEMA_ADDITIONS)
    assert result["status"] == "PASS"
    assert set(result["allowed_added_files"]) == expected
    assert result["unexpected_added_files"] == []


def test_exact_pilot_protocol_schemas_are_permitted_without_weakening_unknown_paths(
    tmp_path: Path,
) -> None:
    expected = {
        "schemas/pilot_protocol.schema.json",
        "schemas/pilot_dataset_selection.schema.json",
        "schemas/pilot_condition.schema.json",
        "schemas/pilot_condition_inventory.schema.json",
        "schemas/pilot_metric_policy.schema.json",
        "schemas/pilot_decision_policy.schema.json",
        "schemas/pilot_protocol_validation.schema.json",
    }
    assert expected == validator.PILOT_PROTOCOL_SCHEMA_ADDITIONS
    assert expected.issubset(validator.ALLOWED_REPAIR_SCHEMA_ADDITIONS)
    before = _write_baseline(tmp_path)
    for relative in expected:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("pilot protocol schema", encoding="utf-8")

    result = _compare(tmp_path, before, validator.ALLOWED_REPAIR_SCHEMA_ADDITIONS)
    assert result["status"] == "PASS"
    assert set(result["allowed_added_files"]) == expected
    assert result["unexpected_added_files"] == []


def test_removing_an_expected_schema_from_allowlist_fails(tmp_path: Path) -> None:
    before = _write_baseline(tmp_path)
    expected = "schemas/scheduler_run_plan.schema.json"
    path = tmp_path / expected
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("downstream schema", encoding="utf-8")

    additions = set(validator.ALLOWED_REPAIR_SCHEMA_ADDITIONS) - {expected}
    result = _compare(tmp_path, before, additions)
    assert result["status"] == "FAIL"
    assert result["unexpected_added_files"] == [expected]


def test_unknown_schema_addition_fails(tmp_path: Path) -> None:
    before = _write_baseline(tmp_path)
    unexpected = "schemas/unregistered.schema.json"
    path = tmp_path / unexpected
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("unexpected", encoding="utf-8")

    result = _compare(tmp_path, before, validator.ALLOWED_REPAIR_SCHEMA_ADDITIONS)
    assert result["status"] == "FAIL"
    assert result["unexpected_added_files"] == [unexpected]


@pytest.mark.parametrize(
    "path",
    (
        "C:/outside/example.schema.json",
        "schemas/../outside.schema.json",
        "../schemas/example.schema.json",
        "schemas/*.schema.json",
    ),
)
def test_absolute_traversal_and_pattern_allowlist_paths_are_rejected(path: str) -> None:
    with pytest.raises(ValueError, match="invalid exact schema addition path"):
        validator.validate_schema_addition_paths({path})


@pytest.mark.parametrize(
    "relative",
    (
        "src/schemaguard/transformations/codec.py",
        "schemas/transformation_inventory.schema.json",
    ),
)
def test_modified_transformation_source_or_schema_fails_closed(
    tmp_path: Path, relative: str
) -> None:
    root = tmp_path / "repository"
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("approved", encoding="utf-8")
    subprocess.run(["git", "init", str(root)], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(root), "config", "user.name", "SchemaGuard test"],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(root), "config", "user.email", "schemaguard@example.invalid"],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(root), "add", relative], check=True, capture_output=True
    )
    subprocess.run(
        ["git", "-C", str(root), "commit", "-m", "baseline"],
        check=True,
        capture_output=True,
    )
    baseline = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    validator.assert_frozen_transformation_tree(
        root, baseline_commit=baseline, paths=(relative,)
    )
    target.write_text("modified", encoding="utf-8")
    with pytest.raises(ValueError, match="transformation implementation or protected schema"):
        validator.assert_frozen_transformation_tree(
            root, baseline_commit=baseline, paths=(relative,)
        )


def test_alternate_ignored_inventory_output_preserves_tracked_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repository"
    tracked_inventory = root / "artifacts/handoff/transformation_inventory.json"
    tracked_inventory.parent.mkdir(parents=True, exist_ok=True)
    tracked_inventory.write_bytes(b"accepted tracked inventory")
    before = root / "artifacts/transformation_engine/review/repair_hashes_before.json"
    before.parent.mkdir(parents=True, exist_ok=True)
    before.write_text("{}", encoding="utf-8")
    original_bytes = tracked_inventory.read_bytes()
    alternate = root / "artifacts/transformation_engine/review/repeat.json"

    class Inventory:
        def __init__(self, status: str, comparison: dict[str, str]) -> None:
            self.status = status
            self.protected_hash_comparison = comparison

        def model_copy(self, *, update: dict[str, object]) -> Inventory:
            return Inventory(
                str(update["status"]),
                dict(update["protected_hash_comparison"]),  # type: ignore[arg-type]
            )

        def canonical_dict(self) -> dict[str, object]:
            return {
                "status": self.status,
                "protected_hash_comparison": self.protected_hash_comparison,
            }

    monkeypatch.setattr(validator, "ROOT", root)
    monkeypatch.setattr(validator, "assert_frozen_transformation_tree", lambda: None)
    monkeypatch.setattr(validator, "load_transformation_config", lambda _path: object())
    monkeypatch.setattr(validator, "_validate_split_inventory", lambda: "a" * 64)
    monkeypatch.setattr(validator, "DATASET_IDS", [])
    monkeypatch.setattr(validator, "SEEDS", [])
    monkeypatch.setattr(validator, "VIEW_IDS", [])
    def fake_write_inventory(
        _root: Path, _config: object, _before: Path, output: Path
    ) -> Inventory:
        assert output == alternate
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("temporary generated inventory", encoding="utf-8")
        return Inventory("FAIL", {"status": "FAIL"})

    monkeypatch.setattr(validator, "write_inventory", fake_write_inventory)
    monkeypatch.setattr(
        validator,
        "compare_protected_snapshot",
        lambda *_args, **_kwargs: {"status": "PASS"},
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "validate_transformation_engine.py",
            "--all",
            "--offline",
            "--inventory-output",
            str(alternate),
        ],
    )

    assert validator.main() == 0
    assert tracked_inventory.read_bytes() == original_bytes
    assert json.loads(alternate.read_text(encoding="utf-8"))["status"] == "PASS_PENDING_REVIEW"
