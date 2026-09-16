from __future__ import annotations

import json
from pathlib import Path

from schemaguard.transformations.inventory import _protected_snapshot, compare_protected_snapshot


def _baseline(root: Path) -> Path:
    for relative in (
        "configs/baselines/config.json",
        "data/raw/openml/source.bin",
        "data/processed/openml/features.parquet",
        "data/splits/openml/assignments.parquet",
        "schemas/frozen.schema.json",
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"baseline")
    handoff = root / "artifacts/handoff/split_generation_inventory.json"
    handoff.parent.mkdir(parents=True, exist_ok=True)
    handoff.write_bytes(b"baseline")
    (root / "artifacts/handoff/transformation_inventory.json").write_bytes(b"baseline")
    snapshot = root / "before.json"
    snapshot.write_text(json.dumps(_protected_snapshot(root)), encoding="utf-8")
    return snapshot


def test_unexpected_protected_addition_fails_closed(tmp_path: Path) -> None:
    before = _baseline(tmp_path)
    (tmp_path / "schemas/unexpected.schema.json").write_bytes(b"new")
    result = compare_protected_snapshot(tmp_path, before)
    assert result["status"] == "FAIL"
    assert result["unexpected_added_files"] == ["schemas/unexpected.schema.json"]


def test_unexpected_protected_removal_and_modification_fail_closed(tmp_path: Path) -> None:
    before = _baseline(tmp_path)
    (tmp_path / "schemas/frozen.schema.json").write_bytes(b"changed")
    (tmp_path / "data/raw/openml/source.bin").unlink()
    result = compare_protected_snapshot(tmp_path, before)
    assert result["status"] == "FAIL"
    assert result["unexpected_changed_files"] == ["schemas/frozen.schema.json"]
    assert result["unexpected_removed_files"] == ["data/raw/openml/source.bin"]


def test_explicit_protected_addition_modification_and_removal_are_allowed(tmp_path: Path) -> None:
    before = _baseline(tmp_path)
    (tmp_path / "schemas/allowed.schema.json").write_bytes(b"new")
    (tmp_path / "schemas/frozen.schema.json").write_bytes(b"changed")
    (tmp_path / "data/raw/openml/source.bin").unlink()
    result = compare_protected_snapshot(
        tmp_path,
        before,
        allowed_added_paths={"schemas/allowed.schema.json"},
        allowed_changed_paths={"schemas/frozen.schema.json"},
        allowed_removed_paths={"data/raw/openml/source.bin"},
    )
    assert result["status"] == "PASS"
    assert result["unexpected_added_files"] == []
    assert result["unexpected_changed_files"] == []
    assert result["unexpected_removed_files"] == []


def test_after_snapshot_hash_is_stable_for_an_allowed_self_referential_inventory(
    tmp_path: Path,
) -> None:
    before = _baseline(tmp_path)
    inventory = tmp_path / "artifacts/handoff/transformation_inventory.json"
    allowed_change = {"artifacts/handoff/transformation_inventory.json"}

    inventory.write_bytes(b"first generated inventory")
    first = compare_protected_snapshot(
        tmp_path, before, allowed_changed_paths=allowed_change
    )
    inventory.write_bytes(b"second generated inventory with new timestamp")
    second = compare_protected_snapshot(
        tmp_path, before, allowed_changed_paths=allowed_change
    )

    assert first["status"] == second["status"] == "PASS"
    assert first["after_snapshot_hash"] == second["after_snapshot_hash"]
    assert first["changed_files"] == second["changed_files"] == sorted(allowed_change)
