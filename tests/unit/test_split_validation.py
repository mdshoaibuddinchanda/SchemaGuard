import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from schemaguard.constants import ROW_ID_COLUMN, TARGET_CODE_COLUMN
from schemaguard.splits import generation, validation
from schemaguard.splits.contracts import SplitGenerationConfig
from schemaguard.splits.generation import generate_split
from schemaguard.splits.validation import (
    SplitValidationError,
    _reject_temporary_artifacts,
    validate_split,
)

from .test_split_contracts import valid_payload


def test_missing_split_is_rejected(tmp_path: Path) -> None:
    config = SplitGenerationConfig.model_validate(valid_payload())
    with pytest.raises(SplitValidationError):
        validate_split(tmp_path, config, 3, 1729)


def _config() -> SplitGenerationConfig:
    return SplitGenerationConfig.model_validate(valid_payload())


def _synthetic_inputs() -> tuple[
    object, object, pd.DataFrame, pd.DataFrame, dict[str, object], dict[str, object]
]:
    row_ids = [f"r-{index:03d}" for index in range(100)]
    features = pd.DataFrame(
        {ROW_ID_COLUMN: row_ids, "x": range(100), "category": [index % 3 for index in range(100)]}
    )
    targets = pd.DataFrame(
        {ROW_ID_COLUMN: row_ids, TARGET_CODE_COLUMN: [index % 2 for index in range(100)]}
    )
    spec = SimpleNamespace(id=3, version="1", name="synthetic", ignore_attributes=[])
    return spec, features, features.copy(), targets, {}, {"feature_columns": ["x", "category"]}


def _make_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, SplitGenerationConfig]:
    config = _config()
    _, features, _unused, targets, quality, data_manifest = _synthetic_inputs()
    spec = SimpleNamespace(id=3, version="1", name="synthetic", ignore_attributes=[])

    def provider(
        _root: Path, _dataset_id: int
    ) -> tuple[object, object, pd.DataFrame, pd.DataFrame, dict[str, object], dict[str, object]]:
        return spec, spec, features.copy(), targets.copy(), quality, data_manifest

    monkeypatch.setattr(generation, "_read_local_inputs", provider)
    monkeypatch.setattr(validation, "_read_local_inputs", provider)
    monkeypatch.setattr(generation, "_git_commit", lambda _root: "synthetic-commit")
    processed = tmp_path / "data" / "processed" / "openml" / "3"
    processed.mkdir(parents=True)
    features.to_parquet(processed / "features.parquet", index=False)
    targets.to_parquet(processed / "targets.parquet", index=False)
    (processed / "data_manifest.json").write_text(json.dumps(data_manifest), encoding="utf-8")
    result = generate_split(tmp_path, config, 3, 1729)
    assert result["status"] == "GENERATED"
    return tmp_path, config


def test_fold_provenance_is_row_level_and_partition_consistent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, config = _make_artifact(tmp_path, monkeypatch)
    assignment = pd.read_parquet(
        root
        / "data"
        / "splits"
        / "openml"
        / "3"
        / config.strategy
        / "seed_1729"
        / "assignments.parquet"
    )
    manifest = json.loads(
        (
            root
            / "data"
            / "splits"
            / "openml"
            / "3"
            / config.strategy
            / "seed_1729"
            / "split_manifest.json"
        ).read_text()
    )
    assert set(assignment.loc[assignment.partition == "train", "fold_id"]) == {0, 1, 2}
    assert set(assignment.loc[assignment.partition == "calibration", "fold_id"]) == {
        manifest["fold_assignment"]["calibration"][0]
    }
    assert set(assignment.loc[assignment.partition == "test", "fold_id"]) == {
        manifest["fold_assignment"]["test"][0]
    }
    assert validate_split(root, config, 3, 1729)["status"] == "PASS"


@pytest.mark.parametrize("status", ["FAILED", "CONSTRAINT_INFEASIBLE"])
def test_non_passing_manifest_status_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: str
) -> None:
    root, config = _make_artifact(tmp_path, monkeypatch)
    manifest_path = (
        root
        / "data"
        / "splits"
        / "openml"
        / "3"
        / config.strategy
        / "seed_1729"
        / "split_manifest.json"
    )
    payload = json.loads(manifest_path.read_text())
    payload["validation_status"] = status
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SplitValidationError, match="manifest is not passing"):
        validate_split(root, config, 3, 1729)


def test_group_tampering_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root, config = _make_artifact(tmp_path, monkeypatch)
    path = (
        root
        / "data"
        / "splits"
        / "openml"
        / "3"
        / config.strategy
        / "seed_1729"
        / "assignments.parquet"
    )
    assignment = pd.read_parquet(path)
    assignment.loc[0, "predictor_group_id"] = "tampered"
    assignment.to_parquet(path, index=False)
    with pytest.raises(SplitValidationError, match="predictor groups"):
        validate_split(root, config, 3, 1729)


def test_fold_tampering_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root, config = _make_artifact(tmp_path, monkeypatch)
    path = (
        root
        / "data"
        / "splits"
        / "openml"
        / "3"
        / config.strategy
        / "seed_1729"
        / "assignments.parquet"
    )
    assignment = pd.read_parquet(path)
    assignment.loc[0, "fold_id"] = 4 if assignment.loc[0, "fold_id"] != 4 else 3
    assignment.to_parquet(path, index=False)
    with pytest.raises(SplitValidationError, match="fold provenance"):
        validate_split(root, config, 3, 1729)


def test_cache_identity_tampering_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, config = _make_artifact(tmp_path, monkeypatch)
    path = (
        root
        / "data"
        / "splits"
        / "openml"
        / "3"
        / config.strategy
        / "seed_1729"
        / "split_manifest.json"
    )
    payload = json.loads(path.read_text())
    payload["cache_identity_hash"] = "0" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SplitValidationError, match="cache identity"):
        validate_split(root, config, 3, 1729)


def test_split_implementation_tampering_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, config = _make_artifact(tmp_path, monkeypatch)
    path = (
        root
        / "data"
        / "splits"
        / "openml"
        / "3"
        / config.strategy
        / "seed_1729"
        / "split_manifest.json"
    )
    payload = json.loads(path.read_text())
    payload["split_implementation_hash"] = "0" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SplitValidationError, match="split implementation"):
        validate_split(root, config, 3, 1729)


def test_assignment_checksum_tampering_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, config = _make_artifact(tmp_path, monkeypatch)
    path = (
        root
        / "data"
        / "splits"
        / "openml"
        / "3"
        / config.strategy
        / "seed_1729"
        / "assignments.parquet"
    )
    path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(SplitValidationError):
        validate_split(root, config, 3, 1729)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda frame: frame.iloc[:-1].copy(),
        lambda frame: pd.concat([frame, frame.iloc[[0]]], ignore_index=True),
        lambda frame: frame.assign(
            __sg_row_id=lambda value: value[ROW_ID_COLUMN].mask(value.index == 0, "unknown")
        ),
    ],
)
def test_row_coverage_tampering_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: object
) -> None:
    root, config = _make_artifact(tmp_path, monkeypatch)
    path = (
        root
        / "data"
        / "splits"
        / "openml"
        / "3"
        / config.strategy
        / "seed_1729"
        / "assignments.parquet"
    )
    assignment = pd.read_parquet(path)
    mutated = mutation(assignment)  # type: ignore[operator]
    mutated.to_parquet(path, index=False)
    with pytest.raises(SplitValidationError, match="row count|row-ID|row ID"):
        validate_split(root, config, 3, 1729)


def test_empty_and_partial_temporary_directories_are_rejected(tmp_path: Path) -> None:
    split_root = tmp_path / "data" / "splits" / "openml" / "3" / "stratified_group_5fold_v1"
    (split_root / ".seed_1729.empty").mkdir(parents=True)
    (split_root / "split-incomplete-3-1729").mkdir()
    with pytest.raises(SplitValidationError, match="temporary or partial"):
        _reject_temporary_artifacts(tmp_path)
