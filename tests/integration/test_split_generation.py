import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from schemaguard.constants import ROW_ID_COLUMN, TARGET_CODE_COLUMN
from schemaguard.splits import generation, validation
from schemaguard.splits.contracts import SplitGenerationConfig
from schemaguard.splits.generation import generate_split


def _config() -> SplitGenerationConfig:
    return SplitGenerationConfig.model_validate(
        {
            "schema_version": 1,
            "strategy": "stratified_group_5fold_v1",
            "grouping_method": "typed_predictor_sha256_v1",
            "group_by": "predictors",
            "group_folds": 5,
            "train_fraction": 0.60,
            "calibration_fraction": 0.20,
            "test_fraction": 0.20,
            "minimum_class_count_per_partition": 5,
            "seeds": [1729, 2718, 31415, 57721, 161803],
            "datasets": [3, 23, 29, 31, 36, 37, 38, 44, 46, 50, 54, 1067, 1464, 1489],
            "max_workers": 2,
        }
    )


@pytest.mark.integration
def test_grouped_split_generation_write_validate_promote_reload_and_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    row_ids = [f"r-{index:03d}" for index in range(100)]
    features = pd.DataFrame({ROW_ID_COLUMN: row_ids, "x": range(100)})
    targets = pd.DataFrame(
        {ROW_ID_COLUMN: row_ids, TARGET_CODE_COLUMN: [index % 2 for index in range(100)]}
    )
    spec = SimpleNamespace(id=3, version="1", name="synthetic", ignore_attributes=[])
    data_manifest = {"feature_columns": ["x"]}

    def provider(
        _root: Path, _dataset_id: int
    ) -> tuple[object, object, pd.DataFrame, pd.DataFrame, dict[str, object], dict[str, object]]:
        return spec, spec, features.copy(), targets.copy(), {}, data_manifest

    monkeypatch.setattr(generation, "_read_local_inputs", provider)
    monkeypatch.setattr(validation, "_read_local_inputs", provider)
    monkeypatch.setattr(generation, "_git_commit", lambda _root: "synthetic-commit")
    processed = tmp_path / "data" / "processed" / "openml" / "3"
    processed.mkdir(parents=True)
    features.to_parquet(processed / "features.parquet", index=False)
    targets.to_parquet(processed / "targets.parquet", index=False)
    (processed / "data_manifest.json").write_text(json.dumps(data_manifest), encoding="utf-8")

    config = _config()
    generated = generate_split(tmp_path, config, 3, 1729)
    assert generated["status"] == "GENERATED"
    validated = validation.validate_split(tmp_path, config, 3, 1729)
    assert validated["status"] == "PASS"
    cached = generate_split(tmp_path, config, 3, 1729)
    assert cached["status"] == "CACHE_HIT"
    assert cached["logical_assignment_hash"] == generated["logical_assignment_hash"]
