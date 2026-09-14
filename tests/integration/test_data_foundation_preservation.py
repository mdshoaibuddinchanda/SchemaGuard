from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_data_foundation_grouped_split_remains_valid() -> None:
    manifest = json.loads(
        (
            ROOT / "data/splits/openml/1464/stratified_group_5fold_v1/seed_1729/split_manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert manifest["strategy"] == "stratified_group_5fold_v1"
    assert manifest["row_counts"] == {"train": 449, "calibration": 150, "test": 149}
    assert manifest["total_predictor_groups"] == 502
    assert manifest["duplicate_predictor_groups"] == 69
    assert manifest["conflicting_target_groups"] == 31
    assert manifest["predictor_duplicate_groups_crossing_splits"] == 0
