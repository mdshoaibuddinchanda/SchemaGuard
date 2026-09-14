from pathlib import Path

import pandas as pd

from schemaguard.data.pipeline import load_smoke_config
from schemaguard.data.splits import generate_splits, validate_split_assignments

CONFIG = Path(__file__).parents[2] / "configs" / "datasets" / "smoke_blood_transfusion.yaml"


def split_config():
    config = load_smoke_config(CONFIG)
    return config.model_copy(
        update={
            "split": config.split.model_copy(
                update={"train_fraction": 0.6, "calibration_fraction": 0.2, "test_fraction": 0.2}
            )
        }
    )


def make_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    row_ids = [f"row-{index:03d}" for index in range(20)]
    codes = [index % 2 for index in range(20)]
    features = pd.DataFrame({"__sg_row_id": row_ids, "x": range(20)})
    targets = pd.DataFrame(
        {
            "__sg_row_id": row_ids,
            "target_label": [str(code) for code in codes],
            "target_code": codes,
        }
    )
    return features, targets


def test_splits_are_disjoint_complete_and_repeatable(tmp_path: Path) -> None:
    features, targets = make_tables()
    first = generate_splits(features, targets, split_config(), tmp_path / "first", "a" * 64)
    second = generate_splits(
        features.sample(frac=1, random_state=5),
        targets.sample(frac=1, random_state=7),
        split_config(),
        tmp_path / "second",
        "a" * 64,
    )
    assert first.assignments.equals(second.assignments)
    assert first.manifest.row_counts == {"train": 12, "calibration": 4, "test": 4}
    assert first.manifest.assignment_file_sha256 == second.manifest.assignment_file_sha256
    details = validate_split_assignments(first.assignments, targets, split_config())
    assert details["splits_disjoint"]
    assert details["both_classes_in_every_split"]
