from pathlib import Path

from schemaguard.data.arff_parser import parse_arff
from schemaguard.data.pipeline import load_smoke_config
from schemaguard.data.process import process_dataset, row_id_for_position
from schemaguard.data.validate import validate_processed_tables

FIXTURE = Path(__file__).parents[1] / "fixtures" / "tiny_valid.arff"
CONFIG = Path(__file__).parents[2] / "configs" / "datasets" / "smoke_blood_transfusion.yaml"


def test_processing_preserves_values_duplicates_and_alignment(tmp_path: Path) -> None:
    config = load_smoke_config(CONFIG).model_copy(
        update={
            "expected": load_smoke_config(CONFIG).expected.model_copy(
                update={
                    "rows": 3,
                    "predictor_columns": 2,
                    "total_columns": 3,
                    "numeric_predictors": 1,
                    "categorical_predictors": 1,
                }
            )
        }
    )
    parsed = parse_arff(FIXTURE)
    result = process_dataset(parsed, config, "a" * 64, tmp_path, "b" * 64)
    assert len(result.features) == 3
    assert result.features["age"].tolist() == [10.0, 20.0, 10.0]
    assert (
        len(
            result.features.drop(columns=["__sg_row_id"])[
                result.features.drop(columns=["__sg_row_id"]).duplicated(keep=False)
            ]
        )
        == 2
    )
    assert result.targets["target_code"].tolist() == [0, 1, 0]
    validation = validate_processed_tables(
        result.features,
        result.targets,
        parsed,
        config,
        {"no": 0, "yes": 1},
    )
    assert validation.passed


def test_row_ids_are_deterministic_and_depend_on_raw_hash() -> None:
    first = row_id_for_position(1464, "a" * 64, 0)
    assert first == row_id_for_position(1464, "a" * 64, 0)
    assert first != row_id_for_position(1464, "b" * 64, 0)
    assert len(first) == 32
