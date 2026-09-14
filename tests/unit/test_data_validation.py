from pathlib import Path

import pytest

from schemaguard.data.arff_parser import parse_arff
from schemaguard.data.contracts import ExpectedDatasetProperties
from schemaguard.data.pipeline import load_smoke_config
from schemaguard.data.validate import validate_source_table

FIXTURES = Path(__file__).parents[1] / "fixtures"
CONFIG = Path(__file__).parents[2] / "configs" / "datasets" / "smoke_blood_transfusion.yaml"


def tiny_config(rows: int = 3, numeric: int = 1, categorical: int = 1):
    config = load_smoke_config(CONFIG)
    return config.model_copy(
        update={
            "expected": ExpectedDatasetProperties(
                rows=rows,
                predictor_columns=numeric + categorical,
                total_columns=numeric + categorical + 1,
                classes=2,
                numeric_predictors=numeric,
                categorical_predictors=categorical,
                missing_values=0,
            )
        }
    )


def test_valid_fixture_passes_and_duplicates_warn() -> None:
    result = validate_source_table(parse_arff(FIXTURES / "tiny_valid.arff"), tiny_config())
    assert result.passed
    assert "duplicate_complete_rows" in result.report.warnings


def test_missing_target_fails() -> None:
    result = validate_source_table(
        parse_arff(FIXTURES / "tiny_missing_target.arff"), tiny_config(rows=2, categorical=0)
    )
    assert not result.passed
    assert any(
        check.name == "missing_target" and check.status == "failed"
        for check in result.report.checks
    )


def test_infinite_numeric_value_fails() -> None:
    result = validate_source_table(
        parse_arff(FIXTURES / "tiny_infinite_value.arff"), tiny_config(rows=2, categorical=0)
    )
    assert not result.passed
    assert any(
        check.name == "finite_numeric_values" and check.status == "failed"
        for check in result.report.checks
    )


def test_unexpected_reserved_prefix_fails_at_processing_boundary() -> None:
    parsed = parse_arff(FIXTURES / "tiny_valid.arff")
    parsed.frame.rename(columns={"age": "__sg_age"}, inplace=True)
    with pytest.raises(Exception):
        from schemaguard.data.process import process_dataset

        process_dataset(parsed, tiny_config(), "a" * 64, Path("unused"), "b" * 64)
