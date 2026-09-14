from pathlib import Path

import pytest

from schemaguard.data.arff_parser import ArffParseError, parse_arff

FIXTURES = Path(__file__).parents[1] / "fixtures"


def test_valid_arff_preserves_relation_columns_and_order() -> None:
    parsed = parse_arff(FIXTURES / "tiny_valid.arff")
    assert parsed.relation == "tiny_valid"
    assert [item.name for item in parsed.attributes] == ["age", "city", "Class"]
    assert parsed.numeric_columns == ("age",)
    assert parsed.categorical_columns == ("city", "Class")
    assert parsed.source_row_positions == (0, 1, 2)
    assert list(parsed.frame.columns) == ["age", "city", "Class", "__sg_source_row_position"]


def test_bad_row_width_is_rejected() -> None:
    with pytest.raises(ArffParseError, match="width"):
        parse_arff(FIXTURES / "tiny_bad_width.arff")


def test_duplicate_attribute_names_are_rejected() -> None:
    with pytest.raises(ArffParseError, match="duplicate"):
        parse_arff(FIXTURES / "tiny_duplicate_columns.arff")


def test_missing_target_is_kept_as_null_until_validation() -> None:
    parsed = parse_arff(FIXTURES / "tiny_missing_target.arff")
    assert parsed.frame["Class"].isna().sum() == 1
