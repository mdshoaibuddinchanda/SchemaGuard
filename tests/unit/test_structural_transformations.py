from __future__ import annotations

import pytest

from schemaguard.transformations.certificates import validate_roundtrip
from schemaguard.transformations.column_permutation import ColumnPermutationTransformation
from schemaguard.transformations.duplicate_feature import DuplicateFeatureTransformation
from schemaguard.transformations.quotient_remainder import QuotientRemainderTransformation
from schemaguard.transformations.redundant_affine import RedundantAffineTransformation
from schemaguard.transformations.row_permutation import RowPermutationTransformation

from .transformation_helpers import sample_frame, sample_schema, transformation_config


@pytest.mark.parametrize(
    "factory",
    [
        DuplicateFeatureTransformation,
        RedundantAffineTransformation,
        QuotientRemainderTransformation,
        ColumnPermutationTransformation,
        RowPermutationTransformation,
    ],
)
def test_structural_views_roundtrip(factory) -> None:
    frame = sample_frame()
    transformation = factory(transformation_config()).fit(frame, 1464, 1729, sample_schema(frame))
    output = transformation.transform(frame, "train")
    certificate = transformation.certificate_for(frame, output, "train")
    restored = transformation.reconstruct(output, certificate)
    assert validate_roundtrip(frame, output, restored, certificate)["status"] == "PASS"


def test_row_permutation_does_not_reorder_calibration_or_test() -> None:
    frame = sample_frame()
    transformation = RowPermutationTransformation(transformation_config()).fit(
        frame, 1464, 1729, sample_schema(frame)
    )
    assert (
        transformation.transform(frame, "calibration")["__sg_row_id"].tolist()
        == frame["__sg_row_id"].tolist()
    )
