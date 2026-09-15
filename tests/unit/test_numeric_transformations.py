from __future__ import annotations

import pytest

from schemaguard.transformations.certificates import validate_roundtrip
from schemaguard.transformations.numeric_affine import NumericAffineTransformation
from schemaguard.transformations.numeric_asinh import NumericAsinhTransformation

from .transformation_helpers import sample_frame, sample_schema, target_hash, transformation_config


@pytest.mark.parametrize("factory", [NumericAffineTransformation, NumericAsinhTransformation])
def test_numeric_views_roundtrip_from_training_fit(factory) -> None:
    frame = sample_frame()
    transformation = factory(transformation_config()).fit(
        frame.iloc[:8], 1464, 1729, sample_schema(frame.iloc[:8])
    )
    output = transformation.transform(frame, "test")
    certificate = transformation.certificate_for(
        frame,
        output,
        "test",
        source_target_hash=target_hash(frame),
        output_target_hash=target_hash(frame),
    )
    restored = transformation.reconstruct(output, certificate)
    assert validate_roundtrip(frame, output, restored, certificate)["status"] == "PASS"


def test_numeric_views_reject_nonfinite_input() -> None:
    frame = sample_frame()
    frame.loc[0, "amount"] = float("inf")
    transformation = NumericAffineTransformation(transformation_config()).fit(
        sample_frame(), 1464, 1729, sample_schema()
    )
    with pytest.raises(ValueError):
        transformation.transform(frame, "test")
