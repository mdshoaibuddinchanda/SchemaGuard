from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from schemaguard.transformations.certificates import validate_roundtrip
from schemaguard.transformations.numeric_affine import NumericAffineTransformation

from .transformation_helpers import sample_frame, sample_schema


@settings(max_examples=1000, deadline=None, derandomize=True)
@given(st.lists(st.integers(min_value=-1000, max_value=1000), min_size=5, max_size=20))
def test_affine_roundtrip_property(values: list[int]) -> None:
    frame = sample_frame().iloc[[index % 12 for index in range(len(values))]].reset_index(drop=True)
    frame["__sg_row_id"] = [f"property-row-{index}" for index in range(len(values))]
    frame["amount"] = [float(value) for value in values]
    transformation = NumericAffineTransformation({"max_numeric_columns": 1}).fit(
        frame, 1464, 1729, sample_schema(frame)
    )
    output = transformation.transform(frame, "train")
    certificate = transformation.certificate_for(frame, output, "train")
    restored = transformation.reconstruct(output, certificate)
    assert validate_roundtrip(frame, output, restored, certificate)["status"] == "PASS"
