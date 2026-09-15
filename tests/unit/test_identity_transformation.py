from __future__ import annotations

from schemaguard.transformations.certificates import validate_roundtrip
from schemaguard.transformations.identity import IdentityTransformation

from .transformation_helpers import sample_frame, sample_schema, transformation_config


def test_identity_preserves_rows_columns_values_and_certificate() -> None:
    frame = sample_frame()
    transformation = IdentityTransformation(transformation_config()).fit(
        frame, 1464, 1729, sample_schema(frame)
    )
    output = transformation.transform(frame, "train")
    certificate = transformation.certificate_for(frame, output, "train")
    restored = transformation.reconstruct(output, certificate)
    assert output.equals(frame)
    assert restored.equals(frame)
    assert validate_roundtrip(frame, output, restored, certificate)["status"] == "PASS"
