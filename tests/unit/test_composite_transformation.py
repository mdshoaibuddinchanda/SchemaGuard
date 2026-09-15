from __future__ import annotations

from schemaguard.transformations.certificates import validate_roundtrip
from schemaguard.transformations.composite_migration import CompositeMigrationTransformation

from .transformation_helpers import sample_frame, sample_schema, transformation_config


def test_composite_has_ordered_component_certificates_and_roundtrips() -> None:
    frame = sample_frame()
    transformation = CompositeMigrationTransformation(transformation_config()).fit(
        frame, 1464, 1729, sample_schema(frame)
    )
    output = transformation.transform(frame, "train")
    certificate = transformation.certificate_for(frame, output, "train")
    restored = transformation.reconstruct(output, certificate)
    assert certificate.parameters["components"] == ["V01", "V03", "V08"]
    assert len(certificate.parameters["component_certificates"]) == 3
    assert validate_roundtrip(frame, output, restored, certificate)["status"] == "PASS"
