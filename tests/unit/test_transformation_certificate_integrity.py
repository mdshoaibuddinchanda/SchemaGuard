from __future__ import annotations

import copy

import pytest
from pydantic import ValidationError

from schemaguard.transformations.certificates import validate_roundtrip
from schemaguard.transformations.identity import IdentityTransformation

from .transformation_helpers import sample_frame, sample_schema, target_hash, transformation_config


def _certificate():
    source = sample_frame()
    transformation = IdentityTransformation(transformation_config()).fit(
        source, 1464, 1729, sample_schema(source)
    )
    output = transformation.transform(source, "test")
    certificate = transformation.certificate_for(
        source,
        output,
        "test",
        source_target_hash=target_hash(source),
        output_target_hash=target_hash(source),
    )
    return source, output, transformation, certificate


def test_independent_validation_cannot_claim_configuration_identity() -> None:
    source, output, transformation, certificate = _certificate()
    restored = transformation.reconstruct(output, certificate)
    with pytest.raises(ValueError, match="fitted transformation"):
        validate_roundtrip(source, output, restored, certificate)


def test_reserved_view_name_and_policy_values_are_closed_world() -> None:
    source, output, transformation, certificate = _certificate()
    bad_parameters = copy.deepcopy(certificate.parameters)
    bad_parameters["_view_name"] = "not_identity"
    with pytest.raises(ValidationError):
        type(certificate).model_validate(
            certificate.model_copy(update={"parameters": bad_parameters}).canonical_dict()
        )
    with pytest.raises(ValidationError):
        type(certificate).model_validate(
            certificate.model_copy(update={"missing_mask_policy": "best_effort"}).canonical_dict()
        )
    with pytest.raises(ValueError, match="tolerances"):
        restored = transformation.reconstruct(output, certificate)
        validate_roundtrip(
            source, output, restored, certificate, transformation=transformation, rtol=1.0e-9
        )
