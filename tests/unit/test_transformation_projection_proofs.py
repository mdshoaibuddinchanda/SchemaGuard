from __future__ import annotations

import copy

import numpy as np
import pytest

from schemaguard.transformations.certificates import _validate_projection_relationship
from schemaguard.transformations.duplicate_feature import DuplicateFeatureTransformation
from schemaguard.transformations.redundant_affine import RedundantAffineTransformation

from .transformation_helpers import sample_frame, sample_schema, target_hash, transformation_config


def _projection(factory):
    source = sample_frame()
    transformation = factory(transformation_config()).fit(
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
    return source, output, certificate


@pytest.mark.parametrize("factory", [DuplicateFeatureTransformation, RedundantAffineTransformation])
def test_projection_proof_accepts_the_declared_relationship(factory) -> None:
    source, output, certificate = _projection(factory)
    _validate_projection_relationship(source, output, certificate)


def test_projection_proof_rejects_parent_missing_but_generated_present() -> None:
    source, output, certificate = _projection(RedundantAffineTransformation)
    parent = certificate.parameters["selected_columns"][0]
    source.loc[0, parent] = np.nan
    with pytest.raises(ValueError, match="missing masks"):
        _validate_projection_relationship(source, output, certificate)


def test_projection_proof_rejects_parent_present_but_generated_missing() -> None:
    source, output, certificate = _projection(RedundantAffineTransformation)
    generated = certificate.generated_columns[0]
    output.loc[0, generated] = np.nan
    with pytest.raises(ValueError, match="missing masks"):
        _validate_projection_relationship(source, output, certificate)


def test_projection_proof_rejects_altered_generated_value() -> None:
    source, output, certificate = _projection(RedundantAffineTransformation)
    generated = certificate.generated_columns[0]
    output.loc[0, generated] += 1.0
    with pytest.raises(ValueError, match="relationship is invalid"):
        _validate_projection_relationship(source, output, certificate)


def test_projection_proof_rejects_missing_generated_column() -> None:
    source, output, certificate = _projection(DuplicateFeatureTransformation)
    generated = certificate.generated_columns[0]
    output = output.drop(columns=[generated])
    with pytest.raises(ValueError, match="missing a declared generated"):
        _validate_projection_relationship(source, output, certificate)


def test_projection_proof_rejects_undeclared_relationship() -> None:
    source, output, certificate = _projection(DuplicateFeatureTransformation)
    parameters = copy.deepcopy(certificate.parameters)
    parameters["generated_relationship"]["unexpected"] = {
        "parent": certificate.parameters["selected_columns"][0],
        "relationship": "equal",
    }
    tampered = certificate.model_copy(update={"parameters": parameters})
    with pytest.raises(ValueError, match="cover exactly"):
        _validate_projection_relationship(source, output, tampered)


def test_projection_proof_rejects_changed_parent() -> None:
    source, output, certificate = _projection(DuplicateFeatureTransformation)
    parent = certificate.parameters["selected_columns"][0]
    other = next(column for column in source.columns if column not in {"__sg_row_id", parent})
    parameters = copy.deepcopy(certificate.parameters)
    generated = certificate.generated_columns[0]
    parameters["generated_relationship"][generated]["parent"] = other
    tampered = certificate.model_copy(update={"parameters": parameters})
    with pytest.raises(ValueError, match="relationship is invalid"):
        _validate_projection_relationship(source, output, tampered)
