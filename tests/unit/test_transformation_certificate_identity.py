from __future__ import annotations

from copy import deepcopy

import pytest

from schemaguard.transformations.certificates import (
    certificate_hash,
    certificate_identity_hash,
)
from schemaguard.transformations.identity import IdentityTransformation

from .transformation_helpers import sample_frame, sample_schema, target_hash, transformation_config


def _certificate():
    source = sample_frame()
    transformation = IdentityTransformation(transformation_config()).fit(
        source, 1464, 1729, sample_schema(source)
    )
    output = transformation.transform(source, "test")
    return transformation.certificate_for(
        source,
        output,
        "test",
        dataset_version="identity-test",
        source_target_hash=target_hash(source),
        output_target_hash=target_hash(source),
    )


def test_logical_identity_excludes_generation_provenance_only() -> None:
    certificate = _certificate()
    changed_time = certificate.model_copy(update={"created_at": "2027-01-01T00:00:00Z"})
    changed_commit = certificate.model_copy(update={"source_commit": "different-commit"})

    assert certificate_identity_hash(changed_time) == certificate_identity_hash(certificate)
    assert certificate_identity_hash(changed_commit) == certificate_identity_hash(certificate)
    assert certificate_hash(certificate) == certificate_identity_hash(certificate)


def test_nested_certificate_provenance_is_excluded_but_nested_science_is_bound() -> None:
    certificate = _certificate()
    first = certificate.model_copy(
        update={
            "parameters": {
                "component_certificate": {
                    "source_commit": "first-commit",
                    "created_at": "2026-01-01T00:00:00Z",
                    "implementation_hash": "a" * 64,
                }
            }
        }
    )
    provenance_changed = first.model_copy(
        update={
            "parameters": {
                "component_certificate": {
                    "source_commit": "second-commit",
                    "created_at": "2027-01-01T00:00:00Z",
                    "implementation_hash": "a" * 64,
                }
            }
        }
    )
    scientific_changed = first.model_copy(
        update={
            "parameters": {
                "component_certificate": {
                    "source_commit": "first-commit",
                    "created_at": "2026-01-01T00:00:00Z",
                    "implementation_hash": "b" * 64,
                }
            }
        }
    )

    assert certificate_identity_hash(first) == certificate_identity_hash(provenance_changed)
    assert certificate_identity_hash(first) != certificate_identity_hash(scientific_changed)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("configuration_hash", "b" * 64),
        ("implementation_hash", "c" * 64),
        ("schema_version", 3),
        ("view_name", "changed-view-name"),
        ("scientific_role", "PRIMARY_MIGRATION"),
        ("dataset_version", "different-data-version"),
        ("split_strategy", "different-split-strategy"),
        ("fit_scope", "train_only"),
        ("source_schema_hash", "1" * 64),
        ("output_schema_hash", "2" * 64),
        ("source_artifact_hash", "d" * 64),
        ("output_artifact_hash", "e" * 64),
        ("source_row_id_hash", "3" * 64),
        ("output_row_id_hash", "4" * 64),
        ("source_target_hash", "f" * 64),
        ("output_target_hash", "0" * 64),
        ("selected_columns", ["different-column"]),
        ("generated_columns", ["different-generated-column"]),
        ("parameters", {"identity-relevant": True}),
        ("inverse_parameters", {"restore-relevant": True}),
        ("missing_mask_policy", "changed-policy"),
        ("dtype_policy", "changed-dtype-policy"),
        ("numerical_tolerance", {"rtol": 1.0, "atol": 1.0}),
        ("validation_status", "FAIL"),
        ("validation_results", {"reconstruction_exact": False}),
        ("not_applicable_reason", "changed-reason"),
        ("dataset_id", "different-dataset"),
        ("seed", 2718),
        ("partition", "train"),
        ("view_id", "V08"),
        ("certificate_type", "PERMUTATION"),
    ],
)
def test_every_scientific_or_provenance_relevant_field_changes_identity(
    field: str, value: object
) -> None:
    certificate = _certificate()
    changed = certificate.model_copy(update={field: deepcopy(value)})

    assert certificate_identity_hash(changed) != certificate_identity_hash(certificate)
