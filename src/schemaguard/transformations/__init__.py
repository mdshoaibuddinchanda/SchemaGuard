"""Certified, lossless tabular schema transformations."""

from .base import FeatureSchema, Transformation, TransformOutput, feature_schema_from_frame
from .contracts import (
    ApplicabilityRecord,
    TransformationCertificate,
    TransformationConfig,
    TransformationInventory,
    TransformationInventoryRecord,
    TransformationManifest,
    TransformationPropertyEvidence,
    TransformationValidationRecord,
    TransformationValidationReport,
)
from .registry import VIEW_REGISTRY, get_transformation, registered_views

__all__ = [
    "ApplicabilityRecord",
    "FeatureSchema",
    "TransformOutput",
    "Transformation",
    "TransformationCertificate",
    "TransformationConfig",
    "TransformationInventory",
    "TransformationInventoryRecord",
    "TransformationManifest",
    "TransformationPropertyEvidence",
    "TransformationValidationRecord",
    "TransformationValidationReport",
    "VIEW_REGISTRY",
    "feature_schema_from_frame",
    "get_transformation",
    "registered_views",
]
