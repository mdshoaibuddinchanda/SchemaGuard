"""Composition of the frozen affine, categorical, and column controls."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .base import BaseTransformation, FeatureSchema, feature_schema_from_frame
from .category_permutation import CategoryPermutationTransformation
from .column_permutation import ColumnPermutationTransformation
from .contracts import TransformationCertificate
from .numeric_affine import NumericAffineTransformation


class CompositeMigrationTransformation(BaseTransformation):
    view_id = "V10"
    view_name = "composite_migration"
    certificate_type = "COMPOSITION"

    def _fit(
        self, X_train: pd.DataFrame, dataset_id: int | str, seed: int, feature_schema: FeatureSchema
    ) -> dict[str, Any]:
        self.affine = NumericAffineTransformation(self.config)
        self.category = CategoryPermutationTransformation(self.config)
        self.columns = ColumnPermutationTransformation(self.config)
        self.affine.fit(X_train, dataset_id, seed, feature_schema)
        affine_train = self.affine.transform(X_train, "train")
        self.category.fit(affine_train, dataset_id, seed, feature_schema)
        category_train = self.category.transform(affine_train, "train")
        self.columns.fit(
            category_train, dataset_id, seed, feature_schema_from_frame(category_train)
        )
        return {
            "selected_columns": [],
            "generated_columns": [],
            "components": ["V01", "V03", "V08"],
            "operation": "V01_then_V03_then_V08",
            "inverse_operation": "reverse_order_reconstruct",
        }

    def component_implementation_hashes(self) -> list[str]:
        if not hasattr(self, "affine"):
            return []
        return [
            self.affine.implementation_hash(),
            self.category.implementation_hash(),
            self.columns.implementation_hash(),
        ]

    def _transform(self, X: pd.DataFrame, partition: str) -> tuple[pd.DataFrame, dict[str, Any]]:
        first = self.affine.transform(X, partition)
        second = self.category.transform(first, partition)
        third = self.columns.transform(second, partition)
        self._intermediates[partition] = (X.copy(deep=True), first, second, third)
        return third, dict(self._fit_parameters)

    def _reconstruct(
        self, X_transformed: pd.DataFrame, certificate: TransformationCertificate
    ) -> pd.DataFrame:
        components = [
            TransformationCertificate.model_validate(item)
            for item in certificate.parameters["component_certificates"]
        ]
        step = self.columns.reconstruct(X_transformed, components[2])
        step = self.category.reconstruct(step, components[1])
        return self.affine.reconstruct(step, components[0])

    def fit(
        self, X_train: pd.DataFrame, dataset_id: int | str, seed: int, feature_schema: FeatureSchema
    ):
        self._intermediates: dict[str, tuple[pd.DataFrame, ...]] = {}
        return super().fit(X_train, dataset_id, seed, feature_schema)

    def certificate_for(
        self, source: pd.DataFrame, transformed: pd.DataFrame, partition: str, **kwargs: Any
    ) -> TransformationCertificate:
        source_frame, first, second, third = self._intermediates[partition]
        certificates = [
            self.affine.certificate_for(source_frame, first, partition, **kwargs),
            self.category.certificate_for(first, second, partition, **kwargs),
            self.columns.certificate_for(second, third, partition, **kwargs),
        ]
        params = {
            "components": ["V01", "V03", "V08"],
            "component_certificates": [item.canonical_dict() for item in certificates],
            "selected_columns": [],
            "generated_columns": [],
        }
        old = self._partition_parameters.get(partition, {})
        self._partition_parameters[partition] = params
        try:
            return super().certificate_for(source, transformed, partition, **kwargs)
        finally:
            self._partition_parameters[partition] = old
