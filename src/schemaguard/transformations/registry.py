"""Frozen transformation registry and strict configuration loading."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .categorical_onehot import CategoricalOneHotTransformation
from .category_permutation import CategoryPermutationTransformation
from .column_permutation import ColumnPermutationTransformation
from .composite_migration import CompositeMigrationTransformation
from .contracts import TransformationConfig
from .duplicate_feature import DuplicateFeatureTransformation
from .identity import IdentityTransformation
from .numeric_affine import NumericAffineTransformation
from .numeric_asinh import NumericAsinhTransformation
from .quotient_remainder import QuotientRemainderTransformation
from .redundant_affine import RedundantAffineTransformation
from .row_permutation import RowPermutationTransformation


@dataclass(frozen=True)
class ViewSpec:
    view_id: str
    name: str
    certificate_type: str
    scientific_role: str
    factory: Callable[[dict[str, Any]], Any]


VIEW_REGISTRY: tuple[ViewSpec, ...] = (
    ViewSpec("V00", "identity", "BIJECTION", "CONTROL", IdentityTransformation),
    ViewSpec(
        "V01", "numeric_affine_units", "BIJECTION", "PRIMARY_MIGRATION", NumericAffineTransformation
    ),
    ViewSpec("V02", "numeric_asinh", "BIJECTION", "PRIMARY_MIGRATION", NumericAsinhTransformation),
    ViewSpec(
        "V03",
        "category_permutation",
        "BIJECTION",
        "PRIMARY_MIGRATION",
        CategoryPermutationTransformation,
    ),
    ViewSpec(
        "V04",
        "categorical_onehot",
        "BIJECTION",
        "PRIMARY_MIGRATION",
        CategoricalOneHotTransformation,
    ),
    ViewSpec(
        "V05",
        "duplicate_feature",
        "PROJECTION",
        "PRIMARY_MIGRATION",
        DuplicateFeatureTransformation,
    ),
    ViewSpec(
        "V06",
        "redundant_affine_feature",
        "PROJECTION",
        "PRIMARY_MIGRATION",
        RedundantAffineTransformation,
    ),
    ViewSpec(
        "V07",
        "integer_quotient_remainder",
        "BIJECTION",
        "PRIMARY_MIGRATION",
        QuotientRemainderTransformation,
    ),
    ViewSpec(
        "V08",
        "column_permutation_control",
        "PERMUTATION",
        "CONTROL",
        ColumnPermutationTransformation,
    ),
    ViewSpec(
        "V09", "row_permutation_control", "PERMUTATION", "CONTROL", RowPermutationTransformation
    ),
    ViewSpec(
        "V10",
        "composite_migration",
        "COMPOSITION",
        "PRIMARY_MIGRATION",
        CompositeMigrationTransformation,
    ),
)


def registered_views() -> tuple[ViewSpec, ...]:
    return VIEW_REGISTRY


def get_view(view_id: str) -> ViewSpec:
    for view in VIEW_REGISTRY:
        if view.view_id == view_id:
            return view
    raise KeyError(f"unknown transformation view: {view_id}")


def get_transformation(view_id: str, config: dict[str, Any] | None = None):
    view = get_view(view_id)
    transformation = view.factory(config or {})
    if transformation.view_id != view.view_id or transformation.view_name != view.name:
        raise ValueError("registry factory identity mismatch")
    return transformation


def registry_payload() -> list[dict[str, str]]:
    return [
        {
            "id": view.view_id,
            "name": view.name,
            "certificate_type": view.certificate_type,
            "scientific_role": view.scientific_role,
        }
        for view in VIEW_REGISTRY
    ]


def registry_hash() -> str:
    from ..utils.hashing import sha256_canonical_json

    return sha256_canonical_json(registry_payload())


def load_transformation_config(path: str | Path) -> TransformationConfig:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError("transformation configuration must be a mapping")
    allowed = {
        "schema_version",
        "engine_name",
        "split_strategy",
        "max_numeric_columns",
        "category_minimum",
        "quotient_modulus",
        "numerical_rtol",
        "numerical_atol",
        "cpu_workers",
        "gpu_enabled",
        "schema_registry_hash",
        "views",
    }
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"unknown transformation configuration key(s): {unknown}")
    config = TransformationConfig.model_validate(raw)
    actual = {item["id"] for item in registry_payload()}
    observed = {str(item.get("id")) for item in config.views}
    if actual != observed:
        raise ValueError("configuration views must match the frozen transformation registry")
    for item in config.views:
        if set(item) != {"id", "name", "certificate_type", "scientific_role"}:
            raise ValueError("each transformation view configuration has exactly four keys")
    if config.schema_registry_hash != registry_hash():
        raise ValueError("schema_registry_hash does not match the frozen view registry")
    return config


__all__ = [
    "VIEW_REGISTRY",
    "ViewSpec",
    "get_transformation",
    "load_transformation_config",
    "registry_hash",
    "registered_views",
]
