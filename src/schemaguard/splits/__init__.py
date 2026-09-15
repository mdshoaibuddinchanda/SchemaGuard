"""Deterministic, predictor-group-aware split generation for SchemaOrbit-14."""

from .contracts import SplitGenerationConfig, SplitManifestContract
from .generation import generate_split
from .grouping import predictor_group_ids
from .implementation import SPLIT_IMPLEMENTATION_HASH
from .validation import validate_all, validate_split

__all__ = [
    "SplitGenerationConfig",
    "SplitManifestContract",
    "generate_split",
    "predictor_group_ids",
    "SPLIT_IMPLEMENTATION_HASH",
    "validate_all",
    "validate_split",
]
