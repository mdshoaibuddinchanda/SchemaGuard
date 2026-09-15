from __future__ import annotations

import pytest

from schemaguard.transformations.categorical_onehot import CategoricalOneHotTransformation
from schemaguard.transformations.category_permutation import CategoryPermutationTransformation
from schemaguard.transformations.certificates import validate_roundtrip

from .transformation_helpers import sample_frame, sample_schema, target_hash, transformation_config


@pytest.mark.parametrize(
    "factory", [CategoryPermutationTransformation, CategoricalOneHotTransformation]
)
def test_categorical_views_roundtrip_and_preserve_missingness(factory) -> None:
    frame = sample_frame()
    transformation = factory(transformation_config()).fit(
        frame.iloc[:8], 1464, 1729, sample_schema(frame.iloc[:8])
    )
    output = transformation.transform(frame, "test")
    certificate = transformation.certificate_for(
        frame,
        output,
        "test",
        source_target_hash=target_hash(frame),
        output_target_hash=target_hash(frame),
    )
    restored = transformation.reconstruct(output, certificate)
    assert validate_roundtrip(frame, output, restored, certificate)["status"] == "PASS"


def test_categorical_views_reject_unseen_categories() -> None:
    frame = sample_frame()
    transformation = CategoryPermutationTransformation(transformation_config()).fit(
        frame.iloc[:8], 1464, 1729, sample_schema(frame.iloc[:8])
    )
    frame.loc[0, "category"] = "unseen"
    with pytest.raises(ValueError, match="unseen category"):
        transformation.transform(frame, "test")
