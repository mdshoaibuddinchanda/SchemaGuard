from __future__ import annotations

from schemaguard.transformations.caching import transformation_cache_key


def test_cache_identity_changes_for_every_relevant_input() -> None:
    base = dict(
        dataset_id=1,
        dataset_version="v1",
        feature_hash="a" * 64,
        target_hash="b" * 64,
        split_logical_hash="c" * 64,
        partition="train",
        view_id="V01",
        view_config={},
        implementation_hash="d" * 64,
        fit_parameter_hash="e" * 64,
        certificate_schema_version=1,
    )
    key = transformation_cache_key(**base)
    for field, value in (
        ("view_id", "V02"),
        ("partition", "test"),
        ("fit_parameter_hash", "f" * 64),
        ("implementation_hash", "g" * 64),
    ):
        changed = dict(base, **{field: value})
        assert transformation_cache_key(**changed) != key
