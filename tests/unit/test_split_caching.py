from schemaguard.splits.caching import cache_identity, cache_key, logical_assignment_hash


def test_cache_identity_changes_for_each_material_input() -> None:
    identity = cache_identity(
        3,
        "1",
        "a" * 64,
        "b" * 64,
        "c" * 64,
        1729,
        "stratified_group_5fold_v1",
        "v1",
        "d" * 64,
        "e" * 64,
        "commit-a",
    )
    assert len(identity["split_implementation_hash"]) == 64
    assert cache_key(identity) == cache_key(dict(identity))
    for field, value in {
        "dataset_id": 23,
        "dataset_version": "2",
        "feature_artifact_hash": "f" * 64,
        "target_artifact_hash": "g" * 64,
        "dataset_manifest_hash": "h" * 64,
        "seed": 2718,
        "strategy": "different_strategy",
        "strategy_version": "v2",
        "grouping_implementation_hash": "i" * 64,
        "split_configuration_hash": "j" * 64,
        "split_implementation_hash": "k" * 64,
        "source_commit": "commit-b",
        "artifact_schema_version": 2,
    }.items():
        changed = dict(identity)
        changed[field] = value
        assert cache_key(changed) != cache_key(identity)


def test_logical_hash_is_independent_of_physical_row_order() -> None:
    rows = [
        {"__sg_row_id": "b", "predictor_group_id": "g2", "partition": "test"},
        {"__sg_row_id": "a", "predictor_group_id": "g1", "partition": "train"},
    ]
    assert logical_assignment_hash(rows) == logical_assignment_hash(list(reversed(rows)))
