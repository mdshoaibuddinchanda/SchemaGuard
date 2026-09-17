from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from schemaguard.models.adapters.cache import AdapterArtifactCache, CacheIntegrityError
from schemaguard.models.adapters.contracts import AdapterCacheIdentity


def _identity(**changes: object) -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": 1,
        "source_data_sha256": "a" * 64,
        "fixture_sha256": "a" * 64,
        "row_ids_sha256": "b" * 64,
        "model_spec_sha256": "c" * 64,
        "parameter_sha256": "d" * 64,
        "adapter_sha256": "e" * 64,
        "preprocessing_sha256": "f" * 64,
        "checkpoint_sha256": None,
        "source_commit": "1" * 40,
        "package_runtime": "scikit-learn==1.9.1;python=3.12",
        "seed": 1729,
        "device_policy": "cpu",
        "partition": "test",
        "split_identity": "split-1",
        "transformation_identity": None,
    }
    value.update(changes)
    return value


def test_prediction_cache_roundtrip_and_tamper_detection(tmp_path) -> None:
    cache = AdapterArtifactCache(tmp_path)
    identity = AdapterCacheIdentity.model_validate(_identity())
    value = {"row_ids": ["a", "b"], "probabilities": [[0.8, 0.2], [0.1, 0.9]]}
    cache.store_json("prediction", identity, value)
    assert cache.load_json("prediction", identity) == value
    path = cache.path_for("prediction", identity)
    path.write_text("{truncated", encoding="utf-8")
    with pytest.raises(CacheIntegrityError):
        cache.load_json("prediction", identity)


def test_changed_identity_never_reuses_previous_entry(tmp_path) -> None:
    cache = AdapterArtifactCache(tmp_path)
    identity = AdapterCacheIdentity.model_validate(_identity())
    cache.store_json("prediction", identity, {"row_ids": ["a"]})
    changed = AdapterCacheIdentity.model_validate(_identity(fixture_sha256="9" * 64))
    with pytest.raises(CacheIntegrityError):
        cache.load_json("prediction", changed)


def test_concurrent_identical_writes_produce_one_validated_artifact(tmp_path) -> None:
    cache = AdapterArtifactCache(tmp_path)
    identity = AdapterCacheIdentity.model_validate(_identity())
    value = {"row_ids": ["a"], "probabilities": [[1.0]]}
    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(lambda _: cache.store_json("prediction", identity, value), range(2)))
    assert cache.load_json("prediction", identity) == value
    assert len(list(tmp_path.glob("*.prediction.json"))) == 1


def test_model_bundle_roundtrip_rejects_payload_tampering(tmp_path) -> None:
    cache = AdapterArtifactCache(tmp_path)
    identity = AdapterCacheIdentity.model_validate(_identity())
    cache.store_bytes("model", identity, b"serialized-model")
    assert cache.load_bytes("model", identity) == b"serialized-model"
    path = cache.path_for("model", identity)
    path.write_bytes(path.read_bytes() + b"x")
    with pytest.raises(CacheIntegrityError):
        cache.load_bytes("model", identity)


def test_equivalent_model_identity_reuses_one_validated_serialization(tmp_path) -> None:
    cache = AdapterArtifactCache(tmp_path)
    identity = AdapterCacheIdentity.model_validate(_identity())
    first = cache.store_bytes("model", identity, b"first-state", reuse_validated_existing=True)
    second = cache.store_bytes(
        "model", identity, b"equivalent-second-state", reuse_validated_existing=True
    )
    assert first == second
    assert cache.load_bytes("model", identity) == b"first-state"
    assert len(list(tmp_path.glob("*.model.json"))) == 1
