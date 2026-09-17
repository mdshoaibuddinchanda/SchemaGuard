from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from schemaguard.models.adapters import cache as cache_module
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


def test_identical_model_payload_is_a_validated_cache_hit(tmp_path) -> None:
    cache = AdapterArtifactCache(tmp_path)
    identity = AdapterCacheIdentity.model_validate(_identity())
    first = cache.store_bytes("model", identity, b"same-serialized-model")
    second = cache.store_bytes("model", identity, b"same-serialized-model")
    assert first == second
    assert cache.load_bytes("model", identity) == b"same-serialized-model"


def test_concurrent_conflicting_model_payloads_reject_one_writer(tmp_path) -> None:
    cache = AdapterArtifactCache(tmp_path)
    identity = AdapterCacheIdentity.model_validate(_identity())
    payloads = [b"serialized-model-A", b"serialized-model-B"]
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(
            executor.map(lambda payload: _store_result(cache, identity, payload), payloads)
        )
    assert sum(result is None for result in results) == 1
    assert sum(isinstance(result, CacheIntegrityError) for result in results) == 1
    assert cache.load_bytes("model", identity) in payloads


def _store_result(cache, identity, payload):
    try:
        cache.store_bytes("model", identity, payload)
    except CacheIntegrityError as exc:
        return exc
    return None


@pytest.mark.parametrize(
    "corruption",
    ["truncated_json", "corrupt_base64", "changed_hash", "changed_size", "unknown_field"],
)
def test_corrupt_model_envelopes_fail_closed(tmp_path, corruption: str) -> None:
    cache = AdapterArtifactCache(tmp_path)
    identity = AdapterCacheIdentity.model_validate(_identity())
    cache.store_bytes("model", identity, b"serialized-model")
    path = cache.path_for("model", identity)
    if corruption == "truncated_json":
        path.write_text('{"schema_version":', encoding="utf-8")
    else:
        envelope = json.loads(path.read_text(encoding="utf-8"))
        if corruption == "corrupt_base64":
            envelope["payload_base64"] = "!!!"
        elif corruption == "changed_hash":
            envelope["payload_sha256"] = "0" * 64
        elif corruption == "changed_size":
            envelope["payload_size_bytes"] += 1
        elif corruption == "unknown_field":
            envelope["unrecognized"] = True
        path.write_text(json.dumps(envelope), encoding="utf-8")
    with pytest.raises(CacheIntegrityError):
        cache.load_bytes("model", identity)
    with pytest.raises(CacheIntegrityError):
        cache.store_bytes("model", identity, b"replacement")
    assert list(tmp_path.glob("*.quarantine*")) or not path.exists()


def test_different_model_payload_cannot_reuse_the_same_identity(tmp_path) -> None:
    cache = AdapterArtifactCache(tmp_path)
    identity = AdapterCacheIdentity.model_validate(_identity())
    cache.store_bytes("model", identity, b"first-state")
    with pytest.raises(CacheIntegrityError, match="different payload bytes"):
        cache.store_bytes("model", identity, b"equivalent-second-state")
    assert cache.load_bytes("model", identity) == b"first-state"
    assert len(list(tmp_path.glob("*.model.json"))) == 1


def test_unexpected_temporary_file_is_never_trusted_as_a_cache_hit(tmp_path) -> None:
    cache = AdapterArtifactCache(tmp_path)
    identity = AdapterCacheIdentity.model_validate(_identity())
    stray = cache.path_for("model", identity).with_suffix(".json.tmp")
    stray.write_text("partial cache envelope", encoding="utf-8")
    with pytest.raises(CacheIntegrityError):
        cache.load_bytes("model", identity)
    cache.store_bytes("model", identity, b"complete-payload")
    assert cache.load_bytes("model", identity) == b"complete-payload"


def test_interrupted_atomic_write_does_not_create_a_trusted_cache_entry(
    tmp_path, monkeypatch
) -> None:
    cache = AdapterArtifactCache(tmp_path)
    identity = AdapterCacheIdentity.model_validate(_identity())

    def interrupt(destination, payload):
        destination.with_suffix(destination.suffix + ".tmp").write_text(
            "partial", encoding="utf-8"
        )
        raise OSError("simulated interrupted atomic write")

    monkeypatch.setattr(cache_module, "atomic_write_json", interrupt)
    with pytest.raises(OSError, match="interrupted"):
        cache.store_bytes("model", identity, b"payload")
    assert not cache.path_for("model", identity).exists()
    monkeypatch.undo()
    cache.store_bytes("model", identity, b"payload")
    assert cache.load_bytes("model", identity) == b"payload"
