from __future__ import annotations

import json

import pytest

from schemaguard.compatibility.checkpoint_cache import (
    CacheError,
    CheckpointMetadata,
    checkpoint_cache_key,
    register_checkpoint,
    resolve_offline_checkpoint,
)


def identity(**overrides):
    values = dict(
        model_id="TPFN3-8.5",
        package_version="8.5.0",
        checkpoint_identifier="x.ckpt",
        checkpoint_sha256=None,
        device_policy="cpu",
        model_parameters={"n": 8},
        python_major_minor="3.12",
        pytorch_version="2.6.0+cu124",
        code_commit="abc",
    )
    values.update(overrides)
    return values


def test_cache_identity_changes_with_relevant_inputs() -> None:
    base = checkpoint_cache_key(**identity())
    assert checkpoint_cache_key(**identity(model_parameters={"n": 9})) != base
    assert checkpoint_cache_key(**identity(package_version="8.5.1")) != base
    assert checkpoint_cache_key(**identity(device_policy="cuda")) != base
    assert checkpoint_cache_key(**identity(fixture_hash="different")) != base
    assert checkpoint_cache_key(**identity(code_commit="def")) != base


def test_register_and_validate_offline_checkpoint(tmp_path) -> None:
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"checkpoint-content")
    metadata = register_checkpoint(
        checkpoint,
        cache_dir=tmp_path / "cache",
        model_id="TPFN3-8.5",
        package_version="8.5.0",
        checkpoint_identifier="x.ckpt",
        device_policy="cpu",
        model_parameters={"n": 8},
        python_major_minor="3.12",
        pytorch_version="2.6.0+cu124",
        code_commit="abc",
    )
    assert (
        resolve_offline_checkpoint(tmp_path / "cache", metadata.cache_key).checkpoint_sha256
        == metadata.checkpoint_sha256
    )


def test_truncated_manifest_and_invalid_hash_are_rejected(tmp_path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "bad.json").write_text("{", encoding="utf-8")
    with pytest.raises(CacheError):
        resolve_offline_checkpoint(cache, "bad")
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"data")
    metadata = CheckpointMetadata(
        cache_key="bad-hash",
        model_id="TPFN3-8.5",
        package_version="8.5.0",
        checkpoint_identifier="x",
        checkpoint_sha256="0" * 64,
        size_bytes=4,
        path=str(checkpoint),
        device_policy="cpu",
        model_parameters={},
        python_major_minor="3.12",
        pytorch_version=None,
        code_commit="abc",
        validated=True,
    )
    (cache / "bad-hash.json").write_text(json.dumps(metadata.model_dump()), encoding="utf-8")
    with pytest.raises(CacheError):
        resolve_offline_checkpoint(cache, "bad-hash")
