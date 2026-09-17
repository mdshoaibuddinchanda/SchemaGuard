from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from schemaguard.cache.contracts import CacheIdentity, CacheSchedulerConfig
from schemaguard.cache.keys import dependency_lock_hash, implementation_hash
from tests.unit.cache_scheduler_helpers import cache_identity


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dataset_sha256", "a" * 64),
        ("split_sha256", "b" * 64),
        ("view_certificate_sha256", "c" * 64),
        ("model_spec_sha256", "d" * 64),
        ("model_parameters_sha256", "e" * 64),
        ("checkpoint_sha256", "f" * 64),
        ("dependency_lock_sha256", "8" * 64),
        ("source_implementation_sha256", "9" * 64),
        ("seed", 1730),
        ("device_policy", "cuda"),
        ("artifact_kind", "prediction"),
    ],
)
def test_each_causal_identity_change_changes_cache_key(field: str, value: object) -> None:
    original = cache_identity()
    changed = CacheIdentity.model_validate({**original.model_dump(mode="json"), field: value})
    assert original.cache_key != changed.cache_key


def test_canonical_identity_is_stable_across_field_insertion_order() -> None:
    original = cache_identity()
    reversed_values = dict(reversed(list(original.model_dump(mode="json").items())))
    assert CacheIdentity.model_validate(reversed_values).cache_key == original.cache_key


def test_identity_rejects_zero_hashes_unknown_fields_and_missing_hashes() -> None:
    base = cache_identity().model_dump(mode="json")
    with pytest.raises(ValidationError):
        CacheIdentity.model_validate({**base, "dataset_sha256": "0" * 64})
    with pytest.raises(ValidationError):
        CacheIdentity.model_validate({**base, "unexpected": "value"})
    missing = dict(base)
    del missing["split_sha256"]
    with pytest.raises(ValidationError):
        CacheIdentity.model_validate(missing)


def test_configuration_rejects_unknown_missing_and_unsafe_paths() -> None:
    base = {
        "schema_version": 1,
        "cpu_workers": 2,
        "cpu_threads_per_worker": 2,
        "gpu_workers": 1,
        "ram_soft_limit_mib": 24576,
        "ram_hard_limit_mib": 28672,
        "vram_soft_limit_mib": 3600,
        "gpu_headroom_mib": 512,
        "lock_timeout_seconds": 120,
        "heartbeat_interval_seconds": 5,
        "abandoned_lease_seconds": 60,
        "cache_root": "data/cache/conditions",
        "state_root": "artifacts/cache_scheduler/runtime",
    }
    assert CacheSchedulerConfig.model_validate(base).cpu_workers == 2
    with pytest.raises(ValidationError):
        CacheSchedulerConfig.model_validate({**base, "unexpected": True})
    incomplete = dict(base)
    del incomplete["cache_root"]
    with pytest.raises(ValidationError):
        CacheSchedulerConfig.model_validate(incomplete)
    with pytest.raises(ValidationError):
        CacheSchedulerConfig.model_validate({**base, "cache_root": "D:/private/cache"})
    with pytest.raises(ValidationError):
        CacheSchedulerConfig.model_validate({**base, "state_root": "../../outside"})


def test_implementation_hash_is_order_independent_and_rejects_absolute_paths(
    tmp_path: Path,
) -> None:
    (tmp_path / "a.py").write_text("value = 1\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("value = 2\n", encoding="utf-8")
    assert implementation_hash(tmp_path, ["b.py", "a.py"]) == implementation_hash(
        tmp_path, ["a.py", "b.py"]
    )
    with pytest.raises(ValueError):
        implementation_hash(tmp_path, [str((tmp_path / "a.py").resolve())])
    with pytest.raises(ValueError):
        implementation_hash(tmp_path, ["../outside.py"])


def test_dependency_lock_hash_is_stable_across_checkout_line_endings(tmp_path: Path) -> None:
    lf_path = tmp_path / "uv-lf.lock"
    crlf_path = tmp_path / "uv-crlf.lock"
    lf_path.write_bytes(b"version = 1\npackage = 'example'\n")
    crlf_path.write_bytes(b"version = 1\r\npackage = 'example'\r\n")
    assert dependency_lock_hash(lf_path) == dependency_lock_hash(crlf_path)
