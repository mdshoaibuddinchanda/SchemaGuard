from __future__ import annotations

from schemaguard.models.adapters.base import canonical_parameter_payload
from schemaguard.utils.hashing import sha256_canonical_json


def _payload(
    *,
    model_path: str,
    changes: dict[str, object] | None = None,
    checkpoint_sha256: str = "a" * 64,
    device: str = "cpu",
) -> dict[str, object]:
    parameters: dict[str, object] = {
        "device": device,
        "model_path": model_path,
        "kv_cache": False,
        "allow_auto_download": False,
        "n_estimators": 8,
    }
    if changes:
        parameters.update(changes)
    return canonical_parameter_payload(
        parameters,
        model_id="TICL2-2.2",
        package_name="tabicl",
        package_version="2.2.0",
        seed=1729,
        device="cpu",
        checkpoint_identifier="tabicl-classifier-v2-20260212.ckpt",
        checkpoint_sha256=checkpoint_sha256,
        python_major_minor="3.12",
        pytorch_version="2.6.0+cu124",
    )


def test_parameter_identity_excludes_machine_specific_checkpoint_path() -> None:
    first = _payload(model_path=r"D:\DR2\SchemaGuard\data\tabicl.ckpt")
    second = _payload(model_path=r"E:\cache\tabicl.ckpt")
    assert first == second
    assert sha256_canonical_json(first) == sha256_canonical_json(second)


def test_parameter_identity_binds_frozen_and_runtime_parameters() -> None:
    baseline = _payload(model_path="one", changes={"n_estimators": 8})
    changed = _payload(model_path="two", changes={"n_estimators": 9})
    changed_download = _payload(model_path="two", changes={"allow_auto_download": True})
    changed_cache = _payload(model_path="two", changes={"kv_cache": True})
    changed_checkpoint = _payload(model_path="two", checkpoint_sha256="b" * 64)
    changed_device = _payload(model_path="two", device="cuda")
    assert sha256_canonical_json(baseline) != sha256_canonical_json(changed)
    assert sha256_canonical_json(baseline) != sha256_canonical_json(changed_download)
    assert sha256_canonical_json(baseline) != sha256_canonical_json(changed_cache)
    assert sha256_canonical_json(baseline) != sha256_canonical_json(changed_checkpoint)
    assert sha256_canonical_json(baseline) != sha256_canonical_json(changed_device)
