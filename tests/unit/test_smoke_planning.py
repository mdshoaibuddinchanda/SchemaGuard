from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from schemaguard.experiments.contracts import MODEL_ORDER
from schemaguard.experiments.evidence import estimate_smoke_runtime
from schemaguard.experiments.planning import (
    SmokeConfig,
    _verified_checkpoint_digest,
    index_experiment_registry_rows,
    load_smoke_config,
    reconcile_registry_parameters,
)
from tests.integration.smoke_experiment_support import tiny_smoke_plan

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_registry_order_is_canonical_even_when_input_order_changes() -> None:
    rows = [{"id": model_id, "parameters": {}} for model_id in reversed(MODEL_ORDER)]
    indexed = index_experiment_registry_rows(rows)
    assert tuple(indexed) == MODEL_ORDER
    assert tuple(index_experiment_registry_rows(list(reversed(rows)))) == MODEL_ORDER


def test_registry_rejects_duplicate_or_unrecognized_models() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        index_experiment_registry_rows(
            [{"id": model_id} for model_id in MODEL_ORDER] + [{"id": MODEL_ORDER[0]}]
        )
    with pytest.raises(ValueError, match="exact frozen model set"):
        index_experiment_registry_rows(
            [{"id": model_id} for model_id in MODEL_ORDER[:-1]] + [{"id": "NEW"}]
        )


def test_only_the_two_reviewed_yaml_registry_translations_are_accepted() -> None:
    decisions = reconcile_registry_parameters(
        "CAT-1.2",
        {"bootstrap_type": False, "depth": 6},
        {"bootstrap_type": "No", "depth": 6},
        None,
    )
    assert decisions == ["catboost_yaml_no_scalar"]

    checkpoint = "tabicl-classifier-v2-20260212.ckpt"
    decisions = reconcile_registry_parameters(
        "TICL2-2.2",
        {"n_estimators": 8},
        {"n_estimators": 8, "checkpoint_version": checkpoint},
        checkpoint,
    )
    assert decisions == ["tabicl_checkpoint_version_from_checkpoint_field"]

    with pytest.raises(ValueError, match="parameter mismatch"):
        reconcile_registry_parameters("CAT-1.2", {"depth": 5}, {"depth": 6}, None)


def test_smoke_configuration_rejects_unknown_missing_and_unsafe_values(tmp_path: Path) -> None:
    source = REPOSITORY_ROOT / "configs/runtime/smoke_experiment.yaml"
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    assert isinstance(SmokeConfig.model_validate(payload), SmokeConfig)

    unknown = {**payload, "unexpected_key": 1}
    unknown_path = tmp_path / "unknown.yaml"
    unknown_path.write_text(yaml.safe_dump(unknown), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid smoke experiment config"):
        load_smoke_config(unknown_path)

    missing = dict(payload)
    missing.pop("protocol_version")
    with pytest.raises(ValidationError):
        SmokeConfig.model_validate(missing)

    unsafe = {**payload, "output_paths": {**payload["output_paths"], "runs": "C:/private"}}
    with pytest.raises(ValidationError):
        SmokeConfig.model_validate(unsafe)


def test_checkpoint_digest_reuses_only_unchanged_file_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import schemaguard.experiments.planning as planning_module

    path = tmp_path / "checkpoint.bin"
    path.write_bytes(b"one")
    expected = hashlib.sha256(b"one").hexdigest()
    original = planning_module.sha256_file
    calls = 0

    def counted_sha256(source: str | Path) -> str:
        nonlocal calls
        calls += 1
        return original(source)

    monkeypatch.setattr(planning_module, "sha256_file", counted_sha256)
    observed, identity = _verified_checkpoint_digest(path, expected)
    assert observed == expected
    assert _verified_checkpoint_digest(path, expected, identity)[0] == expected
    assert calls == 1

    old_stat = path.stat()
    path.write_bytes(b"two")
    os.utime(path, ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns + 1_000_000))
    with pytest.raises(ValueError, match="frozen identity"):
        _verified_checkpoint_digest(path, expected, identity)
    assert calls == 2


def test_runtime_estimate_uses_prior_exact_model_device_probe_records(tmp_path: Path) -> None:
    source_inventory = REPOSITORY_ROOT / "artifacts/handoff/model_adapter_inventory.json"
    destination_inventory = tmp_path / "artifacts/handoff/model_adapter_inventory.json"
    destination_inventory.parent.mkdir(parents=True)
    shutil.copyfile(source_inventory, destination_inventory)
    plan = tiny_smoke_plan(tmp_path)
    estimate = estimate_smoke_runtime(tmp_path, plan)
    assert len(estimate.condition_seconds) == 10
    assert len(estimate.cpu_condition_ids) == 6
    assert len(estimate.gpu_condition_ids) == 4
    assert estimate.estimated_total_seconds > estimate.expected_cpu_seconds
    assert estimate.estimated_total_seconds > estimate.expected_gpu_seconds
