from __future__ import annotations

import pytest
from pydantic import ValidationError

from schemaguard.compatibility.contracts import ProbeResult
from schemaguard.models.registry import RegistryError, load_runtime_config


def test_valid_runtime_configuration_parses() -> None:
    config = load_runtime_config("configs/runtime/model_compatibility.yaml")
    assert config.schema_version == 1
    assert len(config.models) == 5


def test_unknown_configuration_key_fails(tmp_path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("schema_version: 1\nunknown: true\n", encoding="utf-8")
    with pytest.raises(RegistryError):
        load_runtime_config(path)


def test_missing_required_configuration_key_fails(tmp_path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("schema_version: 1\n", encoding="utf-8")
    with pytest.raises(RegistryError):
        load_runtime_config(path)


def test_unknown_nested_configuration_key_fails(tmp_path) -> None:
    source = open("configs/runtime/model_compatibility.yaml", encoding="utf-8").read()
    path = tmp_path / "bad.yaml"
    path.write_text(
        source.replace("cpu_workers: 2", "cpu_workers: 2\n  unexpected: true"), encoding="utf-8"
    )
    with pytest.raises(RegistryError):
        load_runtime_config(path)


def test_negative_timeout_fails(tmp_path) -> None:
    source = open("configs/runtime/model_compatibility.yaml", encoding="utf-8").read()
    path = tmp_path / "bad.yaml"
    path.write_text(source.replace("import_seconds: 120", "import_seconds: -1"), encoding="utf-8")
    with pytest.raises(RegistryError):
        load_runtime_config(path)


def test_probe_contract_forbids_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        ProbeResult.model_validate({"model_id": "LR-1.9", "unexpected": True})
