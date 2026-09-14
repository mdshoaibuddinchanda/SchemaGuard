from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from schemaguard.artifact_contracts import (
    SCHEMA_CONTRACTS,
    ConditionManifestContract,
    schema_documents,
)


def test_all_artifact_schemas_are_valid_and_deterministic() -> None:
    first = schema_documents()
    second = schema_documents()
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    for document in first.values():
        Draft202012Validator.check_schema(document)
        assert document["additionalProperties"] is False


def test_generated_schema_files_match_authoritative_contracts() -> None:
    root = Path(__file__).resolve().parents[2] / "schemas"
    for filename, document in schema_documents().items():
        observed = json.loads((root / filename).read_text(encoding="utf-8"))
        assert observed == document


def test_condition_manifest_cross_field_contract_and_unknown_key() -> None:
    valid = {
        "schema_version": 2,
        "stage": "condition_manifest",
        "scope": "pilot",
        "benchmark": "SchemaOrbit-14",
        "config_checksum": "a" * 64,
        "scheduled_count": 1,
        "records": [{"dataset_id": 31, "seed": 1729, "model_id": "LR-1.9", "view_id": "V00"}],
    }
    assert ConditionManifestContract.model_validate(valid).scheduled_count == 1
    with pytest.raises(ValidationError):
        ConditionManifestContract.model_validate({**valid, "unexpected": True})
    with pytest.raises(ValidationError):
        ConditionManifestContract.model_validate({**valid, "scheduled_count": 2})


def test_every_named_contract_rejects_unknown_top_level_keys() -> None:
    assert set(SCHEMA_CONTRACTS) == {
        "condition_manifest.schema.json",
        "model_compatibility.schema.json",
        "dataset_inventory.schema.json",
        "gpu_capacity_report.schema.json",
        "repository_validation.schema.json",
    }
    for contract in SCHEMA_CONTRACTS.values():
        with pytest.raises(ValidationError):
            contract.model_validate({"unexpected": True})
