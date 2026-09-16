from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from schemaguard.transformations.contracts import (
    TransformationInventory,
    _current_property_runner_hashes,
)
from schemaguard.utils.hashing import source_file_hashes


def _payload() -> dict:
    return json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "artifacts/handoff/transformation_inventory.json"
        ).read_text(encoding="utf-8")
    )


def test_current_inventory_has_exact_frozen_matrix() -> None:
    inventory = TransformationInventory.model_validate(_payload())
    assert len(inventory.records) == 770
    assert inventory.pass_count == 545
    assert inventory.not_applicable_count == 225


def test_property_runner_hash_validation_is_line_ending_stable() -> None:
    hashes = _current_property_runner_hashes()
    assert len(hashes) == 2
    assert all(len(value) == 64 for value in hashes)


def test_source_hashes_accept_line_endings_but_reject_semantic_edits(tmp_path: Path) -> None:
    source = tmp_path / "runner.py"
    source.write_bytes(b"first\nsecond\n")
    lf_hashes = source_file_hashes(source)
    source.write_bytes(b"first\r\nsecond\r\n")
    assert source_file_hashes(source) == lf_hashes
    source.write_bytes(b"first\r\nchanged\r\n")
    assert source_file_hashes(source) != lf_hashes


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload["records"].__setitem__(1, copy.deepcopy(payload["records"][0])),
        lambda payload: payload["records"].__setitem__(
            0, {**payload["records"][0], "dataset_id": 999}
        ),
        lambda payload: payload["records"].__setitem__(0, {**payload["records"][0], "seed": 999}),
        lambda payload: payload["records"].__setitem__(
            0, {**payload["records"][0], "view_id": "V99"}
        ),
        lambda payload: payload.__setitem__("pass_count", payload["pass_count"] + 1),
        lambda payload: payload["protected_hash_comparison"].__setitem__("status", "FAIL"),
        lambda payload: payload["property_evidence"].__setitem__(
            0, {**payload["property_evidence"][0], "test_implementation_hash": "0" * 64}
        ),
        lambda payload: payload["property_evidence"].append(
            copy.deepcopy(payload["property_evidence"][0])
        ),
    ],
)
def test_inventory_tampering_is_rejected(mutation) -> None:
    payload = _payload()
    mutation(payload)
    with pytest.raises(ValidationError):
        TransformationInventory.model_validate(payload)
