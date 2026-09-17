from __future__ import annotations

import hashlib
import json
import struct

import numpy as np
import pytest

from schemaguard.models.adapters.contracts import AdapterFailure, FailureCategory
from schemaguard.models.adapters.factory import create_adapter
from tests.unit.adapter_test_support import FIXTURES, fit_predict


@pytest.mark.integration
@pytest.mark.parametrize("model_id", ["LR-1.9", "CAT-1.2", "XGB-3.4"])
def test_classical_adapter_save_reload_and_prediction_identity(model_id: str) -> None:
    fixture = FIXTURES["binary_numerical"]
    adapter, before = fit_predict(model_id, fixture)
    restored = None
    try:
        cache_identity = adapter.save_fitted()
        restored = create_adapter(model_id, device="cpu")
        restored.load_fitted(cache_identity)
        after = restored.predict_proba(
            fixture.test,
            partition="test",
            fixture_id=fixture.name,
            fixture_sha256=fixture.sha256,
            split_identity=f"fixture:{fixture.name}:seed=1729",
        )
        assert before.row_ids == after.row_ids
        assert before.class_order == after.class_order
        assert before.logical_identity_sha256 == after.logical_identity_sha256
        assert (
            np.max(np.abs(np.asarray(before.probabilities) - np.asarray(after.probabilities)))
            <= 1.0e-10
        )
    finally:
        if restored is not None:
            restored.release()
        adapter.release()


@pytest.mark.integration
def test_prediction_cache_rejects_changed_payload_under_same_identity() -> None:
    model_id = "LR-1.9"
    fixture = FIXTURES["binary_numerical"]
    adapter, prediction = fit_predict(model_id, fixture)
    try:
        adapter.roundtrip_prediction(prediction)
        changed = np.asarray(prediction.probabilities, dtype="float64")[:, ::-1]
        raw = b"".join(struct.pack("<d", value) for row in changed for value in row)
        conflicting = prediction.model_copy(
            update={
                "probabilities": changed.tolist(),
                "probabilities_sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
        with pytest.raises(AdapterFailure) as error:
            adapter.roundtrip_prediction(conflicting)
        assert error.value.category == FailureCategory.FAIL_CACHE_INTEGRITY
    finally:
        adapter.release()


@pytest.mark.integration
def test_catboost_serialization_difference_requires_a_valid_prediction_certificate() -> None:
    fixture = FIXTURES["binary_numerical"]
    first_adapter, _ = fit_predict("CAT-1.2", fixture)
    second_adapter = None
    restored = None
    try:
        identity = first_adapter.save_fitted()
        second_adapter, _ = fit_predict("CAT-1.2", fixture)
        assert second_adapter.save_fitted() == identity
        envelope_path = first_adapter.cache.path_for("model", identity)
        envelope = json.loads(envelope_path.read_text(encoding="utf-8"))
        certificate = envelope["semantic_equivalence"]
        assert certificate["equivalent"] is True
        assert certificate["maximum_probability_difference"] <= certificate["tolerance"]
        restored = create_adapter("CAT-1.2", device="cpu")
        restored.load_fitted(identity)

        changed_adapter, _ = fit_predict("LR-1.9", fixture)
        try:
            changed_identity = changed_adapter.save_fitted()
            changed_adapter.model.coef_[:] = 0.0
            changed_adapter.model.intercept_[:] = 0.0
            with pytest.raises(AdapterFailure) as error:
                changed_adapter.save_fitted()
            assert error.value.category == FailureCategory.FAIL_CACHE_INTEGRITY
            assert changed_identity is not None
        finally:
            changed_adapter.release()
    finally:
        if restored is not None:
            restored.release()
        if second_adapter is not None:
            second_adapter.release()
        first_adapter.release()
