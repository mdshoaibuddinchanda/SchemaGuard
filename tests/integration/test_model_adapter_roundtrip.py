from __future__ import annotations

import numpy as np
import pytest

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
