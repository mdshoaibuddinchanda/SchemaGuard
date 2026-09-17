from __future__ import annotations

import pytest

from schemaguard.models.adapters.factory import create_adapter, load_adapter_config

pytestmark = pytest.mark.foundation_model


def test_tabicl_adapter_keeps_kv_cache_disabled_and_exact_checkpoint() -> None:
    adapter = create_adapter("TICL2-2.2", device="cpu")
    try:
        expected = load_adapter_config().checkpoints["TICL2-2.2"]
        assert adapter.spec.checkpoint == expected.identifier
        assert adapter.spec.parameters["kv_cache"] is False
        assert adapter.spec.parameters["batch_size"] == 1
        assert adapter.expected_peak_vram_mib == 198.0
        assert adapter.preprocessing_strategy == "native_tabicl"
    finally:
        adapter.release()
