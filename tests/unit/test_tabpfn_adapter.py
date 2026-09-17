from __future__ import annotations

import pytest

from schemaguard.models.adapters.factory import create_adapter, load_adapter_config

pytestmark = pytest.mark.foundation_model


def test_tabpfn_adapter_is_frozen_and_uses_validated_checkpoint_identity() -> None:
    adapter = create_adapter("TPFN3-8.5", device="cpu")
    try:
        expected = load_adapter_config().checkpoints["TPFN3-8.5"]
        assert adapter.spec.checkpoint == expected.identifier
        assert adapter.expected_peak_vram_mib == 376.0
        assert adapter.supports_model_serialization is False
        assert adapter.preprocessing_strategy == "native_tabpfn"
    finally:
        adapter.release()
