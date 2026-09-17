"""Adapter for the frozen TabICLv2 checkpoint with KV caching disabled."""

from .base import ModelAdapterBase


class TabICLAdapter(ModelAdapterBase):
    adapter_id = "TICL2-2.2"
    preprocessing_strategy = "native_tabicl"
    supports_model_serialization = False
    expected_peak_vram_mib = 198.0
