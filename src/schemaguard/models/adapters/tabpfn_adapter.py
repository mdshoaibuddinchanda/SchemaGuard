"""Adapter for the frozen TabPFN-3 checkpoint and memory policy."""

from .base import ModelAdapterBase


class TabPFNAdapter(ModelAdapterBase):
    adapter_id = "TPFN3-8.5"
    preprocessing_strategy = "native_tabpfn"
    supports_model_serialization = False
    expected_peak_vram_mib = 376.0
