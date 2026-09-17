"""Adapter for the frozen CPU XGBoost encoded-numerical model."""

from .base import ModelAdapterBase


class XGBoostAdapter(ModelAdapterBase):
    adapter_id = "XGB-3.4"
    preprocessing_strategy = "common_onehot_no_scaling"
