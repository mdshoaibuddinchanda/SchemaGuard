"""Adapter for the frozen LR-1.9 logistic-regression control."""

from .base import ModelAdapterBase


class LogisticRegressionAdapter(ModelAdapterBase):
    adapter_id = "LR-1.9"
    preprocessing_strategy = "common_onehot_standard"
