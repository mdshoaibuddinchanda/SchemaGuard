"""Adapter for the frozen CPU CatBoost native-categorical model."""

from .base import ModelAdapterBase


class CatBoostAdapter(ModelAdapterBase):
    adapter_id = "CAT-1.2"
    preprocessing_strategy = "native_catboost"
