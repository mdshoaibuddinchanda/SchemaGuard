"""Strict model adapters; imports remain lazy to keep ordinary tests lightweight."""

from .contracts import AdapterFailure, FailureCategory, PredictionResult

__all__ = ["AdapterFailure", "FailureCategory", "PredictionResult"]
