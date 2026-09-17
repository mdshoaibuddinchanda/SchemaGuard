from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from schemaguard.experiments.evaluation import _ece, _js_divergences, _metrics_for


def _prediction(probabilities: list[tuple[float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model_id": ["LR-1.9"] * len(probabilities),
            "view_id": ["V00"] * len(probabilities),
            "device": ["cpu"] * len(probabilities),
            "partition": ["test"] * len(probabilities),
            "p_0": [value[0] for value in probabilities],
            "p_1": [value[1] for value in probabilities],
            "predicted_class": [int(value[1] > value[0]) for value in probabilities],
        }
    )


def test_smoke_metrics_brier_logloss_auc_and_tied_argmax() -> None:
    prediction = _prediction([(0.9, 0.1), (0.5, 0.5)])
    metrics = _metrics_for(
        prediction,
        np.asarray([0, 1]),
        epsilon=1e-15,
        ece_bins=10,
        probability_tolerance=1e-6,
        prediction_sha256="a" * 64,
        target_sha256="b" * 64,
    )
    assert metrics.brier_score == pytest.approx(0.26)
    assert metrics.roc_auc == pytest.approx(1.0)
    assert metrics.roc_auc_status == "defined"
    assert metrics.accuracy == pytest.approx(0.5)
    assert np.isfinite(metrics.log_loss)


def test_single_class_auc_is_explicit_and_extreme_probabilities_are_clipped() -> None:
    prediction = _prediction([(1.0, 0.0), (1.0, 0.0)])
    metrics = _metrics_for(
        prediction,
        np.asarray([0, 0]),
        epsilon=1e-15,
        ece_bins=10,
        probability_tolerance=1e-6,
        prediction_sha256="c" * 64,
        target_sha256="d" * 64,
    )
    assert metrics.roc_auc is None
    assert metrics.roc_auc_status == "undefined_single_class"
    assert np.isfinite(metrics.log_loss)


@pytest.mark.parametrize(
    ("probabilities", "labels", "message"),
    [
        ([], [], "empty partition"),
        ([(float("nan"), 1.0)], [0], "invalid probabilities"),
        ([(-0.1, 1.1)], [0], "invalid probabilities"),
        ([(0.2, 0.2)], [0], "simplex tolerance"),
    ],
)
def test_invalid_probability_and_empty_metric_inputs_fail_closed(
    probabilities: list[tuple[float, float]], labels: list[int], message: str
) -> None:
    prediction = _prediction(probabilities)
    with pytest.raises(ValueError, match=message):
        _metrics_for(
            prediction,
            np.asarray(labels),
            epsilon=1e-15,
            ece_bins=10,
            probability_tolerance=1e-6,
            prediction_sha256="e" * 64,
            target_sha256="f" * 64,
        )


def test_sii_js_divergence_symmetry_and_ece_upper_boundary() -> None:
    first = np.asarray([[1.0, 0.0], [0.5, 0.5]])
    opposite = np.asarray([[0.0, 1.0], [0.5, 0.5]])
    assert np.array_equal(_js_divergences(first, first), np.zeros(2))
    assert np.allclose(_js_divergences(first, opposite), _js_divergences(opposite, first))
    assert _js_divergences(first[:1], opposite[:1])[0] == pytest.approx(1.0)
    assert _ece(np.asarray([0]), np.asarray([[1.0, 0.0]]), bins=10) == 0.0
