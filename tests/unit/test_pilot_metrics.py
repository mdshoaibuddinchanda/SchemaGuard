from __future__ import annotations

import numpy as np
import pytest

from schemaguard.experiments.pilot_contracts import (
    AurocPolicy,
    LeakageEvent,
    MetricEvaluationRecord,
    NonRankEquivalencePolicy,
)
from schemaguard.experiments.pilot_validation import (
    PilotValidationError,
    chunked_binary_auroc,
    classify_numerical_tie_sensitivity,
    label_flip_summary,
    multiclass_brier_score,
    numerical_tie_only_change,
    semantic_instability_index,
    validate_event_sequence,
    worst_view_brier_degradation,
)


def test_multiclass_brier_and_worst_view_degradation() -> None:
    labels = np.array([0, 1])
    baseline = np.array([[0.8, 0.2], [0.25, 0.75]])
    view = np.array([[0.6, 0.4], [0.2, 0.8]])
    assert multiclass_brier_score(labels, baseline) == pytest.approx(0.1025)
    result = worst_view_brier_degradation(labels, baseline, {"V01": view})
    assert result["worst_view"] == "V01"
    assert result["wbd"] == pytest.approx(
        multiclass_brier_score(labels, view) - multiclass_brier_score(labels, baseline)
    )
    assert result["rwbd"] == pytest.approx(result["wbd"] / max(result["baseline_brier"], 1.0e-12))


def test_brier_rejects_misaligned_labels_and_invalid_probability_rows() -> None:
    with pytest.raises(PilotValidationError, match="aligned"):
        multiclass_brier_score([0], [[0.7, 0.3], [0.2, 0.8]])
    with pytest.raises(PilotValidationError, match="sum to one"):
        multiclass_brier_score([0], [[0.6, 0.3]])


def test_relative_degradation_uses_epsilon_floor_for_perfect_baseline() -> None:
    result = worst_view_brier_degradation(
        [0], [[1.0, 0.0]], {"V01": [[0.9, 0.1]]}, relative_epsilon=0.01
    )
    assert result["baseline_brier"] == 0.0
    assert result["rwbd"] == pytest.approx(result["wbd"] / 0.01)
    with pytest.raises(PilotValidationError, match="epsilon"):
        worst_view_brier_degradation([0], [[1.0, 0.0]], {"V01": [[0.9, 0.1]]}, relative_epsilon=0)


def test_sii_uses_max_pairwise_jsd_in_bits_and_records_responsible_pairs() -> None:
    result = semantic_instability_index(
        {"V00": [[1.0, 0.0]], "V01": [[0.0, 1.0]], "V02": [[1.0, 0.0]]}
    )
    assert result["per_row"][0] == pytest.approx(1.0, abs=1.0e-12)
    assert result["maximum"] == pytest.approx(1.0, abs=1.0e-12)
    assert result["maximum_row_pair"] == ("V00", "V01")
    assert result["maximum_mean_pair"] == ("V00", "V01")


def test_sii_identical_views_and_zero_probabilities_are_finite() -> None:
    result = semantic_instability_index(
        {"V00": [[1.0, 0.0], [0.25, 0.75]], "V01": [[1.0, 0.0], [0.25, 0.75]]}
    )
    assert np.isfinite(result["per_row"]).all()
    assert result["maximum"] == pytest.approx(0.0, abs=1.0e-15)


def test_sii_is_symmetric_in_view_order_and_aligned_rows() -> None:
    first = semantic_instability_index(
        {"V00": [[0.9, 0.1], [0.2, 0.8]], "V01": [[0.2, 0.8], [0.9, 0.1]]}
    )
    second = semantic_instability_index(
        {"V01": [[0.9, 0.1], [0.2, 0.8]], "V00": [[0.2, 0.8], [0.9, 0.1]]}
    )
    assert first["mean"] == pytest.approx(second["mean"])
    assert first["maximum"] == pytest.approx(second["maximum"])


def test_label_flip_summary_is_aligned_to_v00_and_any_view() -> None:
    result = label_flip_summary(
        {
            "V00": [[0.8, 0.2], [0.4, 0.6]],
            "V01": [[0.2, 0.8], [0.45, 0.55]],
            "V02": [[0.7, 0.3], [0.6, 0.4]],
        },
        {"V00": "identity", "V01": "category_permutation", "V02": "column_permutation"},
    )
    assert result["pairwise_vs_v00"] == {"V01": 0.5, "V02": 0.5}
    assert result["worst_view"] == "V01"
    assert result["worst_view_flip_rate"] == 0.5
    assert result["worst_view_family"] == "category_permutation"
    assert result["any_flip_rate"] == 1.0


def test_label_flip_summary_uses_v00_independent_of_mapping_insertion_order() -> None:
    result = label_flip_summary(
        {"V02": [[0.1, 0.9]], "V00": [[0.9, 0.1]], "V01": [[0.8, 0.2]]},
        {"V02": "numeric", "V00": "control", "V01": "categorical"},
    )
    assert result["any_flip_rate"] == 1.0
    assert result["worst_view"] == "V02"
    assert result["worst_view_family"] == "numeric"


def test_label_flip_summary_requires_family_for_each_applicable_view() -> None:
    with pytest.raises(PilotValidationError, match="transformation family"):
        label_flip_summary({"V00": [[0.8, 0.2]], "V01": [[0.2, 0.8]]}, {"V00": "control"})


def test_chunked_auroc_distinguishes_raw_roundoff_from_tolerance_ties() -> None:
    labels = [0, 1]
    scores = [0.5, 0.5000000000000002]
    result = chunked_binary_auroc(labels, scores, chunk_size=1)
    assert result["raw_auroc"] == 1.0
    assert result["tolerance_aware_auroc"] == 0.5
    assert result["tolerance_tie_pair_count"] == 1
    assert chunked_binary_auroc(labels, scores, chunk_size=20) == result


def test_chunked_auroc_matches_pairwise_reference_and_ignores_input_order() -> None:
    labels = np.array([0, 1, 0, 1, 1, 0])
    scores = np.array([0.2, 0.7, 0.7, 0.2, 0.9, 0.5])
    differences = scores[labels == 1, None] - scores[labels == 0][None, :]
    raw_reference = float(np.mean((differences > 0) + 0.5 * (differences == 0)))
    tolerant_reference = float(
        np.mean((differences > 1.0e-15) + 0.5 * (np.abs(differences) <= 1.0e-15))
    )
    result = chunked_binary_auroc(labels, scores, chunk_size=2)
    permuted = chunked_binary_auroc(labels[::-1], scores[::-1], chunk_size=3)
    assert result["raw_auroc"] == pytest.approx(raw_reference)
    assert result["tolerance_aware_auroc"] == pytest.approx(tolerant_reference)
    assert result["raw_auroc"] == permuted["raw_auroc"]
    assert result["tolerance_aware_auroc"] == permuted["tolerance_aware_auroc"]


@pytest.mark.parametrize(
    ("labels", "scores"),
    (([0, 2], [0.1, 0.2]), ([0, 1], [float("nan"), 0.2]), ([0, 1], [0.1, float("inf")])),
)
def test_auroc_rejects_invalid_labels_and_nonfinite_scores(
    labels: list[int], scores: list[float]
) -> None:
    with pytest.raises(PilotValidationError):
        chunked_binary_auroc(labels, scores)


def test_auroc_preserves_true_ordering_and_half_credit_for_exact_ties() -> None:
    true_difference = chunked_binary_auroc([0, 1], [0.5, 0.500000000000002])
    exact_tie = chunked_binary_auroc([0, 1], [0.5, 0.5])
    assert true_difference["tolerance_aware_auroc"] == 1.0
    assert exact_tie["raw_auroc"] == 0.5
    assert exact_tie["tolerance_aware_auroc"] == 0.5


def test_auroc_single_class_is_explicitly_undefined() -> None:
    result = chunked_binary_auroc([1, 1], [0.1, 0.8])
    assert result == {
        "raw_auroc": None,
        "tolerance_aware_auroc": None,
        "tolerance_tie_pair_count": 0,
        "status": "UNDEFINED_SINGLE_CLASS",
    }


def test_numerical_tie_only_uses_separate_non_rank_tolerance() -> None:
    assert numerical_tie_only_change(
        [0.5, 0.25], [0.5 + 2e-16, 0.25], absolute_atol=1e-12, relative_rtol=1e-10
    )
    assert not numerical_tie_only_change(
        [0.5, 0.25], [0.51, 0.25], absolute_atol=1e-12, relative_rtol=1e-10
    )


def test_smoke_logistic_auroc_change_is_classified_as_numerical_tie_sensitivity() -> None:
    policy = AurocPolicy(
        binary_only=True,
        numerical_tie_tolerance=1.0e-15,
        chunk_size=256,
        tie_credit=0.5,
        report_raw_and_tolerance_aware=True,
        record_tolerance_tie_pair_count=True,
        tie_only_changes_excluded_from_scientific_instability=True,
        tie_sensitivity_probability_max_abs_diff=1.0e-15,
        tie_sensitivity_label_flip_count=0,
        tie_sensitivity_non_rank_metrics=("brier", "log_loss"),
        tie_sensitivity_tolerance_aware_auroc_atol=1.0e-12,
        tie_sensitivity_classification="NUMERICAL_TIE_SENSITIVITY",
    )
    non_rank = NonRankEquivalencePolicy(absolute_atol=1.0e-12, relative_rtol=1.0e-10)
    result = classify_numerical_tie_sensitivity(
        probabilities_before=[[0.7, 0.3], [0.4, 0.6]],
        probabilities_after=[[0.7 + 2.22e-16, 0.3 - 2.22e-16], [0.4, 0.6]],
        label_flip_count=0,
        brier_before=0.2,
        brier_after=0.2,
        log_loss_before=0.4,
        log_loss_after=0.4,
        raw_auroc_before=0.6647869674185464,
        raw_auroc_after=0.6640350877192983,
        tolerance_aware_auroc_before=0.664,
        tolerance_aware_auroc_after=0.664,
        policy=policy,
        non_rank_policy=non_rank,
    )
    assert result == "NUMERICAL_TIE_SENSITIVITY"


@pytest.mark.parametrize(
    "changes",
    (
        {"label_flip_count": 1},
        {"brier_after": 0.21},
        {"log_loss_after": 0.41},
        {"raw_auroc_after": 0.6647869674185464},
        {"tolerance_aware_auroc_after": 0.6641},
        {"probabilities_after": [[0.7 + 2.0e-15, 0.3 - 2.0e-15], [0.4, 0.6]]},
    ),
)
def test_every_numerical_tie_classification_condition_is_required(
    changes: dict[str, object],
) -> None:
    policy = AurocPolicy(
        binary_only=True,
        numerical_tie_tolerance=1.0e-15,
        chunk_size=256,
        tie_credit=0.5,
        report_raw_and_tolerance_aware=True,
        record_tolerance_tie_pair_count=True,
        tie_only_changes_excluded_from_scientific_instability=True,
        tie_sensitivity_probability_max_abs_diff=1.0e-15,
        tie_sensitivity_label_flip_count=0,
        tie_sensitivity_non_rank_metrics=("brier", "log_loss"),
        tie_sensitivity_tolerance_aware_auroc_atol=1.0e-12,
        tie_sensitivity_classification="NUMERICAL_TIE_SENSITIVITY",
    )
    arguments: dict[str, object] = {
        "probabilities_before": [[0.7, 0.3], [0.4, 0.6]],
        "probabilities_after": [[0.7 + 2.22e-16, 0.3 - 2.22e-16], [0.4, 0.6]],
        "label_flip_count": 0,
        "brier_before": 0.2,
        "brier_after": 0.2,
        "log_loss_before": 0.4,
        "log_loss_after": 0.4,
        "raw_auroc_before": 0.6647869674185464,
        "raw_auroc_after": 0.6640350877192983,
        "tolerance_aware_auroc_before": 0.664,
        "tolerance_aware_auroc_after": 0.664,
        "policy": policy,
        "non_rank_policy": NonRankEquivalencePolicy(absolute_atol=1.0e-12, relative_rtol=1.0e-10),
    }
    arguments.update(changes)
    assert classify_numerical_tie_sensitivity(**arguments) == "NOT_NUMERICAL_TIE_ONLY"


def test_leakage_sequence_opens_test_labels_only_after_structural_validation() -> None:
    events = (
        "plan_frozen",
        "test_labels_sealed",
        "training_completed",
        "calibration_predictions_completed",
        "test_predictions_completed",
        "prediction_structure_validated",
        "test_labels_opened",
        "metrics_generated",
        "results_sealed",
    )
    validate_event_sequence(events, events)
    early = list(events)
    early[4], early[6] = early[6], early[4]
    with pytest.raises(PilotValidationError, match="frozen sealed-label order"):
        validate_event_sequence(early, events)
    records = [
        LeakageEvent(event=name, sequence=index, evidence_sha256="0" * 64)
        for index, name in enumerate(events)
    ]
    validate_event_sequence(records, events)


def test_metric_record_requires_explicit_reason_when_undefined() -> None:
    with pytest.raises(ValueError, match="explicit reason"):
        MetricEvaluationRecord(status="UNDEFINED")
    assert MetricEvaluationRecord(status="UNDEFINED", reason="single_class_test_partition")
