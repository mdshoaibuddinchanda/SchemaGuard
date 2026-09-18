"""Validation and protocol-metric utilities for the plan-only representation pilot.

No function in this module reads a dataset. Metric functions accept caller-provided
arrays so their policies can be unit-tested on synthetic values without opening pilot
test labels.
"""

from __future__ import annotations

import itertools
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .pilot_contracts import (
    AurocPolicy,
    DecisionOutcome,
    DecisionPolicy,
    LeakageEvent,
    NonRankEquivalencePolicy,
    PilotConditionInventory,
    PilotProtocolArtifact,
    PilotProtocolValidation,
    PilotStatus,
    ProtocolValidationGate,
)


class PilotValidationError(ValueError):
    """Raised when a frozen protocol or a metric input violates its contract."""


def validate_protocol_artifacts(
    protocol_path: str | Path, inventory_path: str | Path
) -> tuple[PilotProtocolArtifact, PilotConditionInventory]:
    """Load and cross-check the two deterministic, metadata-only plan artifacts."""
    protocol_raw = _read_json(protocol_path)
    inventory_raw = _read_json(inventory_path)
    try:
        protocol = PilotProtocolArtifact.model_validate(protocol_raw)
        inventory = PilotConditionInventory.model_validate(inventory_raw)
    except Exception as exc:
        raise PilotValidationError(f"strict protocol contract validation failed: {exc}") from exc
    if protocol.condition_inventory_sha256 != inventory.inventory_sha256:
        raise PilotValidationError("protocol does not bind the condition inventory hash")
    if protocol.protocol_sha256 != inventory.protocol_sha256:
        raise PilotValidationError("condition inventory refers to a different protocol")
    if tuple(protocol.seed_selection.seeds) != inventory.selected_seeds:
        raise PilotValidationError("protocol seeds and inventory seeds differ")
    if (
        tuple(item.dataset_id for item in protocol.selected_datasets)
        != inventory.selected_dataset_ids
    ):
        raise PilotValidationError("protocol datasets and inventory datasets differ")
    expected_na = protocol.summary["not_applicable_view_count"] * 1
    if expected_na != inventory.not_applicable_view_count:
        raise PilotValidationError("protocol and condition inventory applicability counts differ")
    return protocol, inventory


def validate_event_sequence(events: Sequence[LeakageEvent | str], expected: Sequence[str]) -> None:
    """Require the frozen leakage boundary and prohibit opening labels early."""
    observed = tuple(item.event if isinstance(item, LeakageEvent) else str(item) for item in events)
    if observed != tuple(expected):
        raise PilotValidationError("leakage events do not match the frozen sealed-label order")
    if "test_labels_opened" in observed:
        opened = observed.index("test_labels_opened")
        structural = observed.index("prediction_structure_validated")
        if structural >= opened:
            raise PilotValidationError(
                "test labels were opened before prediction structure validation"
            )


def multiclass_brier_score(
    labels: Sequence[int] | np.ndarray,
    probabilities: Sequence[Sequence[float]] | np.ndarray,
    *,
    row_sum_atol: float = 1.0e-6,
) -> float:
    """Return mean per-row multiclass sum-of-squared-error Brier score."""
    matrix, target = _probability_inputs(labels, probabilities, row_sum_atol)
    one_hot = np.zeros_like(matrix)
    one_hot[np.arange(target.size), target] = 1.0
    return float(np.square(matrix - one_hot).sum(axis=1).mean())


def worst_view_brier_degradation(
    labels: Sequence[int] | np.ndarray,
    baseline: Sequence[Sequence[float]] | np.ndarray,
    views: Mapping[str, Sequence[Sequence[float]] | np.ndarray],
    *,
    relative_epsilon: float = 1.0e-12,
    row_sum_atol: float = 1.0e-6,
) -> dict[str, float | str]:
    """Compute absolute and relative worst-view Brier degradation against V00."""
    if not views or "V00" in views:
        raise PilotValidationError("at least one applicable transformed view is required")
    baseline_matrix, target = _probability_inputs(labels, baseline, row_sum_atol)
    base = multiclass_brier_score(target, baseline_matrix, row_sum_atol=row_sum_atol)
    by_view: dict[str, float] = {}
    for view_id, prediction in views.items():
        matrix, view_target = _probability_inputs(labels, prediction, row_sum_atol)
        if matrix.shape != baseline_matrix.shape or not np.array_equal(target, view_target):
            raise PilotValidationError("baseline and transformed probability matrices must align")
        by_view[view_id] = multiclass_brier_score(view_target, matrix, row_sum_atol=row_sum_atol)
    worst_view = min(by_view, key=lambda view_id: (-(by_view[view_id] - base), view_id))
    degradation = by_view[worst_view] - base
    if not math.isfinite(relative_epsilon) or relative_epsilon <= 0:
        raise PilotValidationError("relative degradation epsilon must be finite and positive")
    return {
        "baseline_brier": base,
        "worst_view": worst_view,
        "worst_view_brier": by_view[worst_view],
        "wbd": degradation,
        "rwbd": degradation / max(base, relative_epsilon),
    }


def semantic_instability_index(
    probabilities_by_view: Mapping[str, Sequence[Sequence[float]] | np.ndarray],
    *,
    probability_clip: float = 1.0e-15,
    row_sum_atol: float = 1.0e-6,
) -> dict[str, Any]:
    """Compute per-row max pairwise Jensen-Shannon divergence in bits."""
    if len(probabilities_by_view) < 2:
        raise PilotValidationError("SII requires at least two applicable views")
    view_ids = tuple(sorted(probabilities_by_view))
    matrices = [
        _clip_and_normalize(
            _probability_matrix(probabilities_by_view[key], row_sum_atol), probability_clip
        )
        for key in view_ids
    ]
    shape = matrices[0].shape
    if any(matrix.shape != shape for matrix in matrices[1:]):
        raise PilotValidationError("SII view probability matrices must have identical shapes")
    maximum = np.zeros(shape[0], dtype=np.float64)
    maximum_pairs = [("", "")] * shape[0]
    for left_idx, right_idx in itertools.combinations(range(len(matrices)), 2):
        left = matrices[left_idx]
        right = matrices[right_idx]
        midpoint = 0.5 * (left + right)
        divergence = 0.5 * np.sum(left * np.log2(left / midpoint), axis=1)
        divergence += 0.5 * np.sum(right * np.log2(right / midpoint), axis=1)
        update = divergence > maximum
        maximum[update] = divergence[update]
        for row in np.flatnonzero(update):
            maximum_pairs[int(row)] = (view_ids[left_idx], view_ids[right_idx])
    return {
        "per_row": maximum,
        "mean": float(np.mean(maximum)),
        "median": float(np.median(maximum)),
        "p90": float(np.quantile(maximum, 0.90)),
        "p95": float(np.quantile(maximum, 0.95)),
        "maximum": float(np.max(maximum)),
        "maximum_mean_pair": _maximum_mean_pair(matrices, view_ids, probability_clip),
        "maximum_row_pair": maximum_pairs[int(np.argmax(maximum))],
    }


def _maximum_mean_pair(
    matrices: Sequence[np.ndarray], view_ids: Sequence[str], clip: float
) -> tuple[str, str]:
    best_pair = (view_ids[0], view_ids[1])
    best_mean = -math.inf
    normalized = [_clip_and_normalize(matrix, clip) for matrix in matrices]
    for left_idx, right_idx in itertools.combinations(range(len(normalized)), 2):
        left = normalized[left_idx]
        right = normalized[right_idx]
        midpoint = 0.5 * (left + right)
        values = 0.5 * np.sum(left * np.log2(left / midpoint), axis=1)
        values += 0.5 * np.sum(right * np.log2(right / midpoint), axis=1)
        mean = float(np.mean(values))
        pair = (view_ids[left_idx], view_ids[right_idx])
        if mean > best_mean or (mean == best_mean and pair < best_pair):
            best_pair, best_mean = pair, mean
    return best_pair


def label_flip_summary(
    probabilities_by_view: Mapping[str, Sequence[Sequence[float]] | np.ndarray],
    view_families: Mapping[str, str],
) -> dict[str, Any]:
    """Summarize V00 flips and name the transformation causing the highest rate."""
    if "V00" not in probabilities_by_view or len(probabilities_by_view) < 2:
        raise PilotValidationError("label flips require V00 and at least one other view")
    if set(view_families) != set(probabilities_by_view) or any(
        not isinstance(family, str) or not family.strip() for family in view_families.values()
    ):
        raise PilotValidationError(
            "every applicable view must have exactly one named transformation family"
        )
    classes: dict[str, np.ndarray] = {}
    row_count: int | None = None
    for view_id, values in probabilities_by_view.items():
        matrix = _probability_matrix(values, 1.0e-6)
        if row_count is None:
            row_count = matrix.shape[0]
        elif row_count != matrix.shape[0]:
            raise PilotValidationError("label-flip view row counts differ")
        classes[view_id] = np.argmax(matrix, axis=1)
    baseline = classes["V00"]
    pairwise = {
        view_id: float(np.mean(prediction != baseline))
        for view_id, prediction in sorted(classes.items())
        if view_id != "V00"
    }
    worst_view = min(pairwise, key=lambda view_id: (-pairwise[view_id], view_id))
    flips_any = np.any(
        np.stack(
            [prediction != baseline for view_id, prediction in classes.items() if view_id != "V00"]
        ),
        axis=0,
    )
    return {
        "pairwise_vs_v00": pairwise,
        "worst_view": worst_view,
        "worst_view_flip_rate": pairwise[worst_view],
        "worst_view_family": view_families[worst_view],
        "any_flip_rate": float(np.mean(flips_any)),
        "flipped_row_count": int(np.sum(flips_any)),
        "row_count": int(row_count or 0),
    }


def chunked_binary_auroc(
    labels: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    *,
    numerical_tie_tolerance: float = 1.0e-15,
    chunk_size: int = 256,
) -> dict[str, float | int | str | None]:
    """Compute exact pairwise raw and tolerance-aware binary AUROC in bounded chunks."""
    target = np.asarray(labels)
    values = np.asarray(scores, dtype=np.float64)
    if target.ndim != 1 or values.ndim != 1 or target.shape != values.shape:
        raise PilotValidationError("AUROC labels and scores must be aligned one-dimensional arrays")
    if target.size == 0 or not np.isin(target, (0, 1)).all():
        raise PilotValidationError("AUROC requires nonempty binary labels encoded as 0 and 1")
    if not np.isfinite(values).all():
        raise PilotValidationError("AUROC scores must be finite")
    if numerical_tie_tolerance <= 0 or chunk_size <= 0:
        raise PilotValidationError("AUROC tie tolerance and chunk size must be positive")
    positive = values[target == 1]
    negative = values[target == 0]
    if not positive.size or not negative.size:
        return {
            "raw_auroc": None,
            "tolerance_aware_auroc": None,
            "tolerance_tie_pair_count": 0,
            "status": "UNDEFINED_SINGLE_CLASS",
        }
    raw_credit = 0.0
    tolerant_credit = 0.0
    tolerance_ties = 0
    for positive_start in range(0, positive.size, chunk_size):
        current_positive = positive[positive_start : positive_start + chunk_size, None]
        for negative_start in range(0, negative.size, chunk_size):
            current_negative = negative[None, negative_start : negative_start + chunk_size]
            difference = current_positive - current_negative
            raw_credit += float(np.count_nonzero(difference > 0))
            raw_credit += 0.5 * float(np.count_nonzero(difference == 0))
            tied = np.abs(difference) <= numerical_tie_tolerance
            tolerance_ties += int(np.count_nonzero(tied))
            tolerant_credit += float(np.count_nonzero(difference > numerical_tie_tolerance))
            tolerant_credit += 0.5 * float(np.count_nonzero(tied))
    denominator = float(positive.size * negative.size)
    return {
        "raw_auroc": raw_credit / denominator,
        "tolerance_aware_auroc": tolerant_credit / denominator,
        "tolerance_tie_pair_count": tolerance_ties,
        "status": "DEFINED",
    }


def numerical_tie_only_change(
    before: Sequence[float] | np.ndarray,
    after: Sequence[float] | np.ndarray,
    *,
    absolute_atol: float,
    relative_rtol: float,
) -> bool:
    """Return true only when scores are numerically equivalent under the frozen policy."""
    left = np.asarray(before, dtype=np.float64)
    right = np.asarray(after, dtype=np.float64)
    return bool(
        left.shape == right.shape
        and np.isfinite(left).all()
        and np.isfinite(right).all()
        and np.allclose(left, right, atol=absolute_atol, rtol=relative_rtol, equal_nan=False)
    )


def classify_numerical_tie_sensitivity(
    *,
    probabilities_before: Sequence[Sequence[float]] | np.ndarray,
    probabilities_after: Sequence[Sequence[float]] | np.ndarray,
    label_flip_count: int,
    brier_before: float,
    brier_after: float,
    log_loss_before: float,
    log_loss_after: float,
    raw_auroc_before: float,
    raw_auroc_after: float,
    tolerance_aware_auroc_before: float,
    tolerance_aware_auroc_after: float,
    policy: AurocPolicy,
    non_rank_policy: NonRankEquivalencePolicy,
) -> str:
    """Classify only a verified machine-precision AUROC tie-ordering artifact."""
    before = _probability_matrix(probabilities_before, 1.0e-6)
    after = _probability_matrix(probabilities_after, 1.0e-6)
    metrics = (
        brier_before,
        brier_after,
        log_loss_before,
        log_loss_after,
        raw_auroc_before,
        raw_auroc_after,
        tolerance_aware_auroc_before,
        tolerance_aware_auroc_after,
    )
    if before.shape != after.shape or not all(math.isfinite(value) for value in metrics):
        raise PilotValidationError("tie-sensitivity inputs must be aligned and finite")
    if label_flip_count < 0:
        raise PilotValidationError("label-flip count cannot be negative")
    probabilities_equivalent = float(np.max(np.abs(before - after))) <= (
        policy.tie_sensitivity_probability_max_abs_diff
    )
    non_rank_equivalent = np.isclose(
        brier_before,
        brier_after,
        atol=non_rank_policy.absolute_atol,
        rtol=non_rank_policy.relative_rtol,
    ) and np.isclose(
        log_loss_before,
        log_loss_after,
        atol=non_rank_policy.absolute_atol,
        rtol=non_rank_policy.relative_rtol,
    )
    tolerance_auroc_equivalent = (
        abs(tolerance_aware_auroc_before - tolerance_aware_auroc_after)
        <= policy.tie_sensitivity_tolerance_aware_auroc_atol
    )
    qualifies = (
        probabilities_equivalent
        and label_flip_count == policy.tie_sensitivity_label_flip_count
        and non_rank_equivalent
        and raw_auroc_before != raw_auroc_after
        and tolerance_auroc_equivalent
    )
    return policy.tie_sensitivity_classification if qualifies else "NOT_NUMERICAL_TIE_ONLY"


def pilot_feasibility_passes(
    *,
    v00_planned: int,
    v00_completed: int,
    transformed_applicable: int,
    transformed_completed: int,
    every_failure_classified: bool,
    frozen_model_configuration_unchanged: bool,
    resume_cache_integrity_pass: bool,
    resource_ceilings_respected: bool,
    policy: DecisionPolicy,
) -> bool:
    """Evaluate the frozen completion, integrity, cache, and resource feasibility gate."""
    counts = (v00_planned, v00_completed, transformed_applicable, transformed_completed)
    if any(value < 0 for value in counts):
        raise PilotValidationError("feasibility condition counts cannot be negative")
    if v00_planned == 0 or transformed_applicable == 0:
        raise PilotValidationError(
            "feasibility evidence must include V00 and transformed conditions"
        )
    if v00_completed > v00_planned or transformed_completed > transformed_applicable:
        raise PilotValidationError("completed condition counts cannot exceed planned counts")
    return bool(
        v00_completed == v00_planned
        and transformed_completed / transformed_applicable
        >= policy.minimum_transformed_condition_completion_rate
        and every_failure_classified
        and frozen_model_configuration_unchanged
        and resume_cache_integrity_pass
        and resource_ceilings_respected
    )


def evaluate_pilot_outcome(
    *,
    integrity_pass: bool,
    feasible: bool,
    partial_capability: bool,
    model_evidence: Mapping[str, Mapping[str, Any]],
    policy: DecisionPolicy,
) -> DecisionOutcome:
    """Apply the frozen six-outcome pilot decision vocabulary to summarized evidence."""
    if not integrity_pass:
        return "REPAIR_REQUIRED"
    if partial_capability:
        return "PARTIAL_CAPABILITY"
    if not feasible:
        return "REPAIR_REQUIRED"
    qualifying: dict[str, Mapping[str, Any]] = {}
    for model_id, evidence in model_evidence.items():
        if model_id not in policy.foundation_model_ids:
            continue
        if policy.exclude_numerical_tie_only and evidence.get("tie_only", False):
            continue
        if policy.require_frozen_preprocessing_residual and not evidence.get(
            "preprocessing_residual_pass", False
        ):
            continue
        if (
            float(evidence.get("median_wbd", -math.inf)) >= policy.nontrivial_absolute_wbd
            or float(evidence.get("median_rwbd", -math.inf)) >= policy.nontrivial_relative_wbd
        ):
            qualifying[model_id] = evidence
    if not qualifying:
        return "STOP_OR_REDESIGN"
    reproducible = {
        model_id: item
        for model_id, item in qualifying.items()
        if int(item.get("datasets_meeting_threshold", 0)) >= 1
        and int(item.get("minimum_seed_directions_per_counted_dataset", 0))
        >= policy.required_seed_direction_count
    }
    if not reproducible:
        return "STOP_OR_REDESIGN"
    broad_foundation_models = {
        model_id
        for model_id, item in reproducible.items()
        if int(item.get("datasets_meeting_threshold", 0)) >= policy.breadth_dataset_count
        and int(item.get("transformation_family_count", 0))
        >= policy.required_transformation_families
        and int(item.get("minimum_seed_directions_per_counted_dataset", 0))
        >= policy.required_seed_direction_count
    }
    if broad_foundation_models:
        for model_id, item in model_evidence.items():
            if model_id in policy.foundation_model_ids:
                continue
            if (
                not (policy.exclude_numerical_tie_only and item.get("tie_only", False))
                and int(item.get("datasets_meeting_threshold", 0)) >= policy.breadth_dataset_count
                and int(item.get("transformation_family_count", 0))
                >= policy.required_transformation_families
                and int(item.get("minimum_seed_directions_per_counted_dataset", 0))
                >= policy.required_seed_direction_count
            ):
                return "ADVANCE_BROAD_METHOD"
        return "ADVANCE_TFM_FOCUSED"
    return "NARROW_CASE_STUDY"


def build_validation_report(
    *,
    source_commit: str,
    protocol: PilotProtocolArtifact,
    inventory: PilotConditionInventory,
    gates: Sequence[ProtocolValidationGate],
) -> PilotProtocolValidation:
    """Create a strict forty-gate validation result, never a verified-science verdict."""
    ordered = tuple(sorted(gates, key=lambda gate: gate.gate_id))
    passed = sum(item.result == "PASS" for item in ordered)
    failed = sum(item.result == "FAIL" for item in ordered)
    not_verified = sum(item.result == "NOT_VERIFIED" for item in ordered)
    if failed:
        status: PilotStatus = "REPAIR_REQUIRED"
    elif not_verified:
        status = "BLOCKED"
    elif (
        protocol.runtime_estimate.preferred_wall_limit_passed
        and protocol.runtime_estimate.preferred_storage_limit_passed
    ):
        status = "PASS_PENDING_REVIEW"
    else:
        status = "RESOURCE_REVIEW_REQUIRED"
    return PilotProtocolValidation(
        status=status,
        source_commit=source_commit,
        protocol_sha256=protocol.protocol_sha256,
        condition_inventory_sha256=inventory.inventory_sha256,
        passed_count=passed,
        failed_count=failed,
        not_verified_count=not_verified,
        gates=ordered,
        pilot_execution_performed=False,
        test_labels_accessed=False,
    )


def _probability_inputs(
    labels: Sequence[int] | np.ndarray,
    probabilities: Sequence[Sequence[float]] | np.ndarray,
    row_sum_atol: float,
) -> tuple[np.ndarray, np.ndarray]:
    matrix = _probability_matrix(probabilities, row_sum_atol)
    target = np.asarray(labels, dtype=np.int64)
    if target.ndim != 1 or target.size != matrix.shape[0]:
        raise PilotValidationError("labels and probability rows must be aligned")
    if target.size == 0 or target.min() < 0 or target.max() >= matrix.shape[1]:
        raise PilotValidationError(
            "labels must use contiguous target codes represented in probabilities"
        )
    return matrix, target


def _probability_matrix(
    probabilities: Sequence[Sequence[float]] | np.ndarray, row_sum_atol: float
) -> np.ndarray:
    matrix = np.asarray(probabilities, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] < 2:
        raise PilotValidationError("probability input must be a nonempty rows-by-classes matrix")
    if not np.isfinite(matrix).all() or np.any(matrix < 0.0) or np.any(matrix > 1.0):
        raise PilotValidationError("probabilities must be finite and lie in [0, 1]")
    if not np.allclose(matrix.sum(axis=1), 1.0, atol=row_sum_atol, rtol=0.0):
        raise PilotValidationError("probability rows do not sum to one within the frozen tolerance")
    return matrix


def _clip_and_normalize(matrix: np.ndarray, clip: float) -> np.ndarray:
    if not math.isfinite(clip) or clip <= 0 or clip >= 0.01:
        raise PilotValidationError("SII probability clipping must be finite and in (0, 0.01)")
    clipped = np.clip(matrix, clip, 1.0)
    return clipped / clipped.sum(axis=1, keepdims=True)


def _read_json(path: str | Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PilotValidationError(f"cannot read JSON protocol artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PilotValidationError(f"JSON protocol artifact must be an object: {path}")
    return value


__all__ = [
    "PilotValidationError",
    "build_validation_report",
    "classify_numerical_tie_sensitivity",
    "chunked_binary_auroc",
    "evaluate_pilot_outcome",
    "label_flip_summary",
    "multiclass_brier_score",
    "numerical_tie_only_change",
    "pilot_feasibility_passes",
    "semantic_instability_index",
    "validate_event_sequence",
    "validate_protocol_artifacts",
    "worst_view_brier_degradation",
]
