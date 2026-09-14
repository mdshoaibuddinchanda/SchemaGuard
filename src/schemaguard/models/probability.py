"""Canonical probability output validation."""

from __future__ import annotations

from typing import Any

import numpy as np


class ProbabilityValidationError(ValueError):
    pass


def normalize_probability_matrix(
    probabilities: Any,
    observed_classes: Any,
    canonical_classes: Any,
    *,
    row_count: int | None = None,
    sum_atol: float = 1.0e-6,
) -> tuple[np.ndarray, list[Any], float]:
    """Validate and reorder probabilities to the declared canonical class order."""

    matrix = np.asarray(probabilities, dtype=float)
    classes = list(np.asarray(observed_classes).tolist())
    canonical = list(np.asarray(canonical_classes).tolist())
    if matrix.ndim != 2:
        raise ProbabilityValidationError("Probability output must be a two-dimensional matrix")
    if len(classes) != matrix.shape[1]:
        raise ProbabilityValidationError("Probability column count does not match class order")
    if len(classes) != len(set(classes)):
        raise ProbabilityValidationError("Duplicate class labels")
    if len(canonical) != len(set(canonical)) or set(classes) != set(canonical):
        raise ProbabilityValidationError("Missing or unexpected class label")
    if row_count is not None and matrix.shape[0] != row_count:
        raise ProbabilityValidationError("Probability row count mismatch")
    if not np.isfinite(matrix).all():
        raise ProbabilityValidationError("Probability output contains NaN or infinity")
    if (matrix < -sum_atol).any() or (matrix > 1.0 + sum_atol).any():
        raise ProbabilityValidationError("Probability outside [0, 1]")
    order = [classes.index(label) for label in canonical]
    normalized = matrix[:, order]
    sums = normalized.sum(axis=1)
    max_error = float(np.max(np.abs(sums - 1.0))) if len(sums) else 0.0
    if max_error > sum_atol:
        raise ProbabilityValidationError(f"Probability rows do not sum to one: {max_error}")
    if (normalized < 0).any() or (normalized > 1).any():
        # Only permit tiny floating-point excursions to be clipped after validation.
        normalized = np.clip(normalized, 0.0, 1.0)
    return normalized, canonical, max_error


def validate_repeated_predictions(first: Any, second: Any, *, max_abs_diff: float) -> float:
    a = np.asarray(first, dtype=float)
    b = np.asarray(second, dtype=float)
    if a.shape != b.shape:
        raise ProbabilityValidationError("Repeated prediction shapes differ")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ProbabilityValidationError("Repeated predictions are not finite")
    difference = float(np.max(np.abs(a - b))) if a.size else 0.0
    if difference > max_abs_diff:
        raise ProbabilityValidationError(
            f"Repeated predictions are non-deterministic: {difference}"
        )
    return difference
