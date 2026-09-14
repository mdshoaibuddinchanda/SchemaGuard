from __future__ import annotations

import numpy as np
import pytest

from schemaguard.models.probability import (
    ProbabilityValidationError,
    normalize_probability_matrix,
    validate_repeated_predictions,
)


def valid() -> tuple[np.ndarray, list[int], list[int]]:
    return np.array([[0.25, 0.75], [0.9, 0.1]]), [1, 0], [0, 1]


def test_valid_binary_matrix_is_reordered_to_canonical_classes() -> None:
    matrix, observed, canonical = valid()
    normalized, classes, error = normalize_probability_matrix(
        matrix, observed, canonical, row_count=2
    )
    np.testing.assert_allclose(normalized, [[0.75, 0.25], [0.1, 0.9]])
    assert classes == canonical
    assert error == 0


@pytest.mark.parametrize(
    "matrix",
    [
        [[np.nan, 1.0]],
        [[np.inf, 0.0]],
        [[-0.1, 1.1]],
        [[0.2, 0.2]],
    ],
)
def test_invalid_probability_values_are_rejected(matrix) -> None:
    with pytest.raises(ProbabilityValidationError):
        normalize_probability_matrix(matrix, [0, 1], [0, 1])


def test_shape_and_class_errors_are_rejected() -> None:
    with pytest.raises(ProbabilityValidationError):
        normalize_probability_matrix([[1.0, 0.0]], [0, 1], [0, 1], row_count=2)
    with pytest.raises(ProbabilityValidationError):
        normalize_probability_matrix([1.0, 0.0], [0, 1], [0, 1])
    with pytest.raises(ProbabilityValidationError):
        normalize_probability_matrix([[1.0, 0.0]], [0, 0], [0, 1])
    with pytest.raises(ProbabilityValidationError):
        normalize_probability_matrix([[1.0, 0.0]], [0, 1], [1, 2])


def test_multiclass_and_tiny_sum_error_are_accepted() -> None:
    matrix = [[0.2, 0.3, 0.5000000001]]
    normalized, classes, error = normalize_probability_matrix(matrix, [0, 1, 2], [0, 1, 2])
    assert normalized.shape == (1, 3)
    assert classes == [0, 1, 2]
    assert error <= 1.0e-6


def test_repeat_difference_is_checked() -> None:
    assert validate_repeated_predictions([[0.5, 0.5]], [[0.5, 0.5]], max_abs_diff=1e-10) == 0
    with pytest.raises(ProbabilityValidationError):
        validate_repeated_predictions([[0.5, 0.5]], [[0.6, 0.4]], max_abs_diff=1e-10)
