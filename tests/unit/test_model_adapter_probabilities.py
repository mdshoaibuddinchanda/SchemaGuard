from __future__ import annotations

import numpy as np
import pytest

from schemaguard.models.adapters.contracts import AdapterFailure, FailureCategory
from schemaguard.models.adapters.probability import align_adapter_probabilities


def test_probability_validation_reorders_to_canonical_class_order() -> None:
    probabilities, classes, error = align_adapter_probabilities(
        np.asarray([[0.25, 0.75], [0.8, 0.2]]), [1, 0], [0, 1], row_count=2
    )
    np.testing.assert_array_equal(probabilities, [[0.75, 0.25], [0.2, 0.8]])
    assert classes == [0, 1]
    assert error == 0.0


@pytest.mark.parametrize(
    "probabilities,observed,canonical",
    [
        ([[float("nan"), 0.0]], [0, 1], [0, 1]),
        ([[float("inf"), 0.0]], [0, 1], [0, 1]),
        ([[-0.01, 1.01]], [0, 1], [0, 1]),
        ([[0.7, 0.2]], [0, 1], [0, 1]),
        ([[0.5, 0.5]], [0, 0], [0, 1]),
        ([[0.5, 0.5]], [0, 1], [0, 2]),
    ],
)
def test_invalid_probability_and_class_contracts_fail(probabilities, observed, canonical) -> None:
    with pytest.raises(AdapterFailure) as error:
        align_adapter_probabilities(probabilities, observed, canonical, row_count=1)
    assert error.value.category in {
        FailureCategory.FAIL_CLASS_ORDER,
        FailureCategory.FAIL_PROBABILITY,
    }


def test_probability_row_count_and_one_dimensional_binary_outputs_fail() -> None:
    with pytest.raises(AdapterFailure):
        align_adapter_probabilities([[0.5, 0.5]], [0, 1], [0, 1], row_count=2)
    with pytest.raises(AdapterFailure):
        align_adapter_probabilities([0.2, 0.8], [0, 1], [0, 1], row_count=2)
