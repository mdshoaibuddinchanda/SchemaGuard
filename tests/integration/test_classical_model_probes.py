from __future__ import annotations

import importlib.metadata

import pytest

from schemaguard.compatibility.fixtures import binary_numerical, multiclass_numerical
from schemaguard.models.probability import normalize_probability_matrix
from schemaguard.models.registry import build_parameters, import_class, load_model_registry


@pytest.mark.integration
@pytest.mark.slow
def test_classical_models_fit_binary_and_multiclass() -> None:
    models = load_model_registry("configs/runtime/model_compatibility.yaml")[:3]
    for spec in models:
        assert importlib.metadata.version(spec.package) == spec.expected_version
        cls = import_class(spec.class_path)
        binary = binary_numerical()
        train = binary.train.drop(columns=["row_id"])
        test = binary.test.drop(columns=["row_id"])
        model = cls(**build_parameters(spec, seed=1729, target_classes=2))
        model.fit(train, binary.target)
        probabilities, classes, _ = normalize_probability_matrix(
            model.predict_proba(test), model.classes_, [0, 1]
        )
        assert probabilities.shape == (16, 2)
        assert classes == [0, 1]
        multi = multiclass_numerical()
        multi_model = cls(**build_parameters(spec, seed=1729, target_classes=3))
        multi_model.fit(multi.train.drop(columns=["row_id"]), multi.target)
        multi_probabilities, multi_classes, _ = normalize_probability_matrix(
            multi_model.predict_proba(multi.test.drop(columns=["row_id"])),
            multi_model.classes_,
            [0, 1, 2],
        )
        assert multi_probabilities.shape == (18, 3)
        assert multi_classes == [0, 1, 2]
