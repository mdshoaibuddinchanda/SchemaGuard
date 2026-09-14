from __future__ import annotations

import pytest

from schemaguard.models.registry import (
    EXPECTED_MODELS,
    RegistryError,
    build_parameters,
    load_model_registry,
    resolve_catboost_loss,
)


def test_exact_frozen_registry_and_order() -> None:
    models = load_model_registry("configs/runtime/model_compatibility.yaml")
    observed = tuple(
        (m.id, m.package, m.expected_version, m.class_path, m.checkpoint) for m in models
    )
    assert observed == EXPECTED_MODELS


def test_catboost_auto_loss_resolves_for_both_target_grains() -> None:
    model = next(
        m
        for m in load_model_registry("configs/runtime/model_compatibility.yaml")
        if m.id == "CAT-1.2"
    )
    assert build_parameters(model, seed=1729, target_classes=2)["loss_function"] == "Logloss"
    assert build_parameters(model, seed=1729, target_classes=3)["loss_function"] == "MultiClass"
    assert resolve_catboost_loss(2) != resolve_catboost_loss(3)


def test_frozen_random_seeds_are_explicit() -> None:
    models = load_model_registry("configs/runtime/model_compatibility.yaml")
    assert "random_state" in build_parameters(models[0], seed=1729)
    assert "random_seed" in build_parameters(models[1], seed=1729)


def test_invalid_resource_limit_fails(tmp_path) -> None:
    source = open("configs/runtime/model_compatibility.yaml", encoding="utf-8").read()
    path = tmp_path / "bad.yaml"
    path.write_text(source.replace("gpu_workers: 1", "gpu_workers: 0"), encoding="utf-8")
    with pytest.raises(RegistryError):
        load_model_registry(path)


def test_wrong_checkpoint_name_fails(tmp_path) -> None:
    source = open("configs/runtime/model_compatibility.yaml", encoding="utf-8").read()
    path = tmp_path / "bad.yaml"
    path.write_text(
        source.replace("tabicl-classifier-v2-20260212.ckpt", "wrong.ckpt"), encoding="utf-8"
    )
    with pytest.raises(RegistryError):
        load_model_registry(path)
