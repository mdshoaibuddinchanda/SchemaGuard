from __future__ import annotations

import random
import sys
from types import SimpleNamespace

import numpy as np

from schemaguard.models.adapters.factory import create_adapter


def test_foundation_adapter_resets_python_numpy_and_torch_rngs(monkeypatch) -> None:
    torch_seeds: list[int] = []
    adapter = create_adapter("TICL2-2.2", seed=1729, device="cpu")
    fake_torch = SimpleNamespace(
        manual_seed=torch_seeds.append,
        random=SimpleNamespace(
            get_rng_state=lambda: b"before",
            set_rng_state=lambda state: None,
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    try:
        with adapter._seeded_model_runtime():
            first = (random.random(), float(np.random.random()))
        with adapter._seeded_model_runtime():
            second = (random.random(), float(np.random.random()))
        assert second == first
        assert torch_seeds == [1729, 1729]
    finally:
        adapter.release()


def test_classical_adapter_does_not_seed_torch(monkeypatch) -> None:
    torch_seeds: list[int] = []
    adapter = create_adapter("LR-1.9", seed=1729, device="cpu")
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(manual_seed=torch_seeds.append))
    try:
        with adapter._seeded_model_runtime():
            _ = random.random()
        assert torch_seeds == []
    finally:
        adapter.release()
