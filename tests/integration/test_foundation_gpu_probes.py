from __future__ import annotations

import os

import pytest


@pytest.mark.integration
@pytest.mark.foundation_model
@pytest.mark.gpu
@pytest.mark.slow
def test_foundation_gpu_probe_is_capability_recorded() -> None:
    if os.environ.get("SCHEMAGUARD_RUN_FOUNDATION_TESTS") != "1":
        pytest.skip("GPU probes are controlled by scripts/02_check_model_compatibility.py")
