from __future__ import annotations

import os

import pytest


@pytest.mark.integration
@pytest.mark.foundation_model
@pytest.mark.slow
def test_foundation_cpu_probe_is_run_by_phase_script() -> None:
    if os.environ.get("SCHEMAGUARD_RUN_FOUNDATION_TESTS") != "1":
        pytest.skip("Foundation probes are controlled by scripts/02_check_model_compatibility.py")
