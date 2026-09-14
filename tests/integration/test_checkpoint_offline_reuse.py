from __future__ import annotations

import os

import pytest


@pytest.mark.integration
@pytest.mark.foundation_model
@pytest.mark.network
def test_offline_checkpoint_reuse_is_controlled_by_gate() -> None:
    if os.environ.get("SCHEMAGUARD_RUN_FOUNDATION_TESTS") != "1":
        pytest.skip("Offline foundation reuse is executed by the Phase 02A gate")
