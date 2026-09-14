"""Deterministic component seed derivation."""

from __future__ import annotations

import hashlib

COMPONENT_NAMES = ("train_temp_split", "calibration_test_split")


def derive_component_seed(master_seed: int, component_name: str) -> int:
    """Derive an unsigned 32-bit seed from a master seed and component name."""
    if master_seed < 0:
        raise ValueError("master_seed must be non-negative")
    digest = hashlib.sha256(f"{master_seed}:{component_name}".encode()).hexdigest()
    return int(digest[:8], 16)
