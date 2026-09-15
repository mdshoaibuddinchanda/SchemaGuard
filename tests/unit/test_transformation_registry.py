from __future__ import annotations

import pytest

from schemaguard.transformations.registry import VIEW_REGISTRY, get_transformation, get_view


def test_registry_has_exact_frozen_order_and_names() -> None:
    assert [item.view_id for item in VIEW_REGISTRY] == [f"V{index:02d}" for index in range(11)]
    assert len({item.name for item in VIEW_REGISTRY}) == 11


def test_unknown_view_and_factory_identity_fail() -> None:
    with pytest.raises(KeyError):
        get_view("V99")
    assert get_transformation("V00").view_name == "identity"
