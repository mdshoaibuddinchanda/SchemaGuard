from __future__ import annotations

import pytest

from schemaguard.models.adapters.contracts import AdapterFailure, FailureCategory
from schemaguard.models.adapters.factory import create_adapter, load_adapter_config
from schemaguard.models.registry import EXPECTED_MODELS


def test_factory_resolves_exactly_the_five_frozen_model_ids() -> None:
    adapters = [create_adapter(model_id) for model_id, *_ in EXPECTED_MODELS]
    assert [adapter.model_id for adapter in adapters] == [row[0] for row in EXPECTED_MODELS]
    assert len({adapter.model_id for adapter in adapters}) == 5
    for adapter in adapters:
        adapter.release()


def test_adapter_configuration_references_the_single_existing_registry() -> None:
    config = load_adapter_config()
    assert config.registry_config == "model_compatibility.yaml"
    assert config.experiment_registry == "../experiment_registry.yaml"
    assert set(config.checkpoints) == {"TPFN3-8.5", "TICL2-2.2"}


def test_unknown_model_fails_with_closed_category() -> None:
    with pytest.raises(AdapterFailure) as error:
        create_adapter("RF-1.0")
    assert error.value.category == FailureCategory.FAIL_CONTRACT


def test_adapter_implementation_identity_is_line_ending_portable_and_sensitive() -> None:
    from schemaguard.models.adapters.contracts import implementation_digest

    assert implementation_digest("def predict():\n    return 1\n") == implementation_digest(
        "def predict():\r\n    return 1\r\n"
    )
    assert implementation_digest("def predict():\n    return 1\n") != implementation_digest(
        "def predict():\n    return 2\n"
    )
