import json
from pathlib import Path

import pytest

from schemaguard.config import ConfigError, ExperimentConfig, atomic_write_json, load_config
from schemaguard.manifest import build_manifest, manifest_payload

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "experiment_registry.yaml"


@pytest.fixture(scope="module")
def config() -> ExperimentConfig:
    return load_config(CONFIG_PATH)


def test_frozen_counts(config: ExperimentConfig) -> None:
    assert config.scheduled_count("pilot") == 990
    assert config.scheduled_count("main") == 3850


def test_registry_sizes_and_frozen_ids(config: ExperimentConfig) -> None:
    assert len(config.models) == 5
    assert len(config.datasets) == 14
    assert [view.id for view in config.views] == [f"V{i:02d}" for i in range(11)]
    assert config.pilot_datasets == (31, 36, 38, 44, 50, 1489)
    assert config.smoke_dataset == 1464


def test_unknown_root_key_is_rejected(config: ExperimentConfig) -> None:
    payload = config.canonical_payload()
    payload["refresh_bug"] = True
    with pytest.raises(ConfigError, match="refresh_bug"):
        ExperimentConfig.from_mapping(payload)


def test_duplicate_model_id_is_rejected(config: ExperimentConfig) -> None:
    payload = config.canonical_payload()
    payload["models"] = [*payload["models"], dict(payload["models"][0])]
    with pytest.raises(ConfigError, match="Duplicate model ID"):
        ExperimentConfig.from_mapping(payload)


def test_manifest_is_complete_and_unique(config: ExperimentConfig) -> None:
    records = build_manifest(config, "pilot")
    keys = [(item.dataset_id, item.seed, item.model_id, item.view_id) for item in records]
    assert len(records) == 990
    assert len(keys) == len(set(keys))
    assert {item.status for item in records} == {"scheduled"}


def test_manifest_order_is_stable(config: ExperimentConfig) -> None:
    first = manifest_payload(config, "pilot")
    second = manifest_payload(config, "pilot")
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_atomic_refresh_is_idempotent(tmp_path: Path) -> None:
    target = tmp_path / "state.json"
    payload = {"status": "PASS", "records": [1, 2, 3]}
    assert atomic_write_json(target, payload) is True
    first = target.read_text(encoding="utf-8")
    assert atomic_write_json(target, payload) is False
    assert target.read_text(encoding="utf-8") == first
