"""Offline unit coverage for SchemaOrbit-14 acquisition contracts."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from schemaguard.data.schemaorbit import (
    OfflineCacheMiss,
    SchemaOrbitError,
    acquire_raw,
    load_schemaorbit_config,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "datasets" / "schemaorbit14.yaml"


def test_schemaorbit_registry_has_exact_frozen_identities() -> None:
    config = load_schemaorbit_config(CONFIG)
    assert [item.id for item in config.datasets] == [
        3,
        23,
        29,
        31,
        36,
        37,
        38,
        44,
        46,
        50,
        54,
        1067,
        1464,
        1489,
    ]
    assert [
        (item.id, item.file_id, item.version) for item in config.datasets if item.id in {1067, 1489}
    ] == [(1067, 53950, "1"), (1489, 1592281, "1")]


def test_splice_identifier_is_explicitly_ignored() -> None:
    config = load_schemaorbit_config(CONFIG)
    splice = next(item for item in config.datasets if item.id == 46)
    assert splice.target == "Class"
    assert splice.ignore_attributes == ["Instance_name"]
    assert splice.predictors == 60


def test_configuration_rejects_unknown_keys(tmp_path: Path) -> None:
    raw = json.loads(json.dumps(load_schemaorbit_config(CONFIG).model_dump(), default=str))
    raw["unexpected"] = True
    import yaml

    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(SchemaOrbitError):
        load_schemaorbit_config(path)


def test_offline_cache_miss_never_requires_network(tmp_path: Path) -> None:
    config = load_schemaorbit_config(CONFIG)
    spec = next(item for item in config.datasets if item.id == 3)
    with pytest.raises(OfflineCacheMiss):
        acquire_raw(None, tmp_path, config, spec, offline=True, allow_network=False)


def test_dataset_registry_gitignore_policy() -> None:
    def ignored(path: str) -> bool:
        result = subprocess.run(
            ["git", "check-ignore", "--no-index", "--quiet", "--", path],
            cwd=ROOT,
            check=False,
        )
        return result.returncode == 0

    assert not ignored("schemas/example.schema.json")
    assert not ignored("tests/fixtures/example.csv")
    assert ignored("data/raw/example.csv")
    assert ignored("data/processed/example.parquet")
    assert ignored("results/example.csv")
    assert not ignored("artifacts/handoff/example.md")
