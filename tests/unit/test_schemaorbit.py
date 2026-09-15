"""Offline unit coverage for SchemaOrbit-14 acquisition contracts."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from schemaguard.data import schemaorbit
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


def test_legacy_smoke_validation_requires_ordered_row_alignment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_schemaorbit_config(CONFIG)
    spec = next(item for item in config.datasets if item.id == 1464)
    row_ids = [f"{index:032x}" for index in range(spec.rows)]
    features = pd.DataFrame({"__sg_row_id": row_ids, **{f"V{index}": 0.0 for index in range(1, 5)}})
    targets = pd.DataFrame(
        {
            "__sg_row_id": list(reversed(row_ids)),
            "target_label": ["1"] * 374 + ["2"] * 374,
            "target_code": [0] * 374 + [1] * 374,
        }
    )
    (tmp_path / "data_manifest.json").write_text(
        json.dumps(
            {
                "raw_sha256": "a" * 64,
                "feature_columns": ["V1", "V2", "V3", "V4"],
                "target_columns": ["__sg_row_id", "target_label", "target_code"],
                "artifact_hashes": {},
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "label_mapping.json").write_text(
        json.dumps(
            {
                "original_to_code": {"1": 0, "2": 1},
                "code_to_original": {"0": "1", "1": "2"},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(schemaorbit, "_read_processed", lambda _: (features, targets))
    with pytest.raises(SchemaOrbitError, match="identical in order"):
        schemaorbit._validate_legacy_smoke_processed(tmp_path, config, spec, "a" * 64)


def test_transactional_materialization_cleans_or_quarantines_write_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_schemaorbit_config(CONFIG)
    spec = next(item for item in config.datasets if item.id == 3)
    raw_path = tmp_path / "source.arff"
    raw_path.write_text("synthetic", encoding="utf-8")
    source_manifest = tmp_path / "data/raw/openml" / str(spec.id) / "source_manifest.json"
    source_manifest.parent.mkdir(parents=True, exist_ok=True)
    source_manifest.write_text('{"computed_sha256": "' + "b" * 64 + '"}\n', encoding="utf-8")
    features = pd.DataFrame({"__sg_row_id": ["a"], "feature": [1.0]})
    targets = pd.DataFrame({"__sg_row_id": ["a"], "target_label": ["0"], "target_code": [0]})
    monkeypatch.setattr(
        schemaorbit,
        "parse_arff",
        lambda _: SimpleNamespace(frame=pd.DataFrame(index=range(spec.rows))),
    )
    monkeypatch.setattr(
        schemaorbit,
        "_normalise_frame",
        lambda *args: (features, targets, {"schema": {"synthetic": True}, "classes": ["0"]}),
    )
    monkeypatch.setattr(
        schemaorbit,
        "_quality",
        lambda *_: {"row_count": spec.rows, "predictor_count": 1, "class_count": 1},
    )
    original_parquet = schemaorbit.atomic_write_parquet
    original_json = schemaorbit.atomic_write_json
    for fail_at in range(1, 7):
        counter = {"calls": 0}

        def parquet_writer(*args, **kwargs):
            counter["calls"] += 1
            if counter["calls"] == fail_at:
                raise OSError("injected materialization failure")
            return original_parquet(*args, **kwargs)

        def json_writer(*args, **kwargs):
            counter["calls"] += 1
            if counter["calls"] == fail_at:
                raise OSError("injected materialization failure")
            return original_json(*args, **kwargs)

        with monkeypatch.context() as context:
            context.setattr(schemaorbit, "atomic_write_parquet", parquet_writer)
            context.setattr(schemaorbit, "atomic_write_json", json_writer)
            with pytest.raises(OSError, match="injected materialization failure"):
                schemaorbit.process_dataset(
                    tmp_path,
                    config,
                    spec,
                    raw_path,
                    [],
                    {"computed_sha256": "b" * 64},
                )
        assert not list((tmp_path / "data/processed/openml").glob(".processed-*"))


@pytest.mark.parametrize("failure_kind", ["validation", "promotion"])
def test_transactional_materialization_handles_validation_and_promotion_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_kind: str
) -> None:
    config = load_schemaorbit_config(CONFIG)
    spec = next(item for item in config.datasets if item.id == 3)
    raw_path = tmp_path / "source.arff"
    raw_path.write_text("synthetic", encoding="utf-8")
    source_manifest = tmp_path / "data/raw/openml" / str(spec.id) / "source_manifest.json"
    source_manifest.parent.mkdir(parents=True, exist_ok=True)
    source_manifest.write_text('{"computed_sha256": "' + "b" * 64 + '"}\n', encoding="utf-8")
    features = pd.DataFrame({"__sg_row_id": ["a"], "feature": [1.0]})
    targets = pd.DataFrame(
        {"__sg_row_id": ["a"], "target_label": ["0"], "target_code": [0]}
    )
    monkeypatch.setattr(
        schemaorbit,
        "parse_arff",
        lambda _: SimpleNamespace(frame=pd.DataFrame(index=range(spec.rows))),
    )
    monkeypatch.setattr(
        schemaorbit,
        "_normalise_frame",
        lambda *args: (features, targets, {"schema": {"synthetic": True}, "classes": ["0"]}),
    )
    monkeypatch.setattr(
        schemaorbit,
        "_quality",
        lambda *_: {"row_count": spec.rows, "predictor_count": 1, "class_count": 1},
    )
    if failure_kind == "validation":
        monkeypatch.setattr(
            schemaorbit,
            "_validate_processed_directory",
            lambda *args: (_ for _ in ()).throw(SchemaOrbitError("injected validation failure")),
        )
    else:
        monkeypatch.setattr(schemaorbit, "_validate_processed_directory", lambda *args: None)
        original_replace = schemaorbit.os.replace

        def promotion_failure(source: str | bytes, destination: str | bytes) -> None:
            if Path(destination).name == str(spec.id):
                raise OSError("injected promotion failure")
            original_replace(source, destination)

        monkeypatch.setattr(schemaorbit.os, "replace", promotion_failure)
    with pytest.raises((OSError, SchemaOrbitError), match="injected"):
        schemaorbit.process_dataset(
            tmp_path,
            config,
            spec,
            raw_path,
            [],
            {"computed_sha256": "b" * 64},
        )
    assert not list((tmp_path / "data/processed/openml").glob(".processed-*"))
