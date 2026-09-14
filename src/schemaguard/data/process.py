"""Conservative processing of the smoke dataset into immutable artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import pandas as pd

from schemaguard.constants import (
    DEFAULT_TARGET_NAME,
    OPENML_DATA_ID,
    RESERVED_COLUMN_PREFIX,
    ROW_ID_COLUMN,
    TARGET_CODE_COLUMN,
    TARGET_LABEL_COLUMN,
)
from schemaguard.utils.hashing import sha256_bytes, sha256_canonical_json, sha256_file
from schemaguard.utils.io import atomic_write_json, atomic_write_parquet

from .arff_parser import ParsedArff
from .contracts import ProcessedSchema, SmokeDatasetConfig


class ProcessingError(ValueError):
    """Raised when conservative processing cannot preserve the source contract."""


@dataclass(frozen=True)
class ProcessedArtifacts:
    features: pd.DataFrame
    targets: pd.DataFrame
    features_path: Path
    targets_path: Path
    schema_path: Path
    label_mapping_path: Path
    data_manifest_path: Path
    artifact_hashes: dict[str, str]
    data_manifest_hash: str


def row_id_for_position(dataset_id: int, raw_sha256: str, position: int) -> str:
    """Build the frozen 32-character row ID for a source row."""
    if position < 0:
        raise ValueError("source row position must be non-negative")
    payload = f"openml:{dataset_id}:{raw_sha256}:{position}".encode()
    return sha256_bytes(payload)[:32]


def _canonical_label(value: Any) -> str:
    if value is None or pd.isna(value):
        raise ProcessingError("Target contains a missing value")
    return str(value)


def _processing_config_hash(config: SmokeDatasetConfig) -> str:
    return sha256_canonical_json(config.processing.canonical_dict())


def process_dataset(
    parsed: ParsedArff,
    config: SmokeDatasetConfig,
    raw_sha256: str,
    output_dir: str | Path,
    source_manifest_hash: str,
) -> ProcessedArtifacts:
    """Create feature/target tables, mappings, schema, and a lineage manifest."""
    frame = parsed.frame.copy(deep=True)
    source_columns = [item.name for item in parsed.attributes]
    reserved = [name for name in source_columns if name.startswith(RESERVED_COLUMN_PREFIX)]
    if reserved:
        raise ProcessingError(
            f"Source columns use reserved prefix {RESERVED_COLUMN_PREFIX}: {reserved}"
        )
    target_name = config.task.expected_target_name
    if target_name != DEFAULT_TARGET_NAME or target_name not in source_columns:
        raise ProcessingError(
            f"Expected target {target_name!r} is not present in the source schema"
        )

    feature_columns = [name for name in source_columns if name != target_name]
    row_ids = [
        row_id_for_position(OPENML_DATA_ID, raw_sha256, position)
        for position in parsed.source_row_positions
    ]
    if len(row_ids) != len(set(row_ids)):
        raise ProcessingError("Generated row IDs are not unique")

    original_target = frame[target_name].map(_canonical_label)
    labels = sorted(original_target.unique().tolist())
    expected_classes = config.expected.classes
    if len(labels) != expected_classes:
        raise ProcessingError(f"Expected {expected_classes} target classes, found {len(labels)}")
    mapping = {label: code for code, label in enumerate(labels)}

    features = frame[feature_columns].copy()
    for attribute in parsed.attributes:
        if attribute.is_target or attribute.name not in feature_columns:
            continue
        if attribute.kind == "numeric":
            try:
                features[attribute.name] = pd.to_numeric(features[attribute.name], errors="raise")
            except (TypeError, ValueError) as exc:
                raise ProcessingError(
                    f"Numeric feature {attribute.name!r} cannot be represented numerically"
                ) from exc
    features.insert(0, ROW_ID_COLUMN, row_ids)
    targets = pd.DataFrame(
        {
            ROW_ID_COLUMN: row_ids,
            TARGET_LABEL_COLUMN: original_target.astype("string"),
            TARGET_CODE_COLUMN: original_target.map(mapping).astype("int64"),
        }
    )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    features_path = output / "features.parquet"
    targets_path = output / "targets.parquet"
    schema_path = output / "schema.json"
    label_mapping_path = output / "label_mapping.json"
    data_manifest_path = output / "data_manifest.json"
    atomic_write_parquet(
        features_path,
        features,
        compression=config.processing.parquet_compression,
        compression_level=config.processing.parquet_compression_level,
    )
    atomic_write_parquet(
        targets_path,
        targets,
        compression=config.processing.parquet_compression,
        compression_level=config.processing.parquet_compression_level,
    )

    roundtrip_features = pd.read_parquet(features_path)
    roundtrip_targets = pd.read_parquet(targets_path)
    pd.testing.assert_frame_equal(
        features, roundtrip_features, check_dtype=True, check_index_type=True
    )
    pd.testing.assert_frame_equal(
        targets, roundtrip_targets, check_dtype=True, check_index_type=True
    )

    attributes = [
        attribute.model_copy(update={"is_target": attribute.name == target_name})
        for attribute in parsed.attributes
    ]
    schema = ProcessedSchema(
        internal_dataset_id=config.dataset.internal_id,
        raw_sha256=raw_sha256,
        row_id_column=cast(Literal["__sg_row_id"], ROW_ID_COLUMN),
        target_column=target_name,
        feature_columns=feature_columns,
        attributes=attributes,
        feature_row_count=len(features),
        target_row_count=len(targets),
    )
    atomic_write_json(schema_path, schema.canonical_dict())
    atomic_write_json(
        label_mapping_path,
        {
            "schema_version": 1,
            "raw_sha256": raw_sha256,
            "original_to_code": mapping,
            "code_to_original": {str(code): label for label, code in mapping.items()},
        },
    )

    artifact_hashes = {
        "features.parquet": sha256_file(features_path),
        "targets.parquet": sha256_file(targets_path),
        "schema.json": sha256_file(schema_path),
        "label_mapping.json": sha256_file(label_mapping_path),
    }
    manifest_payload = {
        "schema_version": 1,
        "internal_dataset_id": config.dataset.internal_id,
        "openml_data_id": config.dataset.openml_data_id,
        "raw_sha256": raw_sha256,
        "source_manifest_sha256": source_manifest_hash,
        "processing_config_sha256": _processing_config_hash(config),
        "row_id_formula": "sha256(openml:1464:raw_file_sha256:source_row_position)[:32]",
        "row_count": len(features),
        "feature_columns": feature_columns,
        "target_columns": [ROW_ID_COLUMN, TARGET_LABEL_COLUMN, TARGET_CODE_COLUMN],
        "artifact_hashes": artifact_hashes,
        "compression": config.processing.parquet_compression,
        "compression_level": config.processing.parquet_compression_level,
    }
    atomic_write_json(data_manifest_path, manifest_payload)
    return ProcessedArtifacts(
        features=features,
        targets=targets,
        features_path=features_path,
        targets_path=targets_path,
        schema_path=schema_path,
        label_mapping_path=label_mapping_path,
        data_manifest_path=data_manifest_path,
        artifact_hashes={**artifact_hashes, "data_manifest.json": sha256_file(data_manifest_path)},
        data_manifest_hash=sha256_file(data_manifest_path),
    )
