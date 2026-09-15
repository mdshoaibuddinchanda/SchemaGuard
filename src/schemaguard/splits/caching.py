"""Content-addressed identities for split artifacts."""

from __future__ import annotations

from typing import Any

from schemaguard.utils.hashing import sha256_canonical_json

from .contracts import CacheIdentityContract


def cache_identity(
    dataset_id: int,
    dataset_version: str,
    feature_artifact_hash: str,
    target_artifact_hash: str,
    dataset_manifest_hash: str,
    seed: int,
    strategy: str,
    strategy_version: str,
    grouping_implementation_hash: str,
    split_configuration_hash: str,
    source_commit: str,
    artifact_schema_version: int = 1,
) -> dict[str, Any]:
    identity = {
        "dataset_id": dataset_id,
        "dataset_version": dataset_version,
        "feature_artifact_hash": feature_artifact_hash,
        "target_artifact_hash": target_artifact_hash,
        "dataset_manifest_hash": dataset_manifest_hash,
        "seed": seed,
        "strategy": strategy,
        "strategy_version": strategy_version,
        "grouping_implementation_hash": grouping_implementation_hash,
        "split_configuration_hash": split_configuration_hash,
        "source_commit": source_commit,
        "artifact_schema_version": artifact_schema_version,
    }
    return CacheIdentityContract.model_validate(identity).canonical_dict()


def cache_key(identity: dict[str, Any]) -> str:
    return sha256_canonical_json(identity)


def logical_assignment_hash(rows: list[dict[str, Any]]) -> str:
    canonical = sorted(
        (
            {
                "__sg_row_id": str(row["__sg_row_id"]),
                "predictor_group_id": str(row["predictor_group_id"]),
                "partition": str(row["partition"]),
            }
            for row in rows
        ),
        key=lambda row: row["__sg_row_id"],
    )
    return sha256_canonical_json(canonical)
