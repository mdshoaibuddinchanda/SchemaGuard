"""Materialize one transformation only after validating all written artifacts."""

from __future__ import annotations

import argparse
import ast
import json
import os
import shutil
import sys
import uuid
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from schemaguard.transformations.applicability import assess_applicability  # noqa: E402
from schemaguard.transformations.base import feature_schema_from_frame  # noqa: E402
from schemaguard.transformations.caching import (  # noqa: E402
    TransformationCache,
    transformation_cache_key,
)
from schemaguard.transformations.contracts import TransformationManifest  # noqa: E402
from schemaguard.transformations.registry import (  # noqa: E402
    get_transformation,
    load_transformation_config,
)
from schemaguard.transformations.serialization import (  # noqa: E402
    validate_manifest_directory,
    write_certificate,
    write_manifest_last,
    write_partition,
)
from schemaguard.transformations.validation import validate_transformation  # noqa: E402
from schemaguard.utils.hashing import (  # noqa: E402
    hash_dataframe_logically,
    sha256_canonical_json,
    sha256_file,
)
from schemaguard.utils.process_lock import ProcessLock  # noqa: E402


def _load_inputs(dataset_id: int) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object], Path]:
    processed = ROOT / "data/processed/openml" / str(dataset_id)
    features = pd.read_parquet(processed / "features.parquet")
    targets = pd.read_parquet(processed / "targets.parquet")
    schema_payload = json.loads((processed / "schema.json").read_text(encoding="utf-8"))
    categorical_values = {
        item["name"]: list(ast.literal_eval(item["raw_type"]))
        for item in schema_payload.get("attributes", [])
        if item.get("kind") == "categorical" and str(item.get("raw_type", "")).startswith("[")
    }
    return features, targets, categorical_values, processed


def _assignment(dataset_id: int, seed: int) -> pd.DataFrame:
    return pd.read_parquet(
        ROOT
        / "data/splits/openml"
        / str(dataset_id)
        / "stratified_group_5fold_v1"
        / f"seed_{seed}"
        / "assignments.parquet"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/runtime/transformation_engine.yaml"
    )
    parser.add_argument("--dataset-id", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--view", required=True)
    parser.add_argument("--output-directory", type=Path, default=ROOT / "data/transformed")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    config = load_transformation_config(args.config)
    features, targets, categorical_values, processed = _load_inputs(args.dataset_id)
    schema = feature_schema_from_frame(features, categorical_values)
    view = next(
        item
        for item in config.views
        if str(item["id"]) == args.view or str(item["name"]) == args.view
    )
    view_id = str(view["id"])
    applicability = assess_applicability(
        view_id, str(view["name"]), args.dataset_id, args.seed, schema
    )
    if applicability.status != "APPLICABLE":
        raise SystemExit(f"NOT_APPLICABLE: {applicability.reason_code}: {applicability.reason}")
    assignment = _assignment(args.dataset_id, args.seed)
    partition_column = "partition" if "partition" in assignment.columns else "split"
    feature_index = features.set_index("__sg_row_id", drop=False)
    target_index = targets.set_index("__sg_row_id", drop=False)
    sources = {}
    for partition in ("train", "calibration", "test"):
        row_ids = assignment.loc[assignment[partition_column] == partition, "__sg_row_id"].tolist()
        sources[partition] = (
            feature_index.loc[row_ids].reset_index(drop=True),
            target_index.loc[row_ids].reset_index(drop=True),
        )
    train, _ = sources["train"]
    transformation = get_transformation(view_id, config.model_dump())
    transformation.fit(train, args.dataset_id, args.seed, schema)
    final = (
        args.output_directory
        / "openml"
        / str(args.dataset_id)
        / "stratified_group_5fold_v1"
        / f"seed_{args.seed}"
        / str(view["name"])
    )
    lock = args.output_directory / ".locks" / f"{args.dataset_id}-{args.seed}-{view_id}.lock"
    cache = TransformationCache(args.output_directory / ".cache")
    with ProcessLock(lock, timeout=120):
        if final.exists():
            try:
                validate_manifest_directory(final, project_root=ROOT)
                print(f"validated_cache_hit:{final.relative_to(ROOT).as_posix()}")
                return 0
            except Exception as exc:
                quarantine = final.with_name(f"{final.name}.quarantine-{uuid.uuid4().hex}")
                shutil.move(str(final), str(quarantine))
                print(f"quarantined_invalid_materialization:{type(exc).__name__}")
        temporary = final.with_name(f".{final.name}.tmp-{uuid.uuid4().hex}")
        temporary.mkdir(parents=True, exist_ok=False)
        try:
            source_feature_hash = hash_dataframe_logically(features)
            source_target_hash = hash_dataframe_logically(targets)
            split_assignment_hash = hash_dataframe_logically(assignment)
            configuration_hash = transformation.configuration_hash()
            implementation_hash = transformation.implementation_hash()
            common_identity = {
                "dataset_id": args.dataset_id,
                "seed": args.seed,
                "view_id": view_id,
                "source_feature_hash": source_feature_hash,
                "source_target_hash": source_target_hash,
                "split_assignment_hash": split_assignment_hash,
                "configuration_hash": configuration_hash,
                "implementation_hash": implementation_hash,
            }
            cache_identity = sha256_canonical_json(common_identity)
            partition_paths: dict[str, str] = {}
            certificate_paths: dict[str, str] = {}
            source_hashes: dict[str, str] = {}
            target_hashes: dict[str, str] = {}
            transformed_feature_hashes: dict[str, str] = {}
            certificate_hashes: dict[str, str] = {}
            row_order_hashes: dict[str, str] = {}
            certificates = {}
            for partition, (source, target) in sources.items():
                transformed = transformation.transform(source, partition)
                target_hash = hash_dataframe_logically(target)
                certificate = transformation.certificate_for(
                    source,
                    transformed,
                    partition,
                    dataset_version="local",
                    split_strategy=config.split_strategy,
                    source_target_hash=target_hash,
                    output_target_hash=target_hash,
                )
                restored = transformation.reconstruct(transformed, certificate)
                validate_transformation(
                    source,
                    transformed,
                    restored,
                    certificate,
                    transformation=transformation,
                )
                feature_name = f"{partition}_features.parquet"
                certificate_name = f"{partition}_certificate.json"
                feature_path = temporary / feature_name
                certificate_path = temporary / certificate_name
                write_partition(feature_path, transformed)
                write_certificate(certificate_path, certificate)
                partition_paths[partition] = feature_name
                certificate_paths[partition] = certificate_name
                source_hashes[partition] = hash_dataframe_logically(source)
                target_hashes[partition] = target_hash
                transformed_feature_hashes[partition] = sha256_file(feature_path)
                certificate_hashes[partition] = sha256_file(certificate_path)
                row_order_hashes[partition] = hash_dataframe_logically(transformed[["__sg_row_id"]])
                fit_parameter_hash = sha256_canonical_json(transformation.parameters_for(partition))
                key = transformation_cache_key(
                    dataset_id=args.dataset_id,
                    dataset_version="local",
                    feature_hash=source_hashes[partition],
                    target_hash=target_hash,
                    split_logical_hash=split_assignment_hash,
                    partition=partition,
                    view_id=view_id,
                    view_config=config.model_dump(),
                    implementation_hash=implementation_hash,
                    fit_parameter_hash=fit_parameter_hash,
                    certificate_schema_version=certificate.schema_version,
                )
                cache.publish(
                    key,
                    feature_path,
                    certificate_path,
                    {"partition": partition, "cache_identity": cache_identity},
                )
                if cache.read_validated(key) is None:
                    raise ValueError("published cache entry failed independent validation")
                certificates[partition] = certificate
            manifest = TransformationManifest(
                schema_version=2,
                dataset_id=args.dataset_id,
                dataset_version="local",
                seed=args.seed,
                split_strategy=config.split_strategy,
                view_id=view_id,
                view_name=str(view["name"]),
                source_feature_path=(processed / "features.parquet").relative_to(ROOT).as_posix(),
                target_reference_path=(processed / "targets.parquet").relative_to(ROOT).as_posix(),
                split_assignment_path=(
                    ROOT
                    / "data/splits/openml"
                    / str(args.dataset_id)
                    / "stratified_group_5fold_v1"
                    / f"seed_{args.seed}"
                    / "assignments.parquet"
                )
                .relative_to(ROOT)
                .as_posix(),
                partition_feature_paths=partition_paths,
                certificate_paths=certificate_paths,
                source_hashes=source_hashes,
                target_hashes=target_hashes,
                source_feature_hash=source_feature_hash,
                source_target_hash=source_target_hash,
                split_assignment_hash=split_assignment_hash,
                transformed_feature_hashes=transformed_feature_hashes,
                certificate_hashes=certificate_hashes,
                partition_row_order_hashes=row_order_hashes,
                configuration_hash=configuration_hash,
                implementation_hash=implementation_hash,
                cache_identity=cache_identity,
                manifest_status="PASS",
                created_at=certificates["train"].created_at,
            )
            write_manifest_last(temporary / "manifest.json", manifest)
            validate_manifest_directory(temporary, project_root=ROOT)
            os.replace(temporary, final)
            validate_manifest_directory(final, project_root=ROOT)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
    print(final.relative_to(ROOT).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
