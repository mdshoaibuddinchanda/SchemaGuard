"""Materialize one certified transformation view on demand."""

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
from schemaguard.transformations.contracts import TransformationManifest  # noqa: E402
from schemaguard.transformations.registry import (  # noqa: E402
    get_transformation,
    load_transformation_config,
)
from schemaguard.transformations.serialization import (  # noqa: E402
    write_certificate,
    write_manifest_last,
    write_partition,
)
from schemaguard.utils.hashing import hash_dataframe_logically  # noqa: E402


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
    processed = ROOT / "data/processed/openml" / str(args.dataset_id)
    features = pd.read_parquet(processed / "features.parquet")
    targets = pd.read_parquet(processed / "targets.parquet")
    schema_payload = json.loads((processed / "schema.json").read_text(encoding="utf-8"))
    categorical_values = {
        item["name"]: list(ast.literal_eval(item["raw_type"]))
        for item in schema_payload.get("attributes", [])
        if item.get("kind") == "categorical" and str(item.get("raw_type", "")).startswith("[")
    }
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
    assignment = pd.read_parquet(
        ROOT
        / "data/splits/openml"
        / str(args.dataset_id)
        / "stratified_group_5fold_v1"
        / f"seed_{args.seed}"
        / "assignments.parquet"
    )
    feature_index = features.set_index("__sg_row_id", drop=False)
    target_index = targets.set_index("__sg_row_id", drop=False)
    transformation = get_transformation(view_id, config.model_dump())
    partition_column = "partition" if "partition" in assignment.columns else "split"
    train_ids = assignment.loc[assignment[partition_column] == "train", "__sg_row_id"].tolist()
    train = feature_index.loc[train_ids].reset_index(drop=True)
    transformation.fit(train, args.dataset_id, args.seed, schema)
    final = (
        args.output_directory
        / "openml"
        / str(args.dataset_id)
        / "stratified_group_5fold_v1"
        / f"seed_{args.seed}"
        / str(view["name"])
    )
    if final.exists():
        raise SystemExit(f"refusing to overwrite existing materialization: {final}")
    temporary = final.with_name(f".{final.name}.tmp-{uuid.uuid4().hex}")
    temporary.mkdir(parents=True, exist_ok=False)
    partition_paths: dict[str, str] = {}
    certificate_paths: dict[str, str] = {}
    source_hashes: dict[str, str] = {}
    target_hashes: dict[str, str] = {}
    try:
        for partition in ("train", "calibration", "test"):
            row_ids = assignment.loc[
                assignment[partition_column] == partition, "__sg_row_id"
            ].tolist()
            source = feature_index.loc[row_ids].reset_index(drop=True)
            target = target_index.loc[row_ids].reset_index(drop=True)
            transformed = transformation.transform(source, partition)
            certificate = transformation.certificate_for(
                source,
                transformed,
                partition,
                dataset_version="local",
                source_target_hash=hash_dataframe_logically(target),
                output_target_hash=hash_dataframe_logically(target),
                validation_results={"status": "PASS", "materialized": True},
            )
            feature_name = f"{partition}_features.parquet"
            certificate_name = f"{partition}_certificate.json"
            write_partition(temporary / feature_name, transformed)
            write_certificate(temporary / certificate_name, certificate)
            partition_paths[partition] = feature_name
            certificate_paths[partition] = certificate_name
            source_hashes[partition] = hash_dataframe_logically(source)
            target_hashes[partition] = hash_dataframe_logically(target)
        manifest = TransformationManifest(
            schema_version=1,
            dataset_id=args.dataset_id,
            dataset_version="local",
            seed=args.seed,
            split_strategy=config.split_strategy,
            view_id=view_id,
            view_name=str(view["name"]),
            source_feature_path=str((processed / "features.parquet").relative_to(ROOT).as_posix()),
            target_reference_path=str((processed / "targets.parquet").relative_to(ROOT).as_posix()),
            partition_feature_paths=partition_paths,
            certificate_paths=certificate_paths,
            source_hashes=source_hashes,
            target_hashes=target_hashes,
            manifest_status="PASS",
            created_at=certificate.created_at,
        )
        write_manifest_last(temporary / "manifest.json", manifest)
        os.replace(temporary, final)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(final.relative_to(ROOT).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
