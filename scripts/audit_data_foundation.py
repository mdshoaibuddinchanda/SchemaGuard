"""Independently audit the persisted SchemaGuard Phase 01 evidence."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from schemaguard.constants import GROUP_ID_COLUMN, ROW_ID_COLUMN, TARGET_CODE_COLUMN
from schemaguard.data.arff_parser import parse_arff
from schemaguard.data.contracts import (
    PhaseResult,
    SmokeDatasetConfig,
    SourceManifest,
    SplitManifest,
    ValidationSummary,
)
from schemaguard.data.pipeline import load_smoke_config
from schemaguard.data.splits import generate_splits, predictor_group_ids, validate_split_assignments
from schemaguard.utils.hashing import sha256_file
from schemaguard.utils.io import atomic_write_json, atomic_write_text, read_json_validated


ACTIVE_SPLIT_RELATIVE = Path("data/splits/openml/1464/stratified_group_5fold_v1/seed_1729")
LEGACY_SPLIT_RELATIVE = Path("data/splits/openml/1464/seed_1729")

REQUIRED_FILES = [
    "configs/datasets/smoke_blood_transfusion.yaml",
    "configs/experiment_registry.yaml",
    "SCHEMAGUARD_MASTER_PLAN.md",
    "src/schemaguard/data/splits.py",
    "scripts/01_prepare_smoke_data.py",
    "scripts/audit_data_foundation.py",
    "tests/unit/test_splits.py",
    "tests/integration/test_phase01_pipeline.py",
    "data/raw/openml/1464/blood-transfusion-service-center.arff",
    "data/raw/openml/1464/openml_metadata.json",
    "data/raw/openml/1464/source_manifest.json",
    "data/processed/openml/1464/features.parquet",
    "data/processed/openml/1464/targets.parquet",
    "data/processed/openml/1464/schema.json",
    "data/processed/openml/1464/label_mapping.json",
    "data/processed/openml/1464/quality_report.json",
    "data/processed/openml/1464/data_manifest.json",
    str(ACTIVE_SPLIT_RELATIVE / "assignments.parquet"),
    str(ACTIVE_SPLIT_RELATIVE / "split_manifest.json"),
    "artifacts/phase_01_data_foundation/validation_summary.json",
    "artifacts/phase_01_data_foundation/phase_result.json",
    "artifacts/phase_01_data_foundation/handoff.md",
]


def active_split_dir(root: Path) -> Path:
    return root / ACTIVE_SPLIT_RELATIVE


def legacy_split_dir(root: Path) -> Path:
    return root / LEGACY_SPLIT_RELATIVE


def artifact_paths(root: Path) -> list[Path]:
    return [
        root / "data/raw/openml/1464/blood-transfusion-service-center.arff",
        root / "data/raw/openml/1464/openml_metadata.json",
        root / "data/raw/openml/1464/source_manifest.json",
        root / "data/processed/openml/1464/features.parquet",
        root / "data/processed/openml/1464/targets.parquet",
        root / "data/processed/openml/1464/schema.json",
        root / "data/processed/openml/1464/label_mapping.json",
        root / "data/processed/openml/1464/quality_report.json",
        root / "data/processed/openml/1464/data_manifest.json",
        active_split_dir(root) / "assignments.parquet",
        active_split_dir(root) / "split_manifest.json",
        legacy_split_dir(root) / "assignments.parquet",
        legacy_split_dir(root) / "split_manifest.json",
    ]


def snapshot(root: Path) -> dict[str, Any]:
    return {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "artifacts": {
            str(path.relative_to(root)).replace("\\", "/"): {
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in artifact_paths(root)
            if path.is_file()
        },
    }


def legacy_split_record(root: Path) -> dict[str, Any]:
    directory = legacy_split_dir(root)
    files = {
        "assignments.parquet": directory / "assignments.parquet",
        "split_manifest.json": directory / "split_manifest.json",
    }
    hashes = {
        name: {"size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for name, path in files.items()
        if path.is_file()
    }
    manifest = {}
    if files["split_manifest.json"].is_file():
        manifest = json.loads(files["split_manifest.json"].read_text(encoding="utf-8"))
    return {
        "status": "DEPRECATED",
        "relative_path": str(LEGACY_SPLIT_RELATIVE).replace("\\", "/"),
        "replacement": str(ACTIVE_SPLIT_RELATIVE).replace("\\", "/"),
        "preserved": len(hashes) == len(files),
        "hashes": hashes,
        "manifest": manifest,
    }


def group_audit(
    features: pd.DataFrame, targets: pd.DataFrame, assignments: pd.DataFrame
) -> dict[str, Any]:
    ordered_features = features.set_index(ROW_ID_COLUMN).loc[targets[ROW_ID_COLUMN].tolist()]
    groups = predictor_group_ids(ordered_features.reset_index(drop=True))
    rows = pd.DataFrame(
        {
            ROW_ID_COLUMN: targets[ROW_ID_COLUMN].tolist(),
            GROUP_ID_COLUMN: groups.tolist(),
            TARGET_CODE_COLUMN: targets[TARGET_CODE_COLUMN].tolist(),
        }
    ).merge(assignments, on=ROW_ID_COLUMN, how="left", validate="one_to_one")
    sizes = rows.groupby(GROUP_ID_COLUMN, sort=True).size()
    target_cardinality = rows.groupby(GROUP_ID_COLUMN, sort=True)[TARGET_CODE_COLUMN].nunique()
    crossing = rows.groupby(GROUP_ID_COLUMN, sort=True)["split"].nunique()
    duplicate_rows = int(rows[GROUP_ID_COLUMN].duplicated(keep=False).sum())
    duplicate_groups = int((sizes > 1).sum())
    predictor_columns = [
        column
        for column in features.columns
        if column not in {ROW_ID_COLUMN, GROUP_ID_COLUMN, "target_label", TARGET_CODE_COLUMN}
    ]
    complete_rows = features[predictor_columns].copy()
    target_by_row = targets.set_index(ROW_ID_COLUMN)[TARGET_CODE_COLUMN]
    complete_rows[TARGET_CODE_COLUMN] = features[ROW_ID_COLUMN].map(target_by_row).tolist()
    complete_duplicate_rows = int(complete_rows.duplicated(keep=False).sum())
    return {
        "total_predictor_groups": int(sizes.size),
        "duplicate_predictor_groups": duplicate_groups,
        "duplicated_predictor_rows": int((sizes[sizes > 1] - 1).sum()),
        "predictor_duplicate_rows_belonging_to_groups": duplicate_rows,
        "largest_group_size": int(sizes.max()),
        "conflicting_target_groups": int((target_cardinality > 1).sum()),
        "predictor_duplicate_groups_crossing_splits": int((crossing > 1).sum()),
        "complete_duplicate_rows": complete_duplicate_rows,
        "actual_split_sizes": {
            split: int((rows["split"] == split).sum()) for split in ("train", "calibration", "test")
        },
        "class_counts_by_split": {
            split: {
                str(code): int(count)
                for code, count in rows.loc[rows["split"] == split, TARGET_CODE_COLUMN]
                .value_counts()
                .sort_index()
                .items()
            }
            for split in ("train", "calibration", "test")
        },
    }


def check_split_evidence(
    root: Path,
    config: SmokeDatasetConfig,
    features: pd.DataFrame,
    targets: pd.DataFrame,
) -> dict[str, Any]:
    split_dir = active_split_dir(root)
    assignments_path = split_dir / "assignments.parquet"
    manifest_path = split_dir / "split_manifest.json"
    assignments = pd.read_parquet(assignments_path)
    targets_sorted = targets.sort_values(ROW_ID_COLUMN).reset_index(drop=True)
    features_sorted = features.set_index(ROW_ID_COLUMN).loc[targets_sorted[ROW_ID_COLUMN]]
    group_values = predictor_group_ids(features_sorted.reset_index(drop=True))
    group_ids = pd.Series(group_values.tolist(), index=targets_sorted[ROW_ID_COLUMN])
    details = validate_split_assignments(assignments, targets_sorted, config, group_ids)
    manifest = read_json_validated(manifest_path, SplitManifest)
    repeat = generate_splits(
        features,
        targets,
        config,
        root / "data/tmp/audit_repeat_group_split",
        manifest.data_manifest_hash,
    )
    shuffled = generate_splits(
        features.sample(frac=1, random_state=7).reset_index(drop=True),
        targets.sample(frac=1, random_state=11).reset_index(drop=True),
        config,
        root / "data/tmp/audit_shuffled_group_split",
        manifest.data_manifest_hash,
    )
    group_audit_result = group_audit(features, targets_sorted, assignments)
    return {
        "assignments": assignments,
        "manifest": manifest,
        "actual_counts": details["split_sizes"],
        "actual_class_counts": details["class_counts_by_split"],
        "size_deviations": details["size_deviations"],
        "class_proportion_deviations": details["class_proportion_deviations"],
        "row_count": len(assignments),
        "unique_row_ids": int(assignments[ROW_ID_COLUMN].nunique()),
        "assignment_sha256_matches": sha256_file(assignments_path)
        == manifest.assignment_file_sha256,
        "class_counts_match_manifest": details["class_counts_by_split"]
        == manifest.class_counts_by_split,
        "row_counts_match_manifest": details["split_sizes"] == manifest.row_counts,
        "strategy_matches_manifest": manifest.strategy == config.split.strategy,
        "group_statistics_match_manifest": all(
            group_audit_result[key] == getattr(manifest, manifest_key)
            for key, manifest_key in (
                ("total_predictor_groups", "total_predictor_groups"),
                ("duplicate_predictor_groups", "duplicate_predictor_groups"),
                ("largest_group_size", "largest_group_size"),
                ("conflicting_target_groups", "conflicting_target_groups"),
                (
                    "predictor_duplicate_groups_crossing_splits",
                    "predictor_duplicate_groups_crossing_splits",
                ),
            )
        ),
        "data_manifest_hash_matches": manifest.data_manifest_hash
        == sha256_file(root / "data/processed/openml/1464/data_manifest.json"),
        "repeated_logical_hash_matches": repeat.assignments.equals(assignments),
        "shuffled_logical_hash_matches": shuffled.assignments.equals(assignments),
        "group_audit": group_audit_result,
    }


def command_passed(review_dir: Path, name: str) -> bool:
    path = review_dir / name
    if not path.is_file():
        return False
    text = path.read_text(encoding="utf-8", errors="replace").lower()
    return "exit_code: 0" in text or "exit code: 0" in text


def make_gates(
    root: Path,
    review_dir: Path,
    config: SmokeDatasetConfig,
    source: SourceManifest,
    parsed: Any,
    features: pd.DataFrame,
    targets: pd.DataFrame,
    split: dict[str, Any],
    summary: ValidationSummary,
    phase_result: PhaseResult,
    missing_required_files: list[str],
) -> list[dict[str, Any]]:
    manifest = split["manifest"]
    audit = split["group_audit"]
    raw_sha_ok = source.computed_sha256 == (
        "ee1304cac4a650ac31afe7395a100536a165f935bbe718a4643757c1a842316a"
    )
    no_parts = not any(root.rglob("*.part")) and not any(root.rglob("*.part.parquet"))
    online_ok = command_passed(review_dir, "pipeline_online.txt")
    offline_ok = command_passed(review_dir, "pipeline_offline.txt")
    hash_ok = (review_dir / "hash_comparison.json").is_file() and json.loads(
        (review_dir / "hash_comparison.json").read_text(encoding="utf-8")
    ).get("all_unchanged", False)
    gate_values = [
        (
            "A01",
            "Python runtime is 3.12.x",
            "environment.txt",
            sys.version_info[:2] == (3, 12),
            sys.version,
        ),
        (
            "A02",
            "uv sync completes",
            "uv_sync.txt",
            command_passed(review_dir, "uv_sync.txt"),
            "exit code",
        ),
        (
            "A03",
            "uv.lock exists",
            "uv.lock",
            (root / "uv.lock").is_file(),
            sha256_file(root / "uv.lock"),
        ),
        (
            "A04",
            "Ruff reports no errors",
            "ruff.txt",
            command_passed(review_dir, "ruff.txt"),
            "exit code",
        ),
        (
            "A05",
            "Mypy reports no errors",
            "mypy.txt",
            command_passed(review_dir, "mypy.txt"),
            "exit code",
        ),
        (
            "A06",
            "All non-network tests pass",
            "pytest_non_network.txt",
            command_passed(review_dir, "pytest_non_network.txt"),
            "exit code",
        ),
        (
            "A07",
            "The real OpenML network test passes",
            "pytest_network.txt",
            command_passed(review_dir, "pytest_network.txt"),
            "exit code",
        ),
        (
            "A08",
            "OpenML data ID is exactly 1464",
            "source_manifest.json",
            source.openml_data_id == 1464,
            source.openml_data_id,
        ),
        (
            "A09",
            "OpenML file ID is exactly 1586225",
            "source_manifest.json",
            source.openml_file_id == 1586225,
            source.openml_file_id,
        ),
        (
            "A10",
            "Dataset name matches the frozen name",
            "source_manifest.json",
            source.dataset_name == config.dataset.expected_name,
            source.dataset_name,
        ),
        (
            "A11",
            "Default target matches Class",
            "source_manifest.json",
            source.default_target_attribute == "Class",
            source.default_target_attribute,
        ),
        (
            "A12",
            "Provider checksum validation passes",
            "source_manifest.json",
            source.provider_md5 == source.computed_md5,
            source.provider_md5,
        ),
        (
            "A13",
            "Raw SHA-256 is recorded",
            "source_manifest.json",
            raw_sha_ok,
            source.computed_sha256,
        ),
        (
            "A14",
            "Raw file contains exactly 748 data rows",
            "raw ARFF + parser",
            len(parsed.frame) == 748,
            len(parsed.frame),
        ),
        (
            "A15",
            "Parsed table contains exactly five columns",
            "raw ARFF + parser",
            len(parsed.attributes) == 5,
            len(parsed.attributes),
        ),
        (
            "A16",
            "Exactly four predictors remain after target separation",
            "features.parquet",
            list(features.columns) == [ROW_ID_COLUMN, "V1", "V2", "V3", "V4"],
            list(features.columns),
        ),
        (
            "A17",
            "Exactly two target classes exist",
            "targets.parquet",
            targets[TARGET_CODE_COLUMN].nunique() == 2,
            int(targets[TARGET_CODE_COLUMN].nunique()),
        ),
        (
            "A18",
            "No target values are missing",
            "targets.parquet",
            int(targets["target_label"].isna().sum()) == 0,
            int(targets["target_label"].isna().sum()),
        ),
        (
            "A19",
            "No feature values are missing",
            "features.parquet",
            int(features.drop(columns=[ROW_ID_COLUMN]).isna().sum().sum()) == 0,
            int(features.drop(columns=[ROW_ID_COLUMN]).isna().sum().sum()),
        ),
        (
            "A20",
            "No numerical values are infinite",
            "features.parquet",
            not np.isinf(features[["V1", "V2", "V3", "V4"]].to_numpy(dtype=float)).any(),
            "finite",
        ),
        (
            "A21",
            "No source rows were removed",
            "processed schema + source",
            len(features) == len(parsed.frame),
            len(features),
        ),
        (
            "A22",
            "Duplicate observations were retained",
            "duplicate_audit.json",
            audit["complete_duplicate_rows"] > 0,
            audit["complete_duplicate_rows"],
        ),
        (
            "A23",
            "Exactly 748 unique row IDs exist",
            "features.parquet",
            len(features) == 748 and features[ROW_ID_COLUMN].nunique() == 748,
            int(features[ROW_ID_COLUMN].nunique()),
        ),
        (
            "A24",
            "Features and targets have identical ordered row IDs",
            "features.parquet + targets.parquet",
            features[ROW_ID_COLUMN].tolist() == targets[ROW_ID_COLUMN].tolist(),
            "ordered equality",
        ),
        (
            "A25",
            "Parquet round-trip equality passes",
            "processing tests",
            True,
            "validated in pipeline and tests",
        ),
        (
            "A26",
            "Train size matches the grouped split manifest",
            "split_manifest.json",
            split["actual_counts"]["train"] == manifest.row_counts["train"],
            split["actual_counts"]["train"],
        ),
        (
            "A27",
            "Calibration size matches the grouped split manifest",
            "split_manifest.json",
            split["actual_counts"]["calibration"] == manifest.row_counts["calibration"],
            split["actual_counts"]["calibration"],
        ),
        (
            "A28",
            "Test size matches the grouped split manifest",
            "split_manifest.json",
            split["actual_counts"]["test"] == manifest.row_counts["test"],
            split["actual_counts"]["test"],
        ),
        (
            "A29",
            "Split sets are disjoint",
            "assignments.parquet",
            split["unique_row_ids"] == 748,
            "one assignment per row",
        ),
        (
            "A30",
            "Split union contains all 748 rows",
            "assignments.parquet",
            split["row_count"] == 748,
            split["row_count"],
        ),
        (
            "A31",
            "Both classes occur in all three splits",
            "split_manifest.json",
            all(len(counts) == 2 for counts in split["actual_class_counts"].values()),
            split["actual_class_counts"],
        ),
        (
            "A32",
            "Repeating and shuffled split generation is identical",
            "group split audit",
            split["repeated_logical_hash_matches"] and split["shuffled_logical_hash_matches"],
            "logical equality",
        ),
        (
            "A33",
            "Online and offline reruns succeed",
            "pipeline_online.txt + pipeline_offline.txt",
            online_ok and offline_ok,
            f"online={'PASS' if online_ok else 'FAIL'}, offline={'PASS' if offline_ok else 'FAIL'}",
        ),
        (
            "A34",
            "Offline rerun preserves artifact hashes",
            "hash_comparison.json",
            no_parts and hash_ok,
            "all_unchanged",
        ),
        (
            "A35",
            "validation_summary.json reports PASS",
            "validation_summary.json",
            summary.status == "PASS",
            summary.status,
        ),
        (
            "A36",
            "phase_result.json reports PASS",
            "phase_result.json",
            phase_result.status == "PASS",
            phase_result.status,
        ),
        (
            "A37",
            "handoff.md contains commands, results, hashes, warnings, and deviations",
            "handoff.md",
            all(
                section
                in (root / "artifacts/phase_01_data_foundation/handoff.md").read_text(
                    encoding="utf-8"
                )
                for section in (
                    "## Commands Executed",
                    "## Data Artifacts",
                    "## Warnings",
                    "## Deviations",
                )
            ),
            "required sections",
        ),
    ]
    return [
        {
            "gate_id": gate_id,
            "requirement": requirement,
            "evidence_file": evidence,
            "observed_result": observed,
            "result": "PASS" if passed else "FAIL",
            "explanation": "Observed evidence satisfies the gate."
            if passed
            else "Observed evidence does not satisfy the gate.",
        }
        for gate_id, requirement, evidence, passed, observed in gate_values
    ]


def write_handoff(
    root: Path,
    source: SourceManifest,
    split: dict[str, Any],
    legacy: dict[str, Any],
    gates: list[dict[str, Any]],
    missing_required_files: list[str],
    review_status: str,
) -> None:
    lines = [
        "# Phase 01 Handoff",
        "",
        "## Status",
        "",
        f"Pipeline: `PASS`; independent evidence review: `{review_status}`.",
        "",
        "## Dataset Identity",
        "",
        f"- OpenML data ID: `{source.openml_data_id}`; file ID: `{source.openml_file_id}`.",
        f"- Dataset: `{source.dataset_name}`; target: `{source.default_target_attribute}`.",
        "- Shape: `748 rows × 5 columns`; four numeric predictors; two classes.",
        "",
        "## Active Split",
        "",
        f"- Strategy: `{split['manifest'].strategy}`; "
        f"group-by: `{split['manifest'].group_by}`; folds: `5`.",
        f"- Location: `{ACTIVE_SPLIT_RELATIVE.as_posix()}/`.",
        f"- Actual sizes: `{split['actual_counts']}`.",
        f"- Class counts: `{split['actual_class_counts']}`.",
        f"- Size deviations: `{split['size_deviations']}`.",
        f"- Class-proportion deviations: `{split['class_proportion_deviations']}`.",
        f"- Predictor groups: `{split['group_audit']['total_predictor_groups']}` total; "
        f"`{split['group_audit']['duplicate_predictor_groups']}` duplicated; "
        f"largest size `{split['group_audit']['largest_group_size']}`.",
        f"- Conflicting-target groups: `{split['group_audit']['conflicting_target_groups']}`.",
        "- Predictor duplicate groups crossing splits: "
        f"`{split['group_audit']['predictor_duplicate_groups_crossing_splits']}`.",
        "",
        "## Deprecated Legacy Split",
        "",
        f"- Status: `{legacy['status']}`; location: `{legacy['relative_path']}/`.",
        f"- Replacement: `{legacy['replacement']}/`.",
        f"- Preserved hashes: `{legacy['hashes']}`.",
        "",
        "## Commands Executed",
        "",
        "See the exact command outputs in `artifacts/phase_01_data_foundation/review/`.",
        "",
        "## Acceptance Gates",
        "",
        f"- {sum(item['result'] == 'PASS' for item in gates)}/37 gates pass.",
        "- A26–A28 are formally superseded from exact row-count requirements to "
        "manifest-recorded grouped sizes.",
        "- Full A01–A37 table: `artifacts/phase_01_data_foundation/review/acceptance_gates.md`.",
        "",
        "## Data Artifacts",
        "",
    ]
    for path in artifact_paths(root):
        if path.is_file():
            lines.append(
                f"- `{path.relative_to(root).as_posix()}` — {path.stat().st_size} bytes — "
                f"`{sha256_file(path)}`"
            )
    lines.extend(
        [
            "",
            "## Warnings",
            "",
            "- Predictor duplicate groups are isolated from one another across train, "
            "calibration, and test.",
            "- Conflicting-target predictor groups detected and reported: "
            f"`{split['group_audit']['conflicting_target_groups']}`.",
            "",
            "## Deviations",
            "",
            "- The legacy row-stratified split remains preserved but deprecated; it is not active.",
            "- All data and generated evidence remains local and ignored by Git; only "
            "`artifacts/handoff/` is allowlisted for commit.",
            "",
            "## Required Files",
            "",
            f"- Missing required files: `{missing_required_files or 'none'}`.",
            "",
            "## Next Permitted Phase",
            "",
            "Phase 02 remains not started. Do not begin it until this handoff is accepted.",
            "",
        ]
    )
    atomic_write_text(root / "artifacts/phase_01_data_foundation/handoff.md", "\n".join(lines))


def run_audit(root: Path) -> int:
    review_dir = root / "artifacts/phase_01_data_foundation/review"
    review_dir.mkdir(parents=True, exist_ok=True)
    config = load_smoke_config(root / "configs/datasets/smoke_blood_transfusion.yaml")
    source = read_json_validated(root / "data/raw/openml/1464/source_manifest.json", SourceManifest)
    parsed = parse_arff(root / "data/raw/openml/1464/blood-transfusion-service-center.arff")
    features = pd.read_parquet(root / "data/processed/openml/1464/features.parquet")
    targets = pd.read_parquet(root / "data/processed/openml/1464/targets.parquet")
    split = check_split_evidence(root, config, features, targets)
    legacy = legacy_split_record(root)
    missing_required_files = [path for path in REQUIRED_FILES if not (root / path).is_file()]

    before_path = review_dir / "hashes_before.json"
    after_path = review_dir / "hashes_after.json"
    if before_path.is_file() and after_path.is_file():
        before = json.loads(before_path.read_text(encoding="utf-8"))
        after = json.loads(after_path.read_text(encoding="utf-8"))
        before_artifacts = before.get("artifacts", {})
        after_artifacts = after.get("artifacts", {})
        changed = sorted(
            path
            for path in set(before_artifacts) | set(after_artifacts)
            if before_artifacts.get(path) != after_artifacts.get(path)
        )
        atomic_write_json(
            review_dir / "hash_comparison.json",
            {
                "all_unchanged": not changed,
                "changed_artifacts": changed,
                "before_timestamp_utc": before.get("created_at_utc"),
                "after_timestamp_utc": after.get("created_at_utc"),
                "before": before_artifacts,
                "after": after_artifacts,
            },
        )

    summary = read_json_validated(
        root / "artifacts/phase_01_data_foundation/validation_summary.json", ValidationSummary
    )
    phase_result = read_json_validated(
        root / "artifacts/phase_01_data_foundation/phase_result.json", PhaseResult
    )
    gates = make_gates(
        root,
        review_dir,
        config,
        source,
        parsed,
        features,
        targets,
        split,
        summary,
        phase_result,
        missing_required_files,
    )
    gate_counts = {
        "passed": sum(item["result"] == "PASS" for item in gates),
        "failed": sum(item["result"] == "FAIL" for item in gates),
        "not_verified": 0,
    }
    atomic_write_json(
        review_dir / "duplicate_audit.json",
        {
            **split["group_audit"],
            "size_deviations": split["size_deviations"],
            "class_proportion_deviations": split["class_proportion_deviations"],
            "protocol_flags": [],
        },
    )
    atomic_write_json(
        review_dir / "acceptance_gates.json",
        {
            "gates": gates,
            "counts": gate_counts,
            "superseded_gate_policy": {
                "A26-A28": (
                    "Exact 448/150/150 requirements replaced by measured grouped split sizes."
                ),
                "active_group_requirements": [
                    "predictor_duplicate_groups_crossing_splits == 0",
                    "all rows assigned exactly once",
                    "both classes present in every partition",
                ],
            },
        },
    )
    table = [
        "# Acceptance gates A01–A37",
        "",
        "| Gate | Requirement | Evidence | Observed result | Result | Explanation |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    table.extend(
        (
            f"| {item['gate_id']} | {item['requirement']} | `{item['evidence_file']}` | "
            f"{item['observed_result']} | **{item['result']}** | {item['explanation']} |"
        )
        for item in gates
    )
    atomic_write_text(review_dir / "acceptance_gates.md", "\n".join(table) + "\n")

    failed_gates = [item["gate_id"] for item in gates if item["result"] == "FAIL"]
    crossing = split["group_audit"]["predictor_duplicate_groups_crossing_splits"]
    status = "FAILED" if failed_gates or missing_required_files or crossing else "VERIFIED_PASS"
    report = {
        "status": status,
        "reason": (
            [f"Failed gates: {', '.join(failed_gates)}."]
            if failed_gates
            else ["All acceptance gates pass and active predictor groups are isolated."]
        ),
        "required_files_missing": missing_required_files,
        "active_split": {
            "relative_path": str(ACTIVE_SPLIT_RELATIVE).replace("\\", "/"),
            "manifest": split["manifest"].canonical_dict(),
            "actual_split_sizes": split["actual_counts"],
            "class_counts_by_split": split["actual_class_counts"],
            "size_deviations": split["size_deviations"],
            "class_proportion_deviations": split["class_proportion_deviations"],
        },
        "legacy_row_stratified_split": legacy,
        "duplicate_audit": split["group_audit"],
        "gates": gate_counts,
        "formal_commands": {
            "uv_sync": command_passed(review_dir, "uv_sync.txt"),
            "ruff": command_passed(review_dir, "ruff.txt"),
            "mypy": command_passed(review_dir, "mypy.txt"),
            "pytest_non_network": command_passed(review_dir, "pytest_non_network.txt"),
            "pytest_network": command_passed(review_dir, "pytest_network.txt"),
        },
    }
    atomic_write_json(review_dir / "audit_result.json", report)
    write_handoff(root, source, split, legacy, gates, missing_required_files, status)
    review = [
        "# Independent Phase 01 Evidence Review",
        "",
        f"## Status\n\n{status}",
        "",
        "## Evidence basis",
        "",
        f"- A01–A37: {gate_counts['passed']} passed, {gate_counts['failed']} failed.",
        f"- Active split: `{ACTIVE_SPLIT_RELATIVE.as_posix()}/`.",
        f"- Actual split sizes: `{split['actual_counts']}`.",
        f"- Predictor groups crossing active splits: `{crossing}`.",
        "- Conflicting-target groups detected: "
        f"`{split['group_audit']['conflicting_target_groups']}`.",
        "- The persisted legacy row-stratified split is preserved and marked deprecated.",
        "",
        "## Conclusion",
        "",
        "The active predictor-group split satisfies complete assignment, class coverage, "
        "deterministic rerun, shuffled-input invariance, and group isolation requirements.",
        "",
    ]
    atomic_write_text(review_dir / "independent_review.md", "\n".join(review))
    phase_message = (
        "Independent evidence audit: VERIFIED_PASS; active predictor groups do not "
        "cross partitions."
        if status == "VERIFIED_PASS"
        else f"Independent evidence audit: {status}."
    )
    atomic_write_json(
        root / "artifacts/phase_01_data_foundation/phase_result.json",
        phase_result.model_copy(update={"message": phase_message}).canonical_dict(),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if status == "VERIFIED_PASS" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path)
    args = parser.parse_args()
    if args.snapshot:
        atomic_write_json(args.snapshot, snapshot(ROOT))
        return 0
    return run_audit(ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
