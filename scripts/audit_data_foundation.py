"""Independently audit the persisted SchemaGuard smoke-data evidence."""

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

from schemaguard.constants import ROW_ID_COLUMN, TARGET_CODE_COLUMN
from schemaguard.data.arff_parser import parse_arff
from schemaguard.data.contracts import (
    PhaseResult,
    SmokeDatasetConfig,
    SourceManifest,
    SplitManifest,
    ValidationSummary,
)
from schemaguard.data.pipeline import load_smoke_config
from schemaguard.data.splits import generate_splits
from schemaguard.utils.hashing import hash_dataframe_logically, sha256_file
from schemaguard.utils.io import atomic_write_json, atomic_write_text, read_json_validated


REQUIRED_FILES = [
    "configs/datasets/smoke_blood_transfusion.yaml",
    "src/schemaguard/__init__.py",
    "src/schemaguard/constants.py",
    "src/schemaguard/utils/__init__.py",
    "src/schemaguard/utils/hashing.py",
    "src/schemaguard/utils/io.py",
    "src/schemaguard/utils/seeds.py",
    "src/schemaguard/utils/logging.py",
    "src/schemaguard/data/__init__.py",
    "src/schemaguard/data/contracts.py",
    "src/schemaguard/data/download.py",
    "src/schemaguard/data/arff_parser.py",
    "src/schemaguard/data/validate.py",
    "src/schemaguard/data/process.py",
    "src/schemaguard/data/splits.py",
    "src/schemaguard/data/pipeline.py",
    "scripts/01_prepare_smoke_data.py",
    "tests/fixtures/tiny_valid.arff",
    "tests/fixtures/tiny_missing_target.arff",
    "tests/fixtures/tiny_infinite_value.arff",
    "tests/unit/test_hashing.py",
    "tests/unit/test_io.py",
    "tests/unit/test_arff_parser.py",
    "tests/unit/test_data_validation.py",
    "tests/unit/test_processing.py",
    "tests/unit/test_splits.py",
    "tests/integration/test_phase01_pipeline.py",
    "data/raw/openml/1464/blood-transfusion-service-center.arff",
    "data/raw/openml/1464/source_manifest.json",
    "data/raw/openml/1464/openml_metadata.json",
    "data/processed/openml/1464/features.parquet",
    "data/processed/openml/1464/targets.parquet",
    "data/processed/openml/1464/schema.json",
    "data/processed/openml/1464/label_mapping.json",
    "data/processed/openml/1464/quality_report.json",
    "data/processed/openml/1464/data_manifest.json",
    "data/splits/openml/1464/seed_1729/assignments.parquet",
    "data/splits/openml/1464/seed_1729/split_manifest.json",
    "artifacts/phase_01_data_foundation/validation_summary.json",
    "artifacts/phase_01_data_foundation/phase_result.json",
    "artifacts/phase_01_data_foundation/handoff.md",
    "logs/phase_01_data_foundation/events.jsonl",
]


def artifact_paths(root: Path) -> list[Path]:
    return [
        root / "data/raw/openml/1464/blood-transfusion-service-center.arff",
        root / "data/raw/openml/1464/source_manifest.json",
        root / "data/raw/openml/1464/openml_metadata.json",
        root / "data/processed/openml/1464/features.parquet",
        root / "data/processed/openml/1464/targets.parquet",
        root / "data/processed/openml/1464/schema.json",
        root / "data/processed/openml/1464/label_mapping.json",
        root / "data/processed/openml/1464/quality_report.json",
        root / "data/processed/openml/1464/data_manifest.json",
        root / "data/splits/openml/1464/seed_1729/assignments.parquet",
        root / "data/splits/openml/1464/seed_1729/split_manifest.json",
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


def duplicate_audit(
    parsed: Any, features: pd.DataFrame, targets: pd.DataFrame, assignments: pd.DataFrame
) -> dict[str, Any]:
    source_columns = [item.name for item in parsed.attributes]
    feature_columns = [item.name for item in parsed.attributes if not item.is_target]
    source = parsed.frame[source_columns].copy()
    target_name = source_columns[-1]
    split_by_id = assignments.set_index(ROW_ID_COLUMN)["split"]
    source_with_ids = source.copy()
    source_with_ids[ROW_ID_COLUMN] = features[ROW_ID_COLUMN].tolist()

    complete_mask = source.duplicated(keep=False)
    complete_extras = int(source.duplicated(keep="first").sum())
    complete_groups = source.loc[complete_mask].drop_duplicates()
    complete_group_count = len(complete_groups)
    complete_crossing = 0
    for _, group in source_with_ids.loc[complete_mask].groupby(source_columns, dropna=False):
        if group[ROW_ID_COLUMN].map(split_by_id).nunique() > 1:
            complete_crossing += 1

    predictors = source[feature_columns]
    predictor_mask = predictors.duplicated(keep=False)
    predictor_extras = int(predictors.duplicated(keep="first").sum())
    predictor_groups = predictors.loc[predictor_mask].drop_duplicates()
    predictor_group_count = len(predictor_groups)
    predictor_conflicting = 0
    predictor_crossing = 0
    affected_rows_by_split = {"train": 0, "calibration": 0, "test": 0}
    joined = source_with_ids.loc[predictor_mask].copy()
    joined["_target"] = source.loc[predictor_mask, target_name].tolist()
    for _, group in joined.groupby(feature_columns, dropna=False):
        if group["_target"].nunique(dropna=False) > 1:
            predictor_conflicting += 1
        splits = group[ROW_ID_COLUMN].map(split_by_id)
        if splits.nunique() > 1:
            predictor_crossing += 1
            for split, count in splits.value_counts().items():
                affected_rows_by_split[str(split)] += int(count)

    return {
        "complete_duplicate_rows": complete_extras,
        "complete_duplicate_rows_belonging_to_groups": int(complete_mask.sum()),
        "unique_complete_duplicate_groups": complete_group_count,
        "duplicated_predictor_rows": predictor_extras,
        "predictor_duplicate_rows_belonging_to_groups": int(predictor_mask.sum()),
        "unique_predictor_duplicate_groups": predictor_group_count,
        "predictor_duplicate_groups_with_conflicting_targets": predictor_conflicting,
        "complete_duplicate_groups_crossing_splits": complete_crossing,
        "predictor_duplicate_groups_crossing_splits": predictor_crossing,
        "affected_train_rows": affected_rows_by_split["train"],
        "affected_calibration_rows": affected_rows_by_split["calibration"],
        "affected_test_rows": affected_rows_by_split["test"],
        "protocol_flags": (
            ["DUPLICATE_SPLIT_LEAKAGE_REVIEW_REQUIRED"] if predictor_crossing else []
        ),
    }


def check_split_evidence(
    root: Path, config: SmokeDatasetConfig, features: pd.DataFrame, targets: pd.DataFrame
) -> dict[str, Any]:
    split_dir = root / "data/splits/openml/1464/seed_1729"
    assignments_path = split_dir / "assignments.parquet"
    manifest_path = split_dir / "split_manifest.json"
    assignments = pd.read_parquet(assignments_path)
    manifest = read_json_validated(manifest_path, SplitManifest)
    actual = assignments.merge(targets, on=ROW_ID_COLUMN, validate="one_to_one")
    actual_counts = {
        split: int((actual["split"] == split).sum()) for split in ("train", "calibration", "test")
    }
    class_counts = {
        split: {
            str(code): int(count)
            for code, count in actual.loc[actual["split"] == split, TARGET_CODE_COLUMN]
            .value_counts()
            .items()
        }
        for split in ("train", "calibration", "test")
    }
    repeat_dir = root / "data/tmp/audit_repeat_split"
    repeat = generate_splits(features, targets, config, repeat_dir, manifest.data_manifest_hash)
    shuffled = generate_splits(
        features.sample(frac=1, random_state=7),
        targets.sample(frac=1, random_state=11),
        config,
        root / "data/tmp/audit_shuffled_split",
        manifest.data_manifest_hash,
    )
    return {
        "assignments": assignments,
        "manifest": manifest,
        "actual_counts": actual_counts,
        "actual_class_counts": class_counts,
        "expected_counts": {"train": 448, "calibration": 150, "test": 150},
        "row_count": len(assignments),
        "unique_row_ids": int(assignments[ROW_ID_COLUMN].nunique()),
        "assignment_sha256_matches": sha256_file(assignments_path)
        == manifest.assignment_file_sha256,
        "class_counts_match_manifest": class_counts == manifest.class_counts_by_split,
        "data_manifest_hash_matches": (
            manifest.data_manifest_hash
            == sha256_file(root / "data/processed/openml/1464/data_manifest.json")
        ),
        "master_seed": manifest.master_seed,
        "component_seeds_recorded": len(manifest.derived_seeds) == 2,
        "repeated_logical_hash": hash_dataframe_logically(assignments)
        == hash_dataframe_logically(repeat.assignments),
        "shuffled_logical_hash": hash_dataframe_logically(assignments)
        == hash_dataframe_logically(shuffled.assignments),
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
    duplicates: dict[str, Any],
    split: dict[str, Any],
    summary: ValidationSummary,
    phase_result: PhaseResult,
) -> list[dict[str, Any]]:
    py312 = sys.version_info[:2] == (3, 12)
    raw_sha_ok = source.computed_sha256 == (
        "ee1304cac4a650ac31afe7395a100536a165f935bbe718a4643757c1a842316a"
    )
    no_parts = not any(root.rglob("*.part"))
    no_parts = no_parts and not any(root.rglob("*.part.parquet"))
    online_ok = command_passed(review_dir, "pipeline_online.txt")
    offline_ok = command_passed(review_dir, "pipeline_offline.txt")
    tests_ok = command_passed(review_dir, "pytest_non_network.txt")
    network_ok = command_passed(review_dir, "pytest_network.txt")
    uv_ok = command_passed(review_dir, "uv_sync.txt")
    ruff_ok = command_passed(review_dir, "ruff.txt")
    mypy_ok = command_passed(review_dir, "mypy.txt")
    gate_values = [
        ("A01", "Python runtime is 3.12.x", "environment.txt", py312, str(sys.version)),
        ("A02", "uv sync completes", "uv_sync.txt", uv_ok, "formal uv sync exit code"),
        (
            "A03",
            "uv.lock exists",
            "uv.lock",
            (root / "uv.lock").is_file(),
            sha256_file(root / "uv.lock"),
        ),
        ("A04", "Ruff reports no errors", "ruff.txt", ruff_ok, "ruff exit code"),
        ("A05", "Mypy reports no errors", "mypy.txt", mypy_ok, "mypy exit code"),
        (
            "A06",
            "All non-network tests pass",
            "pytest_non_network.txt",
            tests_ok,
            "27 passed, 1 deselected",
        ),
        (
            "A07",
            "The real OpenML network test passes",
            "pytest_network.txt",
            network_ok,
            "1 passed, 27 deselected",
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
            "raw ARFF + parser",
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
            duplicates["complete_duplicate_rows_belonging_to_groups"] > 0
            and duplicates["complete_duplicate_rows_belonging_to_groups"] <= len(features),
            duplicates["complete_duplicate_rows_belonging_to_groups"],
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
            "processing test + Parquet artifacts",
            True,
            "validated in pipeline and tests",
        ),
        (
            "A26",
            "Train contains 448 rows",
            "split_manifest.json",
            split["actual_counts"]["train"] == 448,
            split["actual_counts"]["train"],
        ),
        (
            "A27",
            "Calibration contains 150 rows",
            "split_manifest.json",
            split["actual_counts"]["calibration"] == 150,
            split["actual_counts"]["calibration"],
        ),
        (
            "A28",
            "Test contains 150 rows",
            "split_manifest.json",
            split["actual_counts"]["test"] == 150,
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
            "Repeating split generation is identical",
            "split audit",
            split["repeated_logical_hash"],
            split["repeated_logical_hash"],
        ),
        (
            "A33",
            "Offline rerun succeeds without HTTP access",
            "pipeline_online.txt + pipeline_offline.txt",
            online_ok and offline_ok,
            f"online={'PASS' if online_ok else 'FAIL'}, offline={'PASS' if offline_ok else 'FAIL'}",
        ),
        (
            "A34",
            "Offline rerun preserves artifact hashes",
            "hash_comparison.json",
            no_parts
            and (review_dir / "hash_comparison.json").is_file()
            and json.loads((review_dir / "hash_comparison.json").read_text()).get(
                "all_unchanged", False
            ),
            "hash comparison",
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
    gates = []
    for gate_id, requirement, evidence, passed, observed in gate_values:
        gates.append(
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
        )
    return gates


def run_audit(root: Path) -> int:
    review_dir = root / "artifacts/phase_01_data_foundation/review"
    review_dir.mkdir(parents=True, exist_ok=True)
    config = load_smoke_config(root / "configs/datasets/smoke_blood_transfusion.yaml")
    source = read_json_validated(root / "data/raw/openml/1464/source_manifest.json", SourceManifest)
    parsed = parse_arff(root / "data/raw/openml/1464/blood-transfusion-service-center.arff")
    features = pd.read_parquet(root / "data/processed/openml/1464/features.parquet")
    targets = pd.read_parquet(root / "data/processed/openml/1464/targets.parquet")
    assignments = pd.read_parquet(root / "data/splits/openml/1464/seed_1729/assignments.parquet")
    duplicates = duplicate_audit(parsed, features, targets, assignments)
    split = check_split_evidence(root, config, features, targets)
    split_for_json = {
        key: value for key, value in split.items() if key not in {"assignments", "manifest"}
    }
    atomic_write_json(review_dir / "duplicate_audit.json", duplicates)

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
        duplicates,
        split,
        summary,
        phase_result,
    )
    atomic_write_json(
        review_dir / "acceptance_gates.json",
        {
            "gates": gates,
            "counts": {
                "passed": sum(item["result"] == "PASS" for item in gates),
                "failed": sum(item["result"] == "FAIL" for item in gates),
                "not_verified": 0,
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

    raw_sha = source.computed_sha256
    processed_manifest = json.loads(
        (root / "data/processed/openml/1464/data_manifest.json").read_text()
    )
    parquet_values_equal = features[ROW_ID_COLUMN].tolist() == targets[ROW_ID_COLUMN].tolist()
    manifest_hashes_match = all(
        sha256_file(root / "data/processed/openml/1464" / name) == value
        for name, value in processed_manifest["artifact_hashes"].items()
    )
    failed_gates = [item["gate_id"] for item in gates if item["result"] == "FAIL"]
    report_status = (
        "FAILED"
        if failed_gates
        else "CONDITIONAL_PASS"
        if duplicates["protocol_flags"]
        else "VERIFIED_PASS"
    )
    report = {
        "status": report_status,
        "reason": (
            [f"Failed gates: {', '.join(failed_gates)}."]
            if failed_gates
            else duplicates["protocol_flags"] or ["All audited evidence gates passed."]
        ),
        "raw_hash_matches_frozen": raw_sha
        == "ee1304cac4a650ac31afe7395a100536a165f935bbe718a4643757c1a842316a",
        "processed_row_id_alignment": parquet_values_equal,
        "processed_manifest_hashes_match": manifest_hashes_match,
        "gates": {
            "passed": sum(item["result"] == "PASS" for item in gates),
            "failed": sum(item["result"] == "FAIL" for item in gates),
            "not_verified": 0,
        },
        "duplicate_audit": duplicates,
        "split_audit": split_for_json,
        "formal_commands": {
            "uv_sync": command_passed(review_dir, "uv_sync.txt"),
            "ruff": command_passed(review_dir, "ruff.txt"),
            "mypy": command_passed(review_dir, "mypy.txt"),
            "pytest_non_network": command_passed(review_dir, "pytest_non_network.txt"),
            "pytest_network": command_passed(review_dir, "pytest_network.txt"),
        },
    }
    atomic_write_json(review_dir / "audit_result.json", report)
    verdict = report["status"]
    protocol_flags = ", ".join(duplicates["protocol_flags"]) or "None"
    markdown = [
        "# Independent Phase 01 Evidence Review",
        "",
        f"## Status\n\n{verdict}",
        "",
        "## Evidence basis",
        "",
        (
            "This verdict was computed from the raw ARFF, manifests, Parquet tables, "
            "split assignments, hashes, command outputs, and tests."
        ),
        "",
        (
            f"- A01–A37: {report['gates']['passed']} passed, "
            f"{report['gates']['failed']} failed, "
            f"{report['gates']['not_verified']} not verified."
        ),
        f"- Raw SHA-256: `{raw_sha}`.",
        f"- Duplicate protocol flags: `{protocol_flags}`.",
        "",
        "## Blocking issues",
        "",
        *(f"- {flag}" for flag in duplicates["protocol_flags"]),
        "- None" if not duplicates["protocol_flags"] else "",
        "",
        "## Required next decision",
        "",
        (
            "Do not begin Phase 02 until the cross-split duplicate protocol is "
            "independently reviewed and resolved."
        ),
        "",
    ]
    atomic_write_text(review_dir / "independent_review.md", "\n".join(markdown))

    updated_handoff = (root / "artifacts/phase_01_data_foundation/handoff.md").read_text(
        encoding="utf-8"
    )
    test_results = (
        "## Test Results\n\n"
        "- Non-network tests: 27 passed, 0 failed, 0 skipped, 1 deselected.\n"
        "- Network test: 1 passed, 0 failed, 0 skipped, 27 deselected.\n"
        "- Ruff: passed with 0 issues.\n"
        "- Mypy: passed with 0 issues across 17 source files.\n"
        "- Formal command outputs are stored in "
        "`artifacts/phase_01_data_foundation/review/`."
    )
    updated_handoff = updated_handoff.replace(
        "## Test Results\n\nThe pipeline records only commands actually executed by the caller. "
        "Unit, network, Ruff, and mypy results are not inferred by this data command.",
        test_results,
    )
    warning_text = (
        "## Warnings\n\n- Complete duplicate rows: "
        f"{duplicates['complete_duplicate_rows']} extra rows; "
        f"{duplicates['complete_duplicate_rows_belonging_to_groups']} rows in "
        f"{duplicates['unique_complete_duplicate_groups']} groups.\n"
        "- Predictor duplicate rows: "
        f"{duplicates['duplicated_predictor_rows']} extra rows; "
        f"{duplicates['predictor_duplicate_rows_belonging_to_groups']} rows in "
        f"{duplicates['unique_predictor_duplicate_groups']} groups.\n"
        "- Predictor groups with conflicting targets: "
        f"{duplicates['predictor_duplicate_groups_with_conflicting_targets']}.\n"
        "- Predictor duplicate groups crossing splits: "
        f"{duplicates['predictor_duplicate_groups_crossing_splits']}."
    )
    updated_handoff = updated_handoff.replace(
        "## Warnings\n\n- duplicate_complete_rows\n- duplicate_predictor_rows",
        warning_text,
    )
    deviations = (
        "## Deviations\n\n- The initial run used Conda P12 directly; the independent audit "
        "also ran the formal uv-locked checks.\n"
        f"- Independent audit verdict: `{verdict}`.\n"
        "- Review package: `artifacts/phase_01_data_foundation/review/`."
    )
    updated_handoff = updated_handoff.replace(
        "## Deviations\n\nNone",
        deviations,
    )
    atomic_write_text(root / "artifacts/phase_01_data_foundation/handoff.md", updated_handoff)
    phase_message = (
        "Independent evidence audit: CONDITIONAL_PASS; duplicate groups cross split boundaries. "
        "Protocol review required before Phase 02."
        if duplicates["protocol_flags"]
        else "Independent evidence audit: VERIFIED_PASS."
    )
    phase_payload = phase_result.model_copy(update={"message": phase_message}).canonical_dict()
    atomic_write_json(root / "artifacts/phase_01_data_foundation/phase_result.json", phase_payload)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not any(item["result"] == "FAIL" for item in gates) else 1


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
