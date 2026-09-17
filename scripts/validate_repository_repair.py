"""Validate repository structure independently from optional local evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

FORBIDDEN_COMPONENT = re.compile(r"^(?:phase[_-]?\d+|step[_-]?\d+|\d{1,2}_.+)$", re.I)
FORBIDDEN_REFERENCE = re.compile(
    r"(?:scripts/[0-9]{1,2}_[A-Za-z0-9_]+\.py|phase_[0-9]+[A-Za-z]*|phase[0-9]+|"
    r"phase-[0-9]+[A-Za-z-]+|PHASE[0-9]+|FAIL_PHASE[0-9]+|phase_numbering)"
)
PRIVATE_DOCUMENT = "SchemaGuard_Complete_Research_and_Engineering_Plan.docx"
REQUIRED_MODELS = ("LR-1.9", "CAT-1.2", "XGB-3.4", "TPFN3-8.5", "TICL2-2.2")


def _git(root: Path, *args: str) -> tuple[int, str, str]:
    result = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=False)
    return result.returncode, result.stdout.strip(), result.stderr.strip()


def _tracked_paths(root: Path) -> list[str]:
    code, stdout, stderr = _git(root, "ls-files")
    if code != 0:
        raise RuntimeError(stderr or "git ls-files failed")
    return [line for line in stdout.splitlines() if line]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_private_document(root: Path) -> tuple[str, str]:
    path = root / PRIVATE_DOCUMENT
    if not path.exists():
        return "PASS", "private document absent is acceptable"
    code, _, _ = _git(root, "ls-files", "--error-unmatch", "--", PRIVATE_DOCUMENT)
    if code == 0:
        return "FAIL", "private document is tracked"
    code, stdout, _ = _git(root, "status", "--porcelain=v1", "--", PRIVATE_DOCUMENT)
    if code != 0:
        return "FAIL", "private document status could not be determined"
    if stdout.startswith("?? "):
        return "PASS", "private document is present and untracked"
    return "FAIL", f"private document has staged or tracked status: {stdout}"


def _naming_violations(root: Path, paths: list[str]) -> list[str]:
    violations: list[str] = []
    migration_file = "artifacts/handoff/repository_repair_review.md"
    for relative in paths:
        for component in Path(relative).parts:
            if component.lower().startswith("fig") and component[3:].split("_", 1)[0].isdigit():
                continue
            if FORBIDDEN_COMPONENT.match(component):
                violations.append(f"path:{relative}")
        path = root / relative
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            continue
        if relative == "scripts/validate_repository_repair.py":
            continue  # This file contains the audit pattern definitions themselves.
        in_migration_table = False
        for line_number, line in enumerate(lines, 1):
            if relative == migration_file:
                if line.startswith("## Naming migration"):
                    in_migration_table = True
                elif in_migration_table and line.startswith("## "):
                    in_migration_table = False
                if in_migration_table and line.startswith("|"):
                    continue
            if FORBIDDEN_REFERENCE.search(line):
                violations.append(f"reference:{relative}:{line_number}")
    return violations


def _load_contract_module(root: Path) -> Any:
    source = str(root / "src")
    if source not in sys.path:
        sys.path.insert(0, source)
    from schemaguard import artifact_contracts

    return artifact_contracts


def _check_schemas(root: Path, contracts: Any) -> tuple[str, str]:
    from jsonschema import Draft202012Validator

    documents = contracts.schema_documents()
    missing = []
    for filename, document in documents.items():
        path = root / "schemas" / filename
        if not path.is_file():
            missing.append(filename)
            continue
        observed = json.loads(path.read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(observed)
        if observed != document:
            raise ValueError(f"generated schema differs from contract: {filename}")
    if missing:
        return "FAIL", f"missing tracked schemas: {', '.join(missing)}"
    return "PASS", f"validated {len(documents)} deterministic JSON schemas"


def _check_baseline(root: Path, contracts: Any) -> tuple[str, str]:
    path = root / "configs/baselines/data_foundation.json"
    if not path.is_file():
        return "FAIL", "tracked data-foundation baseline is missing"
    baseline = contracts.DataFoundationBaselineContract.model_validate(
        json.loads(path.read_text(encoding="utf-8"))
    )
    if baseline.dataset.openml_data_id != 1464:
        return "FAIL", "unexpected data-foundation OpenML ID"
    return "PASS", "tracked baseline contract and immutable identity validated"


def _check_dependencies(root: Path) -> tuple[str, str]:
    import tomllib

    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    dev = pyproject["project"]["optional-dependencies"]["dev"]
    if not any(item.startswith("jsonschema>=4.23,<5") for item in dev):
        return "FAIL", "jsonschema is absent from the development extra"
    if not (root / "requirements/p12_windows_constraints.txt").is_file():
        return "FAIL", "P12 constraints file is missing"
    if any(
        path.startswith("requirements/") and path.endswith("_windows.lock")
        for path in _tracked_paths(root)
    ):
        return "FAIL", "misleading short P12 lock filename remains"
    lock_text = (root / "uv.lock").read_text(encoding="utf-8")
    if 'name = "torch"' not in lock_text or 'version = "2.6.0"' not in lock_text:
        return "FAIL", "uv.lock does not resolve the intended PyTorch version"
    return "PASS", "development dependency, P12 constraints, and uv resolution agree"


def _check_configs(root: Path) -> tuple[str, str]:
    try:
        from schemaguard.data.schemaorbit import load_schemaorbit_config
        from schemaguard.models.registry import load_model_registry

        models = load_model_registry(root / "configs/runtime/model_compatibility.yaml")
        load_schemaorbit_config(root / "configs/datasets/schemaorbit14.yaml")
    except Exception as exc:
        return "FAIL", f"configuration validation failed: {type(exc).__name__}: {exc}"
    ids = tuple(model.id for model in models)
    if ids != REQUIRED_MODELS:
        return "FAIL", f"unexpected frozen model registry: {ids}"
    return "PASS", "frozen model and dataset configurations validate"


def _check_workflow(root: Path) -> tuple[str, str]:
    text = (root / ".github/workflows/quality.yml").read_text(encoding="utf-8")
    required = (
        "core-quality:",
        "classical-model-validation:",
        "uv sync --extra dev --extra monitoring --no-managed-python",
        "uv sync --extra dev --extra models-cpu --extra monitoring --no-managed-python",
        "tests/unit",
        "tests/integration/test_classical_model_probes.py",
    )
    missing = [item for item in required if item not in text]
    return (
        ("FAIL", f"workflow missing: {missing}")
        if missing
        else ("PASS", "CI jobs match test dependencies")
    )


def _check_tracked_generated_paths(paths: list[str]) -> tuple[str, str]:
    generated_prefixes = (
        "data/raw/",
        "data/processed/",
        "data/splits/",
        "data/cache/",
        "results/",
        "logs/",
    )
    generated = [
        path for path in paths if path.startswith(generated_prefixes) or path.endswith(".ckpt")
    ]
    return (
        ("FAIL", f"generated artifacts are tracked: {generated}")
        if generated
        else ("PASS", "no generated datasets, reports, or checkpoints are tracked")
    )


def _check_local_baseline(root: Path, contracts: Any) -> tuple[str, str]:
    baseline = contracts.DataFoundationBaselineContract.model_validate(
        json.loads((root / "configs/baselines/data_foundation.json").read_text(encoding="utf-8"))
    )
    raw = root / "data/raw/openml/1464/source_manifest.json"
    processed = root / "data/processed/openml/1464"
    split = root / "data/splits/openml/1464/stratified_group_5fold_v1/seed_1729"
    if not raw.exists() or not processed.exists() or not split.exists():
        return "NOT_APPLICABLE_LOCAL_ARTIFACTS_ABSENT", "local data-foundation artifacts are absent"
    source = json.loads(raw.read_text(encoding="utf-8"))
    manifest = json.loads((processed / "data_manifest.json").read_text(encoding="utf-8"))
    expected = baseline.dataset.processed_artifact_sha256
    observed = {name: _sha256(processed / name) for name in expected}
    split_manifest = json.loads((split / "split_manifest.json").read_text(encoding="utf-8"))
    valid = (
        source["openml_data_id"] == baseline.dataset.openml_data_id
        and source["openml_file_id"] == baseline.dataset.openml_file_id
        and source["computed_sha256"] == baseline.dataset.raw_source_sha256
        and manifest["row_count"] == baseline.dataset.row_count
        and len(manifest["feature_columns"]) == baseline.dataset.predictor_count
        and observed == expected
        and split_manifest["strategy"] == baseline.split_protocol.name
        and split_manifest["predictor_duplicate_groups_crossing_splits"]
        == baseline.split_protocol.cross_split_duplicate_group_count
    )
    return (
        ("PASS", "current artifacts match tracked data-foundation baseline")
        if valid
        else ("FAIL", "current artifacts differ from tracked baseline")
    )


def _check_local_reports(root: Path, contracts: Any) -> tuple[str, str]:
    report_paths = {
        "dataset": root / "results/validation/dataset_registry_report.json",
        "gpu": root / "results/validation/gpu_capacity_report.json",
        "model": root / "results/validation/model_compatibility_report.json",
    }
    if not all(path.exists() for path in report_paths.values()):
        return "NOT_APPLICABLE_LOCAL_ARTIFACTS_ABSENT", "local validation reports are absent"
    contracts.DatasetRegistryReportContract.model_validate(
        json.loads(report_paths["dataset"].read_text(encoding="utf-8"))
    )
    contracts.GpuCapacityReportContract.model_validate(
        json.loads(report_paths["gpu"].read_text(encoding="utf-8"))
    )
    model = contracts.ModelCompatibilityReportContract.model_validate(
        json.loads(report_paths["model"].read_text(encoding="utf-8"))
    )
    if len(model.probes) != len(REQUIRED_MODELS) or any(
        probe.status != "PASS" or probe.device != "cpu" for probe in model.probes
    ):
        return "FAIL", "local model compatibility evidence is incomplete"
    return "PASS", "local validation reports satisfy strict contracts"


def validate_repository(
    root: Path, *, expected_head: str | None, local_evidence: bool
) -> dict[str, Any]:
    contracts = _load_contract_module(root)
    paths = _tracked_paths(root)
    checks: list[dict[str, str]] = []

    def add(gate_id: str, requirement: str, result: str, evidence: str) -> None:
        checks.append(
            {
                "gate_id": gate_id,
                "requirement": requirement,
                "result": result,
                "evidence": evidence,
            }
        )

    head_code, head, _ = _git(root, "rev-parse", "HEAD")
    branch_code, branch, _ = _git(root, "branch", "--show-current")
    head_ok = head_code == 0 and bool(head) and (expected_head is None or head == expected_head)
    branch_text = branch or "<detached>"
    add(
        "R01",
        "repository has a valid HEAD and optional expected commit matches",
        "PASS" if head_ok else "FAIL",
        f"HEAD={head}; branch={branch_text}; branch_command={branch_code}",
    )
    document_result, document_evidence = _check_private_document(root)
    add(
        "R02",
        "private reference document is absent or safely untracked",
        document_result,
        document_evidence,
    )
    naming = _naming_violations(root, paths)
    add(
        "R03",
        "tracked repository paths and references use semantic names",
        "PASS" if not naming else "FAIL",
        "no forbidden numbered names" if not naming else "; ".join(naming),
    )
    structural_checks = (
        ("R04", "tracked schemas validate against authoritative contracts", _check_schemas),
        ("R05", "tracked data-foundation baseline is valid", _check_baseline),
        ("R06", "dependency declarations are reproducible", _check_dependencies),
        ("R07", "frozen configuration identities validate", _check_configs),
        ("R08", "CI workflow provisions the dependencies it tests", _check_workflow),
    )
    for gate_id, requirement, function in structural_checks:
        result, evidence = (
            function(root, contracts) if gate_id == "R04" or gate_id == "R05" else function(root)
        )
        add(gate_id, requirement, result, evidence)
    result, evidence = _check_tracked_generated_paths(paths)
    add("R09", "generated artifacts are not tracked", result, evidence)
    local_baseline = _check_local_baseline(root, contracts)
    local_reports = _check_local_reports(root, contracts)
    if not local_evidence:
        local_baseline = (
            "NOT_APPLICABLE_LOCAL_ARTIFACTS_ABSENT",
            "local evidence mode was not requested",
        )
        local_reports = (
            "NOT_APPLICABLE_LOCAL_ARTIFACTS_ABSENT",
            "local evidence mode was not requested",
        )
    add("R10", "local data-foundation integrity is explicit", *local_baseline)
    add("R11", "local reports are independently validated", *local_reports)
    status = "PASS" if all(check["result"] != "FAIL" for check in checks) else "FAIL"
    return {"schema_version": 1, "stage": "repository_repair", "status": status, "checks": checks}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-head")
    parser.add_argument("--local-evidence", action="store_true")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-directory", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    report = validate_repository(
        root, expected_head=args.expected_head, local_evidence=args.local_evidence
    )
    contracts = _load_contract_module(root)
    contracts.RepositoryValidationReportContract.model_validate(report)
    output = args.output_directory or root / "artifacts/repository_repair"
    output.mkdir(parents=True, exist_ok=True)
    (output / "structural_validation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": report["status"], "checks": report["checks"]}, indent=2))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
