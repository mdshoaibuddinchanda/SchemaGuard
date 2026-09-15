from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from validate_repository_repair import _check_private_document, validate_repository  # noqa: E402


def test_structural_validator_is_rerunnable_without_local_evidence() -> None:
    first = validate_repository(ROOT, expected_head=None, local_evidence=False)
    second = validate_repository(ROOT, expected_head=None, local_evidence=False)
    assert first == second
    assert first["status"] == "PASS"
    assert {check["result"] for check in first["checks"]} == {
        "PASS",
        "NOT_APPLICABLE_LOCAL_ARTIFACTS_ABSENT",
    }


def test_expected_head_is_optional_and_mismatch_is_reported() -> None:
    report = validate_repository(ROOT, expected_head="not-the-current-commit", local_evidence=False)
    assert report["status"] == "FAIL"
    assert report["checks"][0]["result"] == "FAIL"


def test_absent_private_document_passes(tmp_path: Path) -> None:
    assert _check_private_document(tmp_path)[0] == "PASS"


def test_staged_private_document_fails(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "--quiet"], cwd=tmp_path, check=True)
    document = tmp_path / "SchemaGuard_Complete_Research_and_Engineering_Plan.docx"
    document.write_bytes(b"synthetic")
    subprocess.run(["git", "add", document.name], cwd=tmp_path, check=True)
    assert _check_private_document(tmp_path)[0] == "FAIL"
