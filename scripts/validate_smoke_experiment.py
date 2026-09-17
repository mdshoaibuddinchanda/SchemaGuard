"""Independently validate smoke evidence and record a sanitized handoff."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from schemaguard.experiments.contracts import (  # noqa: E402
    PlanEnvelope,
    ProtectedFoundationHashComparison,
    ProtectedSplitValidation,
)
from schemaguard.experiments.evidence import (  # noqa: E402
    make_inventory,
    render_review,
    validate_smoke_report,
    write_json,
)
from schemaguard.experiments.planning import (  # noqa: E402
    build_smoke_plan,
    verify_runtime_sources_after_plan,
)
from schemaguard.utils.hashing import sha256_file  # noqa: E402
from schemaguard.utils.io import atomic_write_text  # noqa: E402


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()


def _assert_no_tracked_edits() -> None:
    _git("diff", "--quiet", "HEAD", "--")
    _git("diff", "--cached", "--quiet")


def _plan_from_disk(plan_sha256: str | None):
    if plan_sha256 is None:
        candidates = sorted((ROOT / "results/smoke/runtime/plans").glob("*.json"))
        if len(candidates) != 1:
            raise ValueError("provide a plan hash when zero or multiple local smoke plans exist")
        path = candidates[0]
    else:
        path = ROOT / "results/smoke/runtime/plans" / f"{plan_sha256}.json"
    envelope = PlanEnvelope.model_validate(json.loads(path.read_text(encoding="utf-8")))
    if path.stem != envelope.plan_sha256:
        raise ValueError("plan filename and canonical plan identity disagree")
    return envelope.plan


def _check_ignore_policy() -> None:
    for relative in (
        "results/smoke/predictions/example.parquet",
        "data/cache/conditions/example/payload.bin",
    ):
        result = subprocess.run(
            ["git", "check-ignore", "-q", relative],
            cwd=ROOT,
            check=False,
            timeout=10,
        )
        if result.returncode != 0:
            raise ValueError(f"generated smoke/cache path is not ignored: {relative}")
    for relative in (
        "artifacts/handoff/smoke_experiment_review.md",
        "artifacts/handoff/smoke_experiment_inventory.json",
    ):
        result = subprocess.run(
            ["git", "check-ignore", "-q", relative],
            cwd=ROOT,
            check=False,
            timeout=10,
        )
        if result.returncode == 0:
            raise ValueError(f"sanitized handoff path must remain trackable: {relative}")


def validate(report_relative: str | None, plan_sha256: str | None, record_evidence: bool) -> int:
    _assert_no_tracked_edits()
    plan = _plan_from_disk(plan_sha256)
    rebuilt = build_smoke_plan(
        ROOT,
        "configs/runtime/smoke_experiment.yaml",
        source_commit_override=plan.source_commit,
    )
    if rebuilt.plan_sha256 != plan.plan_sha256:
        raise ValueError("recomputed smoke plan differs from the persisted content-addressed plan")
    verify_runtime_sources_after_plan(ROOT, plan)
    _check_ignore_policy()
    if report_relative is None:
        report_path = ROOT / "results/smoke/runs" / f"{plan.plan_sha256}_resume.json"
    else:
        report_path = Path(report_relative)
        if not report_path.is_absolute():
            report_path = ROOT / report_path
    report, validation = validate_smoke_report(ROOT, plan, report_path)
    validation_path = ROOT / "results/smoke/runtime/validation_report.json"
    validation_sha = write_json(validation_path, validation)

    if not record_evidence:
        print(
            json.dumps(
                {
                    "status": validation.status,
                    "smoke_status": report.status,
                    "validation_sha256": validation_sha,
                    "plan_sha256": plan.plan_sha256,
                },
                sort_keys=True,
            )
        )
        return 0 if validation.status == "PASS" else 2

    run_report_sha = sha256_file(report_path)
    inventory = make_inventory(plan, report, run_report_sha, validation.gates)
    inventory_path = ROOT / "artifacts/handoff/smoke_experiment_inventory.json"
    inventory_sha = write_json(inventory_path, inventory)
    comparison_path = (
        ROOT / "results/smoke/runtime"
        / f"protected_foundation_hash_comparison_{report.run_mode}.json"
    )
    split_path = ROOT / "results/smoke/runtime/grouped_split_validation.json"
    comparison = ProtectedFoundationHashComparison.model_validate(
        json.loads(comparison_path.read_text(encoding="utf-8"))
    )
    split_validation = ProtectedSplitValidation.model_validate(
        json.loads(split_path.read_text(encoding="utf-8"))
    )
    commands = [
        (
            "conda run -n P12 python scripts/run_smoke_experiment.py "
            "--config configs/runtime/smoke_experiment.yaml"
        ),
        "conda run -n P12 python scripts/validate_smoke_experiment.py --record-evidence",
        "uv run --python D:\\Conda\\P12\\python.exe python -m ruff check .",
        "uv run --python D:\\Conda\\P12\\python.exe python -m mypy src/schemaguard",
        (
            "uv run --python D:\\Conda\\P12\\python.exe python -m pytest -q tests/unit "
            '-m "not network and not gpu and not foundation_model and not evidence" -rA'
        ),
    ]
    review = render_review(
        plan=plan,
        report=report,
        inventory=inventory,
        validation=validation,
        foundation_comparison=comparison,
        split_validation=split_validation,
        commands=commands,
    )
    review_path = ROOT / "artifacts/handoff/smoke_experiment_review.md"
    atomic_write_text(review_path, review)
    print(
        json.dumps(
            {
                "validation_status": validation.status,
                "smoke_status": report.status,
                "inventory_sha256": inventory_sha,
                "review_path": review_path.relative_to(ROOT).as_posix(),
                "plan_sha256": plan.plan_sha256,
            },
            sort_keys=True,
        )
    )
    return 0 if validation.status == "PASS" and report.status == "PASS_PENDING_REVIEW" else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-sha256")
    parser.add_argument("--report")
    parser.add_argument("--record-evidence", action="store_true")
    args = parser.parse_args()
    try:
        return validate(args.report, args.plan_sha256, args.record_evidence)
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "FAIL",
                    "error_category": type(exc).__name__,
                    "error": str(exc),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
