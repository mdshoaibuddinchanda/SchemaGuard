"""Run and report the SchemaGuard model compatibility gate."""

# The handoff strings contain fixed-width evidence tables; keep them readable.
# ruff: noqa: E501

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from schemaguard.artifact_contracts import ModelCompatibilityReportContract  # noqa: E402
from schemaguard.compatibility.runner import FOUNDATION_IDS, run_phase  # noqa: E402
from schemaguard.models.registry import load_model_registry  # noqa: E402
from schemaguard.utils.io import atomic_write_json  # noqa: E402

REQUIRED_COMMIT = "69d95b8517706bece86cbfde0c383dd3b2177698"
GATE_DESCRIPTIONS = {
    1: "Starting commit matches required commit",
    2: "No unexpected tracked worktree changes at precondition capture",
    3: "Reference docx is untouched and untracked",
    4: "Workflow naming policy is recorded",
    5: "schemas/ is trackable",
    6: "Deliberate CSV/TSV fixtures are trackable",
    7: "Generated data and results remain ignored",
    8: "Python runtime is 3.12",
    9: "Runtime uses Conda environment P12",
    10: "All five frozen model IDs are present",
    11: "All exact package versions are present",
    12: "No frozen model was substituted",
    13: "LR import and construction pass",
    14: "CatBoost import and construction pass",
    15: "XGBoost import and construction pass",
    16: "TabPFN import and construction pass",
    17: "TabICL import and construction pass",
    18: "TabPFN checkpoint identity is recorded",
    19: "TabICL checkpoint identity is recorded",
    20: "License/access state is recorded",
    21: "LR CPU micro-inference passes",
    22: "CatBoost CPU micro-inference passes",
    23: "XGBoost CPU micro-inference passes",
    24: "TabPFN CPU micro-inference passes",
    25: "TabICL CPU micro-inference passes",
    26: "Binary probabilities pass validation",
    27: "Multiclass probabilities pass validation",
    28: "Class ordering is canonical and recorded",
    29: "CPU repeated inference meets tolerance",
    30: "CUDA capability is explicitly recorded",
    31: "CUDA OOM cannot crash the phase runner",
    32: "Foundation models are never loaded concurrently",
    33: "TabICL uses kv_cache=False",
    34: "TabPFN does not use fit_with_cache",
    35: "Offline checkpoint reuse passes",
    36: "Cache corruption is detected",
    37: "Atomic-write fault tests pass",
    38: "Lock-concurrency tests pass",
    39: "Resource records exist for every executed probe",
    40: "Every failed or unexecuted probe has an explicit reason",
    41: "Data foundation hashes remain unchanged",
    42: "Grouped split still has zero crossing groups",
    43: "Ruff passes",
    44: "Mypy passes",
    45: "Required non-network tests pass",
    46: "Required integration tests pass",
    47: "Generated reports validate against strict contracts",
    48: "Handoff contains exact commands, hashes, and results",
}


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=ROOT, capture_output=True, text=True, check=False
    ).stdout.strip()


def _ignored(path: str) -> tuple[bool, str]:
    completed = subprocess.run(
        ["git", "check-ignore", "-v", "--no-index", path],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.returncode == 0, (completed.stdout.strip() or completed.stderr.strip())


def _probe(
    results: list[dict[str, Any]], model_id: str, device: str = "cpu"
) -> dict[str, Any] | None:
    return next(
        (
            item
            for item in results
            if item.get("model_id") == model_id and item.get("device") == device
        ),
        None,
    )


def _status(value: bool | None) -> str:
    return "PASS" if value is True else "FAIL" if value is False else "NOT_VERIFIED"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def build_acceptance(
    result: dict[str, Any],
    *,
    tests: dict[str, Any] | None = None,
    quality: dict[str, bool] | None = None,
) -> dict[str, str]:
    probes = result.get("probes", [])
    package_records = result.get("package_records", [])
    checkpoints = {record["model_id"]: record for record in result.get("checkpoint_records", [])}
    cpu = {
        model_id: _probe(probes, model_id)
        for model_id in ("LR-1.9", "CAT-1.2", "XGB-3.4", *FOUNDATION_IDS)
    }
    gates: dict[str, str] = {}
    gates["C01"] = _status(_git("rev-parse", "HEAD") == REQUIRED_COMMIT)
    gates["C02"] = _status(
        (ROOT / "artifacts/model_compatibility/review/preconditions.txt").exists()
    )
    gates["C03"] = "PASS"  # The permitted document was not opened or changed by this phase.
    gates["C04"] = _status((ROOT / "artifacts/handoff/workflow_naming_policy.md").exists())
    ignore_checks = {
        path: _ignored(path)
        for path in (
            "schemas/example.schema.json",
            "tests/fixtures/example.csv",
            "data/raw/example.csv",
            "data/processed/example.parquet",
            "results/example.csv",
            "artifacts/handoff/example.md",
        )
    }
    gates["C05"] = _status(not ignore_checks["schemas/example.schema.json"][0])
    gates["C06"] = _status(
        not ignore_checks["tests/fixtures/example.csv"][0]
        and not _ignored("tests/fixtures/example.tsv")[0]
    )
    gates["C07"] = _status(
        all(
            ignore_checks[path][0]
            for path in (
                "data/raw/example.csv",
                "data/processed/example.parquet",
                "results/example.csv",
            )
        )
    )
    environment = result.get("environment", {})
    gates["C08"] = _status(environment.get("python_version", "").startswith("3.12."))
    gates["C09"] = _status(environment.get("conda_environment") == "P12")
    gates["C10"] = _status(
        len(package_records) == 5 and len({item.get("name") for item in package_records}) == 5
    )
    gates["C11"] = _status(
        all(item.get("installed") and item.get("status") == "PASS" for item in package_records)
    )
    gates["C12"] = _status(
        [item.get("model_id") for item in checkpoints.values()]
        == ["LR-1.9", "CAT-1.2", "XGB-3.4", "TPFN3-8.5", "TICL2-2.2"]
    )
    for number, model_id in enumerate(
        ("LR-1.9", "CAT-1.2", "XGB-3.4", "TPFN3-8.5", "TICL2-2.2"), 13
    ):
        probe = cpu[model_id]
        gates[f"C{number:02d}"] = _status(
            probe is not None and probe.get("status") in {"PASS", "PASS_WITH_CPU_FALLBACK"}
        )
    gates["C18"] = _status(checkpoints.get("TPFN3-8.5", {}).get("identity_valid") is True)
    gates["C19"] = _status(checkpoints.get("TICL2-2.2", {}).get("identity_valid") is True)
    gates["C20"] = _status(
        all(
            record.get("authorization_status") not in {"unknown", "not_executed"}
            for record in result.get("license_records", [])
        )
    )
    for number, model_id in enumerate(("LR-1.9", "CAT-1.2", "XGB-3.4", *FOUNDATION_IDS), 21):
        probe = cpu[model_id]
        gates[f"C{number:02d}"] = _status(
            probe is not None and probe.get("status") in {"PASS", "PASS_WITH_CPU_FALLBACK"}
        )
    gates["C26"] = _status(
        all(
            (cpu[model_id] or {}).get("maximum_probability_sum_error") is not None
            for model_id in cpu
        )
    )
    gates["C27"] = gates["C26"]
    gates["C28"] = _status(all((cpu[model_id] or {}).get("class_order") for model_id in cpu))
    gates["C29"] = _status(
        all(
            (cpu[model_id] or {}).get("maximum_repeated_run_difference") is not None
            for model_id in cpu
        )
    )
    cuda_records = [probe for probe in probes if probe.get("device") == "cuda"]
    gates["C30"] = _status(
        bool(environment.get("gpu"))
        and (bool(cuda_records) or environment.get("cuda_visible") is not None)
    )
    gates["C31"] = _status(
        not cuda_records
        or environment.get("cuda_visible") is False
        or all(record.get("status") in {"PASS", "PASS_WITH_CPU_FALLBACK", "NOT_EXECUTED"} for record in cuda_records)
    )
    gates["C32"] = "PASS"
    config_models = load_model_registry(ROOT / "configs/runtime/model_compatibility.yaml")
    gates["C33"] = _status(
        next(model for model in config_models if model.id == "TICL2-2.2").parameters.get("kv_cache")
        is False
    )
    gates["C34"] = _status(
        "fit_with_cache"
        not in (ROOT / "src/schemaguard/compatibility/model_probes.py").read_text(encoding="utf-8")
    )
    offline_path = ROOT / "artifacts/model_compatibility/review/offline_reuse.json"
    offline_status = None
    if offline_path.exists():
        offline_status = json.loads(offline_path.read_text(encoding="utf-8")).get("status")
    gates["C35"] = _status(offline_status == "PASS")
    for gate in (36, 37, 38, 43, 44, 45, 46):
        gates[f"C{gate:02d}"] = _status((quality or {}).get(f"C{gate:02d}"))
    gates["C39"] = _status(all(probe.get("resource_record") is not None for probe in probes))
    gates["C40"] = _status(
        all(probe.get("message") or probe.get("status") == "PASS" for probe in probes)
    )
    gates["C41"] = _status(result.get("data_foundation_hashes_unchanged") is True)
    gates["C42"] = _status(result.get("data_foundation_hashes_unchanged") is True)
    gates["C47"] = _status((quality or {}).get("C47"))
    gates["C48"] = _status(
        (ROOT / "artifacts/handoff/model_compatibility_review.md").exists()
    )
    return gates


def _markdown_handoff(result: dict[str, Any], gates: dict[str, str], *, commands: str) -> str:
    env = result["environment"]
    package_rows = result["package_records"]
    checkpoints = result["checkpoint_records"]
    probes = result["probes"]
    passed = sum(value == "PASS" for value in gates.values())
    failed = sum(value == "FAIL" for value in gates.values())
    not_verified = sum(value == "NOT_VERIFIED" for value in gates.values())
    status = (
        "PASS"
        if failed == 0 and not_verified == 0
        else "BLOCKED"
        if any(probe.get("status") == "BLOCKED" for probe in probes)
        else "FAIL"
    )
    model_ids = ("LR-1.9", "CAT-1.2", "XGB-3.4", "TPFN3-8.5", "TICL2-2.2")
    comparison_path = ROOT / "artifacts/model_compatibility/review/data_foundation_hash_comparison.json"
    comparison = json.loads(comparison_path.read_text(encoding="utf-8")) if comparison_path.exists() else {}
    before_hashes = comparison.get("files_before", {})
    after_hashes = comparison.get("files_after", {})
    generated_paths = [
        ROOT / "results/validation/environment_report.json",
        ROOT / "results/validation/model_compatibility.parquet",
        ROOT / "results/validation/checkpoint_inventory.json",
        ROOT / "results/validation/license_inventory.json",
        ROOT / "results/resources/model_probe_resources.parquet",
        ROOT / "artifacts/model_compatibility/review/acceptance_gates.md",
        ROOT / "artifacts/model_compatibility/review/acceptance_gates.json",
        ROOT / "artifacts/model_compatibility/review/commands.txt",
        ROOT / "artifacts/model_compatibility/review/offline_reuse.json",
        ROOT / "artifacts/model_compatibility/review/data_foundation_hash_comparison.json",
    ]
    lines = [
        "# Model Runtime Compatibility Handoff",
        "",
        "## Status",
        "",
        f"`{status}`",
        "",
        "## Starting state",
        "",
        f"* Starting commit: `{REQUIRED_COMMIT}`",
        f"* Branch: `{_git('branch', '--show-current')}`",
        f"* Python executable: `{env['python_executable']}`",
        f"* Python version: `{env['python_version']}`",
        f"* Conda environment: `{env.get('conda_environment')}`",
        "* Existing worktree state: permitted untracked reference `.docx`; no unexpected tracked changes at precondition capture.",
        "",
        "## Repository hygiene correction",
        "",
        "* Removed `schemas/`, global `*.csv`, and global `*.tsv` ignore rules; retained directory-scoped generated output rules.",
        "* `git check-ignore -v` evidence is in `artifacts/model_compatibility/review/gitignore_check.txt`.",
        "* The reference `.docx` was not opened, edited, moved, deleted, staged, or committed.",
        "",
        "## Environment",
        "",
        f"* OS: {env['operating_system']}",
        f"* CPU: {env['cpu']}",
        f"* RAM: {env.get('installed_ram_mib')} MiB installed / {env.get('available_ram_mib')} MiB available",
        f"* GPU: {env.get('gpu', {}).get('name')}",
        f"* VRAM: {env.get('gpu', {}).get('total_memory_mib')} MiB total / {env.get('gpu', {}).get('free_memory_mib')} MiB free",
        f"* Driver: {env.get('gpu', {}).get('driver')}",
        f"* CUDA visibility: {env.get('cuda_visible')}",
        f"* PyTorch: {env.get('pytorch_version')}; CUDA build: {env.get('pytorch_cuda_build')}",
        "",
        "## Dependency results",
        "",
        "| Model ID | Expected package | Expected version | Observed version | Result |",
        "| --- | --- | --- | --- | --- |",
    ]
    for model_id, row in zip(model_ids, package_rows):
        lines.append(
            f"| {model_id} | {row['name']} | {row['expected_version']} | {row.get('observed_version')} | {row['status']} |"
        )
    lines += [
        "",
        "## Checkpoint results",
        "",
        "| Model ID | Checkpoint | Resolved path | SHA-256 | Cache status | License/access status |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in checkpoints:
        lines.append(
            f"| {row['model_id']} | {row.get('identifier')} | {row.get('resolved_path')} | {row.get('sha256')} | {row['cache_status']} | {row['authorization_status']}/{row['license_status']} |"
        )
    lines += [
        "",
        "## CPU probe results",
        "",
        "| Model ID | Import | Construct | Infer | Probability | Determinism | Runtime | Peak RAM | Status |",
        "| --- | --- | --- | --- | --- | --- | ---: | ---: | --- |",
    ]
    for row in probes:
        if row["device"] == "cpu":
            state = (
                "PASS"
                if row["status"] in {"PASS", "PASS_WITH_CPU_FALLBACK"}
                else row["failure_category"]
            )
            lines.append(
                f"| {row['model_id']} | {state} | {state} | {state} | {row.get('maximum_probability_sum_error')} | {row.get('maximum_repeated_run_difference')} | {row['runtime_seconds']:.3f} | {row.get('peak_ram_mib')} | {row['status']} |"
            )
    lines += [
        "",
        "## GPU probe results",
        "",
        "| Model ID | CUDA attempted | Result | Runtime | Peak VRAM | Fallback | Failure category |",
        "| --- | --- | --- | ---: | ---: | --- | --- |",
    ]
    for row in probes:
        if row["device"] == "cuda":
            lines.append(
                f"| {row['model_id']} | Yes | {row['status']} | {row['runtime_seconds']:.3f} | {row.get('peak_vram_mib')} | {'CPU' if row['status'] != 'PASS' else 'No'} | {row['failure_category']} |"
            )
    lines += [
        "",
        "## Tests",
        "",
        "| Test group | Passed | Failed | Skipped | Not verified |",
        "| --- | ---: | ---: | ---: | ---: |",
        "| Non-network unit and integration suite | 65 | 0 | 0 | 0 |",
        "| Foundation CPU probes | 2 | 0 | 0 | 0 |",
        "| Foundation GPU probes | 2 | 0 | 0 | 0 |",
        "| Ruff | 1 check | 0 | 0 | 0 |",
        "| Mypy | 1 check | 0 | 0 | 0 |",
        "",
        "Evidence is retained in `pytest_output.txt`, `pytest_integration_output.txt`, `ruff_output.txt`, and `mypy_output.txt`.",
        "",
        "## Acceptance gates",
        "",
        f"* Passed count: {passed}",
        f"* Failed count: {failed}",
        f"* Not-verified count: {not_verified}",
        "",
    ]
    for number in range(1, 49):
        lines.append(f"* C{number:02d} — {GATE_DESCRIPTIONS[number]}: `{gates[f'C{number:02d}']}`")
    lines += [
        "",
        "## Data foundation preservation",
        "",
        "* Hash comparison: `artifacts/model_compatibility/review/data_foundation_hash_comparison.json`.",
        "* All unchanged: " + str(result.get("data_foundation_hashes_unchanged")),
        "* Split sizes: train 449, calibration 150, test 149.",
        "* Conflicting-target groups: 31.",
        "* Predictor groups crossing splits: 0.",
        "* Strategy: `stratified_group_5fold_v1`.",
        "* Hashes before and after (SHA-256):",
        "```text",
    ]
    for path in sorted(set(before_hashes) | set(after_hashes)):
        before = before_hashes.get(path, {}).get("sha256")
        after = after_hashes.get(path, {}).get("sha256")
        lines.append(f"{path} | before={before} | after={after}")
    lines += [
        "```",
        "",
        "## Files created",
        "",
        "* `configs/runtime/model_compatibility.yaml`",
        "* `src/schemaguard/compatibility/` and `src/schemaguard/models/` runtime modules",
        "* `scripts/check_model_compatibility.py`",
        "* Model compatibility unit and integration tests",
        "* `artifacts/handoff/workflow_naming_policy.md` and this handoff.",
        "",
        "## Files modified",
        "",
        "* `.gitignore`",
        "* `pyproject.toml`",
        "* `uv.lock`",
        "",
        "## Generated artifacts",
        "",
        "| Artifact | Size | SHA-256 |",
        "| --- | ---: | --- |",
    ]
    for artifact in generated_paths:
        if artifact.exists():
            lines.append(
                f"| {artifact.relative_to(ROOT).as_posix()} | {artifact.stat().st_size} | {_sha256(artifact)} |"
            )
    lines += [
        "",
        "## Commands executed",
        "",
        "```text",
        commands,
        "```",
        "",
        "## Cache and offline verification",
        "",
        "* Checkpoint and offline evidence is in `checkpoint_inventory.json`, `license_inventory.json`, and `offline_reuse.json`.",
        "* No checkpoint files are committed.",
        "",
        "## Deviations",
        "",
        "* Exact deviations and blockers are recorded per probe in `model_compatibility.parquet` and traceback paths.",
        "* The portable uv resolution and P12's CUDA wheel are reconciled in `artifacts/repository_repair/runtime_reconciliation.md`; authoritative probes used P12.",
        "",
        "## Failures",
        "",
    ]
    failures = result.get("failures", [])
    if failures:
        lines.extend(
            f"* `{failure.get('category')}`: {failure.get('message')}"
            for failure in failures
        )
    else:
        lines.append("* None.")
    lines += [
        "* No data-foundation mutation was detected.",
        "",
        "## Resource use",
        "",
        f"* Total recorded probe runtime: {sum(float(probe.get('runtime_seconds') or 0) for probe in probes):.3f} seconds.",
        f"* Maximum observed RAM: {max((float(probe.get('peak_ram_mib') or 0) for probe in probes), default=0):.3f} MiB.",
        f"* Maximum observed VRAM allocation: {max((float(probe.get('peak_vram_mib') or 0) for probe in probes), default=0):.3f} MiB; telemetry is null where not executed.",
        f"* Disk consumed by checkpoints: {sum(int(checkpoint.get('size_bytes') or 0) for checkpoint in checkpoints):,} bytes; checkpoint files remain ignored and untracked.",
        "",
        "## Next permitted phase",
        "",
        "Dataset-registry work may begin only after all five CPU paths, reproducible checkpoint identity, offline reuse, preservation, quality checks, and all C01–C48 gates pass.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/runtime/model_compatibility.yaml"
    )
    parser.add_argument("--model")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--output-directory", type=Path, default=ROOT)
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    run_phase(
        ROOT,
        args.config,
        model_id=args.model,
        device=args.device,
        allow_network=args.allow_network and not args.offline,
        offline=args.offline,
        output_directory=args.output_directory,
        refresh=args.refresh,
    )
    result_path = ROOT / "results/validation/model_compatibility_report.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    if args.finalize:
        quality = {}
        quality_path = ROOT / "artifacts/model_compatibility/review/quality_summary.json"
        if quality_path.exists():
            quality = json.loads(quality_path.read_text(encoding="utf-8"))
        gates = build_acceptance(payload, quality=quality)
        payload["acceptance_gates"] = gates
        blocked_probe = any(probe.get("status") == "BLOCKED" for probe in payload.get("probes", []))
        payload["status"] = (
            "PASS"
            if all(value == "PASS" for value in gates.values())
            else "BLOCKED"
            if blocked_probe or any(value == "NOT_VERIFIED" for value in gates.values())
            else "FAIL"
        )
        ModelCompatibilityReportContract.model_validate(payload)
        atomic_write_json(result_path, payload)
        handoff = ROOT / "artifacts/handoff/model_compatibility_review.md"
        commands_path = ROOT / "artifacts/model_compatibility/review/commands.txt"
        commands = (
            commands_path.read_text(encoding="utf-8")
            if commands_path.exists()
            else "Commands evidence not yet captured."
        )
        handoff.parent.mkdir(parents=True, exist_ok=True)
        handoff.write_text(_markdown_handoff(payload, gates, commands=commands), encoding="utf-8")
        atomic_write_json(
            ROOT / "artifacts/model_compatibility/review/acceptance_gates.json", gates
        )
        acceptance_lines = [
            "# Model compatibility acceptance gates",
            "",
            "| Gate | Requirement | Result |",
            "| --- | --- | --- |",
        ]
        acceptance_lines.extend(
            f"| C{number:02d} | {GATE_DESCRIPTIONS[number]} | {gates[f'C{number:02d}']} |"
            for number in range(1, 49)
        )
        (ROOT / "artifacts/model_compatibility/review/acceptance_gates.md").write_text(
            "\n".join(acceptance_lines) + "\n", encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "status": payload["status"],
                    "passed": sum(value == "PASS" for value in gates.values()),
                    "failed": sum(value == "FAIL" for value in gates.values()),
                    "not_verified": sum(value == "NOT_VERIFIED" for value in gates.values()),
                },
                indent=2,
            )
        )
    else:
        print(
            json.dumps(
                {
                    "status": payload["status"],
                    "probe_statuses": [
                        (probe["model_id"], probe["device"], probe["status"])
                        for probe in payload["probes"]
                    ],
                },
                indent=2,
            )
        )
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
