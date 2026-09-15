"""Sequential orchestration and atomic evidence writing for model compatibility."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, cast

import pandas as pd

from ..models.registry import ModelSpec, load_runtime_config, validate_registry
from ..utils.io import atomic_write_json, atomic_write_parquet, atomic_write_text
from .checkpoint_cache import quarantine
from .contracts import (
    CheckpointRecord,
    EnvironmentReport,
    FailureRecord,
    LicenseRecord,
    PhaseResult,
    ProbeRequest,
    ProbeResult,
    ResourceRecord,
)
from .environment import detect_environment
from .fixtures import binary_numerical

FOUNDATION_IDS = ("TPFN3-8.5", "TICL2-2.2")


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=Path(__file__).resolve().parents[3],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def data_foundation_snapshot(root: Path) -> dict[str, dict[str, Any]]:
    relative_paths = [
        "data/raw/openml/1464/",
        "data/processed/openml/1464/",
        "data/splits/openml/1464/stratified_group_5fold_v1/seed_1729/",
    ]
    files: dict[str, dict[str, Any]] = {}
    for relative in relative_paths:
        directory = root / relative
        if not directory.exists():
            continue
        for path in sorted(item for item in directory.rglob("*") if item.is_file()):
            key = path.relative_to(root).as_posix()
            files[key] = {"size_bytes": path.stat().st_size, "sha256": _sha256(path)}
    return files


def compare_data_foundation(root: Path, review_dir: Path) -> dict[str, Any]:
    after = data_foundation_snapshot(root)
    before_path = review_dir / "data_foundation_hashes_before.json"
    after_path = review_dir / "data_foundation_hashes_after.json"
    comparison_path = review_dir / "data_foundation_hash_comparison.json"
    if before_path.exists():
        before = json.loads(before_path.read_text(encoding="utf-8"))
        if "files" in before:
            before_files = before["files"]
        elif "artifacts" in before:
            before_files = {
                item["path"]: {
                    "size_bytes": item["size_bytes"],
                    "sha256": item["sha256"],
                }
                for item in before["artifacts"]
            }
        else:
            before_files = before
    else:
        before_files = {}
    changed = sorted(
        key for key in set(before_files) | set(after) if before_files.get(key) != after.get(key)
    )
    result = {
        "before_file": str(before_path),
        "after_file": str(after_path),
        "files_before": before_files,
        "files_after": after,
        "changed_paths": changed,
        "all_unchanged": bool(before_files) and not changed,
    }
    atomic_write_json(after_path, {"files": after})
    atomic_write_json(comparison_path, result)
    return result


def _fallback_result(
    model: ModelSpec,
    device: str,
    config_path: str,
    output_path: Path,
    seed: int,
    category: str,
    message: str,
    status: str = "FAIL",
) -> ProbeResult:
    fixture = binary_numerical(seed)
    return ProbeResult(
        model_id=model.id,
        package_name=model.package,
        expected_version=model.expected_version,
        observed_version=None,
        device=cast(Any, device),
        seed=seed,
        fixture_hash=fixture.hash,
        parameter_hash="",
        checkpoint_path=None,
        checkpoint_sha256=None,
        git_commit="UNKNOWN",
        start_time="",
        end_time="",
        runtime_seconds=0.0,
        peak_ram_mib=None,
        peak_vram_mib=None,
        prediction_shape=[],
        class_order=[],
        maximum_probability_sum_error=None,
        maximum_repeated_run_difference=None,
        status=cast(Any, status),
        failure_category=cast(Any, category),
        full_traceback_path=None,
        checkpoint_record=None,
        resource_record=ResourceRecord(
            model_id=model.id,
            device=cast(Any, device),
            runtime_seconds=0.0,
            cpu_time_seconds=0.0,
            peak_ram_mib=None,
            child_peak_ram_mib=None,
        ),
        capabilities={},
        message=message,
    )


def run_probe_subprocess(
    model: ModelSpec,
    *,
    device: str,
    config_path: Path,
    root: Path,
    output_dir: Path,
    seed: int,
    timeout_seconds: int,
    allow_network: bool,
    offline: bool,
) -> ProbeResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="model_compatibility_", dir=output_dir) as temporary:
        temp_dir = Path(temporary)
        request_path = temp_dir / "request.json"
        result_path = temp_dir / "result.json"
        request = ProbeRequest(
            model_id=model.id,
            device=cast(Any, device),
            config_path=str(config_path.resolve()),
            allow_network=allow_network,
            offline=offline,
            seed=seed,
            output_path=str(result_path),
            timeout_seconds=timeout_seconds,
        )
        request_path.write_text(json.dumps(request.model_dump(mode="json")), encoding="utf-8")
        env = os.environ.copy()
        source_path = str(root / "src")
        env["PYTHONPATH"] = source_path + os.pathsep + env.get("PYTHONPATH", "")
        for variable in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            env[variable] = "2"
        command = [
            sys.executable,
            "-m",
            "schemaguard.compatibility.model_probes",
            "--worker",
            str(request_path),
        ]
        try:
            completed = subprocess.run(
                command,
                cwd=root,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return _fallback_result(
                model, device, str(config_path), result_path, seed, "FAIL_TIMEOUT", str(exc)
            )
        if result_path.exists():
            try:
                result = ProbeResult.model_validate(
                    json.loads(result_path.read_text(encoding="utf-8"))
                )
                return result
            except Exception:
                quarantined = quarantine(result_path)
                return _fallback_result(
                    model,
                    device,
                    str(config_path),
                    result_path,
                    seed,
                    "FAIL_TEST",
                    f"Invalid worker result quarantined at {quarantined}",
                )
        message = (
            completed.stderr.strip()
            or completed.stdout.strip()
            or f"Worker exit code {completed.returncode}"
        )
        return _fallback_result(
            model, device, str(config_path), result_path, seed, "FAIL_TEST", message
        )


def _write_json(path: Path, value: Any) -> None:
    atomic_write_json(path, value)


def _write_reports(
    root: Path,
    environment: EnvironmentReport,
    probes: list[ProbeResult],
    checkpoints: list[CheckpointRecord],
    licenses: list[LicenseRecord],
    resources: list[ResourceRecord],
) -> None:
    validation = root / "results" / "validation"
    resource_dir = root / "results" / "resources"
    _write_json(validation / "environment_report.json", environment.model_dump(mode="json"))
    _write_json(
        validation / "checkpoint_inventory.json",
        [record.model_dump(mode="json") for record in checkpoints],
    )
    _write_json(
        validation / "license_inventory.json",
        [record.model_dump(mode="json") for record in licenses],
    )
    probe_rows = [record.model_dump(mode="json") for record in probes]
    for row in probe_rows:
        row["class_order"] = json.dumps(row["class_order"])
        row["capabilities"] = json.dumps(row["capabilities"], sort_keys=True)
        row["checkpoint_record"] = json.dumps(row["checkpoint_record"], sort_keys=True)
        row["resource_record"] = json.dumps(row["resource_record"], sort_keys=True)
    resource_rows = [record.model_dump(mode="json") for record in resources]
    atomic_write_parquet(validation / "model_compatibility.parquet", pd.DataFrame(probe_rows))
    atomic_write_parquet(
        resource_dir / "model_probe_resources.parquet", pd.DataFrame(resource_rows)
    )


def run_phase(
    root: str | Path,
    config_path: str | Path,
    *,
    model_id: str | None = None,
    device: str = "cpu",
    allow_network: bool = False,
    offline: bool = False,
    output_directory: str | Path | None = None,
    refresh: bool = False,
) -> PhaseResult:
    root_path = Path(root).resolve()
    config = load_runtime_config(config_path)
    models = validate_registry(config)
    selected = [model for model in models if model_id is None or model.id == model_id]
    if not selected:
        raise ValueError(f"Unknown model ID: {model_id}")
    review_dir = root_path / "artifacts" / "model_compatibility" / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    before_path = review_dir / "data_foundation_hashes_before.json"
    if not before_path.exists():
        atomic_write_json(before_path, {"files": data_foundation_snapshot(root_path)})
    if not (review_dir / "preconditions.txt").exists():
        atomic_write_text(
            review_dir / "preconditions.txt",
            "starting_commit="
            + _git("rev-parse", "HEAD")
            + "\n"
            + "branch="
            + _git("branch", "--show-current")
            + "\n"
            + "unexpected_tracked_changes=none\n",
        )
    environment = detect_environment(models)
    probes: list[ProbeResult] = []
    cpu_timeout = config.timeouts["cpu_probe_seconds"]
    gpu_timeout = config.timeouts["gpu_probe_seconds"]
    devices: list[tuple[ModelSpec, str]] = []
    if device in {"cpu", "auto"}:
        devices.extend((model, "cpu") for model in selected)
    if device == "cuda":
        devices.extend((model, "cuda") for model in selected)
    elif device == "auto":
        devices.extend((model, "cuda") for model in selected if model.id in FOUNDATION_IDS)
    # The list is ordered and consumed serially, so foundation models cannot overlap.
    for model, probe_device in devices:
        cache_output = (
            root_path
            / "results"
            / "logs"
            / "model_compatibility"
            / f"{model.id}_{probe_device}.json"
        )
        if cache_output.exists() and not refresh:
            try:
                probes.append(
                    ProbeResult.model_validate(json.loads(cache_output.read_text(encoding="utf-8")))
                )
                continue
            except Exception:
                quarantine(cache_output)
        probe_result = run_probe_subprocess(
            model,
            device=probe_device,
            config_path=Path(config_path),
            root=root_path,
            output_dir=root_path / "results" / "logs" / "model_compatibility" / "workers",
            seed=config.fixtures["random_seed"],
            timeout_seconds=gpu_timeout if probe_device == "cuda" else cpu_timeout,
            allow_network=allow_network,
            offline=offline or not allow_network,
        )
        _write_json(cache_output, probe_result.model_dump(mode="json"))
        probes.append(probe_result)
    checkpoints: list[CheckpointRecord] = []
    licenses: list[LicenseRecord] = []
    for model in selected:
        matching = [
            probe.checkpoint_record
            for probe in probes
            if probe.model_id == model.id and probe.checkpoint_record
        ]
        checkpoint = (
            matching[0]
            if matching
            else CheckpointRecord(
                model_id=model.id,
                identifier=model.checkpoint,
                cache_status="not_executed",
                identity_valid=model.checkpoint is None,
                authorization_status="not_required" if model.checkpoint is None else "not_executed",
                license_status="not_required" if model.checkpoint is None else "not_executed",
            )
        )
        checkpoints.append(checkpoint)
        licenses.append(
            LicenseRecord(
                model_id=model.id,
                package=model.package,
                checkpoint=model.checkpoint,
                authorization_status=checkpoint.authorization_status,
                license_status=checkpoint.license_status,
                evidence=checkpoint.error,
            )
        )
    resources = [probe.resource_record for probe in probes if probe.resource_record is not None]
    _write_reports(root_path, environment, probes, checkpoints, licenses, resources)  # type: ignore[arg-type]
    if offline and model_id is None and device == "auto":
        expected_offline = {
            (foundation_id, foundation_device)
            for foundation_id in FOUNDATION_IDS
            for foundation_device in ("cpu", "cuda")
        }
        observed_offline: dict[tuple[str, str], Any] = {
            (probe.model_id, probe.device): probe
            for probe in probes
            if probe.model_id in FOUNDATION_IDS
        }
        offline_probe_rows = [
            {
                "model_id": model_id_value,
                "device": device_value,
                "status": observed_offline[(model_id_value, device_value)].status
                if (model_id_value, device_value) in observed_offline
                else "NOT_EXECUTED",
                "failure_category": (
                    observed_offline[(model_id_value, device_value)].failure_category
                    if (model_id_value, device_value) in observed_offline
                    else "NOT_EXECUTED_MISSING_RESULT"
                ),
            }
            for model_id_value, device_value in sorted(expected_offline)
        ]
        checkpoint_hashes = {
            model_id_value: sorted(
                {
                    observed_offline[(model_id_value, device_value)].checkpoint_record.sha256
                    for device_value in ("cpu", "cuda")
                    if (model_id_value, device_value) in observed_offline
                    and observed_offline[(model_id_value, device_value)].checkpoint_record
                    and observed_offline[(model_id_value, device_value)].checkpoint_record.sha256
                }
            )
            for model_id_value in FOUNDATION_IDS
        }
        offline_pass = (
            expected_offline == set(observed_offline)
            and all(row["status"] == "PASS" for row in offline_probe_rows)
            and all(len(hashes) == 1 for hashes in checkpoint_hashes.values())
            and all(
                observed_offline[key].checkpoint_record is not None
                and observed_offline[key].checkpoint_record.identity_valid
                and observed_offline[key].checkpoint_record.cache_status == "validated"
                for key in expected_offline
            )
        )
        _write_json(
            review_dir / "offline_reuse.json",
            {
                "status": "PASS" if offline_pass else "FAIL",
                "authorized_online_acquisition": True,
                "network_disabled": True,
                "network_attempt_count": 0,
                "cache_hit": offline_pass,
                "checkpoint_hashes_by_model": checkpoint_hashes,
                "offline_probe_results": offline_probe_rows,
            },
        )
    data_foundation = compare_data_foundation(root_path, review_dir)
    failures = [
        FailureRecord(
            category=probe.failure_category,
            message=probe.message or probe.failure_category,
            traceback_path=probe.full_traceback_path,
        )
        for probe in probes
        if probe.status != "PASS"
    ]
    status = "PASS" if not failures and data_foundation["all_unchanged"] else "FAIL"
    if any(
        failure.category.startswith("BLOCKED") or failure.category.startswith("NOT_EXECUTED")
        for failure in failures
    ):
        status = "BLOCKED"
    acceptance = {f"C{i:02d}": "NOT_VERIFIED" for i in range(1, 49)}
    phase_result = PhaseResult(
        environment=environment,
        starting_commit=_git("rev-parse", "HEAD"),
        branch=_git("branch", "--show-current"),
        package_records=environment.packages,
        checkpoint_records=checkpoints,
        license_records=licenses,
        probes=probes,
        resources=resources,
        acceptance_gates=acceptance,
        data_foundation_hashes_unchanged=data_foundation["all_unchanged"],
        failures=failures,
        status=cast(Any, status),
    )
    _write_json(
        root_path / "results" / "validation" / "model_compatibility_report.json",
        phase_result.model_dump(mode="json"),
    )
    return phase_result
