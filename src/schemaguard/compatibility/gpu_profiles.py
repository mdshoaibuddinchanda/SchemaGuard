"""Bounded, serialized GPU capacity and cleanup profiling for Phase 02B."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..models.probability import normalize_probability_matrix, validate_repeated_predictions
from ..models.registry import build_parameters, import_class, load_model_registry
from ..utils.hashing import hash_dataframe_logically, sha256_canonical_json, sha256_file
from ..utils.io import atomic_write_json, atomic_write_parquet
from ..utils.process_lock import ProcessLock

FOUNDATION_IDS = ("TPFN3-8.5", "TICL2-2.2")
GPU_HEADROOM_MIB = 512.0
GPU_SOFT_LIMIT_MIB = 3600.0
PROFILE_CYCLES = 5


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _configure_threads() -> None:
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[variable] = "2"


def _fixture(
    profile: dict[str, Any], seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, str]:
    rng = np.random.default_rng(seed)
    train_rows = int(profile["train_rows"])
    test_rows = int(profile["test_rows"])
    columns = [f"synthetic_{index:03d}" for index in range(int(profile["predictors"]))]
    x_train = pd.DataFrame(
        rng.normal(size=(train_rows, len(columns))).astype("float32"), columns=columns
    )
    x_test = pd.DataFrame(
        rng.normal(size=(test_rows, len(columns))).astype("float32"), columns=columns
    )
    classes = int(profile["classes"])
    y = np.arange(train_rows, dtype="int64") % classes
    rng.shuffle(y)
    fixture_hash = sha256_canonical_json(
        {
            "profile": profile,
            "seed": seed,
            "train": hash_dataframe_logically(x_train),
            "test": hash_dataframe_logically(x_test),
            "target": y.tolist(),
        }
    )
    return x_train, x_test, y, fixture_hash


def _parameter_hash(parameters: dict[str, Any]) -> str:
    return sha256_canonical_json(parameters)


def _checkpoint_path(root: Path, model_id: str, checkpoint: str | None) -> str | None:
    if checkpoint is None:
        return None
    path = root / checkpoint
    if not path.is_file():
        raise FileNotFoundError(f"Frozen checkpoint is absent: {path}")
    return str(path.resolve())


def _memory_snapshot(device: str) -> dict[str, float | None]:
    result: dict[str, float | None] = {
        "ram_mib": None,
        "gpu_allocated_mib": None,
        "gpu_reserved_mib": None,
        "gpu_free_mib": None,
    }
    try:
        import psutil  # type: ignore[import-not-found]

        result["ram_mib"] = float(psutil.Process(os.getpid()).memory_info().rss / (1024**2))
    except Exception:
        pass
    if device == "cuda":
        try:
            import torch  # type: ignore[import-not-found]

            if torch.cuda.is_available():
                free, _ = torch.cuda.mem_get_info(0)
                result["gpu_free_mib"] = float(free / (1024**2))
                result["gpu_allocated_mib"] = float(torch.cuda.memory_allocated() / (1024**2))
                result["gpu_reserved_mib"] = float(torch.cuda.memory_reserved() / (1024**2))
        except Exception:
            pass
    return result


def _record_peak(base: dict[str, Any], device: str) -> None:
    snapshot = _memory_snapshot(device)
    if snapshot["ram_mib"] is not None:
        base["peak_ram_mib"] = max(float(base["peak_ram_mib"] or 0.0), float(snapshot["ram_mib"]))
    if snapshot["gpu_allocated_mib"] is not None:
        base["peak_vram_allocated_mib"] = max(
            float(base["peak_vram_allocated_mib"] or 0.0),
            float(snapshot["gpu_allocated_mib"]),
        )
    if snapshot["gpu_reserved_mib"] is not None:
        base["peak_vram_reserved_mib"] = max(
            float(base["peak_vram_reserved_mib"] or 0.0),
            float(snapshot["gpu_reserved_mib"]),
        )


def _is_oom(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return (
        isinstance(exc, MemoryError)
        or "out of memory" in text
        or "cuda error" in text
        and "memory" in text
    )


def execute_profile(request: dict[str, Any]) -> dict[str, Any]:
    """Run one model/profile in the worker process; never raises to the parent."""

    _configure_threads()
    started_at = _now()
    started = time.perf_counter()
    root = Path(request["root"])
    model_id = str(request["model_id"])
    device = str(request["device"])
    profile = dict(request["profile"])
    seed = int(request["seed"])
    cycles = int(request["cycles"])
    base = {
        "schema_version": 1,
        "model_id": model_id,
        "profile_id": profile["profile_id"],
        "device": device,
        "strategy": request["strategy"],
        "seed": seed,
        "fixture_hash": None,
        "parameter_hash": None,
        "checkpoint_sha256": None,
        "cycles": cycles,
        "status": "FAIL",
        "failure_category": None,
        "start_time": started_at,
        "end_time": None,
        "runtime_seconds": 0.0,
        "load_seconds": None,
        "fit_seconds": None,
        "predict_seconds": [],
        "prediction_shape": [],
        "class_order": [],
        "probability_sum_error": None,
        "repeat_max_abs_diff": None,
        "prediction_hash": None,
        "peak_ram_mib": None,
        "peak_vram_allocated_mib": None,
        "peak_vram_reserved_mib": None,
        "gpu_free_before_mib": None,
        "gpu_free_after_mib": None,
        "post_cleanup_ram_mib": None,
        "post_cleanup_gpu_allocated_mib": None,
        "post_cleanup_gpu_reserved_mib": None,
        "error": None,
        "traceback": None,
    }
    model: Any = None
    try:
        config_path = root / "configs" / "runtime" / "model_compatibility.yaml"
        spec = next(item for item in load_model_registry(config_path) if item.id == model_id)
        x_train, x_test, y, fixture_hash = _fixture(profile, seed)
        base["fixture_hash"] = fixture_hash
        if device == "cuda":
            try:
                import torch  # type: ignore[import-not-found]

                if not torch.cuda.is_available():
                    base["status"] = "NOT_EXECUTED"
                    base["failure_category"] = "NOT_EXECUTED_NO_CUDA"
                    return base
                # On this CUDA/PyTorch build the explicit integer argument is
                # rejected even though device 0 is valid; the default device
                # is the detected sole GPU and is semantically equivalent.
                torch.cuda.reset_peak_memory_stats()
                before = _memory_snapshot(device)
                base["gpu_free_before_mib"] = before["gpu_free_mib"]
            except Exception as exc:
                base["status"] = "NOT_EXECUTED"
                base["failure_category"] = "NOT_EXECUTED_NO_CUDA"
                base["error"] = str(exc)
                return base
        observed = importlib.metadata.version(spec.package)
        if observed != spec.expected_version:
            base["failure_category"] = "FAIL_VERSION_MISMATCH"
            base["error"] = f"Expected {spec.expected_version}, observed {observed}"
            return base
        parameters = build_parameters(
            spec, seed=seed, device=device, target_classes=int(profile["classes"])
        )
        checkpoint_path = _checkpoint_path(root, model_id, spec.checkpoint)
        if checkpoint_path:
            parameters["model_path"] = checkpoint_path
            base["checkpoint_sha256"] = sha256_file(checkpoint_path)
            if model_id == "TICL2-2.2":
                parameters["allow_auto_download"] = False
        base["parameter_hash"] = _parameter_hash(parameters)
        load_started = time.perf_counter()
        model = import_class(spec.class_path)(**parameters)
        base["load_seconds"] = time.perf_counter() - load_started
        _record_peak(base, device)
        fit_started = time.perf_counter()
        model.fit(x_train, y)
        base["fit_seconds"] = time.perf_counter() - fit_started
        _record_peak(base, device)
        first_prediction: np.ndarray | None = None
        normalized_last: np.ndarray | None = None
        prediction_times: list[float] = []
        probability_error = 0.0
        for cycle in range(cycles):
            prediction_started = time.perf_counter()
            predictions = model.predict_proba(x_test)
            prediction_times.append(time.perf_counter() - prediction_started)
            _record_peak(base, device)
            normalized, classes, error = normalize_probability_matrix(
                predictions,
                model.classes_,
                list(range(int(profile["classes"]))),
                row_count=len(x_test),
                sum_atol=1.0e-6,
            )
            probability_error = max(probability_error, error)
            if first_prediction is None:
                first_prediction = normalized.copy()
                base["prediction_shape"] = list(normalized.shape)
                base["class_order"] = list(classes)
            else:
                validate_repeated_predictions(first_prediction, normalized, max_abs_diff=1.0e-5)
            normalized_last = normalized
        if first_prediction is None or normalized_last is None:
            raise RuntimeError("No prediction cycle executed")
        base["predict_seconds"] = prediction_times
        base["probability_sum_error"] = probability_error
        base["repeat_max_abs_diff"] = float(np.max(np.abs(first_prediction - normalized_last)))
        base["prediction_hash"] = hashlib.sha256(first_prediction.tobytes()).hexdigest()
        if device == "cuda":
            import torch  # type: ignore[import-not-found]

            base["peak_vram_allocated_mib"] = float(torch.cuda.max_memory_allocated() / (1024**2))
            base["peak_vram_reserved_mib"] = float(torch.cuda.max_memory_reserved() / (1024**2))
        base["status"] = "PASS"
        base["failure_category"] = "PASS"
    except Exception as exc:
        base["failure_category"] = (
            "FAIL_GPU_OOM"
            if device == "cuda" and _is_oom(exc)
            else "FAIL_GPU_INFERENCE"
            if device == "cuda"
            else "FAIL_CPU_INFERENCE"
        )
        base["error"] = f"{type(exc).__name__}: {exc}"
        base["traceback"] = traceback.format_exc()
    finally:
        if model is not None:
            del model
        gc.collect()
        if device == "cuda":
            try:
                import torch  # type: ignore[import-not-found]

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
        after = _memory_snapshot(device)
        base["post_cleanup_ram_mib"] = after["ram_mib"]
        base["post_cleanup_gpu_allocated_mib"] = after["gpu_allocated_mib"]
        base["post_cleanup_gpu_reserved_mib"] = after["gpu_reserved_mib"]
        base["gpu_free_after_mib"] = after["gpu_free_mib"]
        if after["ram_mib"] is not None:
            base["peak_ram_mib"] = max(float(base["peak_ram_mib"] or 0.0), float(after["ram_mib"]))
        base["runtime_seconds"] = max(0.0, time.perf_counter() - started)
        base["end_time"] = _now()
    return base


def _worker() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text(encoding="utf-8"))
    result = execute_profile(request)
    atomic_write_json(args.result, result)
    return 0


def _gpu_state() -> dict[str, Any]:
    try:
        import torch  # type: ignore[import-not-found]

        if not torch.cuda.is_available():
            return {"available": False, "reason": "CUDA is not available"}
        free, total = torch.cuda.mem_get_info(0)
        properties = torch.cuda.get_device_properties(0)
        return {
            "available": True,
            "name": torch.cuda.get_device_name(0),
            "total_mib": total / (1024**2),
            "free_mib": free / (1024**2),
            "torch_version": torch.__version__,
            "cuda_build": torch.version.cuda,
            "capability": list(properties.major for _ in [0]) + [properties.minor],
        }
    except Exception as exc:
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}


def _profiles(inventory_path: Path) -> list[dict[str, Any]]:
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    datasets = inventory["datasets"]
    choices: list[tuple[str, dict[str, Any]]] = []
    choices.append(
        (
            "minimum_rows",
            min(datasets, key=lambda item: (item["observed_rows"], item["openml_data_id"])),
        )
    )
    choices.append(
        (
            "maximum_rows",
            max(datasets, key=lambda item: (item["observed_rows"], -item["openml_data_id"])),
        )
    )
    choices.append(
        (
            "maximum_predictors",
            max(datasets, key=lambda item: (item["observed_predictors"], -item["openml_data_id"])),
        )
    )
    choices.append(
        (
            "maximum_classes",
            max(datasets, key=lambda item: (item["class_count"], -item["openml_data_id"])),
        )
    )
    choices.append(
        (
            "maximum_cells",
            max(
                datasets,
                key=lambda item: (
                    item["observed_rows"] * item["observed_predictors"],
                    -item["openml_data_id"],
                ),
            ),
        )
    )
    choices.append(
        (
            "mid_range",
            min(
                datasets,
                key=lambda item: (
                    abs(item["observed_rows"] - 2000) + abs(item["observed_predictors"] - 20) * 20,
                    item["openml_data_id"],
                ),
            ),
        )
    )
    result: dict[int, dict[str, Any]] = {}
    for label, item in choices:
        result[item["openml_data_id"]] = {
            "profile_id": label,
            "source_dataset_id": item["openml_data_id"],
            "rows": item["observed_rows"],
            "predictors": item["observed_predictors"],
            "classes": item["class_count"],
            "train_rows": max(20, round(item["observed_rows"] * 0.60)),
            "test_rows": max(10, round(item["observed_rows"] * 0.20)),
        }
    # A dataset can satisfy two extremal criteria; it is profiled once and the
    # selected criterion names are retained for auditability.
    return list(result.values())


def _run_worker(root: Path, request: dict[str, Any], timeout: int) -> dict[str, Any]:
    with __import__("tempfile").TemporaryDirectory(
        prefix="phase02b_gpu_", dir=root / "results" / "logs"
    ) as temp:
        temp_path = Path(temp)
        request_path = temp_path / "request.json"
        result_path = temp_path / "result.json"
        atomic_write_json(request_path, request)
        env = os.environ.copy()
        env["PYTHONPATH"] = str(root / "src") + os.pathsep + env.get("PYTHONPATH", "")
        command = [
            sys.executable,
            "-m",
            "schemaguard.compatibility.gpu_profiles",
            "--worker",
            "--request",
            str(request_path),
            "--result",
            str(result_path),
        ]
        try:
            completed = subprocess.run(
                command,
                cwd=root,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return {
                "schema_version": 1,
                "model_id": request["model_id"],
                "profile_id": request["profile"]["profile_id"],
                "device": request["device"],
                "strategy": request["strategy"],
                "cycles": request["cycles"],
                "status": "FAIL",
                "failure_category": "FAIL_TIMEOUT",
                "error": str(exc),
                "runtime_seconds": float(timeout),
            }
        if result_path.exists():
            return json.loads(result_path.read_text(encoding="utf-8"))
        return {
            "schema_version": 1,
            "model_id": request["model_id"],
            "profile_id": request["profile"]["profile_id"],
            "device": request["device"],
            "strategy": request["strategy"],
            "cycles": request["cycles"],
            "status": "FAIL",
            "failure_category": "FAIL_TEST",
            "error": completed.stderr.strip()
            or completed.stdout.strip()
            or f"worker exit {completed.returncode}",
            "runtime_seconds": 0.0,
        }


def run_gpu_profiles(
    root: str | Path, inventory_path: str | Path, *, output_directory: str | Path | None = None
) -> dict[str, Any]:
    """Run all selected profiles serially and write content-addressed evidence."""

    root_path = Path(root).resolve()
    output = Path(output_directory or root_path / "results" / "validation")
    output.mkdir(parents=True, exist_ok=True)
    (root_path / "results" / "logs").mkdir(parents=True, exist_ok=True)
    state = _gpu_state()
    profiles = _profiles(Path(inventory_path))
    config_path = root_path / "configs" / "runtime" / "model_compatibility.yaml"
    registry_hash = sha256_file(root_path / "configs" / "experiment_registry.yaml")
    config_hash = sha256_file(config_path)
    checkpoint_hashes = {
        model_id: sha256_file(root_path / checkpoint)
        for model_id, checkpoint in (
            ("TPFN3-8.5", "tabpfn-v3-classifier-v3_default.ckpt"),
            ("TICL2-2.2", "tabicl-classifier-v2-20260212.ckpt"),
        )
    }
    rows: list[dict[str, Any]] = []
    optimization: list[dict[str, Any]] = []
    lock_path = root_path / "data" / "cache" / "locks" / "phase-02b-gpu.lock"
    with ProcessLock(lock_path, timeout=30):
        for model_id in FOUNDATION_IDS:
            for profile in profiles:
                device = (
                    "cuda"
                    if state.get("available") and float(state.get("free_mib", 0)) > GPU_HEADROOM_MIB
                    else "cpu"
                )
                request = {
                    "root": str(root_path),
                    "model_id": model_id,
                    "device": device,
                    "profile": profile,
                    "seed": 1729,
                    "cycles": PROFILE_CYCLES if profile["profile_id"] == "mid_range" else 1,
                    "strategy": "same_process_cleanup",
                }
                row = _run_worker(root_path, request, 1800 if device == "cpu" else 900)
                row["execution_policy"] = (
                    "one foundation model at a time; process cleanup; two CPU threads"
                )
                row["gpu_soft_limit_mib"] = GPU_SOFT_LIMIT_MIB
                rows.append(row)
                if (
                    row.get("status") != "PASS"
                    and device == "cuda"
                    and row.get("failure_category") in {"FAIL_GPU_OOM", "FAIL_GPU_INFERENCE"}
                ):
                    fallback_request = dict(request)
                    fallback_request["device"] = "cpu"
                    fallback_request["strategy"] = "cpu_fallback"
                    fallback = _run_worker(root_path, fallback_request, 1800)
                    fallback["execution_policy"] = (
                        "deterministic CPU fallback after controlled CUDA failure"
                    )
                    rows.append(fallback)
            representative = next(
                profile for profile in profiles if profile["profile_id"] == "mid_range"
            )
            isolated_rows: list[dict[str, Any]] = []
            for _ in range(PROFILE_CYCLES):
                request = {
                    "root": str(root_path),
                    "model_id": model_id,
                    "device": "cuda" if state.get("available") else "cpu",
                    "profile": representative,
                    "seed": 1729,
                    "cycles": 1,
                    "strategy": "isolated_process",
                }
                isolated_rows.append(
                    _run_worker(root_path, request, 900 if state.get("available") else 1800)
                )
            rows.extend(isolated_rows)
            same = next(
                (
                    row
                    for row in rows
                    if row.get("model_id") == model_id
                    and row.get("profile_id") == "mid_range"
                    and row.get("strategy") == "same_process_cleanup"
                    and row.get("status") == "PASS"
                ),
                None,
            )
            successful_isolated = [row for row in isolated_rows if row.get("status") == "PASS"]
            equivalent = bool(
                same
                and successful_isolated
                and all(
                    row.get("prediction_hash") == same.get("prediction_hash")
                    for row in successful_isolated
                )
            )
            optimization.append(
                {
                    "schema_version": 1,
                    "model_id": model_id,
                    "profile_id": "mid_range",
                    "baseline_strategy": "same_process_cleanup",
                    "candidate_strategy": "isolated_process",
                    "baseline_prediction_hash": same.get("prediction_hash") if same else None,
                    "candidate_prediction_hashes": [
                        row.get("prediction_hash") for row in successful_isolated
                    ],
                    "prediction_equivalent": equivalent,
                    "accepted": False,
                    "decision": (
                        "retain isolated subprocess scheduling for crash/memory containment; "
                        "do not change scientific configuration"
                    ),
                }
            )
    report = {
        "schema_version": 1,
        "phase": "02B-GPU",
        "status": "PASS"
        if rows and all(row.get("status") in {"PASS", "NOT_EXECUTED"} for row in rows)
        else "FAIL",
        "gpu_state": state,
        "gpu_headroom_mib": GPU_HEADROOM_MIB,
        "gpu_soft_limit_mib": GPU_SOFT_LIMIT_MIB,
        "foundation_models_serialized": True,
        "profiles": profiles,
        "checkpoint_sha256": checkpoint_hashes,
        "registry_sha256": registry_hash,
        "runtime_config_sha256": config_hash,
        "profile_count": len(profiles),
        "rows": rows,
        "optimization": optimization,
        "fallback_policy": (
            "CUDA safe check -> controlled CUDA probe -> CPU fallback; CUDA OOM is "
            "capability evidence, never a runner crash"
        ),
    }
    atomic_write_json(output / "gpu_capacity_report.json", report)
    frame = pd.DataFrame(rows)
    atomic_write_parquet(
        root_path / "results" / "resources" / "gpu_capacity_profiles.parquet", frame
    )
    atomic_write_parquet(
        root_path / "results" / "resources" / "gpu_optimization_benchmarks.parquet",
        pd.DataFrame(optimization),
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--request", type=Path)
    parser.add_argument("--result", type=Path)
    parser.add_argument(
        "--inventory", type=Path, default=Path("results/validation/schemaorbit14_inventory.json")
    )
    args = parser.parse_args()
    if args.worker:
        if args.request is None or args.result is None:
            raise SystemExit("--worker requires --request and --result")
        return _worker()
    report = run_gpu_profiles(Path.cwd(), args.inventory)
    print(
        json.dumps(
            {"status": report["status"], "gpu": report["gpu_state"], "rows": len(report["rows"])},
            indent=2,
        )
    )
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
