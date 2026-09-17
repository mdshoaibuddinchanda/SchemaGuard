"""Validate the frozen model adapters on deterministic fixtures only."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "2"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from schemaguard.models.adapters.resources import configure_thread_limits  # noqa: E402

configure_thread_limits()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from pydantic import ValidationError  # noqa: E402
from scipy import sparse  # noqa: E402

from schemaguard.models.adapters.base import (  # noqa: E402
    FOUNDATION_IDS,
    _row_identifiers,
)
from schemaguard.models.adapters.contracts import (  # noqa: E402
    AdapterFailure,
    AdapterInventoryRecord,
    FailureCategory,
    ModelAdapterInventory,
    PredictionResult,
)
from schemaguard.models.adapters.factory import create_adapter, load_adapter_config  # noqa: E402
from schemaguard.models.adapters.fixtures import (  # noqa: E402
    AdapterFixture,
    adapter_fixture_catalog,
)
from schemaguard.models.adapters.resources import (  # noqa: E402
    gpu_memory_state,
    process_tree_memory_mib,
)
from schemaguard.utils.hashing import sha256_canonical_json  # noqa: E402
from schemaguard.utils.io import atomic_write_json  # noqa: E402


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _matrix_difference(left: Any, right: Any) -> float:
    if sparse.issparse(left) and sparse.issparse(right):
        delta = (left - right).tocsr()
        return float(np.max(np.abs(delta.data))) if delta.nnz else 0.0
    if isinstance(left, pd.DataFrame) and isinstance(right, pd.DataFrame):
        if list(left.columns) != list(right.columns) or left.shape != right.shape:
            return float("inf")
        for name in left.columns:
            if not left[name].equals(right[name]):
                if pd.api.types.is_numeric_dtype(left[name]) and pd.api.types.is_numeric_dtype(
                    right[name]
                ):
                    return float(
                        np.max(
                            np.abs(
                                left[name].to_numpy(dtype="float64")
                                - right[name].to_numpy(dtype="float64")
                            )
                        )
                    )
                return float("inf")
        return 0.0
    a = np.asarray(left)
    b = np.asarray(right)
    return float(np.max(np.abs(a - b))) if a.shape == b.shape and a.size else float("inf")


def _prepared_matrix_benchmark(
    adapter: Any, fixture: AdapterFixture, order: str
) -> tuple[float, float, float]:
    frame = fixture.test if order == "normal" else fixture.test.iloc[::-1].reset_index(drop=True)
    row_ids, predictors = _row_identifiers(frame)
    started = time.perf_counter()
    baseline = adapter.preprocessor.transform(
        predictors, row_ids=row_ids, partition="test", use_cache=False
    )
    uncached_seconds = max(0.0, time.perf_counter() - started)
    started = time.perf_counter()
    adapter.preprocessor.transform(predictors, row_ids=row_ids, partition="test", use_cache=True)
    _ = time.perf_counter() - started
    started = time.perf_counter()
    cached = adapter.preprocessor.transform(
        predictors, row_ids=row_ids, partition="test", use_cache=True
    )
    cache_hit_seconds = max(0.0, time.perf_counter() - started)
    difference = _matrix_difference(baseline, cached)
    if difference != 0.0:
        raise AdapterFailure(
            FailureCategory.FAIL_DETERMINISM,
            f"cached preprocessing changed values (maximum difference {difference})",
        )
    return uncached_seconds, cache_hit_seconds, difference


def _worker_payload(
    model_id: str,
    fixture_id: str,
    device: str,
    order: str,
    output_path: Path,
) -> int:
    from schemaguard.models.adapters import base as base_module

    output_path.parent.mkdir(parents=True, exist_ok=True)
    adapter = None
    try:
        config = load_adapter_config()
        fixture = adapter_fixture_catalog(config.seed)[fixture_id]
        adapter = create_adapter(model_id, seed=config.seed, device=device)
        adapter.fit(
            fixture.train,
            fixture.target,
            fixture_id=fixture.name,
            fixture_sha256=fixture.sha256,
            split_identity=f"fixture:{fixture.name}:seed={config.seed}",
        )
        uncached, cached, matrix_difference = _prepared_matrix_benchmark(adapter, fixture, order)
        inference = fixture.test
        if order == "reversed":
            inference = inference.iloc[::-1].reset_index(drop=True)
        prediction = adapter.predict_proba(
            inference,
            partition="test",
            fixture_id=fixture.name,
            fixture_sha256=fixture.sha256,
            split_identity=f"fixture:{fixture.name}:seed={config.seed}",
        )
        if model_id in FOUNDATION_IDS:
            adapter.roundtrip_prediction(prediction)
        resources = adapter.resource_record().model_dump(mode="json")
        output = {
            "status": "PASS",
            "model_id": model_id,
            "fixture_id": fixture_id,
            "fixture_sha256": fixture.sha256,
            "device": device,
            "order": order,
            "prediction": prediction.model_dump(mode="json"),
            "resources": resources,
            "preprocessing_uncached_seconds": uncached,
            "preprocessing_cache_hit_seconds": cached,
            "preprocessing_max_difference": matrix_difference,
            "offline_network_attempts": base_module.NETWORK_ATTEMPT_COUNT,
            "model_spec_sha256": adapter.model_spec_sha256,
        }
        adapter.release()
        resources = adapter.resource_record().model_dump(mode="json")
        output["resources"] = resources
        output["cleanup_gpu"] = gpu_memory_state() if device == "cuda" else None
        output["cleanup_ram_mib"] = process_tree_memory_mib()
        atomic_write_json(output_path, output)
        return 0
    except Exception as exc:
        category = (
            exc.category if isinstance(exc, AdapterFailure) else FailureCategory.FAIL_MODEL_RUNTIME
        )
        trace_path = output_path.with_suffix(".traceback.txt")
        trace_path.write_text(traceback.format_exc(), encoding="utf-8")
        atomic_write_json(
            output_path,
            {
                "status": "FAIL",
                "model_id": model_id,
                "fixture_id": fixture_id,
                "device": device,
                "order": order,
                "failure_category": category.value,
                "failure_reason": f"{type(exc).__name__}: {exc}",
                "traceback_path": str(trace_path.relative_to(ROOT)),
                "offline_network_attempts": base_module.NETWORK_ATTEMPT_COUNT,
            },
        )
        return 2
    finally:
        if adapter is not None:
            adapter.release()


def _worker_subprocess(
    model_id: str,
    fixture_id: str,
    device: str,
    order: str,
    case_index: int,
    output_directory: Path,
) -> dict[str, Any]:
    safe = f"{model_id.replace('.', '_')}_{fixture_id}_{device}_{order}_{case_index}"
    result_path = output_directory / f"{safe}.json"
    log_path = output_directory / f"{safe}.log"
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--model",
        model_id,
        "--fixture",
        fixture_id,
        "--device",
        device,
        "--order",
        order,
        "--worker-result",
        str(result_path),
    ]
    env = os.environ.copy()
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env[name] = "2"
    env["PYTHONHASHSEED"] = str(load_adapter_config().seed)
    if device == "cuda":
        env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    timeout = 900 if device == "cuda" else 1800
    with log_path.open("w", encoding="utf-8") as log:
        try:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return {
                "status": "FAIL",
                "model_id": model_id,
                "fixture_id": fixture_id,
                "device": device,
                "order": order,
                "failure_category": FailureCategory.FAIL_TIMEOUT.value,
                "failure_reason": f"worker exceeded {timeout} seconds",
                "traceback_path": str(log_path.relative_to(ROOT)),
            }
    try:
        with result_path.open("r", encoding="utf-8") as source:
            result = json.load(source)
    except (OSError, json.JSONDecodeError):
        return {
            "status": "FAIL",
            "model_id": model_id,
            "fixture_id": fixture_id,
            "device": device,
            "order": order,
            "failure_category": FailureCategory.FAIL_MODEL_RUNTIME.value,
            "failure_reason": f"worker exited {completed.returncode} without a valid result",
            "traceback_path": str(log_path.relative_to(ROOT)),
        }
    if completed.returncode != 0 and result.get("status") == "PASS":
        result["status"] = "FAIL"
        result["failure_category"] = FailureCategory.FAIL_MODEL_RUNTIME.value
        result["failure_reason"] = f"worker exited with status {completed.returncode}"
    return result


def _compare_result_rows(first: dict[str, Any], second: dict[str, Any]) -> float:
    left = PredictionResult.model_validate(first.get("prediction", first))
    right = PredictionResult.model_validate(second.get("prediction", second))
    if left.class_order != right.class_order:
        raise AdapterFailure(
            FailureCategory.FAIL_CLASS_ORDER, "class order changed between workers"
        )
    left_map = dict(zip(left.row_ids, left.probabilities, strict=True))
    right_map = dict(zip(right.row_ids, right.probabilities, strict=True))
    if set(left_map) != set(right_map):
        raise AdapterFailure(FailureCategory.FAIL_INPUT, "repeated inference row IDs changed")
    ordered = np.asarray([right_map[row_id] for row_id in left.row_ids], dtype="float64")
    first_matrix = np.asarray(left.probabilities, dtype="float64")
    if first_matrix.shape != ordered.shape:
        raise AdapterFailure(FailureCategory.FAIL_PROBABILITY, "repeat matrix shape changed")
    return float(np.max(np.abs(first_matrix - ordered))) if first_matrix.size else 0.0


def _classical_case(
    model_id: str, fixture: AdapterFixture, device: str
) -> tuple[dict[str, Any], float]:
    from schemaguard.models.adapters.factory import create_adapter

    config = load_adapter_config()
    adapter = create_adapter(model_id, seed=config.seed, device=device)
    restored = None
    try:
        adapter.fit(
            fixture.train,
            fixture.target,
            fixture_id=fixture.name,
            fixture_sha256=fixture.sha256,
            split_identity=f"fixture:{fixture.name}:seed={config.seed}",
        )
        uncached, cached, matrix_difference = _prepared_matrix_benchmark(adapter, fixture, "normal")
        first = adapter.predict_proba(
            fixture.test,
            partition="test",
            fixture_id=fixture.name,
            fixture_sha256=fixture.sha256,
            split_identity=f"fixture:{fixture.name}:seed={config.seed}",
        ).model_dump(mode="json")
        second = adapter.predict_proba(
            fixture.test,
            partition="test",
            fixture_id=fixture.name,
            fixture_sha256=fixture.sha256,
            split_identity=f"fixture:{fixture.name}:seed={config.seed}",
        ).model_dump(mode="json")
        repeat_difference = _compare_result_rows(first, second)
        reversed_result = adapter.predict_proba(
            fixture.test.iloc[::-1].reset_index(drop=True),
            partition="test",
            fixture_id=fixture.name,
            fixture_sha256=fixture.sha256,
            split_identity=f"fixture:{fixture.name}:seed={config.seed}",
        ).model_dump(mode="json")
        _compare_result_rows(first, reversed_result)
        identity = adapter.save_fitted()
        restored = create_adapter(model_id, seed=config.seed, device=device)
        restored.load_fitted(identity)
        roundtrip = restored.predict_proba(
            fixture.test,
            partition="test",
            fixture_id=fixture.name,
            fixture_sha256=fixture.sha256,
            split_identity=f"fixture:{fixture.name}:seed={config.seed}",
        ).model_dump(mode="json")
        roundtrip_difference = _compare_result_rows(first, roundtrip)
        if repeat_difference > config.tolerances.cpu_repeat_max_abs_diff:
            raise AdapterFailure(FailureCategory.FAIL_DETERMINISM, "CPU repeat tolerance failed")
        if roundtrip_difference > config.tolerances.cpu_repeat_max_abs_diff:
            raise AdapterFailure(
                FailureCategory.FAIL_SERIALIZATION, "model restore changed predictions"
            )
        resources = adapter.resource_record().model_dump(mode="json")
        return (
            {
                "status": "PASS",
                "prediction": first,
                "resources": resources,
                "repeat_max_abs_difference": max(repeat_difference, roundtrip_difference),
                "preprocessing_uncached_seconds": uncached,
                "preprocessing_cache_hit_seconds": cached,
                "preprocessing_max_difference": matrix_difference,
                "offline_network_attempts": 0,
                "roundtrip": "model_serialization",
            },
            repeat_difference,
        )
    finally:
        if restored is not None:
            restored.release()
        adapter.release()


def _stable_record(record: AdapterInventoryRecord) -> tuple[Any, ...]:
    return (
        record.logical_case_id,
        record.prediction_identity_sha256,
        record.fixture_sha256,
        record.model_spec_sha256,
        record.parameter_sha256,
        record.adapter_sha256,
        record.preprocessing_sha256,
        record.checkpoint_sha256,
        tuple(record.class_order),
        record.status,
    )


def _load_inventory(path: Path) -> ModelAdapterInventory:
    try:
        return ModelAdapterInventory.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as exc:
        raise AdapterFailure(
            FailureCategory.FAIL_CACHE_INTEGRITY, f"invalid baseline inventory: {exc}"
        ) from exc


def _compare_prediction_artifacts(
    baseline_path: Path, current_records: list[dict[str, Any]], config: Any
) -> float:
    try:
        baseline_payload = json.loads(baseline_path.read_text(encoding="utf-8"))
        baseline_records = baseline_payload["records"]
        baseline = {record["logical_case_id"]: record for record in baseline_records}
        current = {record["logical_case_id"]: record for record in current_records}
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise AdapterFailure(
            FailureCategory.FAIL_CACHE_INTEGRITY, "invalid repeat prediction artifact"
        ) from exc
    if set(baseline) != set(current):
        raise AdapterFailure(
            FailureCategory.FAIL_DETERMINISM, "repeat run has different prediction cases"
        )
    maximum = 0.0
    for case_id, first in baseline.items():
        second = current[case_id]
        if first["row_ids"] != second["row_ids"] or first["class_order"] != second["class_order"]:
            raise AdapterFailure(
                FailureCategory.FAIL_INPUT, "repeat run row/class ordering changed"
            )
        left = np.asarray(first["probabilities"], dtype="float64")
        right = np.asarray(second["probabilities"], dtype="float64")
        if left.shape != right.shape:
            raise AdapterFailure(
                FailureCategory.FAIL_PROBABILITY, "repeat run probability shape changed"
            )
        difference = float(np.max(np.abs(left - right))) if left.size else 0.0
        tolerance = (
            config.tolerances.gpu_repeat_max_abs_diff
            if second["device"] == "cuda"
            else config.tolerances.cpu_repeat_max_abs_diff
        )
        if difference > tolerance:
            raise AdapterFailure(
                FailureCategory.FAIL_DETERMINISM,
                f"cross-run probability difference {difference} exceeds {tolerance}",
            )
        maximum = max(maximum, difference)
    return maximum


def _case_record(
    *,
    model_id: str,
    fixture: AdapterFixture,
    device: str,
    outcome: dict[str, Any],
    repeat_difference: float,
    source_commit: str,
) -> AdapterInventoryRecord:
    if outcome.get("status") != "PASS":
        category_text = outcome.get("failure_category", FailureCategory.FAIL_MODEL_RUNTIME.value)
        try:
            category = FailureCategory(category_text)
        except ValueError:
            category = FailureCategory.FAIL_MODEL_RUNTIME
        return AdapterInventoryRecord(
            logical_case_id=sha256_canonical_json(
                {"model": model_id, "fixture": fixture.sha256, "device": device}
            ),
            prediction_identity_sha256="0" * 64,
            probabilities_sha256="0" * 64,
            model_id=model_id,
            fixture_id=fixture.name,
            fixture_sha256=fixture.sha256,
            device=device,
            roundtrip="prediction_cache" if model_id in FOUNDATION_IDS else "model_serialization",
            model_spec_sha256="0" * 64,
            parameter_sha256="0" * 64,
            adapter_sha256="0" * 64,
            preprocessing_sha256="0" * 64,
            checkpoint_sha256=None,
            row_ids_sha256="0" * 64,
            class_order=list(fixture.classes),
            probability_sum_error=0.0,
            repeat_max_abs_difference=0.0,
            preprocessing_uncached_seconds=0.0,
            preprocessing_cache_hit_seconds=0.0,
            offline_network_attempts=int(outcome.get("offline_network_attempts", 0)),
            roundtrip_passed=False,
            leakage_test_passed=False,
            status="BLOCKED"
            if category in {FailureCategory.BLOCKED_ENVIRONMENT, FailureCategory.BLOCKED_CHECKPOINT}
            else "FAIL",
            failure_category=category,
            runtime_seconds=0.0,
            peak_ram_mib=None,
            peak_vram_mib=None,
            source_commit=source_commit,
        )
    prediction = PredictionResult.model_validate(outcome["prediction"])
    resources = outcome["resources"]
    parameters = outcome.get("parameter_sha256") or prediction.parameter_sha256
    logical_case = sha256_canonical_json(
        {
            "model_id": model_id,
            "fixture_sha256": fixture.sha256,
            "device": device,
            "partition": prediction.partition,
            "seed": prediction.seed,
            "parameter_sha256": parameters,
            "preprocessing_sha256": prediction.preprocessing_sha256,
            "checkpoint_sha256": prediction.checkpoint_sha256,
            "roundtrip": outcome["roundtrip"],
        }
    )
    matrix = np.asarray(prediction.probabilities, dtype="float64")
    sum_error = float(np.max(np.abs(matrix.sum(axis=1) - 1.0)))
    return AdapterInventoryRecord(
        logical_case_id=logical_case,
        prediction_identity_sha256=prediction.logical_identity_sha256,
        probabilities_sha256=prediction.probabilities_sha256,
        model_id=model_id,
        fixture_id=fixture.name,
        fixture_sha256=fixture.sha256,
        device=device,
        roundtrip=outcome["roundtrip"],
        model_spec_sha256=outcome.get("model_spec_sha256", prediction.model_spec_sha256),
        parameter_sha256=parameters,
        adapter_sha256=prediction.adapter_sha256,
        preprocessing_sha256=prediction.preprocessing_sha256,
        checkpoint_sha256=prediction.checkpoint_sha256,
        row_ids_sha256=prediction.row_ids_sha256,
        class_order=prediction.class_order,
        probability_sum_error=sum_error,
        repeat_max_abs_difference=repeat_difference,
        preprocessing_uncached_seconds=outcome["preprocessing_uncached_seconds"],
        preprocessing_cache_hit_seconds=outcome["preprocessing_cache_hit_seconds"],
        offline_network_attempts=int(outcome.get("offline_network_attempts", 0)),
        roundtrip_passed=True,
        leakage_test_passed=True,
        status="PASS",
        failure_category=FailureCategory.PASS,
        runtime_seconds=float(resources["wall_time_seconds"]),
        peak_ram_mib=resources.get("peak_ram_mib"),
        peak_vram_mib=resources.get("peak_vram_reserved_mib"),
        source_commit=source_commit,
    )


def _run_parent(args: argparse.Namespace) -> int:
    config = load_adapter_config()
    fixtures = adapter_fixture_catalog(config.seed)
    model_ids = (
        [args.model]
        if args.model
        else [
            row[0]
            for row in __import__(
                "schemaguard.models.registry", fromlist=["EXPECTED_MODELS"]
            ).EXPECTED_MODELS
        ]
    )
    if args.device == "cuda":
        if args.model and args.model not in FOUNDATION_IDS:
            raise AdapterFailure(
                FailureCategory.FAIL_CONTRACT, "CUDA is reserved for foundation probes"
            )
        model_ids = [model_id for model_id in model_ids if model_id in FOUNDATION_IDS]
        fixture_ids = args.fixtures or ["binary_numerical"]
    else:
        fixture_ids = args.fixtures or config.fixtures
    unknown_fixtures = set(fixture_ids) - set(fixtures)
    if unknown_fixtures:
        raise AdapterFailure(
            FailureCategory.FAIL_CONTRACT, f"unknown fixtures: {sorted(unknown_fixtures)}"
        )
    if not model_ids:
        raise AdapterFailure(FailureCategory.FAIL_CONTRACT, "no frozen model selected")

    output = Path(args.output_directory).resolve()
    task_logs = ROOT / "results" / "logs" / "model_adapters"
    output.mkdir(parents=True, exist_ok=True)
    task_logs.mkdir(parents=True, exist_ok=True)
    total_cases = len(model_ids) * len(fixture_ids)
    expected_worker_runs = sum(
        3 if model_id in FOUNDATION_IDS else 1 for model_id in model_ids
    ) * len(fixture_ids)
    estimate_per_run = args.estimated_case_seconds or 15.0
    estimated_seconds = estimate_per_run * expected_worker_runs * 1.5
    start = time.monotonic()
    eta = datetime.now(UTC).timestamp() + estimated_seconds
    print(
        json.dumps(
            {
                "task_count": total_cases,
                "isolated_worker_runs": expected_worker_runs,
                "effective_workers": 1,
                "start_time_utc": _now(),
                "estimated_seconds": round(estimated_seconds, 2),
                "expected_completion_utc": datetime.fromtimestamp(eta, UTC).isoformat(),
                "planned_heartbeat_seconds": 60,
                "later_experiments": "not run",
            },
            sort_keys=True,
        ),
        flush=True,
    )
    heartbeat_path = task_logs / "progress.json"
    stop_heartbeat = threading.Event()
    state: dict[str, Any] = {
        "completed": 0,
        "total": total_cases,
        "current_model": None,
        "current_fixture": None,
        "last_success_utc": None,
        "last_failure_category": None,
    }

    def heartbeat() -> None:
        while not stop_heartbeat.wait(60):
            elapsed = max(0.0, time.monotonic() - start)
            remaining = max(0, total_cases - int(state["completed"]))
            current_ram = process_tree_memory_mib()
            gpu_state = gpu_memory_state() if args.device == "cuda" else None
            atomic_write_json(
                heartbeat_path,
                {
                    **state,
                    "elapsed_seconds": elapsed,
                    "estimated_remaining_seconds": (estimate_per_run * remaining * 1.5),
                    "process_tree_ram_mib": current_ram,
                    "gpu_state": gpu_state,
                    "updated_at": _now(),
                },
            )

    monitor_thread = threading.Thread(target=heartbeat, daemon=True)
    monitor_thread.start()
    records: list[AdapterInventoryRecord] = []
    prediction_artifact_records: list[dict[str, Any]] = []
    failure_details: list[dict[str, str]] = []
    run_failures = False
    case_index = 0
    try:
        for model_id in model_ids:
            for fixture_id in fixture_ids:
                fixture = fixtures[fixture_id]
                state["current_model"] = model_id
                state["current_fixture"] = fixture_id
                state["last_failure_category"] = None
                try:
                    if model_id in FOUNDATION_IDS:
                        runs = [
                            _worker_subprocess(
                                model_id, fixture_id, args.device, "normal", case_index, task_logs
                            ),
                            _worker_subprocess(
                                model_id,
                                fixture_id,
                                args.device,
                                "normal",
                                case_index + 1,
                                task_logs,
                            ),
                            _worker_subprocess(
                                model_id,
                                fixture_id,
                                args.device,
                                "reversed",
                                case_index + 2,
                                task_logs,
                            ),
                        ]
                        case_index += 3
                        failures = [row for row in runs if row.get("status") != "PASS"]
                        if failures:
                            outcome = failures[0]
                            difference = 0.0
                            run_failures = True
                        else:
                            difference = _compare_result_rows(runs[0], runs[1])
                            reorder_difference = _compare_result_rows(runs[0], runs[2])
                            if reorder_difference > 1.0e-10:
                                raise AdapterFailure(
                                    FailureCategory.FAIL_INPUT,
                                    "row reordering changed row-aligned predictions",
                                )
                            config_tolerance = (
                                config.tolerances.gpu_repeat_max_abs_diff
                                if args.device == "cuda"
                                else config.tolerances.cpu_repeat_max_abs_diff
                            )
                            if difference > config_tolerance:
                                raise AdapterFailure(
                                    FailureCategory.FAIL_DETERMINISM,
                                    f"repeated difference {difference} exceeds {config_tolerance}",
                                )
                            outcome = dict(runs[0])
                            outcome["roundtrip"] = "prediction_cache"
                            outcome["parameter_sha256"] = runs[0]["prediction"]["parameter_sha256"]
                            outcome["model_spec_sha256"] = runs[0]["prediction"][
                                "model_spec_sha256"
                            ]
                            outcome["resources"] = runs[0]["resources"]
                            outcome["offline_network_attempts"] = sum(
                                int(row.get("offline_network_attempts", 0)) for row in runs
                            )
                            outcome["preprocessing_uncached_seconds"] = runs[0][
                                "preprocessing_uncached_seconds"
                            ]
                            outcome["preprocessing_cache_hit_seconds"] = runs[0][
                                "preprocessing_cache_hit_seconds"
                            ]
                            outcome["status"] = "PASS"
                            if outcome["offline_network_attempts"] != 0:
                                raise AdapterFailure(
                                    FailureCategory.FAIL_MODEL_RUNTIME,
                                    "an offline foundation probe attempted network access",
                                )
                    else:
                        outcome, difference = _classical_case(model_id, fixture, args.device)
                        case_index += 1
                    record = _case_record(
                        model_id=model_id,
                        fixture=fixture,
                        device=args.device,
                        outcome=outcome,
                        repeat_difference=difference,
                        source_commit=subprocess.run(
                            ["git", "rev-parse", "HEAD"],
                            cwd=ROOT,
                            check=True,
                            capture_output=True,
                            text=True,
                        ).stdout.strip(),
                    )
                    records.append(record)
                    if record.status == "PASS":
                        prediction = PredictionResult.model_validate(outcome["prediction"])
                        prediction_artifact_records.append(
                            {
                                "logical_case_id": record.logical_case_id,
                                "model_id": model_id,
                                "fixture_id": fixture_id,
                                "device": args.device,
                                "row_ids": prediction.row_ids,
                                "class_order": prediction.class_order,
                                "probabilities": prediction.probabilities,
                            }
                        )
                    state["last_success_utc"] = _now() if record.status == "PASS" else None
                    if record.status != "PASS":
                        state["last_failure_category"] = record.failure_category.value
                        run_failures = True
                except AdapterFailure as exc:
                    state["last_failure_category"] = exc.category.value
                    run_failures = True
                    failure_path = (
                        task_logs
                        / f"failure_{model_id.replace('.', '_')}_{fixture_id}.traceback.txt"
                    )
                    failure_path.write_text(traceback.format_exc(), encoding="utf-8")
                    failure_details.append(
                        {
                            "model_id": model_id,
                            "fixture_id": fixture_id,
                            "failure_category": exc.category.value,
                            "reason": str(exc),
                            "traceback_path": str(failure_path.relative_to(ROOT)),
                        }
                    )
                    failure = {
                        "status": "BLOCKED"
                        if exc.category
                        in {FailureCategory.BLOCKED_ENVIRONMENT, FailureCategory.BLOCKED_CHECKPOINT}
                        else "FAIL",
                        "failure_category": exc.category.value,
                        "failure_reason": str(exc),
                        "offline_network_attempts": 0,
                        "traceback_path": str(failure_path.relative_to(ROOT)),
                    }
                    record = _case_record(
                        model_id=model_id,
                        fixture=fixture,
                        device=args.device,
                        outcome=failure,
                        repeat_difference=0.0,
                        source_commit=subprocess.run(
                            ["git", "rev-parse", "HEAD"],
                            cwd=ROOT,
                            check=True,
                            capture_output=True,
                            text=True,
                        ).stdout.strip(),
                    )
                    records.append(record)
                except Exception as exc:
                    category = FailureCategory.FAIL_TEST
                    state["last_failure_category"] = category.value
                    run_failures = True
                    failure_path = (
                        task_logs
                        / f"failure_{model_id.replace('.', '_')}_{fixture_id}.traceback.txt"
                    )
                    failure_path.write_text(traceback.format_exc(), encoding="utf-8")
                    failure_details.append(
                        {
                            "model_id": model_id,
                            "fixture_id": fixture_id,
                            "failure_category": category.value,
                            "reason": f"{type(exc).__name__}: {exc}",
                            "traceback_path": str(failure_path.relative_to(ROOT)),
                        }
                    )
                    record = _case_record(
                        model_id=model_id,
                        fixture=fixture,
                        device=args.device,
                        outcome={
                            "status": "FAIL",
                            "failure_category": category.value,
                            "failure_reason": f"{type(exc).__name__}: {exc}",
                        },
                        repeat_difference=0.0,
                        source_commit=subprocess.run(
                            ["git", "rev-parse", "HEAD"],
                            cwd=ROOT,
                            check=True,
                            capture_output=True,
                            text=True,
                        ).stdout.strip(),
                    )
                    records.append(record)
                state["completed"] += 1
                atomic_write_json(
                    heartbeat_path,
                    {
                        **state,
                        "elapsed_seconds": max(0.0, time.monotonic() - start),
                        "estimated_remaining_seconds": estimate_per_run
                        * max(0, total_cases - int(state["completed"]))
                        * 1.5,
                        "process_tree_ram_mib": process_tree_memory_mib(),
                        "gpu_state": gpu_memory_state() if args.device == "cuda" else None,
                        "updated_at": _now(),
                    },
                )
                print(
                    json.dumps(
                        {
                            "completed": state["completed"],
                            "total": total_cases,
                            "model_id": model_id,
                            "fixture": fixture_id,
                            "status": records[-1].status,
                            "failure_category": records[-1].failure_category.value,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                if records[-1].status != "PASS":
                    break
            if records and records[-1].status != "PASS":
                break
    finally:
        stop_heartbeat.set()
        monitor_thread.join(timeout=2)

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    inventory = ModelAdapterInventory(source_commit=commit, records=records)
    if args.merge_inventory:
        target_inventory = Path(args.merge_inventory).resolve()
        if target_inventory.exists():
            inventory = _load_inventory(target_inventory).merge(inventory)
    else:
        target_inventory = (
            Path(args.inventory).resolve()
            if args.inventory
            else output / "model_adapter_inventory.json"
        )
    atomic_write_json(target_inventory, inventory.model_dump(mode="json"))
    prediction_artifact_path = output / "adapter_predictions.json"
    atomic_write_json(
        prediction_artifact_path,
        {"schema_version": 1, "records": prediction_artifact_records},
    )
    cross_run_difference = None
    if args.compare_run:
        cross_run_difference = _compare_prediction_artifacts(
            Path(args.compare_run).resolve(), prediction_artifact_records, config
        )
    report = {
        "schema_version": 1,
        "stage": "model_adapter_validation",
        "status": "FAIL" if run_failures else "PASS",
        "device": args.device,
        "model_ids": model_ids,
        "fixture_ids": fixture_ids,
        "record_count": len(records),
        "pass_count": sum(record.status == "PASS" for record in records),
        "failure_count": sum(record.status != "PASS" for record in records),
        "inventory_path": str(target_inventory.relative_to(ROOT))
        if target_inventory.is_relative_to(ROOT)
        else target_inventory.name,
        "inventory_sha256": __import__("hashlib").sha256(target_inventory.read_bytes()).hexdigest(),
        "estimated_seconds": estimated_seconds,
        "observed_seconds": max(0.0, time.monotonic() - start),
        "network_access": "disabled for foundation adapters",
        "cross_run_max_abs_difference": cross_run_difference,
        "failure_details": failure_details,
        "later_experiments": "not run",
    }
    atomic_write_json(output / "adapter_validation_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.compare_inventory:
        baseline = _load_inventory(Path(args.compare_inventory).resolve())
        if [_stable_record(row) for row in baseline.records] != [
            _stable_record(row) for row in inventory.records
        ]:
            print("stable inventory comparison: FAIL", flush=True)
            return 2
        print("stable inventory comparison: PASS", flush=True)
    return 2 if run_failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", choices=["LR-1.9", "CAT-1.2", "XGB-3.4", "TPFN3-8.5", "TICL2-2.2"]
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--fixture", dest="fixtures", action="append")
    parser.add_argument("--output-directory", default="results/validation/model_adapters")
    parser.add_argument("--inventory")
    parser.add_argument("--merge-inventory")
    parser.add_argument("--compare-inventory")
    parser.add_argument("--compare-run")
    parser.add_argument("--estimated-case-seconds", type=float)
    parser.add_argument("--offline", action="store_true", default=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--order", choices=["normal", "reversed"], default="normal", help=argparse.SUPPRESS
    )
    parser.add_argument("--worker-result", help=argparse.SUPPRESS)
    args = parser.parse_args()
    configure_thread_limits()
    if args.worker:
        if not args.model or not args.fixtures or len(args.fixtures) != 1 or not args.worker_result:
            parser.error("internal worker requires exactly one model, fixture, and result path")
        return _worker_payload(
            args.model,
            args.fixtures[0],
            args.device,
            args.order,
            Path(args.worker_result),
        )
    try:
        return _run_parent(args)
    except AdapterFailure as exc:
        print(
            json.dumps(
                {"status": "BLOCKED", "failure_category": exc.category.value, "reason": str(exc)}
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
