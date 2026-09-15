"""Isolated model probes for the Phase 02A gate.

The worker entry point is deliberately small and subprocess-safe. Optional model
packages are imported only inside the worker, never while importing this module.
"""

# ruff: noqa: E501

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from ..models.probability import (
    ProbabilityValidationError,
    normalize_probability_matrix,
    validate_repeated_predictions,
)
from ..models.registry import (
    ModelSpec,
    build_parameters,
    import_class,
    load_model_registry,
    validate_constructor_parameters,
)
from ..utils.io import atomic_write_json
from ..utils.resource_monitor import ResourceMonitor
from .checkpoint_cache import register_checkpoint, sha256_file, validate_checkpoint_file
from .contracts import CheckpointRecord, ProbeRequest, ProbeResult, ResourceRecord
from .fixtures import FeatureFixture, binary_numerical, multiclass_numerical


class CheckpointAuthorizationRequired(RuntimeError):
    """Raised when a gated checkpoint needs explicit user authorization."""


class OfflineCacheMiss(RuntimeError):
    """Raised when an offline run cannot find a validated local checkpoint."""


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _git_commit() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10, check=True
        )
        return completed.stdout.strip()
    except Exception:
        return "UNKNOWN"


def _parameter_hash(parameters: dict[str, Any]) -> str:
    payload = json.dumps(parameters, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(payload).hexdigest()


def _data_frame(fixture: FeatureFixture, *, native_categorical: bool = False) -> tuple[Any, Any]:

    x_train = fixture.train.drop(columns=["row_id"]).copy()
    x_test = fixture.test.drop(columns=["row_id"]).copy()
    if not native_categorical:
        categorical = x_train.select_dtypes(include=["object", "category"]).columns
        for column in categorical:
            categories = sorted(set(x_train[column].astype(str)))
            mapping = {value: index for index, value in enumerate(categories)}
            x_train[column] = x_train[column].astype(str).map(mapping).astype(float)
            x_test[column] = x_test[column].astype(str).map(mapping).fillna(-1).astype(float)
    return x_train, x_test


def _checkpoint_from_model(
    model: Any,
    spec: ModelSpec,
    *,
    request: ProbeRequest | None = None,
    parameters: dict[str, Any] | None = None,
) -> CheckpointRecord:
    if spec.checkpoint is None:
        return CheckpointRecord(
            model_id=spec.id,
            cache_status="not_applicable",
            identity_valid=True,
            authorization_status="not_required",
            license_status="not_required",
        )
    candidates: list[Any] = []
    for name in (
        "checkpoint_path",
        "model_path",
        "model_path_",
        "_model_path",
        "checkpoint",
        "_checkpoint_path",
    ):
        if hasattr(model, name):
            candidates.append(getattr(model, name))
    if parameters and parameters.get("model_path") is not None:
        candidates.append(parameters["model_path"])
    for candidate in candidates:
        if isinstance(candidate, (str, os.PathLike)):
            path = Path(candidate).expanduser()
            if path.is_file():
                digest = sha256_file(path)
                cache_key: str | None = None
                cache_status = "resolved"
                if request is not None:
                    root = Path(request.config_path).resolve().parents[2]
                    cache_dir = root / "data" / "cache" / "models" / spec.package
                    try:
                        try:
                            pytorch_version = importlib.metadata.version("torch")
                        except importlib.metadata.PackageNotFoundError:
                            pytorch_version = None
                        metadata = register_checkpoint(
                            path,
                            cache_dir=cache_dir,
                            model_id=spec.id,
                            package_version=spec.expected_version,
                            checkpoint_identifier=spec.checkpoint,
                            device_policy=request.device,
                            model_parameters=parameters or {},
                            python_major_minor=f"{sys.version_info.major}.{sys.version_info.minor}",
                            pytorch_version=pytorch_version,
                            code_commit=_git_commit(),
                        )
                        validate_checkpoint_file(metadata)
                        cache_key = metadata.cache_key
                        cache_status = "validated"
                    except Exception as exc:
                        return CheckpointRecord(
                            model_id=spec.id,
                            identifier=spec.checkpoint,
                            resolved_path=str(path.resolve()),
                            sha256=digest,
                            size_bytes=path.stat().st_size,
                            cache_status="invalid",
                            identity_valid=False,
                            authorization_status="resolved",
                            license_status="package_default",
                            error=f"Checkpoint cache validation failed: {exc}",
                        )
                return CheckpointRecord(
                    model_id=spec.id,
                    identifier=spec.checkpoint,
                    resolved_path=str(path.resolve()),
                    sha256=digest,
                    size_bytes=path.stat().st_size,
                    cache_key=cache_key,
                    cache_status=cache_status,
                    identity_valid=True,
                    authorization_status="resolved",
                    license_status="package_default",
                )
    return CheckpointRecord(
        model_id=spec.id,
        identifier=spec.checkpoint,
        cache_status="unresolved",
        identity_valid=False,
        authorization_status="unknown",
        license_status="unknown",
        error="The installed model did not expose a verifiable checkpoint path",
    )


def _base_result(
    request: ProbeRequest,
    spec: ModelSpec,
    fixture_hash: str,
    parameter_hash: str,
    start: str,
    end: str,
    runtime: float,
    *,
    status: str,
    category: str,
    message: str,
    observed_version: str | None = None,
    checkpoint: CheckpointRecord | None = None,
    resource: ResourceRecord | None = None,
    prediction_shape: list[int] | None = None,
    class_order: list[int | str] | None = None,
    probability_error: float | None = None,
    repeated_difference: float | None = None,
    traceback_path: str | None = None,
    capabilities: dict[str, Any] | None = None,
) -> ProbeResult:
    return ProbeResult(
        model_id=spec.id,
        package_name=spec.package,
        expected_version=spec.expected_version,
        observed_version=observed_version,
        device=request.device,
        seed=request.seed,
        fixture_hash=fixture_hash,
        parameter_hash=parameter_hash,
        git_commit=_git_commit(),
        start_time=start,
        end_time=end,
        runtime_seconds=runtime,
        peak_ram_mib=resource.peak_ram_mib if resource else None,
        peak_vram_mib=resource.gpu_allocated_mib if resource else None,
        prediction_shape=prediction_shape or [],
        class_order=class_order or [],
        maximum_probability_sum_error=probability_error,
        maximum_repeated_run_difference=repeated_difference,
        status=status,  # type: ignore[arg-type]
        failure_category=cast(Any, category),
        full_traceback_path=traceback_path,
        checkpoint_record=checkpoint,
        resource_record=resource,
        capabilities=capabilities or {},
        message=message,
    )


def _traceback_path(request: ProbeRequest, spec: ModelSpec, text: str) -> str:
    directory = Path(request.output_path).parent / "tracebacks"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{spec.id.replace('-', '_')}_{request.device}.txt"
    path.write_text(text, encoding="utf-8")
    return str(path)


def _check_device(request: ProbeRequest) -> tuple[bool, str | None, str | None]:
    if request.device == "cpu":
        return True, None, None
    try:
        import torch  # type: ignore[import-not-found]

        if not torch.cuda.is_available():
            return False, "NOT_EXECUTED_NO_CUDA", "CUDA is unavailable"
        free, _ = torch.cuda.mem_get_info(0)
        if free / (1024**2) <= 512:
            return False, "NOT_EXECUTED_UNSAFE_VRAM", "Configured GPU headroom is not available"
    except Exception as exc:
        return False, "NOT_EXECUTED_NO_CUDA", f"CUDA detection failed: {type(exc).__name__}: {exc}"
    return True, None, None


def _disable_network() -> None:
    """Make an offline foundation probe incapable of opening a network socket."""
    import socket

    def blocked(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("OFFLINE_NETWORK_BLOCK: network access is disabled for this probe")

    socket.create_connection = blocked  # type: ignore[assignment]
    socket.socket.connect = blocked  # type: ignore[assignment]


def _foundation_parameters(
    request: ProbeRequest, spec: ModelSpec, parameters: dict[str, Any]
) -> dict[str, Any]:
    """Resolve the installed package's supported checkpoint path without guessing."""
    resolved = dict(parameters)
    root = Path(request.config_path).resolve().parents[2]
    if spec.id == "TPFN3-8.5":
        from tabpfn.constants import ModelVersion  # type: ignore[import-not-found]
        from tabpfn.model_loading import resolve_model_path  # type: ignore[import-not-found]

        path, _, _, _ = resolve_model_path(
            spec.checkpoint, which="classifier", version=ModelVersion.V3.value
        )
        checkpoint_path = path[0]
        if not checkpoint_path.is_file():
            raise CheckpointAuthorizationRequired(
                "TabPFN v3 checkpoint is not cached; its gated license/authentication requires explicit user authorization"
            )
        resolved["model_path"] = str(checkpoint_path)
    elif spec.id == "TICL2-2.2":
        cached_checkpoint_path = (
            root / "data" / "cache" / "models" / "tabicl" / (spec.checkpoint or "checkpoint.ckpt")
        )
        # A user-provided exact checkpoint may be staged at the repository root
        # for an offline verification run.  Prefer it only when its frozen
        # identifier matches exactly; otherwise use the content-addressed cache.
        root_checkpoint_path = root / (spec.checkpoint or "checkpoint.ckpt")
        checkpoint_path = (
            root_checkpoint_path if root_checkpoint_path.is_file() else cached_checkpoint_path
        )
        if not checkpoint_path.is_file() and request.offline:
            raise OfflineCacheMiss(f"TabICL checkpoint is not cached: {checkpoint_path}")
        resolved["model_path"] = str(checkpoint_path)
        resolved["allow_auto_download"] = request.allow_network and not request.offline
    return resolved


def _fit_and_validate(
    model: Any,
    fixture: FeatureFixture,
    spec: ModelSpec,
    seed: int,
    device: str,
    sum_atol: float,
    repeat_atol: float,
) -> tuple[list[int], list[int | str], float, float, dict[str, Any]]:
    capabilities: dict[str, Any] = {"binary": {}, "multiclass": {}}
    if spec.id == "CAT-1.2":
        capabilities["native_categorical"] = "tested_separately"
    binary = binary_numerical(seed)
    x_train, x_test = _data_frame(binary)
    y = binary.target
    model.fit(x_train, y)
    first = model.predict_proba(x_test)
    second = model.predict_proba(x_test)
    normalized, classes, probability_error = normalize_probability_matrix(
        first, model.classes_, [0, 1], row_count=len(x_test), sum_atol=sum_atol
    )
    repeated_difference = validate_repeated_predictions(
        normalized,
        second[:, [list(model.classes_).index(c) for c in [0, 1]]],
        max_abs_diff=repeat_atol,
    )
    capabilities["binary"] = {
        "prediction_shape": list(normalized.shape),
        "fixture_hash": binary.hash,
    }

    multi = multiclass_numerical(seed)
    mx_train, mx_test = _data_frame(multi)
    multi_model = model
    if spec.id == "CAT-1.2":
        multi_parameters = build_parameters(spec, seed=seed, device=device, target_classes=3)
        multi_model = import_class(spec.class_path)(**multi_parameters)
    elif spec.id == "LR-1.9":
        multi_model = import_class(spec.class_path)(
            **build_parameters(spec, seed=seed, device=device, target_classes=3)
        )
    elif spec.id == "XGB-3.4":
        multi_model = import_class(spec.class_path)(
            **build_parameters(spec, seed=seed, device=device, target_classes=3)
        )
    multi_model.fit(mx_train, multi.target)
    multi_predictions = multi_model.predict_proba(mx_test)
    multi_normalized, multi_classes, multi_error = normalize_probability_matrix(
        multi_predictions,
        multi_model.classes_,
        [0, 1, 2],
        row_count=len(mx_test),
        sum_atol=sum_atol,
    )
    capabilities["multiclass"] = {
        "prediction_shape": list(multi_normalized.shape),
        "fixture_hash": multi.hash,
    }
    return (
        list(multi_normalized.shape),
        multi_classes,
        max(probability_error, multi_error),
        repeated_difference,
        capabilities,
    )


def execute_probe(request: ProbeRequest) -> ProbeResult:
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[variable] = "2"
    models = load_model_registry(request.config_path)
    spec = next((model for model in models if model.id == request.model_id), None)
    if spec is None:
        raise ValueError(f"Unknown model ID: {request.model_id}")
    fixture = binary_numerical(request.seed)
    start = _now()
    started = __import__("time").perf_counter()
    parameters = build_parameters(spec, seed=request.seed, device=request.device, target_classes=2)
    if request.offline and spec.id in {"TPFN3-8.5", "TICL2-2.2"}:
        _disable_network()
    foundation_error: Exception | None = None
    if spec.id in {"TPFN3-8.5", "TICL2-2.2"}:
        try:
            parameters = _foundation_parameters(request, spec, parameters)
        except (CheckpointAuthorizationRequired, OfflineCacheMiss) as exc:
            foundation_error = exc
    parameter_hash = _parameter_hash(parameters)
    checkpoint: CheckpointRecord | None = None
    resource: ResourceRecord | None = None
    with ResourceMonitor(spec.id, request.device) as monitor:
        try:
            if foundation_error is not None:
                raise foundation_error
            available, device_category, device_message = _check_device(request)
            if not available:
                resource = monitor.finish()
                return _base_result(
                    request,
                    spec,
                    fixture.hash,
                    parameter_hash,
                    start,
                    _now(),
                    __import__("time").perf_counter() - started,
                    status="NOT_EXECUTED",
                    category=device_category or "NOT_EXECUTED_NO_CUDA",
                    message=device_message or "Device unavailable",
                    observed_version=importlib.metadata.version(spec.package),
                    resource=resource,
                )
            observed = importlib.metadata.version(spec.package)
            if observed != spec.expected_version:
                resource = monitor.finish()
                return _base_result(
                    request,
                    spec,
                    fixture.hash,
                    parameter_hash,
                    start,
                    _now(),
                    __import__("time").perf_counter() - started,
                    status="FAIL",
                    category="FAIL_VERSION_MISMATCH",
                    message=f"Expected {spec.expected_version}, observed {observed}",
                    observed_version=observed,
                    resource=resource,
                )
            constructor = import_class(spec.class_path)
            validate_constructor_parameters(spec, parameters)
            model = constructor(**parameters)
            if spec.id in {"TPFN3-8.5", "TICL2-2.2"}:
                checkpoint = _checkpoint_from_model(
                    model, spec, request=request, parameters=parameters
                )
                if not checkpoint.identity_valid and spec.id == "TPFN3-8.5":
                    resource = monitor.finish()
                    return _base_result(
                        request,
                        spec,
                        fixture.hash,
                        parameter_hash,
                        start,
                        _now(),
                        __import__("time").perf_counter() - started,
                        status="FAIL",
                        category="FAIL_CHECKPOINT_HASH",
                        message=checkpoint.error or "Checkpoint identity unavailable",
                        observed_version=observed,
                        checkpoint=checkpoint,
                        resource=resource,
                    )
                # This is a safety assertion: the frozen API must be used, with no cache shortcut.
                if spec.id == "TICL2-2.2" and parameters.get("kv_cache") is not False:
                    raise ValueError("TabICL kv_cache=False is mandatory")
            shape, classes, probability_error, repeated_difference, capabilities = (
                _fit_and_validate(
                    model,
                    fixture,
                    spec,
                    request.seed,
                    request.device,
                    1.0e-6,
                    1.0e-5 if request.device == "cuda" else 1.0e-10,
                )
            )
            if spec.id == "TICL2-2.2":
                checkpoint = _checkpoint_from_model(
                    model, spec, request=request, parameters=parameters
                )
                if not checkpoint.identity_valid:
                    resource = monitor.finish()
                    return _base_result(
                        request,
                        spec,
                        fixture.hash,
                        parameter_hash,
                        start,
                        _now(),
                        __import__("time").perf_counter() - started,
                        status="FAIL",
                        category="FAIL_CHECKPOINT_HASH",
                        message=checkpoint.error or "Checkpoint identity unavailable",
                        observed_version=observed,
                        checkpoint=checkpoint,
                        resource=resource,
                        prediction_shape=shape,
                        class_order=classes,
                        probability_error=probability_error,
                        repeated_difference=repeated_difference,
                        capabilities=capabilities,
                    )
            resource = monitor.finish()
            if checkpoint is not None and not checkpoint.identity_valid:
                return _base_result(
                    request,
                    spec,
                    fixture.hash,
                    parameter_hash,
                    start,
                    _now(),
                    __import__("time").perf_counter() - started,
                    status="FAIL",
                    category="FAIL_CHECKPOINT_HASH",
                    message="Checkpoint identity was not validated",
                    observed_version=observed,
                    checkpoint=checkpoint,
                    resource=resource,
                    prediction_shape=shape,
                    class_order=classes,
                    probability_error=probability_error,
                    repeated_difference=repeated_difference,
                    capabilities=capabilities,
                )
            return _base_result(
                request,
                spec,
                fixture.hash,
                parameter_hash,
                start,
                _now(),
                __import__("time").perf_counter() - started,
                status="PASS",
                category="PASS",
                message="Micro-inference passed",
                observed_version=observed,
                checkpoint=checkpoint,
                resource=resource,
                prediction_shape=shape,
                class_order=classes,
                probability_error=probability_error,
                repeated_difference=repeated_difference,
                capabilities=capabilities,
            )
        except ProbabilityValidationError as exc:
            resource = monitor.finish()
            path = _traceback_path(request, spec, traceback.format_exc())
            return _base_result(
                request,
                spec,
                fixture.hash,
                parameter_hash,
                start,
                _now(),
                __import__("time").perf_counter() - started,
                status="FAIL",
                category="FAIL_INVALID_PROBABILITY",
                message=str(exc),
                observed_version=importlib.metadata.version(spec.package),
                checkpoint=checkpoint,
                resource=resource,
                traceback_path=path,
            )
        except MemoryError as exc:
            resource = monitor.finish(oom=True)
            path = _traceback_path(request, spec, traceback.format_exc())
            category = "FAIL_GPU_OOM" if request.device == "cuda" else "FAIL_CPU_INFERENCE"
            return _base_result(
                request,
                spec,
                fixture.hash,
                parameter_hash,
                start,
                _now(),
                __import__("time").perf_counter() - started,
                status="FAIL",
                category=category,
                message=str(exc) or "Out of memory",
                observed_version=importlib.metadata.version(spec.package),
                checkpoint=checkpoint,
                resource=resource,
                traceback_path=path,
            )
        except TypeError as exc:
            resource = monitor.finish()
            path = _traceback_path(request, spec, traceback.format_exc())
            return _base_result(
                request,
                spec,
                fixture.hash,
                parameter_hash,
                start,
                _now(),
                __import__("time").perf_counter() - started,
                status="BLOCKED",
                category="BLOCKED_API_MISMATCH",
                message=str(exc),
                observed_version=importlib.metadata.version(spec.package),
                checkpoint=checkpoint,
                resource=resource,
                traceback_path=path,
            )
        except (CheckpointAuthorizationRequired, OfflineCacheMiss) as exc:
            resource = monitor.finish()
            path = _traceback_path(request, spec, traceback.format_exc())
            authorization = (
                "required" if isinstance(exc, CheckpointAuthorizationRequired) else "not_required"
            )
            license_status = (
                "not_accepted" if isinstance(exc, CheckpointAuthorizationRequired) else "unknown"
            )
            checkpoint = CheckpointRecord(
                model_id=spec.id,
                identifier=spec.checkpoint,
                cache_status=(
                    "authorization_required"
                    if isinstance(exc, CheckpointAuthorizationRequired)
                    else "offline_cache_miss"
                ),
                identity_valid=False,
                authorization_status=authorization,
                license_status=license_status,
                error=str(exc),
            )
            category = (
                "BLOCKED_CHECKPOINT_AUTHORIZATION"
                if isinstance(exc, CheckpointAuthorizationRequired)
                else "NOT_EXECUTED_OFFLINE_CACHE_MISS"
            )
            return _base_result(
                request,
                spec,
                fixture.hash,
                parameter_hash,
                start,
                _now(),
                __import__("time").perf_counter() - started,
                status="BLOCKED",
                category=category,
                message=str(exc),
                observed_version=importlib.metadata.version(spec.package),
                checkpoint=checkpoint,
                resource=resource,
                traceback_path=path,
            )
        except Exception as exc:
            resource = monitor.finish(
                oom="out of memory" in str(exc).lower() or "cuda out of memory" in str(exc).lower()
            )
            path = _traceback_path(request, spec, traceback.format_exc())
            text_error = str(exc).lower()
            category = (
                "FAIL_GPU_OOM"
                if request.device == "cuda" and "out of memory" in text_error
                else "FAIL_CPU_INFERENCE"
                if request.device == "cpu"
                else "FAIL_GPU_INFERENCE"
            )
            if any(
                term in text_error
                for term in ("login", "token", "license", "gated", "401", "403", "unauthorized")
            ):
                category = "BLOCKED_CHECKPOINT_AUTHORIZATION"
            return _base_result(
                request,
                spec,
                fixture.hash,
                parameter_hash,
                start,
                _now(),
                __import__("time").perf_counter() - started,
                status="BLOCKED" if category.startswith("BLOCKED") else "FAIL",
                category=category,
                message=str(exc),
                observed_version=importlib.metadata.version(spec.package),
                checkpoint=checkpoint,
                resource=resource,
                traceback_path=path,
            )
        finally:
            if "model" in locals():
                del model
            gc.collect()
            try:
                import torch  # type: ignore[import-not-found]

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass


def _worker(request_path: str) -> int:
    with Path(request_path).open("r", encoding="utf-8") as handle:
        request = ProbeRequest.model_validate(json.load(handle))
    try:
        result = execute_probe(request)
    except Exception as exc:
        spec = next(
            model
            for model in load_model_registry(request.config_path)
            if model.id == request.model_id
        )
        fixture = binary_numerical(request.seed)
        result = _base_result(
            request,
            spec,
            fixture.hash,
            _parameter_hash({}),
            _now(),
            _now(),
            0.0,
            status="FAIL",
            category="FAIL_IMPORT",
            message=str(exc),
            traceback_path=_traceback_path(request, spec, traceback.format_exc()),
        )
    atomic_write_json(Path(request.output_path), result.model_dump(mode="json"))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", required=True)
    args = parser.parse_args()
    return _worker(args.worker)


if __name__ == "__main__":
    raise SystemExit(main())
