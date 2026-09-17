"""Leakage-safe common adapter lifecycle and locally validated round trips."""

from __future__ import annotations

import copy
import gc
import hashlib
import importlib.metadata
import inspect
import pickle
import random
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .resources import configure_thread_limits

configure_thread_limits()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from pydantic import ValidationError  # noqa: E402

from ...compatibility.checkpoint_cache import (  # noqa: E402
    register_checkpoint,
    resolve_offline_checkpoint,
)
from ...constants import ROW_ID_COLUMN  # noqa: E402
from ...utils.hashing import hash_dataframe_logically, sha256_canonical_json  # noqa: E402
from .cache import AdapterArtifactCache, CacheIntegrityError  # noqa: E402
from .contracts import (  # noqa: E402
    FOUNDATION_CHECKPOINTS,
    AdapterCacheIdentity,
    AdapterFailure,
    AdapterResourceRecord,
    FailureCategory,
    ModelAdapterConfig,
    PredictionResult,
    SemanticEquivalenceCertificate,
    implementation_digest,
)
from .preprocessing import TrainingPreprocessor  # noqa: E402
from .probability import align_adapter_probabilities  # noqa: E402
from .resources import (  # noqa: E402
    AdapterResourceTracker,
    GpuExecutionLock,
    ensure_gpu_budget,
    ensure_ram_budget,
)

FOUNDATION_IDS = frozenset({"TPFN3-8.5", "TICL2-2.2"})
GPU_EXPECTED_PEAK_MIB = {"TPFN3-8.5": 376.0, "TICL2-2.2": 198.0}
NETWORK_ATTEMPT_COUNT = 0


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _row_ids_sha256(row_ids: list[str | int]) -> str:
    return sha256_canonical_json(row_ids)


def canonical_parameter_payload(
    parameters: dict[str, Any],
    *,
    model_id: str,
    package_name: str,
    package_version: str,
    seed: int,
    device: str,
    checkpoint_identifier: str | None,
    checkpoint_sha256: str | None,
    python_major_minor: str,
    pytorch_version: str,
) -> dict[str, Any]:
    """Return the complete portable constructor/runtime identity, without local paths."""

    normalized_parameters = dict(parameters)
    if checkpoint_identifier is not None:
        if not checkpoint_sha256 or "model_path" not in normalized_parameters:
            raise ValueError("checkpoint parameters require a path, identifier, and checksum")
        normalized_parameters["model_path"] = {
            "checkpoint_identifier": checkpoint_identifier,
            "checkpoint_sha256": checkpoint_sha256,
        }
    elif "model_path" in normalized_parameters:
        raise ValueError("a model path without a frozen checkpoint identity is not portable")
    return {
        "schema_version": 1,
        "model_id": model_id,
        "package_name": package_name,
        "package_version": package_version,
        "python_major_minor": python_major_minor,
        "pytorch_version": pytorch_version,
        "seed": seed,
        "device": device,
        "checkpoint_identifier": checkpoint_identifier,
        "checkpoint_sha256": checkpoint_sha256,
        "parameters": normalized_parameters,
    }


def _probability_sha256(probabilities: np.ndarray) -> str:
    canonical = np.asarray(probabilities, dtype="<f8", order="C")
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _git_commit(root: Path) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise AdapterFailure(
            FailureCategory.BLOCKED_ENVIRONMENT, "cannot identify the source git commit"
        ) from exc


def _disable_network() -> None:
    def blocked(*args: Any, **kwargs: Any) -> Any:
        global NETWORK_ATTEMPT_COUNT
        NETWORK_ATTEMPT_COUNT += 1
        raise RuntimeError("offline adapter validation prohibits network access")

    socket.create_connection = blocked  # type: ignore[assignment]
    socket.socket.connect = blocked  # type: ignore[assignment]


def _row_identifiers(frame: pd.DataFrame) -> tuple[list[str | int], pd.DataFrame]:
    if not isinstance(frame, pd.DataFrame):
        raise AdapterFailure(FailureCategory.FAIL_INPUT, "feature input must be a DataFrame")
    if frame.columns.duplicated().any():
        raise AdapterFailure(FailureCategory.FAIL_INPUT, "duplicate input column names")
    if ROW_ID_COLUMN not in frame.columns:
        raise AdapterFailure(FailureCategory.FAIL_INPUT, f"missing immutable {ROW_ID_COLUMN}")
    raw_ids = frame[ROW_ID_COLUMN].tolist()
    row_ids: list[str | int] = []
    for value in raw_ids:
        if isinstance(value, np.generic):
            value = value.item()
        if not isinstance(value, (str, int)) or isinstance(value, bool):
            raise AdapterFailure(FailureCategory.FAIL_INPUT, "row IDs must be strings or integers")
        row_ids.append(value)
    if not row_ids or len(row_ids) != len(set(row_ids)):
        raise AdapterFailure(FailureCategory.FAIL_INPUT, "row IDs must be non-empty and unique")
    predictors = frame.drop(columns=[ROW_ID_COLUMN])
    if predictors.empty:
        raise AdapterFailure(FailureCategory.FAIL_INPUT, "at least one predictor is required")
    if any(name in {"target_code", "target_label"} for name in predictors.columns):
        raise AdapterFailure(FailureCategory.FAIL_LEAKAGE, "target columns are forbidden in X")
    if any(str(name).startswith("__sg_") for name in predictors.columns):
        raise AdapterFailure(FailureCategory.FAIL_LEAKAGE, "split/group metadata is forbidden in X")
    return row_ids, predictors


def _canonical_row_order(row_ids: list[str | int]) -> tuple[list[int], list[int]]:
    """Sort only by immutable IDs for model invocation and retain an inverse alignment map."""

    order = sorted(
        range(len(row_ids)),
        key=lambda position: (type(row_ids[position]).__name__, row_ids[position]),
    )
    inverse = [0] * len(order)
    for sorted_position, original_position in enumerate(order):
        inverse[original_position] = sorted_position
    return order, inverse


def _implementation_hash(adapter_type: type[Any]) -> str:
    from . import base as base_module
    from . import evidence as evidence_module
    from . import preprocessing as preprocessing_module
    from . import probability as probability_module

    sources = {
        "base": inspect.getsource(base_module),
        "adapter": inspect.getsource(adapter_type),
        "preprocessing": inspect.getsource(preprocessing_module),
        "probability": inspect.getsource(probability_module),
        "leakage_evidence": inspect.getsource(evidence_module),
    }
    return sha256_canonical_json(
        {name: implementation_digest(source) for name, source in sorted(sources.items())}
    )


def _preprocessing_implementation_hash() -> str:
    from ...utils import hashing as hashing_module
    from . import preprocessing as preprocessing_module

    return sha256_canonical_json(
        {
            "preprocessing": implementation_digest(inspect.getsource(preprocessing_module)),
            "hashing": implementation_digest(inspect.getsource(hashing_module)),
        }
    )


class ModelAdapterBase:
    """Common strict lifecycle. No test/calibration labels enter this object."""

    adapter_id = ""
    preprocessing_strategy = ""
    supports_model_serialization = True
    expected_peak_vram_mib = 0.0

    def __init__(
        self,
        spec: Any,
        config: ModelAdapterConfig,
        *,
        root: str | Path,
        preprocessing_strategy: str,
        seed: int,
        device: str,
    ) -> None:
        if device not in {"cpu", "cuda"}:
            raise AdapterFailure(FailureCategory.FAIL_CONTRACT, "device must be cpu or cuda")
        if device == "cuda" and spec.id not in FOUNDATION_IDS:
            raise AdapterFailure(
                FailureCategory.FAIL_CONTRACT, "CUDA is frozen only for foundation-model adapters"
            )
        self.spec = spec
        self.model_id = spec.id
        self.package_name = spec.package
        self.expected_version = spec.expected_version
        self.seed = seed
        self.device = device
        self.root = Path(root).resolve()
        self.source_commit = _git_commit(self.root)
        self.config = config
        self.preprocessing_strategy = preprocessing_strategy
        self.preprocessor = TrainingPreprocessor(preprocessing_strategy)
        self.model: Any | None = None
        self.fitted = False
        self.class_order: list[int] = []
        self.parameters: dict[str, Any] = {}
        self.parameter_identity: dict[str, Any] = {}
        self.package_version = ""
        self.parameter_sha256 = ""
        self.model_spec_sha256 = sha256_canonical_json(spec.model_dump(mode="json"))
        self.adapter_sha256 = _implementation_hash(type(self))
        self.preprocessing_implementation_sha256 = _preprocessing_implementation_hash()
        self.checkpoint_identifier: str | None = None
        self.checkpoint_sha256: str | None = None
        self.checkpoint_path: Path | None = None
        self.gpu_headroom_passed: bool | None = None
        self.fixture_id = ""
        self.fixture_sha256 = ""
        self.split_identity = ""
        self.transformation_identity: str | None = None
        self.training_data_sha256 = ""
        self.training_row_ids_sha256 = ""
        self.training_row_ids: list[str | int] = []
        self.training_start = time.perf_counter()
        self.training_start_time = _now()
        self.model_load_seconds = 0.0
        self.preprocessing_fit_seconds = 0.0
        self.fit_seconds = 0.0
        self.preprocessing_seconds = 0.0
        self.prediction_seconds = 0.0
        self.serialization_seconds = 0.0
        self.cache_seconds = 0.0
        self.resource_tracker: AdapterResourceTracker | None = None
        self.gpu_lock: GpuExecutionLock | None = None
        self._last_prediction_reference: tuple[
            pd.DataFrame, list[str | int], PredictionResult
        ] | None = None
        self.cache = AdapterArtifactCache(self.root / self.config.cache_directory)

    def capability(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "preprocessing": self.preprocessing_strategy,
            "binary": True,
            "multiclass": True,
            "numeric": True,
            "categorical": True,
            "missing": True,
            "unseen_category": True,
            "model_serialization": self.supports_model_serialization,
            "prediction_cache_roundtrip": True,
            "device": self.device,
        }

    def fit(
        self,
        training_features: pd.DataFrame,
        training_targets: np.ndarray,
        *,
        fixture_id: str,
        fixture_sha256: str,
        split_identity: str,
        transformation_identity: str | None = None,
    ) -> ModelAdapterBase:
        if self.fitted or self.model is not None:
            raise AdapterFailure(FailureCategory.FAIL_CONTRACT, "adapter instances fit only once")
        if not fixture_id or not split_identity:
            raise AdapterFailure(
                FailureCategory.FAIL_CONTRACT, "fixture and split identities are required"
            )
        row_ids, predictors = _row_identifiers(training_features)
        y = np.asarray(training_targets)
        if y.ndim != 1 or len(y) != len(row_ids) or not len(y):
            raise AdapterFailure(
                FailureCategory.FAIL_INPUT, "training targets must align one-to-one"
            )
        if not np.issubdtype(y.dtype, np.integer):
            raise AdapterFailure(
                FailureCategory.FAIL_INPUT, "target codes must be contiguous integers"
            )
        if not np.isfinite(y).all():
            raise AdapterFailure(
                FailureCategory.FAIL_INPUT, "training targets contain non-finite values"
            )
        self.class_order = [int(label) for label in np.unique(y)]
        if len(self.class_order) < 2 or self.class_order != list(range(len(self.class_order))):
            raise AdapterFailure(
                FailureCategory.FAIL_INPUT,
                "training requires at least two contiguous target codes beginning at zero",
            )
        if np.min(np.bincount(y.astype(np.int64))) < 2:
            raise AdapterFailure(
                FailureCategory.FAIL_INPUT, "each target class requires at least two training rows"
            )
        self._verify_package()
        self.fixture_id = fixture_id
        self.fixture_sha256 = fixture_sha256
        self.split_identity = split_identity
        self.transformation_identity = transformation_identity
        self.training_row_ids = row_ids
        self.training_row_ids_sha256 = _row_identifiers_sha256(row_ids)
        self.training_data_sha256 = hash_dataframe_logically(predictors)
        self.parameters = self._build_parameters(len(self.class_order))

        try:
            self._prepare_device()
            self.parameter_identity = self._canonical_parameter_identity()
            self.parameter_sha256 = sha256_canonical_json(self.parameter_identity)
            ensure_ram_budget(hard_ram_gib=self.config.resources.hard_ram_gib)
            self.resource_tracker = AdapterResourceTracker(
                self.device,
                hard_ram_limit_mib=self.config.resources.hard_ram_gib * 1024.0,
                gpu_soft_limit_mib=self.config.resources.gpu_soft_limit_mib,
            )
            started = time.perf_counter()
            self.preprocessor.fit(predictors)
            self.preprocessing_fit_seconds = time.perf_counter() - started
            started = time.perf_counter()
            train_matrix = self._preprocess(predictors, row_ids, "train")
            self.preprocessing_seconds += time.perf_counter() - started
            started = time.perf_counter()
            with self._seeded_model_runtime():
                model_class = self._import_and_validate_model()
                self.model = model_class(**self.parameters)
            self.model_load_seconds = time.perf_counter() - started
            started = time.perf_counter()
            with self._seeded_model_runtime():
                self._fit_estimator(train_matrix, y.astype(np.int64, copy=False))
            self.fit_seconds = time.perf_counter() - started
            self._enforce_resource_limits()
            self.fitted = True
            return self
        except AdapterFailure:
            self.release()
            raise
        except Exception as exc:
            category = _classify_runtime_exception(exc, self.device)
            self.release()
            raise AdapterFailure(category, f"{type(exc).__name__}: {exc}") from exc

    def predict_proba(
        self,
        features: pd.DataFrame,
        *,
        partition: str,
        fixture_id: str,
        fixture_sha256: str,
        split_identity: str,
        transformation_identity: str | None = None,
    ) -> PredictionResult:
        self._ensure_fitted()
        if partition not in {"train", "calibration", "test"}:
            raise AdapterFailure(FailureCategory.FAIL_CONTRACT, "unknown prediction partition")
        if (
            fixture_id != self.fixture_id
            or fixture_sha256 != self.fixture_sha256
            or split_identity != self.split_identity
            or transformation_identity != self.transformation_identity
        ):
            raise AdapterFailure(
                FailureCategory.FAIL_CONTRACT,
                "prediction identity differs from the frozen training fixture/split",
            )
        row_ids, predictors = _row_identifiers(features)
        row_order, inverse_row_order = _canonical_row_order(row_ids)
        ordered_ids = [row_ids[position] for position in row_order]
        ordered_predictors = predictors.iloc[row_order].reset_index(drop=True)
        started_at = _now()
        started = time.perf_counter()
        try:
            prep_started = time.perf_counter()
            matrix = self._preprocess(ordered_predictors, ordered_ids, partition)
            self.preprocessing_seconds += time.perf_counter() - prep_started
            predict_started = time.perf_counter()
            with self._seeded_model_runtime():
                raw = self._predict_estimator(matrix)
            self.prediction_seconds += time.perf_counter() - predict_started
            observed = getattr(self.model, "classes_", None)
            if observed is None:
                raise AdapterFailure(
                    FailureCategory.FAIL_CLASS_ORDER, "estimator exposes no classes_"
                )
            probabilities, classes, _ = align_adapter_probabilities(
                raw,
                observed,
                self.class_order,
                row_count=len(ordered_ids),
                sum_atol=self.config.tolerances.probability_sum_atol,
            )
            if classes != self.class_order:
                raise AdapterFailure(
                    FailureCategory.FAIL_CLASS_ORDER, "canonical class order changed"
                )
            probabilities = probabilities[inverse_row_order, :]
            if self.resource_tracker is not None:
                self.resource_tracker.capture()
                self._enforce_resource_limits()
            prob_sha = _probability_sha256(probabilities)
            row_sha = _row_identifiers_sha256(row_ids)
            logical_identity = sha256_canonical_json(
                {
                    "model_spec_sha256": self.model_spec_sha256,
                    "source_commit": self.source_commit,
                    "parameter_sha256": self.parameter_sha256,
                    "parameter_identity": self.parameter_identity,
                    "adapter_sha256": self.adapter_sha256,
                    "preprocessing_sha256": self.preprocessor.fitted_state_sha256,
                    "checkpoint_sha256": self.checkpoint_sha256,
                    "fixture_sha256": fixture_sha256,
                    "source_data_sha256": hash_dataframe_logically(predictors),
                    "split_identity": split_identity,
                    "transformation_identity": transformation_identity,
                    "seed": self.seed,
                    "device": self.device,
                    "partition": partition,
                    "row_ids_sha256": row_sha,
                    "class_order": classes,
                }
            )
            tracker = self.resource_tracker
            cpu_time = tracker.cpu_time_seconds() if tracker else None
            peak_ram = tracker.peak_ram_mib if tracker else None
            peak_vram = tracker.peak_vram_reserved_mib if tracker else None
            ended_at = _now()
            result = PredictionResult(
                model_id=self.model_id,
                package_name=self.package_name,
                package_version=self.package_version,
                model_spec_sha256=self.model_spec_sha256,
                source_commit=self.source_commit,
                parameter_sha256=self.parameter_sha256,
                adapter_sha256=self.adapter_sha256,
                preprocessing_sha256=self.preprocessor.fitted_state_sha256,
                checkpoint_identifier=self.checkpoint_identifier,
                checkpoint_sha256=self.checkpoint_sha256,
                fixture_id=fixture_id,
                fixture_sha256=fixture_sha256,
                source_data_sha256=hash_dataframe_logically(predictors),
                split_identity=split_identity,
                transformation_identity=transformation_identity,
                seed=self.seed,
                device=self.device,  # type: ignore[arg-type]
                execution_policy=(
                    "fresh_worker_per_inference;canonical_row_id_order;shared_foundation_lock"
                    if self.model_id in FOUNDATION_IDS
                    else "single_process_cpu;canonical_row_id_order;thread_limit=2"
                ),
                partition=partition,  # type: ignore[arg-type]
                row_ids=row_ids,
                row_ids_sha256=row_sha,
                class_order=classes,
                probabilities=probabilities.tolist(),
                probabilities_sha256=prob_sha,
                logical_identity_sha256=logical_identity,
                started_at=started_at,
                ended_at=ended_at,
                wall_time_seconds=max(0.0, time.perf_counter() - started),
                cpu_time_seconds=cpu_time,
                peak_ram_mib=peak_ram,
                peak_vram_mib=peak_vram,
                status="PASS",
                failure_category=FailureCategory.PASS,
            )
            self._last_prediction_reference = (predictors.copy(deep=True), list(row_ids), result)
            return result
        except AdapterFailure:
            raise
        except Exception as exc:
            category = _classify_runtime_exception(exc, self.device)
            raise AdapterFailure(category, f"{type(exc).__name__}: {exc}") from exc

    def save_fitted(self) -> AdapterCacheIdentity:
        """Persist a classical fitted pipeline locally with source-bound integrity metadata."""

        self._ensure_fitted()
        if not self.supports_model_serialization:
            raise AdapterFailure(
                FailureCategory.FAIL_SERIALIZATION,
                "foundation models use prediction-cache round trips, not full model serialization",
            )
        identity = self._cache_identity(self.training_row_ids, "train", self.training_data_sha256)
        started = time.perf_counter()
        try:
            fitted_preprocessor = copy.copy(self.preprocessor)
            fitted_preprocessor._matrix_cache = {}
            fitted_preprocessor.cache_hits = 0
            payload = pickle.dumps(
                {
                    "model_id": self.model_id,
                    "model_spec_sha256": self.model_spec_sha256,
                    "source_commit": self.source_commit,
                    "parameter_sha256": self.parameter_sha256,
                    "parameter_identity": self.parameter_identity,
                    "adapter_sha256": self.adapter_sha256,
                    "preprocessing_sha256": self.preprocessor.fitted_state_sha256,
                    "checkpoint_sha256": self.checkpoint_sha256,
                    "package_version": self.package_version,
                    "class_order": self.class_order,
                    "fixture_id": self.fixture_id,
                    "fixture_sha256": self.fixture_sha256,
                    "split_identity": self.split_identity,
                    "transformation_identity": self.transformation_identity,
                    "training_data_sha256": self.training_data_sha256,
                    "training_row_ids": self.training_row_ids,
                    "training_row_ids_sha256": self.training_row_ids_sha256,
                    "preprocessor": fitted_preprocessor,
                    "model": self.model,
                },
                protocol=pickle.HIGHEST_PROTOCOL,
            )
            try:
                self.cache.store_bytes("model", identity, payload)
            except CacheIntegrityError as exc:
                if "different payload bytes" not in str(exc):
                    raise
                certificate = self._semantic_equivalence_certificate(
                    identity, self.cache.load_bytes("model", identity), payload
                )
                self.cache.store_bytes(
                    "model",
                    identity,
                    payload,
                    semantic_equivalence=certificate,
                )
            self.serialization_seconds += time.perf_counter() - started
            return identity
        except CacheIntegrityError as exc:
            raise AdapterFailure(FailureCategory.FAIL_CACHE_INTEGRITY, str(exc)) from exc
        except AdapterFailure:
            raise
        except Exception as exc:
            raise AdapterFailure(FailureCategory.FAIL_SERIALIZATION, str(exc)) from exc

    def _semantic_equivalence_certificate(
        self,
        identity: AdapterCacheIdentity,
        existing_payload: bytes,
        candidate_payload: bytes,
    ) -> SemanticEquivalenceCertificate:
        """Certify differing model bytes only after fixture-bound prediction comparison."""

        try:
            existing_bundle = pickle.loads(existing_payload)
        except Exception as exc:
            raise CacheIntegrityError(
                "cached model cannot be deserialized for semantic proof"
            ) from exc
        required_metadata = {
            "model_id": self.model_id,
            "model_spec_sha256": self.model_spec_sha256,
            "source_commit": self.source_commit,
            "parameter_sha256": self.parameter_sha256,
            "parameter_identity": self.parameter_identity,
            "adapter_sha256": self.adapter_sha256,
            "preprocessing_sha256": identity.preprocessing_sha256,
            "checkpoint_sha256": identity.checkpoint_sha256,
            "package_version": self.expected_version,
            "fixture_id": self.fixture_id,
            "fixture_sha256": identity.fixture_sha256,
            "split_identity": identity.split_identity,
            "transformation_identity": identity.transformation_identity,
            "training_data_sha256": identity.source_data_sha256,
            "training_row_ids_sha256": identity.row_ids_sha256,
        }
        if any(
            existing_bundle.get(name) != expected
            for name, expected in required_metadata.items()
        ):
            raise CacheIntegrityError(
                "cached model scientific metadata differs from its identity"
            )
        if existing_bundle.get("class_order") != self.class_order:
            raise CacheIntegrityError("cached model class order differs from the candidate")
        reference = self._last_prediction_reference
        if reference is None:
            raise CacheIntegrityError("semantic proof requires an executed reference prediction")
        reference_features, row_ids, candidate_prediction = reference
        if (
            candidate_prediction.fixture_sha256 != identity.fixture_sha256
            or candidate_prediction.row_ids_sha256 != _row_identifiers_sha256(row_ids)
            or candidate_prediction.source_data_sha256
            != hash_dataframe_logically(reference_features)
        ):
            raise CacheIntegrityError("semantic reference fixture is not bound to this model cache")
        existing_preprocessor = existing_bundle.get("preprocessor")
        existing_model = existing_bundle.get("model")
        if (
            existing_preprocessor is None
            or existing_model is None
            or existing_preprocessor.fitted_state_sha256 != identity.preprocessing_sha256
        ):
            raise CacheIntegrityError("cached model preprocessing state is invalid")
        order, inverse = _canonical_row_order(row_ids)
        ordered_ids = [row_ids[index] for index in order]
        ordered_features = reference_features.iloc[order].reset_index(drop=True)
        def predict_reference(
            model: Any, preprocessor: TrainingPreprocessor
        ) -> tuple[np.ndarray, list[Any]]:
            matrix = preprocessor.transform(
                ordered_features,
                row_ids=ordered_ids,
                partition=candidate_prediction.partition,
                use_cache=False,
            )
            active_model = self.model
            try:
                self.model = model
                with self._seeded_model_runtime():
                    raw = self._predict_estimator(matrix)
            finally:
                self.model = active_model
            observed = getattr(model, "classes_", None)
            if observed is None:
                raise CacheIntegrityError("serialized model exposes no reference class order")
            probabilities, class_order, _ = align_adapter_probabilities(
                raw,
                observed,
                self.class_order,
                row_count=len(ordered_ids),
                sum_atol=self.config.tolerances.probability_sum_atol,
            )
            return probabilities[inverse, :], class_order

        existing_probabilities, existing_classes = predict_reference(
            existing_model, existing_preprocessor
        )
        candidate_probabilities, candidate_classes = predict_reference(
            self.model, self.preprocessor
        )
        if (
            existing_classes != candidate_prediction.class_order
            or candidate_classes != candidate_prediction.class_order
        ):
            raise CacheIntegrityError("semantic reference class orders differ")
        recorded_probabilities = np.asarray(candidate_prediction.probabilities, dtype="float64")
        recorded_difference = float(
            np.max(np.abs(candidate_probabilities - recorded_probabilities))
        )
        tolerance = self.config.tolerances.cpu_repeat_max_abs_diff
        if recorded_difference > tolerance:
            raise CacheIntegrityError(
                "candidate model no longer matches its executed reference prediction"
            )
        if existing_probabilities.shape != candidate_probabilities.shape:
            raise CacheIntegrityError("semantic reference prediction shapes differ")
        maximum_difference = float(
            np.max(np.abs(existing_probabilities - candidate_probabilities))
        )
        if maximum_difference > tolerance:
            raise CacheIntegrityError(
                "serialized models disagree on identity-bound reference predictions: "
                f"{maximum_difference} > {tolerance}"
            )
        certificate_payload: dict[str, Any] = {
            "schema_version": 1,
            "existing_payload_sha256": hashlib.sha256(existing_payload).hexdigest(),
            "candidate_payload_sha256": hashlib.sha256(candidate_payload).hexdigest(),
            "model_spec_sha256": identity.model_spec_sha256,
            "parameter_sha256": identity.parameter_sha256,
            "preprocessing_sha256": identity.preprocessing_sha256,
            "training_data_sha256": identity.source_data_sha256,
            "checkpoint_sha256": identity.checkpoint_sha256,
            "reference_fixture_sha256": identity.fixture_sha256,
            "reference_row_ids_sha256": _row_identifiers_sha256(row_ids),
            "reference_features_sha256": hash_dataframe_logically(reference_features),
            "reference_class_order": candidate_classes,
            "reference_prediction_sha256_existing": _probability_sha256(
                existing_probabilities
            ),
            "reference_prediction_sha256_candidate": _probability_sha256(
                candidate_probabilities
            ),
            "maximum_probability_difference": maximum_difference,
            "tolerance": tolerance,
            "equivalent": True,
        }
        certificate_payload["certificate_sha256"] = sha256_canonical_json(certificate_payload)
        certificate = SemanticEquivalenceCertificate.model_validate(certificate_payload)
        certificate.validate_identity(
            identity,
            existing_payload_sha256=certificate.existing_payload_sha256,
            candidate_payload_sha256=certificate.candidate_payload_sha256,
        )
        return certificate

    def load_fitted(self, identity: AdapterCacheIdentity) -> ModelAdapterBase:
        """Restore a validated classical fitted pipeline into a fresh adapter instance."""

        if not self.supports_model_serialization:
            raise AdapterFailure(
                FailureCategory.FAIL_SERIALIZATION, "model restore is not supported"
            )
        started = time.perf_counter()
        try:
            self._verify_package()
            self.package_version = self.expected_version
            payload = self.cache.load_bytes("model", identity)
            bundle = pickle.loads(payload)
            for name, expected in (
                ("model_id", self.model_id),
                ("model_spec_sha256", self.model_spec_sha256),
                ("source_commit", self.source_commit),
                ("parameter_sha256", identity.parameter_sha256),
                ("adapter_sha256", self.adapter_sha256),
                ("package_version", self.expected_version),
                ("checkpoint_sha256", self.checkpoint_sha256),
            ):
                if bundle.get(name) != expected:
                    raise CacheIntegrityError(f"serialized model metadata mismatch: {name}")
            if bundle["preprocessing_sha256"] != identity.preprocessing_sha256:
                raise CacheIntegrityError("serialized preprocessing identity mismatch")
            self.parameters = dict(self._build_parameters(len(bundle["class_order"])))
            self.parameter_identity = self._canonical_parameter_identity()
            if sha256_canonical_json(self.parameter_identity) != identity.parameter_sha256:
                raise CacheIntegrityError(
                    "serialized model parameters differ from the cache identity"
                )
            if bundle.get("parameter_identity") != self.parameter_identity:
                raise CacheIntegrityError("serialized canonical parameter identity mismatch")
            self.parameter_sha256 = identity.parameter_sha256
            self.package_version = bundle["package_version"]
            self.preprocessor = bundle["preprocessor"]
            self.model = bundle["model"]
            self.class_order = list(bundle["class_order"])
            self.fixture_id = bundle["fixture_id"]
            self.fixture_sha256 = bundle["fixture_sha256"]
            self.split_identity = bundle["split_identity"]
            self.transformation_identity = bundle["transformation_identity"]
            self.training_data_sha256 = bundle["training_data_sha256"]
            self.training_row_ids = list(bundle["training_row_ids"])
            self.training_row_ids_sha256 = bundle["training_row_ids_sha256"]
            if (
                identity.partition != "train"
                or identity.source_data_sha256 != self.training_data_sha256
                or identity.fixture_sha256 != self.fixture_sha256
                or identity.row_ids_sha256 != self.training_row_ids_sha256
                or identity.split_identity != self.split_identity
                or identity.transformation_identity != self.transformation_identity
            ):
                raise CacheIntegrityError("serialized training source identity mismatch")
            self.fitted = True
            self.training_start = time.perf_counter()
            self.resource_tracker = AdapterResourceTracker(
                self.device,
                hard_ram_limit_mib=self.config.resources.hard_ram_gib * 1024.0,
                gpu_soft_limit_mib=self.config.resources.gpu_soft_limit_mib,
            )
            self.serialization_seconds += time.perf_counter() - started
            return self
        except CacheIntegrityError as exc:
            raise AdapterFailure(FailureCategory.FAIL_CACHE_INTEGRITY, str(exc)) from exc
        except AdapterFailure:
            raise
        except Exception as exc:
            raise AdapterFailure(FailureCategory.FAIL_SERIALIZATION, str(exc)) from exc

    def roundtrip_prediction(self, prediction: PredictionResult) -> PredictionResult:
        """Validate a foundation prediction cache round trip without serializing model weights."""

        identity = self._cache_identity(
            prediction.row_ids,
            prediction.partition,
            prediction.source_data_sha256,
        )
        started = time.perf_counter()
        try:
            full_payload = prediction.model_dump(mode="json")
            payload = dict(full_payload)
            for volatile in (
                "started_at",
                "ended_at",
                "wall_time_seconds",
                "cpu_time_seconds",
                "peak_ram_mib",
                "peak_vram_mib",
                "traceback_path",
            ):
                payload.pop(volatile, None)
            # Always compare the candidate bytes with a validated existing cache
            # entry; merely finding a path must never authorize reusing different
            # model output under the same scientific identity.
            self.cache.store_json("prediction", identity, payload)
            restored = self.cache.load_json("prediction", identity)
            self.cache_seconds += time.perf_counter() - started
            if (
                restored.get("logical_identity_sha256") != prediction.logical_identity_sha256
                or restored.get("row_ids_sha256") != prediction.row_ids_sha256
            ):
                raise CacheIntegrityError(
                    "prediction cache changed the logical prediction identity"
                )
            if (
                restored.get("class_order") != prediction.class_order
                or restored.get("row_ids") != prediction.row_ids
                or restored.get("fixture_sha256") != prediction.fixture_sha256
                or restored.get("source_data_sha256") != prediction.source_data_sha256
            ):
                raise CacheIntegrityError("prediction cache metadata differs from the request")
            cached_probabilities = np.asarray(restored["probabilities"], dtype="float64")
            current_probabilities = np.asarray(prediction.probabilities, dtype="float64")
            difference = (
                float(np.max(np.abs(cached_probabilities - current_probabilities)))
                if cached_probabilities.shape == current_probabilities.shape
                and current_probabilities.size
                else float("inf")
            )
            tolerance = (
                self.config.tolerances.gpu_repeat_max_abs_diff
                if self.device == "cuda"
                else self.config.tolerances.cpu_repeat_max_abs_diff
            )
            if difference > tolerance:
                raise AdapterFailure(
                    FailureCategory.FAIL_DETERMINISM,
                    f"cached prediction changed by {difference}, above tolerance {tolerance}",
                )
            for volatile in (
                "started_at",
                "ended_at",
                "wall_time_seconds",
                "cpu_time_seconds",
                "peak_ram_mib",
                "peak_vram_mib",
                "traceback_path",
            ):
                restored[volatile] = full_payload[volatile]
            return PredictionResult.model_validate(restored)
        except (CacheIntegrityError, ValidationError) as exc:
            raise AdapterFailure(FailureCategory.FAIL_CACHE_INTEGRITY, str(exc)) from exc

    def resource_record(self) -> AdapterResourceRecord:
        tracker = self.resource_tracker
        if tracker is not None:
            tracker.capture()
        return AdapterResourceRecord(
            model_load_seconds=max(0.0, self.model_load_seconds),
            preprocessing_fit_seconds=max(0.0, self.preprocessing_fit_seconds),
            fit_seconds=max(0.0, self.fit_seconds),
            preprocessing_seconds=max(0.0, self.preprocessing_seconds),
            prediction_seconds=max(0.0, self.prediction_seconds),
            serialization_seconds=max(0.0, self.serialization_seconds),
            cache_seconds=max(0.0, self.cache_seconds),
            wall_time_seconds=tracker.wall_time_seconds() if tracker else 0.0,
            cpu_time_seconds=tracker.cpu_time_seconds() if tracker else None,
            peak_ram_mib=tracker.peak_ram_mib if tracker else None,
            peak_process_tree_ram_mib=tracker.peak_tree_ram_mib if tracker else None,
            peak_vram_allocated_mib=tracker.peak_vram_allocated_mib if tracker else None,
            peak_vram_reserved_mib=tracker.peak_vram_reserved_mib if tracker else None,
            gpu_baseline_allocated_mib=tracker.gpu_baseline_allocated_mib if tracker else None,
            gpu_baseline_reserved_mib=tracker.gpu_baseline_reserved_mib if tracker else None,
            free_vram_before_mib=tracker.free_vram_before_mib if tracker else None,
            free_vram_after_mib=tracker.free_vram_after_mib if tracker else None,
            telemetry_complete=tracker.telemetry_complete if tracker else False,
            telemetry_error=tracker.telemetry_error if tracker else "adapter is not fitted",
            gpu_headroom_passed=self.gpu_headroom_passed,
            resource_limits_passed=tracker.resource_limits_passed if tracker else False,
            resource_limit_failure=tracker.limit_failure_reason if tracker else None,
            cleanup_verified=tracker.cleanup_verified if tracker else False,
            cleanup_gpu_allocated_mib=tracker.cleanup_gpu_allocated_mib if tracker else None,
            cleanup_gpu_reserved_mib=tracker.cleanup_gpu_reserved_mib if tracker else None,
        )

    def release(self) -> None:
        """Drop model references before clearing CUDA allocations, then release locks."""

        self.model = None
        self.fitted = False
        gc.collect()
        if self.device == "cuda":
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.synchronize(0)
                    torch.cuda.empty_cache()
            except Exception as exc:
                if self.resource_tracker is not None:
                    self.resource_tracker._note_error(exc)
        if self.resource_tracker is not None:
            self.resource_tracker.capture()
            self.resource_tracker.verify_cleanup()
            self.resource_tracker.stop()
        if self.gpu_lock is not None:
            self.gpu_lock.release()

    def _verify_package(self) -> None:
        try:
            observed = importlib.metadata.version(self.package_name)
        except importlib.metadata.PackageNotFoundError as exc:
            raise AdapterFailure(
                FailureCategory.BLOCKED_ENVIRONMENT,
                f"required package {self.package_name} is not installed",
            ) from exc
        if observed != self.expected_version:
            raise AdapterFailure(
                FailureCategory.BLOCKED_ENVIRONMENT,
                f"expected {self.package_name}=={self.expected_version}; observed {observed}",
            )
        self.package_version = observed

    @contextmanager
    def _seeded_model_runtime(self) -> Iterator[None]:
        """Seed foundation RNGs locally and restore process-global states afterward.

        Foundation packages combine their estimator-level ``random_state`` with
        lower-level Python, NumPy, and PyTorch routines. Seeding only the estimator
        leaves those routines dependent on process-global state, which can change
        results between otherwise identical isolated workers. Save and restore the
        surrounding process state so inference cannot perturb unrelated code.
        """

        if self.model_id not in FOUNDATION_IDS:
            yield
            return
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        random.seed(self.seed)
        np.random.seed(self.seed)
        import torch

        torch_state = torch.random.get_rng_state()
        cuda_states = None
        if self.device == "cuda" and torch.cuda.is_available():
            cuda_states = torch.cuda.get_rng_state_all()
        try:
            torch.manual_seed(self.seed)
            if cuda_states is not None:
                torch.cuda.manual_seed_all(self.seed)
            yield
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.random.set_rng_state(torch_state)
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)

    def _build_parameters(self, target_classes: int) -> dict[str, Any]:
        from ..registry import build_parameters

        return build_parameters(
            self.spec, seed=self.seed, device=self.device, target_classes=target_classes
        )

    def _canonical_parameter_identity(self) -> dict[str, Any]:
        torch_version = "none"
        if self.model_id in FOUNDATION_IDS:
            import torch

            torch_version = str(torch.__version__)
        return canonical_parameter_payload(
            self.parameters,
            model_id=self.model_id,
            package_name=self.package_name,
            package_version=self.package_version,
            seed=self.seed,
            device=self.device,
            checkpoint_identifier=self.checkpoint_identifier,
            checkpoint_sha256=self.checkpoint_sha256,
            python_major_minor=f"{sys.version_info.major}.{sys.version_info.minor}",
            pytorch_version=torch_version,
        )

    def _enforce_resource_limits(self) -> None:
        tracker = self.resource_tracker
        if tracker is None:
            raise AdapterFailure(
                FailureCategory.FAIL_RESOURCE_LIMIT, "resource tracker was not initialized"
            )
        tracker.capture()
        if not tracker.execution_telemetry_complete:
            raise AdapterFailure(
                FailureCategory.FAIL_RESOURCE_LIMIT,
                "resource telemetry incomplete: "
                f"{tracker.telemetry_error or 'required value missing'}",
            )
        if tracker.limit_exceeded:
            raise AdapterFailure(
                FailureCategory.FAIL_RESOURCE_LIMIT,
                tracker.limit_failure_reason or "configured resource limit was exceeded",
            )

    def _prepare_device(self) -> None:
        if self.model_id not in FOUNDATION_IDS:
            return
        _disable_network()
        self._resolve_checkpoint()
        if self.model_id in FOUNDATION_IDS:
            self.gpu_lock = GpuExecutionLock(self.root)
            self.gpu_lock.acquire()
        if self.device == "cuda":
            ensure_gpu_budget(
                headroom_mib=self.config.resources.gpu_headroom_mib,
                expected_peak_mib=self.expected_peak_vram_mib,
            )
            self.gpu_headroom_passed = True
            import torch

            torch.set_num_threads(self.config.resources.cpu_threads)
            try:
                torch.cuda.reset_peak_memory_stats(0)
            except Exception:
                pass
        else:
            try:
                import torch

                torch.set_num_threads(self.config.resources.cpu_threads)
            except Exception:
                pass

    def _resolve_checkpoint(self) -> None:
        expected = self.config.checkpoints.get(self.model_id)
        if expected is None or expected.identifier != self.spec.checkpoint:
            raise AdapterFailure(
                FailureCategory.FAIL_CONTRACT, "adapter checkpoint identity differs from registry"
            )
        frozen_checkpoint = FOUNDATION_CHECKPOINTS.get(self.model_id)
        if frozen_checkpoint != (expected.identifier, expected.sha256):
            raise AdapterFailure(
                FailureCategory.FAIL_CONTRACT,
                "runtime checkpoint configuration differs from the frozen identity",
            )
        candidates = (
            self.root / expected.identifier,
            self.root / "data" / "cache" / "models" / "tabicl" / expected.identifier,
            self.root / "data" / "cache" / "models" / "tabpfn" / expected.identifier,
        )
        path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if path is None:
            raise AdapterFailure(
                FailureCategory.BLOCKED_CHECKPOINT,
                f"validated offline checkpoint is unavailable: {expected.identifier}",
            )
        if self.model_id == "TICL2-2.2":
            self.parameters["allow_auto_download"] = False
        try:
            import torch

            torch_version = str(torch.__version__)
            code_commit = self.source_commit
            checkpoint_parameters = dict(self.parameters)
            checkpoint_parameters["model_path"] = expected.identifier
            canonical_parameters = canonical_parameter_payload(
                checkpoint_parameters,
                model_id=self.model_id,
                package_name=self.package_name,
                package_version=self.expected_version,
                seed=self.seed,
                device=self.device,
                checkpoint_identifier=expected.identifier,
                checkpoint_sha256=expected.sha256,
                python_major_minor=f"{sys.version_info.major}.{sys.version_info.minor}",
                pytorch_version=torch_version,
            )
            metadata = register_checkpoint(
                path,
                cache_dir=self.root / "data" / "cache" / "model_adapters" / "checkpoints",
                model_id=self.model_id,
                package_version=self.expected_version,
                checkpoint_identifier=expected.identifier,
                device_policy=self.device,
                model_parameters=canonical_parameters,
                python_major_minor=f"{sys.version_info.major}.{sys.version_info.minor}",
                pytorch_version=torch_version,
                code_commit=code_commit,
            )
            if metadata.checkpoint_sha256 != expected.sha256:
                raise AdapterFailure(
                    FailureCategory.FAIL_CACHE_INTEGRITY,
                    f"checkpoint checksum mismatch for {expected.identifier}",
                )
            validated = resolve_offline_checkpoint(
                self.root / "data" / "cache" / "model_adapters" / "checkpoints",
                metadata.cache_key,
            )
        except Exception as exc:
            raise AdapterFailure(FailureCategory.FAIL_CACHE_INTEGRITY, str(exc)) from exc
        if validated.checkpoint_sha256 != expected.sha256:
            raise AdapterFailure(
                FailureCategory.FAIL_CACHE_INTEGRITY, "offline cache hash mismatch"
            )
        self.checkpoint_path = path.resolve()
        self.checkpoint_identifier = expected.identifier
        self.checkpoint_sha256 = expected.sha256
        self.parameters["model_path"] = str(self.checkpoint_path)
        if self.model_id == "TICL2-2.2":
            self.parameters["allow_auto_download"] = False

    def _import_and_validate_model(self) -> type[Any]:
        from ..registry import import_class, validate_constructor_parameters

        model_class = import_class(self.spec.class_path)
        validate_constructor_parameters(self.spec, self.parameters)
        return model_class

    def _preprocess(
        self, predictors: pd.DataFrame, row_ids: list[str | int], partition: str
    ) -> Any:
        return self.preprocessor.transform(predictors, row_ids=row_ids, partition=partition)

    def _fit_estimator(self, matrix: Any, target: np.ndarray) -> None:
        model = self.model
        if model is None:
            raise AdapterFailure(FailureCategory.FAIL_CONTRACT, "model is not constructed")
        if self.model_id == "CAT-1.2":
            kwargs = {"cat_features": list(self.preprocessor.categorical_columns)}
            with _limited_native_threads():
                model.fit(matrix, target, **kwargs)
        else:
            with _limited_native_threads():
                model.fit(matrix, target)

    def _predict_estimator(self, matrix: Any) -> Any:
        model = self.model
        if model is None:
            raise AdapterFailure(FailureCategory.FAIL_CONTRACT, "model is not constructed")
        with _limited_native_threads():
            return model.predict_proba(matrix)

    def _cache_identity(
        self, row_ids: list[str | int], partition: str, source_data_sha256: str
    ) -> AdapterCacheIdentity:
        if not self.parameter_sha256 or not self.package_version:
            raise AdapterFailure(
                FailureCategory.FAIL_CONTRACT, "adapter identity is not initialized"
            )
        if self.model_id in FOUNDATION_IDS:
            import torch

            torch_version = str(torch.__version__)
        else:
            torch_version = "none"
        return AdapterCacheIdentity(
            source_data_sha256=source_data_sha256,
            fixture_sha256=self.fixture_sha256,
            row_ids_sha256=_row_identifiers_sha256(row_ids),
            model_spec_sha256=self.model_spec_sha256,
            parameter_sha256=self.parameter_sha256,
            adapter_sha256=self.adapter_sha256,
            preprocessing_sha256=self.preprocessor.fitted_state_sha256,
            checkpoint_sha256=self.checkpoint_sha256,
            source_commit=self.source_commit,
            package_runtime=(
                f"{self.package_name}=={self.package_version};python="
                f"{sys.version_info.major}.{sys.version_info.minor};torch={torch_version}"
            ),
            seed=self.seed,
            device_policy=self.device,  # type: ignore[arg-type]
            partition=partition,  # type: ignore[arg-type]
            split_identity=self.split_identity,
            transformation_identity=self.transformation_identity,
        )

    def _ensure_fitted(self) -> None:
        if not self.fitted or self.model is None:
            raise AdapterFailure(FailureCategory.FAIL_CONTRACT, "adapter must be fitted first")


def _row_identifiers_sha256(row_ids: list[str | int]) -> str:
    return _row_ids_sha256(row_ids)


def _classify_runtime_exception(exc: BaseException, device: str) -> FailureCategory:
    message = str(exc).lower()
    if "blocked_environment:" in message or "insufficient conservative free vram" in message:
        return FailureCategory.BLOCKED_ENVIRONMENT
    if isinstance(exc, MemoryError):
        return FailureCategory.FAIL_RESOURCE_LIMIT
    if any(term in message for term in ("cuda out of memory", "cublas_status_alloc_failed")):
        return (
            FailureCategory.FAIL_CUDA_OOM
            if device == "cuda"
            else FailureCategory.FAIL_RESOURCE_LIMIT
        )
    if "timeout" in message or "timed out" in message:
        return FailureCategory.FAIL_TIMEOUT
    if type(exc).__name__ == "Timeout":
        return FailureCategory.FAIL_TIMEOUT
    if any(term in message for term in ("out of memory", "cannot allocate memory")):
        return FailureCategory.FAIL_RESOURCE_LIMIT
    if "ram cap" in message or "available ram" in message:
        return FailureCategory.FAIL_RESOURCE_LIMIT
    if isinstance(exc, TypeError):
        return FailureCategory.FAIL_CONTRACT
    if isinstance(exc, ValueError):
        return FailureCategory.FAIL_INPUT
    return FailureCategory.FAIL_MODEL_RUNTIME


def _limited_native_threads() -> Any:
    try:
        from threadpoolctl import threadpool_limits

        return threadpool_limits(limits=2)
    except ImportError:
        from contextlib import nullcontext

        return nullcontext()
