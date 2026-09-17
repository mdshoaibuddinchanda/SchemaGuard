"""Final-boundary metrics for already completed, label-free smoke predictions."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, cast

import numpy as np
import pandas as pd

from ..utils.hashing import sha256_canonical_json, sha256_file
from ..utils.io import atomic_write_parquet
from .contracts import (
    MODEL_ORDER,
    ConditionRun,
    Device,
    MetricName,
    MetricRecord,
    ModelId,
    PairedMetricRecord,
    Partition,
    SmokePlan,
    ViewId,
)

EVALUATION_PARTITIONS: tuple[Partition, ...] = ("calibration", "test")

METRIC_COLUMNS = (
    "model_id",
    "view_id",
    "device",
    "partition",
    "row_count",
    "brier_score",
    "log_loss",
    "accuracy",
    "balanced_accuracy",
    "roc_auc",
    "roc_auc_status",
    "expected_calibration_error",
    "prediction_sha256",
    "target_artifact_sha256",
)
PAIRED_COLUMNS = (
    "model_id",
    "device",
    "partition",
    "row_count",
    "sii_mean_js_divergence",
    "sii_median_js_divergence",
    "sii_p90_js_divergence",
    "sii_max_js_divergence",
    "label_flip_rate",
    "maximum_absolute_probability_difference",
    "mean_absolute_probability_difference",
    "metric_deltas_v01_minus_v00",
    "v00_prediction_sha256",
    "v01_prediction_sha256",
)


def expected_prediction_paths(root: str | Path, runs: list[ConditionRun]) -> list[Path]:
    project = Path(root).resolve()
    if len({run.condition_id for run in runs}) != 10:
        raise ValueError("evaluation requires ten unique condition records")
    if any(run.status != "PASS" for run in runs):
        raise ValueError("evaluation is prohibited until every smoke condition passes")
    records = [file for run in runs if run.status == "PASS" for file in run.prediction_files]
    if len(records) != 20 or len({file.relative_path for file in records}) != 20:
        raise ValueError("all ten conditions and both partitions must pass before evaluation")
    paths: list[Path] = []
    for record in records:
        path = project / record.relative_path
        if not path.is_file() or sha256_file(path) != record.sha256:
            raise ValueError(f"prediction artifact is missing or changed: {record.relative_path}")
        paths.append(path)
    return paths


def _ece(y_true: np.ndarray, probabilities: np.ndarray, bins: int) -> float:
    confidence = probabilities.max(axis=1)
    predicted = probabilities.argmax(axis=1)
    correct = (predicted == y_true).astype(np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(y_true)
    error = 0.0
    for index in range(bins):
        if index == bins - 1:
            mask = (confidence >= edges[index]) & (confidence <= edges[index + 1])
        else:
            mask = (confidence >= edges[index]) & (confidence < edges[index + 1])
        if mask.any():
            error += (
                float(mask.sum())
                / total
                * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
            )
    return error


def _metrics_for(
    prediction: pd.DataFrame,
    y_true: np.ndarray,
    *,
    epsilon: float,
    ece_bins: int,
    probability_tolerance: float,
    prediction_sha256: str,
    target_sha256: str,
) -> MetricRecord:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

    probabilities = prediction[["p_0", "p_1"]].to_numpy(dtype=np.float64)
    if len(y_true) == 0:
        raise ValueError("smoke metrics cannot be computed from an empty partition")
    if probabilities.shape != (len(y_true), 2):
        raise ValueError("smoke output probability matrix has the wrong shape")
    if (
        not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or (probabilities > 1).any()
    ):
        raise ValueError("smoke output contains invalid probabilities")
    if float(np.max(np.abs(probabilities.sum(axis=1) - 1.0))) > probability_tolerance:
        raise ValueError("smoke output probability rows exceed the simplex tolerance")
    predicted = probabilities.argmax(axis=1)
    if not np.array_equal(predicted, prediction["predicted_class"].to_numpy(dtype=np.int64)):
        raise ValueError("persisted predicted class differs from canonical probability argmax")
    from sklearn.metrics import log_loss

    clipped = np.clip(probabilities, epsilon, 1.0 - epsilon)
    clipped /= clipped.sum(axis=1, keepdims=True)
    one_hot = np.eye(2, dtype=np.float64)[y_true]
    brier = float(np.mean(np.square(probabilities - one_hot).sum(axis=1)))
    loss = float(log_loss(y_true, clipped, labels=[0, 1]))
    if len(np.unique(y_true)) == 2:
        auc = float(roc_auc_score(y_true, probabilities[:, 1]))
        auc_status: Literal["defined", "undefined_single_class"] = "defined"
    else:
        auc = None
        auc_status = "undefined_single_class"
    return MetricRecord(
        model_id=prediction["model_id"].iloc[0],
        view_id=prediction["view_id"].iloc[0],
        device=prediction["device"].iloc[0],
        partition=prediction["partition"].iloc[0],
        row_count=len(y_true),
        brier_score=brier,
        log_loss=loss,
        accuracy=float(accuracy_score(y_true, predicted)),
        balanced_accuracy=float(balanced_accuracy_score(y_true, predicted)),
        roc_auc=auc,
        roc_auc_status=auc_status,
        expected_calibration_error=_ece(y_true, probabilities, ece_bins),
        prediction_sha256=prediction_sha256,
        target_artifact_sha256=target_sha256,
    )


def _js_divergences(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    midpoint = (left + right) / 2.0
    left_terms = np.zeros_like(left)
    right_terms = np.zeros_like(right)
    positive_left = left > 0
    positive_right = right > 0
    left_terms[positive_left] = left[positive_left] * np.log2(
        left[positive_left] / midpoint[positive_left]
    )
    right_terms[positive_right] = right[positive_right] * np.log2(
        right[positive_right] / midpoint[positive_right]
    )
    return 0.5 * left_terms.sum(axis=1) + 0.5 * right_terms.sum(axis=1)


def evaluate_smoke(
    root: str | Path,
    plan: SmokePlan,
    runs: list[ConditionRun],
) -> tuple[list[MetricRecord], list[PairedMetricRecord], str, str]:
    """Open calibration/test labels only after verifying all 20 prediction files."""
    project = Path(root).resolve()
    paths = expected_prediction_paths(project, runs)
    from .planning import validate_views_at_evaluation_boundary

    validate_views_at_evaluation_boundary(project)
    # This is the deliberate label-access boundary: all predictions are complete and hashed.
    targets = pd.read_parquet(project / "data/processed/openml/1464/targets.parquet")
    if (
        sha256_file(project / "data/processed/openml/1464/targets.parquet")
        != plan.target_artifact_sha256
    ):
        raise ValueError("target artifact changed between planning and final evaluation")
    target_by_id = targets.set_index("__sg_row_id")["target_code"]
    target_sha = plan.target_artifact_sha256
    prediction_by_key: dict[tuple[ModelId, ViewId, Partition], tuple[pd.DataFrame, str]] = {}
    for path in paths:
        frame = pd.read_parquet(path)
        if frame.empty or frame["row_id"].duplicated().any():
            raise ValueError(f"prediction table is empty or duplicates row IDs: {path.name}")
        condition_id = str(frame["condition_id"].iloc[0])
        condition = next(item for item in plan.conditions if item.condition_id == condition_id)
        partition_value = str(frame["partition"].iloc[0])
        if partition_value not in EVALUATION_PARTITIONS:
            raise ValueError("prediction output has an unknown evaluation partition")
        partition = cast(Partition, partition_value)
        expected_view = next(item for item in plan.views if item.view_id == condition.view_id)
        if len(frame) != expected_view.partition_rows[partition]:
            raise ValueError("prediction output row count differs from the planned partition")
        if (
            sha256_canonical_json(frame["row_id"].tolist())
            != expected_view.partition_row_id_sha256[partition]
        ):
            raise ValueError("prediction output row IDs differ from the planned partition")
        if (
            frame["model_id"].ne(condition.model_id).any()
            or frame["view_id"].ne(condition.view_id).any()
        ):
            raise ValueError("prediction output does not match its frozen condition identity")
        if frame["target_artifact_sha256"].ne(plan.target_artifact_sha256).any():
            raise ValueError("prediction table is not bound to the planned target artifact")
        key = (condition.model_id, condition.view_id, partition)
        if key in prediction_by_key:
            raise ValueError(f"duplicate prediction output for {key}")
        prediction_by_key[key] = (frame, sha256_file(path))

    metric_records: list[MetricRecord] = []
    for condition in plan.conditions:
        for partition in EVALUATION_PARTITIONS:
            frame, file_sha = prediction_by_key[(condition.model_id, condition.view_id, partition)]
            try:
                y_true = target_by_id.loc[frame["row_id"].tolist()].to_numpy(dtype=np.int64)
            except KeyError as exc:
                raise ValueError("prediction output references an unknown target row ID") from exc
            metric_records.append(
                _metrics_for(
                    frame,
                    y_true,
                    epsilon=plan.log_loss_epsilon,
                    ece_bins=plan.ece_bins,
                    probability_tolerance=plan.probability_sum_tolerance,
                    prediction_sha256=file_sha,
                    target_sha256=target_sha,
                )
            )
    metric_map = {(item.model_id, item.view_id, item.partition): item for item in metric_records}
    paired_records: list[PairedMetricRecord] = []
    for model_id in MODEL_ORDER:
        device: Device = "cuda" if model_id in {"TPFN3-8.5", "TICL2-2.2"} else "cpu"
        for partition in EVALUATION_PARTITIONS:
            left, left_sha = prediction_by_key[(model_id, "V00", partition)]
            right, right_sha = prediction_by_key[(model_id, "V01", partition)]
            paired = left[["row_id", "p_0", "p_1", "predicted_class"]].merge(
                right[["row_id", "p_0", "p_1", "predicted_class"]],
                on="row_id",
                how="outer",
                suffixes=("_v00", "_v01"),
                validate="one_to_one",
                indicator=True,
            )
            if len(paired) != len(left) or not paired["_merge"].eq("both").all():
                raise ValueError("cross-view prediction row alignment is incomplete")
            left_probs = paired[["p_0_v00", "p_1_v00"]].to_numpy(dtype=np.float64)
            right_probs = paired[["p_0_v01", "p_1_v01"]].to_numpy(dtype=np.float64)
            sii_values = _js_divergences(left_probs, right_probs)
            absolute_differences = np.abs(left_probs - right_probs)
            delta: dict[MetricName, float | None] = {}
            metric_names: tuple[MetricName, ...] = (
                "brier_score",
                "log_loss",
                "accuracy",
                "balanced_accuracy",
                "roc_auc",
                "expected_calibration_error",
            )
            for metric_name in metric_names:
                a = getattr(metric_map[(model_id, "V00", partition)], metric_name)
                b = getattr(metric_map[(model_id, "V01", partition)], metric_name)
                delta[metric_name] = None if a is None or b is None else float(b - a)
            paired_records.append(
                PairedMetricRecord(
                    model_id=model_id,
                    device=device,
                    partition=partition,
                    row_count=len(paired),
                    sii_mean_js_divergence=float(sii_values.mean()),
                    sii_median_js_divergence=float(np.median(sii_values)),
                    sii_p90_js_divergence=float(np.quantile(sii_values, 0.9)),
                    sii_max_js_divergence=float(sii_values.max()),
                    label_flip_rate=float(
                        np.mean(
                            paired["predicted_class_v00"].to_numpy()
                            != paired["predicted_class_v01"].to_numpy()
                        )
                    ),
                    maximum_absolute_probability_difference=float(absolute_differences.max()),
                    mean_absolute_probability_difference=float(absolute_differences.mean()),
                    metric_deltas_v01_minus_v00=delta,
                    v00_prediction_sha256=left_sha,
                    v01_prediction_sha256=right_sha,
                )
            )

    metric_path = project / "results/smoke/metrics/model_metrics.parquet"
    paired_path = project / "results/smoke/metrics/paired_view_metrics.parquet"
    atomic_write_parquet(
        metric_path,
        pd.DataFrame(
            [item.model_dump(mode="json") for item in metric_records], columns=METRIC_COLUMNS
        ),
    )
    atomic_write_parquet(
        paired_path,
        pd.DataFrame(
            [item.model_dump(mode="json") for item in paired_records], columns=PAIRED_COLUMNS
        ),
    )
    return metric_records, paired_records, sha256_file(metric_path), sha256_file(paired_path)


__all__ = ["evaluate_smoke", "expected_prediction_paths"]
