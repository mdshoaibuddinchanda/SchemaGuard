"""Deterministic fold selection for the frozen five-fold protocol."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from schemaguard.constants import ROW_ID_COLUMN, TARGET_CODE_COLUMN
from schemaguard.utils.seeds import derive_component_seed

from .contracts import SplitGenerationConfig


def derive_fold_seed(seed: int, strategy: str = "stratified_group_5fold_v1") -> int:
    return derive_component_seed(seed, strategy)


def _folds(
    features: pd.DataFrame, targets: pd.DataFrame, config: SplitGenerationConfig, seed: int
) -> dict[str, int]:
    order = targets[ROW_ID_COLUMN].astype(str).sort_values(kind="mergesort").index
    ordered_targets = targets.loc[order].reset_index(drop=True).copy()
    ordered_targets[ROW_ID_COLUMN] = ordered_targets[ROW_ID_COLUMN].astype(str)
    feature_input = features.copy()
    feature_input[ROW_ID_COLUMN] = feature_input[ROW_ID_COLUMN].astype(str)
    feature_by_row = feature_input.set_index(ROW_ID_COLUMN)
    ordered_features = feature_by_row.loc[ordered_targets[ROW_ID_COLUMN].tolist()].reset_index()
    row_ids = ordered_targets[ROW_ID_COLUMN].tolist()
    target = ordered_targets[TARGET_CODE_COLUMN].reset_index(drop=True)
    predictors = ordered_features.drop(columns=[ROW_ID_COLUMN], errors="ignore")
    group_ids = ordered_features["predictor_group_id"].tolist()
    splitter = StratifiedGroupKFold(
        n_splits=config.group_folds,
        shuffle=True,
        random_state=derive_fold_seed(seed, config.strategy),
    )
    result: dict[str, int] = {}
    for fold, (_train, held_out) in enumerate(splitter.split(predictors, target, group_ids)):
        for index in held_out:
            row_id = row_ids[int(index)]
            if row_id in result:
                raise ValueError("a row was assigned to multiple grouped folds")
            result[row_id] = fold
    if set(result) != set(row_ids):
        raise ValueError("group folds do not cover every row")
    return result


def _candidate(
    fold_by_row: dict[str, int],
    row_ids: list[str],
    target_by_row: dict[str, int],
    config: SplitGenerationConfig,
    calibration_fold: int,
    test_fold: int,
) -> dict[str, Any] | None:
    folds = tuple(range(config.group_folds))
    train_folds = tuple(fold for fold in folds if fold not in {calibration_fold, test_fold})
    fold_to_partition = {
        fold: "calibration"
        if fold == calibration_fold
        else "test"
        if fold == test_fold
        else "train"
        for fold in folds
    }
    labels = {row_id: fold_to_partition[fold_by_row[row_id]] for row_id in row_ids}
    counts = {partition: 0 for partition in ("train", "calibration", "test")}
    class_counts: dict[str, dict[str, int]] = {partition: {} for partition in counts}
    for row_id in row_ids:
        partition = labels[row_id]
        code = str(target_by_row[row_id])
        counts[partition] += 1
        class_counts[partition][code] = class_counts[partition].get(code, 0) + 1
    if any(value == 0 for value in counts.values()):
        return None
    required = config.minimum_class_count_per_partition
    if any(
        class_counts[partition].get(str(code), 0) < required
        for partition in counts
        for code in sorted(set(target_by_row.values()))
    ):
        return {"hard_constraint_violation": True, "counts": counts, "class_counts": class_counts}
    target_counts = pd.Series(list(target_by_row.values())).value_counts().sort_index().to_dict()
    total_rows = len(row_ids)
    fractions = {
        "train": config.train_fraction,
        "calibration": config.calibration_fraction,
        "test": config.test_fraction,
    }
    size_deviations = {
        partition: abs(counts[partition] / total_rows - fractions[partition])
        for partition in counts
    }
    class_deviations = {
        partition: {
            str(code): abs(
                class_counts[partition].get(str(code), 0) / counts[partition] - count / total_rows
            )
            for code, count in target_counts.items()
        }
        for partition in counts
    }
    size_score = float(sum(size_deviations.values()))
    class_score = float(sum(sum(values.values()) for values in class_deviations.values()))
    return {
        "hard_constraint_violation": False,
        "labels": labels,
        "fold_assignment": {
            "train": list(train_folds),
            "calibration": [calibration_fold],
            "test": [test_fold],
        },
        "partition_counts": counts,
        "partition_class_counts": class_counts,
        "size_deviations": size_deviations,
        "class_proportion_deviations": class_deviations,
        "selection_score": {
            "size_deviation_sum": size_score,
            "class_proportion_deviation_sum": class_score,
        },
        "sort_key": (size_score, class_score, train_folds, calibration_fold, test_fold),
    }


def select_fold_assignment(
    features: pd.DataFrame, targets: pd.DataFrame, config: SplitGenerationConfig, seed: int
) -> dict[str, Any]:
    """Select the best valid calibration/test fold pair and retain diagnostics."""

    row_ids = targets[ROW_ID_COLUMN].astype(str).tolist()
    target_by_row = dict(zip(row_ids, targets[TARGET_CODE_COLUMN].astype(int), strict=True))
    fold_by_row = _folds(features, targets, config, seed)
    candidates: list[dict[str, Any]] = []
    candidate_diagnostics: list[dict[str, Any]] = []
    hard_violation_candidates = 0
    for calibration_fold in range(config.group_folds):
        for test_fold in range(config.group_folds):
            if calibration_fold == test_fold:
                continue
            item = _candidate(
                fold_by_row, row_ids, target_by_row, config, calibration_fold, test_fold
            )
            if item is None:
                candidate_diagnostics.append(
                    {
                        "calibration_fold": calibration_fold,
                        "test_fold": test_fold,
                        "hard_constraint_violation": True,
                        "reason": "empty_partition",
                    }
                )
                continue
            candidate_diagnostics.append(
                {
                    "calibration_fold": calibration_fold,
                    "test_fold": test_fold,
                    "hard_constraint_violation": item["hard_constraint_violation"],
                    "partition_counts": item["counts"]
                    if item["hard_constraint_violation"]
                    else item["partition_counts"],
                    "partition_class_counts": item["class_counts"]
                    if item["hard_constraint_violation"]
                    else item["partition_class_counts"],
                    "selection_score": item.get("selection_score"),
                }
            )
            if item["hard_constraint_violation"]:
                hard_violation_candidates += 1
                continue
            candidates.append(item)
    if not candidates:
        return {
            "status": "CONSTRAINT_INFEASIBLE",
            "candidate_count": 20,
            "valid_candidate_count": 0,
            "hard_violation_candidate_count": hard_violation_candidates,
            "fold_seed": derive_fold_seed(seed, config.strategy),
            "candidate_diagnostics": candidate_diagnostics,
            "message": "No grouped fold assignment meets the minimum class count constraint",
        }
    selected = min(candidates, key=lambda item: item["sort_key"])
    return {
        **selected,
        "status": "PASS",
        "fold_by_row": fold_by_row,
        "candidate_count": 20,
        "valid_candidate_count": len(candidates),
        "hard_violation_candidate_count": hard_violation_candidates,
        "fold_seed": derive_fold_seed(seed, config.strategy),
        "candidate_diagnostics": candidate_diagnostics,
    }
