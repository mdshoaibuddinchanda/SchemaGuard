"""Execute 1,000 machine-verifiable round-trip examples for every view."""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from schemaguard.transformations.base import feature_schema_from_frame  # noqa: E402
from schemaguard.transformations.certificates import certificate_identity_hash  # noqa: E402
from schemaguard.transformations.implementation import (  # noqa: E402
    transformation_engine_implementation_hash,
)
from schemaguard.transformations.registry import (  # noqa: E402
    get_transformation,
    load_transformation_config,
)
from schemaguard.transformations.validation import validate_transformation  # noqa: E402
from schemaguard.utils.hashing import canonical_source_hash, hash_dataframe_logically  # noqa: E402
from schemaguard.utils.io import atomic_write_json  # noqa: E402

SEED = 1729
VIEW_IDS = [f"V{index:02d}" for index in range(11)]
CATEGORIES = [
    "a:b",
    "quoted'\"",
    r"slash\\value",
    "",
    "Δ-unicode",
    True,
    np.int8(-1),
    np.float32(2.5),
]
CATEGORICAL_VALUES = {"category": CATEGORIES, "bool_category": [False, True]}


def generated_frame(example: int, row_id_prefix: str = "property-row") -> pd.DataFrame:
    rng = np.random.default_rng(SEED + example)
    row_count = 1 if example % 11 == 0 else 12
    values = rng.integers(-1000, 1001, size=row_count)
    nullable = pd.Series(values, dtype="Int64")
    if row_count > 1:
        nullable.iloc[0] = pd.NA
    category_values = [CATEGORIES[index % len(CATEGORIES)] for index in range(row_count)]
    frame = pd.DataFrame(
        {
            "__sg_row_id": [f"{row_id_prefix}-{example}-{index}" for index in range(row_count)],
            "amount_float32": (values.astype(np.float32) / 7.0),
            "amount_float64": values.astype(np.float64) / 7.0,
            "extreme_magnitude": values.astype(np.float64) * 1.0e6,
            "near_zero": np.full(row_count, np.nextafter(0.0, 1.0)),
            "constant_int16": np.full(row_count, 7, dtype=np.int16),
            "count_int8": (values % 100).astype(np.int8),
            "count_int16": (values * 2).astype(np.int16),
            "count_int32": (values * 1000).astype(np.int32),
            "count_int64": values.astype(np.int64),
            "nullable_count": nullable,
            "category": category_values,
            "bool_category": [index % 2 == 0 for index in range(row_count)],
            "feature:quoted": values.astype(np.float64),
            "amount_float32__duplicate": values.astype(np.float32),
            "count_int64__quotient": values.astype(np.int64),
            "count_int64__remainder": values.astype(np.int64),
        }
    )
    # Keep the mixed typed vocabulary in an object column even for one-row cases.
    frame["category"] = pd.Series(category_values, index=frame.index, dtype="object")
    return frame


def target_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "__sg_row_id": frame["__sg_row_id"],
            "__sg_target_code": [index % 2 for index in range(len(frame))],
        }
    )


def assert_property_roundtrip(
    source: pd.DataFrame,
    transformed: pd.DataFrame,
    restored: pd.DataFrame,
    certificate,
    *,
    rtol: float,
    atol: float,
) -> None:
    """Check every declared invariant on one independently certified example."""

    if len(source) != len(transformed) or len(source) != len(restored):
        raise AssertionError("row count changed")
    if transformed["__sg_row_id"].duplicated().any():
        raise AssertionError("transformation duplicated a row identifier")
    if set(source["__sg_row_id"]) != set(transformed["__sg_row_id"]):
        raise AssertionError("transformation changed row IDs")
    if tuple(restored.columns) != tuple(source.columns):
        raise AssertionError("reconstruction columns changed")
    if [str(dtype) for dtype in restored.dtypes] != [str(dtype) for dtype in source.dtypes]:
        raise AssertionError("reconstruction dtypes changed")
    if (
        not source["__sg_row_id"]
        .reset_index(drop=True)
        .equals(restored["__sg_row_id"].reset_index(drop=True))
    ):
        raise AssertionError("reconstruction row order changed")
    for column in source.columns:
        if not source[column].isna().equals(restored[column].isna()):
            raise AssertionError(f"missing mask changed for {column}")
        if pd.api.types.is_numeric_dtype(source[column]):
            mask = source[column].notna() & restored[column].notna()
            if mask.any() and not np.allclose(
                source.loc[mask, column].astype(float),
                restored.loc[mask, column].astype(float),
                rtol=rtol,
                atol=atol,
            ):
                raise AssertionError(f"numeric reconstruction exceeded tolerance for {column}")
        else:
            for source_value, restored_value in zip(
                source[column].tolist(), restored[column].tolist(), strict=True
            ):
                if pd.isna(source_value) and pd.isna(restored_value):
                    continue
                if type(source_value) is not type(restored_value) or source_value != restored_value:
                    raise AssertionError(f"categorical reconstruction changed values for {column}")


def run_view(view_id: str, examples: int, config: object) -> dict[str, object]:
    passed = 0
    failures: list[str] = []
    rtol = float(config.model_dump()["numerical_rtol"])
    atol = float(config.model_dump()["numerical_atol"])
    for example in range(examples):
        try:
            source = generated_frame(example)
            training_example = example + 1 if example % 11 == 0 else example
            training = generated_frame(training_example, "fit-row")
            schema = feature_schema_from_frame(training, CATEGORICAL_VALUES)
            transformation = get_transformation(view_id, config.model_dump())
            transformation.fit(training, "property", SEED, schema)
            target = target_frame(source)
            target_hash = hash_dataframe_logically(target)
            output = transformation.transform(source, "test")
            certificate = transformation.certificate_for(
                source,
                output,
                "test",
                source_target_hash=target_hash,
                output_target_hash=target_hash,
            )
            restored = transformation.reconstruct(output, certificate)
            validation = validate_transformation(
                source,
                output,
                restored,
                certificate,
                rtol=rtol,
                atol=atol,
                transformation=transformation,
            )
            assert_property_roundtrip(
                source, output, restored, certificate, rtol=rtol, atol=atol
            )
            if (
                certificate.source_target_hash != target_hash
                or certificate.output_target_hash != target_hash
            ):
                raise AssertionError("target hash claim changed")
            if validation["source_schema_hash"] != certificate.source_schema_hash:
                raise AssertionError("certificate source schema claim changed")
            repeat = transformation.transform(source, "test")
            repeat_certificate = transformation.certificate_for(
                source,
                repeat,
                "test",
                source_target_hash=target_hash,
                output_target_hash=target_hash,
            )
            repeat_restored = transformation.reconstruct(repeat, repeat_certificate)
            validate_transformation(
                source,
                repeat,
                repeat_restored,
                repeat_certificate,
                rtol=rtol,
                atol=atol,
                transformation=transformation,
            )
            assert_property_roundtrip(
                source, repeat, repeat_restored, repeat_certificate, rtol=rtol, atol=atol
            )
            if hash_dataframe_logically(output) != hash_dataframe_logically(repeat):
                raise AssertionError("repeat output hash changed")
            if certificate_identity_hash(certificate) != certificate_identity_hash(
                repeat_certificate
            ):
                raise AssertionError("repeat certificate identity changed")
            passed += 1
        except Exception as exc:  # pragma: no cover - evidence records the concrete failure
            failures.append(f"example={example}: {type(exc).__name__}: {exc}")
    return {
        "schema_version": 1,
        "view_id": view_id,
        "executed_example_count": examples,
        "passed_example_count": passed,
        "failed_example_count": examples - passed,
        "seed": SEED,
        "deterministic_profile": "synthetic_mixed_numeric_categorical_v1",
        "test_implementation_hash": canonical_source_hash(Path(__file__).resolve()),
        "transformation_engine_implementation_hash": transformation_engine_implementation_hash(),
        "execution_timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "failure_examples": failures[:10],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/runtime/transformation_engine.yaml"
    )
    parser.add_argument("--examples", type=int, default=1000)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/handoff/transformation_property_evidence.json",
    )
    args = parser.parse_args()
    if args.examples < 1000:
        raise SystemExit("--examples must be at least 1000")
    config = load_transformation_config(args.config)
    evidence = [run_view(view_id, args.examples, config) for view_id in VIEW_IDS]
    atomic_write_json(args.output, evidence)
    print(
        {
            "views": len(evidence),
            "examples": sum(item["executed_example_count"] for item in evidence),
        }
    )
    return 0 if all(item["failed_example_count"] == 0 for item in evidence) else 1


if __name__ == "__main__":
    raise SystemExit(main())
