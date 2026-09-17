from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from schemaguard.models.adapters.base import FOUNDATION_IDS
from schemaguard.models.adapters.contracts import PredictionResult
from schemaguard.models.adapters.factory import load_adapter_config
from schemaguard.models.adapters.fixtures import adapter_fixture_catalog
from schemaguard.models.adapters.resources import gpu_memory_state
from schemaguard.utils.hashing import sha256_canonical_json

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "validate_model_adapters.py"


def _run_isolated_worker(
    model_id: str, device: str, order: str, fixture_id: str, result_path: Path, log_path: Path
) -> dict[str, object]:
    command = [
        sys.executable,
        str(SCRIPT),
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
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=900 if device == "cuda" else 1800,
            check=False,
        )
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    if completed.returncode != 0 and payload.get("status") == "PASS":
        raise AssertionError(f"isolated worker returned {completed.returncode}: {log_path}")
    return payload


@pytest.mark.integration
@pytest.mark.foundation_model
@pytest.mark.slow
@pytest.mark.parametrize("model_id", ["TPFN3-8.5", "TICL2-2.2"])
def test_foundation_cpu_prediction_uses_isolated_offline_worker(
    model_id: str, tmp_path: Path
) -> None:
    config = load_adapter_config()
    result_path = tmp_path / f"{model_id}.json"
    log_path = tmp_path / f"{model_id}.log"
    payload = _run_isolated_worker(
        model_id, "cpu", "normal", "binary_numerical", result_path, log_path
    )
    assert payload["status"] == "PASS", payload
    assert payload["offline_network_attempts"] == 0
    result = PredictionResult.model_validate(payload["prediction"])
    assert result.checkpoint_sha256 == config.checkpoints[model_id].sha256
    assert result.class_order == [0, 1]
    assert payload["resources"]["telemetry_complete"] is True


@pytest.mark.integration
@pytest.mark.foundation_model
@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("model_id", ["TPFN3-8.5", "TICL2-2.2"])
def test_foundation_gpu_probe_is_explicitly_gated_and_isolated(
    model_id: str, tmp_path: Path
) -> None:
    state = gpu_memory_state()
    if state["free_mib"] is None:
        capability = {
            "model_id": model_id,
            "device": "cuda",
            "status": "NOT_APPLICABLE",
            "reason": "CUDA memory telemetry unavailable",
        }
        (tmp_path / "gpu_capability.json").write_text(
            json.dumps(capability, sort_keys=True), encoding="utf-8"
        )
        assert capability["reason"]
        return
    result_path = tmp_path / f"{model_id}.cuda.json"
    log_path = tmp_path / f"{model_id}.cuda.log"
    payload = _run_isolated_worker(
        model_id, "cuda", "normal", "binary_numerical", result_path, log_path
    )
    assert payload["status"] == "PASS", payload
    result = PredictionResult.model_validate(payload["prediction"])
    assert result.device == "cuda"
    assert result.class_order == [0, 1]
    assert payload["resources"]["peak_vram_reserved_mib"] is not None


@pytest.mark.integration
@pytest.mark.foundation_model
@pytest.mark.slow
def test_foundation_normal_and_reordered_workers_keep_row_alignment(tmp_path: Path) -> None:
    fixture = adapter_fixture_catalog()["binary_numerical"]
    for model_id in sorted(FOUNDATION_IDS):
        first = _run_isolated_worker(
            model_id,
            "cpu",
            "normal",
            fixture.name,
            tmp_path / f"{model_id}.normal.json",
            tmp_path / f"{model_id}.normal.log",
        )
        second = _run_isolated_worker(
            model_id,
            "cpu",
            "reversed",
            fixture.name,
            tmp_path / f"{model_id}.reversed.json",
            tmp_path / f"{model_id}.reversed.log",
        )
        assert first["status"] == second["status"] == "PASS"
        normal = PredictionResult.model_validate(first["prediction"])
        reversed_result = PredictionResult.model_validate(second["prediction"])
        reversed_map = dict(
            zip(reversed_result.row_ids, reversed_result.probabilities, strict=True)
        )
        aligned = np.asarray([reversed_map[row_id] for row_id in normal.row_ids])
        assert np.max(np.abs(np.asarray(normal.probabilities) - aligned)) <= 1.0e-10
        assert sha256_canonical_json(normal.row_ids) == normal.row_ids_sha256
