"""Validate locally retained generated evidence; intentionally not a clean-suite test."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from validate_repository_repair import _check_local_baseline, _load_contract_module  # noqa: E402

from schemaguard.artifact_contracts import (  # noqa: E402
    DatasetRegistryReportContract,
    GpuCapacityReportContract,
    ModelCompatibilityReportContract,
)

CHECKPOINTS = {
    "TPFN3-8.5": (
        ROOT / "tabpfn-v3-classifier-v3_default.ckpt",
        "d0d865d54dfbc524f5703104be90620182dca7e5fb2c16de72e9959ea18f3988",
    ),
    "TICL2-2.2": (
        ROOT / "tabicl-classifier-v2-20260212.ckpt",
        "bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0",
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path, contract: type) -> None:
    if not path.is_file():
        raise SystemExit(f"LOCAL_EVIDENCE_MISSING: {path}")
    try:
        contract.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except Exception as exc:
        raise SystemExit(f"LOCAL_EVIDENCE_INVALID: {path}: {exc}") from exc


def main() -> int:
    contracts = _load_contract_module(ROOT)
    baseline_status, baseline_message = _check_local_baseline(ROOT, contracts)
    if baseline_status != "PASS":
        raise SystemExit(f"LOCAL_EVIDENCE_BASELINE_{baseline_status}: {baseline_message}")
    _load(ROOT / "results/validation/dataset_registry_report.json", DatasetRegistryReportContract)
    _load(ROOT / "results/validation/gpu_capacity_report.json", GpuCapacityReportContract)
    _load(
        ROOT / "results/validation/model_compatibility_report.json",
        ModelCompatibilityReportContract,
    )
    for model_id, (path, expected_hash) in CHECKPOINTS.items():
        if not path.is_file():
            raise SystemExit(f"LOCAL_EVIDENCE_MISSING_CHECKPOINT: {path}")
        observed_hash = _sha256(path)
        if observed_hash != expected_hash:
            raise SystemExit(
                f"LOCAL_EVIDENCE_CHECKPOINT_HASH_MISMATCH: {model_id}: "
                f"expected {expected_hash}, observed {observed_hash}"
            )
    print("local_evidence_valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
