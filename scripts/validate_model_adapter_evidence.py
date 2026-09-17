"""Independently validate the complete, sanitized model-adapter evidence set."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jsonschema import Draft202012Validator  # noqa: E402
from jsonschema.exceptions import ValidationError as JsonSchemaValidationError  # noqa: E402
from pydantic import ValidationError  # noqa: E402

from schemaguard.artifact_contracts import schema_documents  # noqa: E402
from schemaguard.models.adapters.evidence import validate_inventory_evidence  # noqa: E402


def _load(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load evidence JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"evidence root must be a JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_generated_schema(payload: dict[str, Any], filename: str, root: Path) -> None:
    schema_path = root / "schemas" / filename
    schema = _load(schema_path)
    Draft202012Validator.check_schema(schema)
    if schema != schema_documents()[filename]:
        raise ValueError(f"committed JSON schema is stale: {filename}")
    Draft202012Validator(schema).validate(payload)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inventory", type=Path, default=ROOT / "artifacts/handoff/model_adapter_inventory.json"
    )
    parser.add_argument(
        "--leakage-evidence",
        type=Path,
        default=ROOT / "artifacts/handoff/model_adapter_leakage_evidence.json",
    )
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    try:
        inventory_payload = _load(args.inventory)
        leakage_payload = _load(args.leakage_evidence)
        _validate_generated_schema(
            inventory_payload, "model_adapter_inventory.schema.json", args.root
        )
        _validate_generated_schema(
            leakage_payload, "model_adapter_leakage_evidence.schema.json", args.root
        )
        inventory, leakage = validate_inventory_evidence(
            inventory_payload,
            leakage_payload,
            root=args.root,
        )
    except (OSError, ValueError, ValidationError, JsonSchemaValidationError, KeyError) as exc:
        print(
            json.dumps(
                {
                    "stage": "model_adapter_evidence_validation",
                    "status": "FAIL",
                    "reason": str(exc),
                },
                sort_keys=True,
            )
        )
        return 1
    print(
        json.dumps(
            {
                "stage": "model_adapter_evidence_validation",
                "status": "PASS",
                "source_commit": inventory.source_commit,
                "inventory_records": len(inventory.records),
                "leakage_proofs": len(leakage.records),
                "cpu_cases": sum(record.device == "cpu" for record in inventory.records),
                "cuda_cases": sum(record.device == "cuda" for record in inventory.records),
                "inventory_sha256": _sha256(args.inventory),
                "leakage_evidence_sha256": _sha256(args.leakage_evidence),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
