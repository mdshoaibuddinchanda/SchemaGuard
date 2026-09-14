"""Create and validate deterministic SchemaOrbit condition manifests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    sys.path.insert(0, str(ROOT / "src"))
    from schemaguard.config import load_config
    from schemaguard.manifest import write_manifest

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default=ROOT / "configs" / "experiment_registry.yaml", type=Path
    )
    parser.add_argument("--output-dir", default=ROOT / "results" / "manifests", type=Path)
    args = parser.parse_args()

    config = load_config(args.config)
    expected = {"pilot": 990, "main": 3850}
    for scope, count in expected.items():
        actual = config.scheduled_count(scope)  # type: ignore[arg-type]
        if actual != count:
            raise SystemExit(f"Manifest count failure: {scope} expected {count}, found {actual}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    changed = {}
    for scope in expected:
        path = args.output_dir / f"{scope}_manifest.json"
        changed[scope] = write_manifest(config, scope, str(path))  # type: ignore[arg-type]

    summary = {
        "scope": "experiment_registry",
        "status": "PASS",
        "benchmark": config.benchmark,
        "config_checksum": config.checksum(),
        "pilot_scheduled": config.scheduled_count("pilot"),
        "main_scheduled": config.scheduled_count("main"),
        "manifest_changed": changed,
    }
    validation_path = ROOT / "results" / "validation" / "validation_summary.json"
    from schemaguard.config import atomic_write_json

    atomic_write_json(validation_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
