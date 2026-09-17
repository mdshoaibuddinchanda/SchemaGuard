"""Run a bounded offline synthetic cache/scheduler probe (not a research experiment)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from schemaguard.cache.index import CacheIndex  # noqa: E402
from schemaguard.cache.store import CacheStore  # noqa: E402
from schemaguard.runner.plan import (  # noqa: E402
    current_commit,
    load_config,
    make_probe_plan,
    runtime_configuration_hash,
)
from schemaguard.runner.scheduler import Scheduler  # noqa: E402
from schemaguard.utils.io import atomic_write_json  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--offline", action="store_true", help="disable all outbound connections")
    parser.add_argument(
        "--resume", action="store_true", help="revalidate and resume persisted tasks"
    )
    parser.add_argument("--retry-failed", action="store_true", help="explicitly retry failed tasks")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if not args.offline:
        raise SystemExit("offline mode is mandatory for this synthetic scheduler probe")
    if sys.version_info[:2] != (3, 12):
        raise SystemExit("cache/scheduler probe requires Python 3.12")
    environment = os.environ.get("CONDA_DEFAULT_ENV")
    if environment and Path(environment).name.casefold() != "p12":
        raise SystemExit("cache/scheduler probe must use the existing P12 environment")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    config_path = (ROOT / args.config).resolve() if not args.config.is_absolute() else args.config
    config = load_config(config_path)
    plan, source_hash, lock_hash = make_probe_plan(ROOT, config)
    commit = current_commit(ROOT)
    cache_root = ROOT / config.cache_root
    state_root = ROOT / config.state_root
    store = CacheStore(cache_root, lock_timeout_seconds=config.lock_timeout_seconds)
    index = CacheIndex(state_root / "cache_index.sqlite", cache_root)
    scheduler = Scheduler(
        ROOT,
        config,
        store,
        index,
        source_commit=commit,
    )
    result = scheduler.run(plan, resume=args.resume, retry_failed=args.retry_failed)
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "source_implementation_commit": commit,
        "source_implementation_sha256": source_hash,
        "dependency_lock_sha256": lock_hash,
        "configuration_sha256": runtime_configuration_hash(config_path),
        "plan_hash": plan.plan_hash,
        "mode": result.manifest.mode,
        "run_manifest": result.manifest.model_dump(mode="json"),
        "state_path": result.state_path.resolve().relative_to(ROOT).as_posix(),
    }
    run_directory = state_root / "probe_runs"
    run_directory.mkdir(parents=True, exist_ok=True)
    output_path = run_directory / ("resume_run.json" if args.resume else "cold_run.json")
    atomic_write_json(output_path, receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0 if result.manifest.failed == 0 and result.manifest.blocked == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
