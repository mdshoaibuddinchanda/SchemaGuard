"""Generate the frozen SchemaOrbit-14 grouped split inventory."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from schemaguard.splits.contracts import SplitGenerationConfig  # noqa: E402
from schemaguard.splits.generation import generate_split  # noqa: E402
from schemaguard.splits.inventory import write_protected_snapshot  # noqa: E402
from schemaguard.splits.validation import validate_all, validate_split  # noqa: E402


def _config(path: Path) -> SplitGenerationConfig:
    with path.open("r", encoding="utf-8") as handle:
        return SplitGenerationConfig.model_validate(yaml.safe_load(handle))


def _tasks(
    config: SplitGenerationConfig, dataset_id: int | None, seed: int | None
) -> list[tuple[int, int]]:
    datasets = [dataset_id] if dataset_id is not None else config.datasets
    seeds = [seed] if seed is not None else config.seeds
    if dataset_id is not None and dataset_id not in config.datasets:
        raise ValueError(f"dataset ID {dataset_id} is not in the frozen configuration")
    if seed is not None and seed not in config.seeds:
        raise ValueError(f"seed {seed} is not in the frozen configuration")
    return [(item, value) for item in datasets for value in seeds]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/runtime/split_generation.yaml"
    )
    parser.add_argument("--workers", type=int, choices=(1, 2), default=None)
    parser.add_argument("--offline", action="store_true", help="require local source artifacts")
    parser.add_argument("--force-recompute", action="store_true")
    parser.add_argument("--dataset-id", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    del args.offline  # This workstream has no network code; all inputs are local by design.
    try:
        config = _config(args.config)
        if args.workers is not None:
            config = config.model_copy(update={"max_workers": args.workers})
        snapshot_path = (
            ROOT / "artifacts" / "validation" / "split_generation_hashes_before.json"
        )
        if not snapshot_path.is_file():
            write_protected_snapshot(ROOT, config, snapshot_path)
        tasks = _tasks(config, args.dataset_id, args.seed)
        if args.validate_only:
            result = (
                validate_all(ROOT, config)
                if args.dataset_id is None and args.seed is None
                else [validate_split(ROOT, config, tasks[0][0], tasks[0][1])]
            )
            print(json.dumps({"status": "PASS", "validated": len(result)}, sort_keys=True))
            return 0
        results: list[dict[str, Any]] = []
        errors: list[str] = []
        worker_count = args.workers or config.max_workers
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(
                    generate_split,
                    ROOT,
                    config,
                    dataset_id,
                    seed,
                    force_recompute=args.force_recompute,
                ): (dataset_id, seed)
                for dataset_id, seed in tasks
            }
            for future in as_completed(futures):
                dataset_id, seed = futures[future]
                try:
                    results.append(future.result())
                except Exception as exc:
                    errors.append(f"{dataset_id}/{seed}: {type(exc).__name__}: {exc}")
        results.sort(key=lambda item: (int(item["dataset_id"]), int(item["seed"])))
        summary: dict[str, Any] = {
            "status": "FAIL" if errors else "PASS",
            "task_count": len(tasks),
            "results": results,
            "errors": sorted(errors),
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 1 if errors else 0
    except Exception as exc:
        print(
            json.dumps({"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}, sort_keys=True)
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
