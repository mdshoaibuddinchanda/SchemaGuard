"""Prepare the frozen OpenML 1464 smoke dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "datasets" / "smoke_blood_transfusion.yaml",
    )
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--force-rebuild-processed", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT / "src"))

    from pydantic import ValidationError

    from schemaguard.data.download import OfflineCacheUnavailable, SourceIntegrityError
    from schemaguard.data.pipeline import PipelineError, _write_handoff, run_pipeline
    from schemaguard.utils.io import atomic_write_json

    try:
        outcome = run_pipeline(
            args.config,
            root=ROOT,
            offline=args.offline,
            validate_only=args.validate_only,
            force_rebuild_processed=args.force_rebuild_processed,
        )
    except ValidationError as exc:
        print(f"Configuration failure: {exc}", file=sys.stderr)
        return 2
    except OfflineCacheUnavailable as exc:
        print(f"Offline cache unavailable: {exc}", file=sys.stderr)
        return 5
    except SourceIntegrityError as exc:
        print(f"Source integrity failure: {exc}", file=sys.stderr)
        return 3
    except PipelineError as exc:
        message = str(exc)
        print(f"Pipeline failure: {message}", file=sys.stderr)
        return 4 if "validation" in message.lower() else 1
    except Exception as exc:
        print(f"Pipeline failure: {exc}", file=sys.stderr)
        return 1

    offline_ok = False
    if not args.offline and not args.validate_only:
        try:
            offline_outcome = run_pipeline(
                args.config,
                root=ROOT,
                offline=True,
                validate_only=False,
            )
            offline_ok = (
                outcome.download.source_manifest.computed_sha256
                == offline_outcome.download.source_manifest.computed_sha256
                and outcome.processed.artifact_hashes == offline_outcome.processed.artifact_hashes
                and outcome.splits.manifest.assignment_file_sha256
                == offline_outcome.splits.manifest.assignment_file_sha256
            )
        except Exception as exc:
            print(f"Offline rerun failure: {exc}", file=sys.stderr)
            return 5
    elif args.offline:
        offline_ok = outcome.download.cache_status == "hit"

    summary = outcome.summary.model_copy(update={"offline_rerun_check": offline_ok})
    atomic_write_json(outcome.paths.validation_summary_path, summary.canonical_dict())
    _write_handoff(outcome.paths, summary, outcome.download, outcome.processed, outcome.splits)
    print(f"dataset_id: {outcome.config.dataset.internal_id}")
    print(f"raw_path: {outcome.download.raw_path}")
    print(f"raw_sha256: {outcome.download.source_manifest.computed_sha256}")
    print(f"processed_path: {outcome.processed.features_path}")
    print(f"split_path: {outcome.splits.assignments_path}")
    print(f"validation_status: {summary.status}")
    print(f"cache_status: {outcome.download.cache_status}")
    print(f"offline_rerun: {'PASS' if offline_ok else 'NOT_RUN'}")
    print(f"report_path: {outcome.paths.foundation_dir}")
    return 0 if summary.status == "PASS" else 4


if __name__ == "__main__":
    raise SystemExit(main())
