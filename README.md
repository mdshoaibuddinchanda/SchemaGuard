# SchemaGuard

SchemaGuard studies prediction consistency after certified, lossless tabular schema migrations.

The repository is being implemented from
[`SCHEMAGUARD_MASTER_PLAN.md`](SCHEMAGUARD_MASTER_PLAN.md). The registry foundation provides
the strict experiment registry and deterministic condition manifests.

## Quick start

```powershell
uv sync --extra dev
uv run python scripts/generate_manifests.py
uv run pytest -q -m "not network"
```

The freeze command is safe to run repeatedly. It writes manifests through a temporary file and
atomic rename, and does not append duplicate conditions when the workspace is refreshed or the
command is retried.

All project execution must use the existing `P12` Conda environment. The repository lockfile is
kept for reproducibility, and the active Phase 01 split is deterministic predictor-group-aware
stratification under `stratified_group_5fold_v1`.

## Current status

| Phase | Scope | Status |
| --- | --- | --- |
| Phase 00 | Specification freeze, identifiers, manifest counts | PASS |
| Phase 01 | Reproducible data foundation and predictor-group split audit | VERIFIED_PASS |
| Phase 02 | Models, transformations, and execution | Not started |

The heavy model dependencies are intentionally not installed in the registry foundation. They will be added only
when their corresponding compatibility phase begins.
