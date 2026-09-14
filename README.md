# SchemaGuard

SchemaGuard studies prediction consistency after certified, lossless tabular schema migrations.

The repository is being implemented from
[`SCHEMAGUARD_MASTER_PLAN.md`](SCHEMAGUARD_MASTER_PLAN.md). The registry foundation provides
the strict experiment registry and deterministic condition manifests.

## Quick start

```powershell
conda activate P12
python scripts/generate_manifests.py
python -m pytest
```

The freeze command is safe to run repeatedly. It writes manifests through a temporary file and
atomic rename, and does not append duplicate conditions when the workspace is refreshed or the
command is retried.

All project execution must use the existing `P12` Conda environment. The repository lockfile is
kept for reproducibility, but no project virtual environment is required.

## Current status

| Phase | Scope | Status |
| --- | --- | --- |
| 00 | Specification freeze, identifiers, manifest counts | PASS |
| 01+ | Environment, data, transformations, models, execution | Not started |

The heavy model dependencies are intentionally not installed in the registry foundation. They will be added only
when their corresponding compatibility phase begins.
