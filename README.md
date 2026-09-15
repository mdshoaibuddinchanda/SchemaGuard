# SchemaGuard

SchemaGuard studies prediction consistency after certified, lossless tabular schema migrations.

The repository is being implemented from
[`SCHEMAGUARD_MASTER_PLAN.md`](SCHEMAGUARD_MASTER_PLAN.md). The registry foundation provides
the strict experiment registry and deterministic condition manifests.

## Quick start

```powershell
conda run -n P12 python scripts/generate_manifests.py
conda run -n P12 python -m pytest -q -m "not network and not evidence"
```

The freeze command is safe to run repeatedly. It writes manifests through a temporary file and
atomic rename, and does not append duplicate conditions when the workspace is refreshed or the
command is retried.

All project execution must use the existing `P12` Conda environment. The repository lockfile is
kept for reproducibility, and the active data-foundation split is deterministic predictor-group-aware
stratification under `stratified_group_5fold_v1`.

## Current status

| Stage | Scope | Status |
| --- | --- | --- |
| Experiment registry | Specification freeze, identifiers, manifest counts | PASS |
| Data foundation | Reproducible data foundation and predictor-group split audit | VERIFIED_PASS |
| Model compatibility | Frozen model runtime gate | PASS |
| Dataset registry | SchemaOrbit-14 acquisition and validation | VERIFIED_PASS |
| GPU capacity | Foundation-model resource envelope | VERIFIED_PASS |
| Split generation | Additional grouped splits | VERIFIED_PASS |
| Transformation engine | Lossless schema views | NOT_STARTED |
| Model adapters | Frozen model adapters | NOT_STARTED |
| Smoke experiment | Controlled smoke experiment | NOT_STARTED |
| Pilot experiment | Pilot experiment | NOT_STARTED |
| Main experiment | Main experiment | NOT_STARTED |

All project execution uses the existing `P12` Conda environment. Foundation-model
compatibility is separately gated and does not authorize experiments.
