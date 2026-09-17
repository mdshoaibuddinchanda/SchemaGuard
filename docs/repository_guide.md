# Repository guide

SchemaGuard separates frozen scientific configuration, reusable implementation, validation
evidence, and generated local artifacts.

Use descriptive, responsibility-based path names. Do not add phase numbers, step numbers, or
execution-order prefixes; execution order belongs in the master plan and handoff metadata.

| Path | Responsibility |
| --- | --- |
| `configs/` | Frozen experiment, runtime, dataset, and baseline configuration. |
| `src/schemaguard/` | Data, split, transformation, compatibility, and model-adapter implementation. |
| `scripts/` | Explicit validation, schema-generation, and repository-check commands. |
| `schemas/` | Generated JSON Schemas for tracked evidence contracts. |
| `tests/unit/` | Fast contract, leakage, preprocessing, probability, and cache tests. |
| `tests/integration/` | Model round trips and isolated foundation-model probes. |
| `docs/` | Current status and repository operating guide. |
| `artifacts/handoff/` | Concise tracked reviews and sanitized inventory records. |
| `data/`, `results/`, `logs/`, `cache/` | Generated or private local inputs/outputs; ignored by Git. |

## Runtime and safety

Use only the existing `P12` Conda environment with Python 3.12. CPU work is limited to two
workers/threads; foundation models execute sequentially in isolated processes. TabICL's primary
configuration keeps `kv_cache=False`, and checkpoints remain local, content-addressed, and
untracked. Test/calibration labels are never passed to preprocessing or model fitting.

Before a workstream, check the current branch, commit, and worktree. Do not stage generated datasets,
predictions, model binaries, checkpoints, caches, logs, raw tracebacks, or the private root reference
document. Only concise, schema-validated handoff evidence belongs in `artifacts/handoff/`.

## Adapter validation

Run the synthetic adapter matrix with:

```powershell
conda activate P12
python scripts/validate_model_adapters.py --device cpu
```

The validator records one result per model/fixture case under ignored `results/`. The fixture-only
run is distinct from the full 770-case research matrix and does not authorize the smoke, pilot, or
main experiments. See `project_status.md` and `artifacts/handoff/model_adapters_review.md` for gate
status and the current review boundary.
