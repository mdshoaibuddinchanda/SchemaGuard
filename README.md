# SchemaGuard

SchemaGuard asks whether complete tabular prediction pipelines remain consistent after certified,
lossless schema migrations, and whether deterministic canonicalization can restore consistency
within a bounded inference budget. The [master plan](SCHEMAGUARD_MASTER_PLAN.md) defines the frozen
scientific scope.

The planned benchmark is SchemaOrbit-14 (14 datasets and 11 schema views). Its five frozen model
families are Logistic Regression (`scikit-learn==1.9.1`), CatBoost (`1.2.10`), XGBoost (`3.4.1`),
TabPFN (`8.5.0`), and TabICL (`2.2.0`). Adapter checks establish implementation readiness only;
they are not benchmark results and do not test the research hypothesis.

## Current status

| Workstream | Status | Evidence |
| --- | --- | --- |
| Experiment registry | `PASS` | `artifacts/handoff/experiment_registry_review.md` |
| Data foundation | `VERIFIED_PASS` | `artifacts/handoff/grouped_split_data_foundation_review.md` |
| Model runtime compatibility | `PASS` | `artifacts/handoff/model_compatibility_review.md` |
| SchemaOrbit-14 dataset registry | `VERIFIED_PASS` | `artifacts/handoff/dataset_registry_review.md` |
| GPU capacity and execution policy | `VERIFIED_PASS` | `artifacts/handoff/gpu_capacity_review.md` |
| Grouped split generation | `VERIFIED_PASS` | `artifacts/handoff/split_generation_review.md` |
| Transformation engine | `VERIFIED_PASS` | `artifacts/handoff/transformation_evidence_finalization.md` |
| Model adapters | `VERIFIED_PASS` | Accepted by the current workstream authorization; implementation evidence is indexed in `artifacts/handoff/README.md` |
| Cache and scheduler | `PASS_PENDING_REVIEW` | `artifacts/handoff/cache_scheduler_review.md` |
| Ten-condition smoke test | `VERIFIED_PASS` | Independent review: `artifacts/handoff/smoke_independent_review.md` |
| Pilot and main experiments | `NOT_STARTED` | Not authorized |

`PASS_PENDING_REVIEW` means independent review is still required for the cache and scheduler
workstream. The Model Adapters workstream is accepted as `VERIFIED_PASS` under the current
workstream authorization. The ten-condition smoke has independently passed as an implementation
and feasibility check; it is not a benchmark result and does not establish generalization or
improvement. No full dataset-seed-view matrix, pilot, or main experiment has been run. Pilot
protocol freezing is the next permitted workstream; pilot execution remains unstarted. SCNF, COSA,
statistical analysis, paper figures, and paper claims remain unstarted.

## Reproducibility and resource limits

Use the existing `P12` Conda environment with Python 3.12. CPU work is limited to two threads;
foundation-model probes run sequentially in isolated processes with at least 512 MiB VRAM headroom.
Checkpoints are validated by SHA-256 and used offline. Generated predictions, caches, logs, and
reports remain local and ignored.

## Run the local checks

Use the existing `P12` Conda environment (Python 3.12):

```powershell
conda activate P12
python -m pytest -q tests/unit -m "not network and not gpu and not foundation_model and not evidence"
python scripts/validate_model_adapters.py --device cpu
python scripts/run_cache_scheduler_probe.py --config configs/runtime/cache_scheduler.yaml --offline
python scripts/run_cache_scheduler_probe.py --config configs/runtime/cache_scheduler.yaml --offline --resume
python scripts/validate_cache_scheduler.py --config configs/runtime/cache_scheduler.yaml --inventory artifacts/handoff/cache_scheduler_inventory.json --fault-evidence artifacts/handoff/cache_scheduler_fault_evidence.json
```

The adapter validator uses deterministic synthetic fixtures and the locally validated frozen
checkpoints. It does not download datasets or launch research experiments. Generated predictions,
caches, logs, and reports are local ignored outputs under `results/` and `data/cache/`.

To regenerate the sanitized scheduler evidence after a source implementation commit, run
`conda run -n P12 python scripts/validate_cache_scheduler.py --config configs/runtime/cache_scheduler.yaml --inventory artifacts/handoff/cache_scheduler_inventory.json --fault-evidence artifacts/handoff/cache_scheduler_fault_evidence.json --record`.
The validator checks the committed inventory, probe summary, protected-file comparison, fault matrix,
schemas, and source hashes. GPU scheduling evidence uses synthetic workers and mocked telemetry; it
does not claim CUDA inference or measured VRAM use.

See [project status](docs/project_status.md), the [repository guide](docs/repository_guide.md), and
the [handoff index](artifacts/handoff/README.md) for current scope and review evidence. Browse
[GitHub Actions](https://github.com/mdshoaibuddinchanda/SchemaGuard/actions) for CI results.
