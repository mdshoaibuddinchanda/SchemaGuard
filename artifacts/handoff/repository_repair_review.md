# Repository Consistency and Reproducibility Review

## Status

`SUPERSEDED`

This earlier repair review is superseded. The first clean GitHub Actions run
for this repair failed during pytest collection because `jsonschema` was not
included in the development dependency contract. The verification-integrity
review records the corrected workflow and must be used for approval.

The earlier local evidence package recorded 40/40 local checks, but that
claim is superseded because clean GitHub Actions failed before test collection.
No split generation, transformation, model training, prediction generation,
SCNF, COSA, smoke, pilot, or main experiment was started by this repair.

## Starting state

* Commit: `69d95b8517706bece86cbfde0c383dd3b2177698`
* Branch: `main`
* Worktree: no unexpected tracked modifications; the user reference `.docx`
  remains untracked and untouched
* Runtime: Conda `P12`, `D:\Conda\P12\python.exe`, Python 3.12.14

## Naming migration

| Old path | New path | References updated | Verified |
| -------- | -------- | -----------------: | -------- |
| `scripts/01_prepare_smoke_data.py` | `scripts/prepare_smoke_data.py` | Yes | Yes |
| `scripts/02_check_model_compatibility.py` | `scripts/check_model_compatibility.py` | Yes | Yes |
| `scripts/02b_acquire_schemaorbit.py` | `scripts/acquire_schemaorbit.py` | Yes | Yes |
| `scripts/02b_profile_gpu_capacity.py` | `scripts/profile_gpu_capacity.py` | Yes | Yes |
| `tests/integration/test_phase01_pipeline.py` | `tests/integration/test_data_foundation_pipeline.py` | Yes | Yes |
| `tests/integration/test_phase01_preservation.py` | `tests/integration/test_data_foundation_preservation.py` | Yes | Yes |
| `tests/integration/test_phase02b_preservation.py` | `tests/integration/test_dataset_registry_evidence.py` | Yes | Yes |
| `artifacts/handoff/phase_02a_model_compatibility_review.md` | `artifacts/handoff/model_compatibility_review.md` | Yes | Yes |
| `artifacts/handoff/phase_02b_dataset_gpu_review.md` | `artifacts/handoff/dataset_registry_review.md` | Yes | Yes |
| `artifacts/handoff/phase_02b_gpu_execution_decision.md` | `artifacts/handoff/gpu_execution_policy.md` | Yes | Yes |
| `artifacts/handoff/phase_numbering_decision.md` | `artifacts/handoff/workflow_naming_policy.md` | Yes | Yes |
| `artifacts/phase_01_data_foundation/` | `artifacts/data_foundation/` | Yes | Yes |
| `artifacts/phase_02a_model_compatibility/` | `artifacts/model_compatibility/` | Yes | Yes |
| `artifacts/phase_02b_dataset_gpu/` | `artifacts/dataset_registry/` and `artifacts/gpu_capacity/` | Yes | Yes |

The migration inventory records source/destination hashes in
`artifacts/repository_repair/evidence_migration.json`. Scientific IDs such as
`LR-1.9`, `TPFN3-8.5`, `TICL2-2.2`, and OpenML IDs were not renamed.

## Schema restoration

| Schema | Source contract | Validated | Trackable |
| ------ | --------------- | --------: | --------: |
| `condition_manifest.schema.json` | `ConditionManifestContract` | Yes | Yes |
| `dataset_inventory.schema.json` | `DatasetRegistryReportContract` | Yes | Yes |
| `gpu_capacity_report.schema.json` | `GpuCapacityReportContract` | Yes | Yes |
| `model_compatibility.schema.json` | `ModelCompatibilityReportContract` | Yes | Yes |
| `repository_validation.schema.json` | `RepositoryValidationReportContract` | Yes | Yes |

Schemas are generated deterministically by
`scripts/generate_artifact_schemas.py` and checked by unit tests.

## Runtime reconciliation

* P12 Python: 3.12.14
* P12 PyTorch: 2.6.0+cu124
* CUDA build: 12.4; CUDA visible
* Lockfile PyTorch: `uv.lock` resolves public `torch==2.6.0`; the P12 lock
  records the installed CUDA build
* Dependency consistency: all five frozen model versions match exactly;
  `uv.lock` changed from SHA-256
  `e80be80bfc9334f62e3bd2285367752061960003bab0686549d00569ad139ead` to
  `206275eb2437cf83d50f8ff30bcb73db43d7513619aa7fb4ff2f725cd4495997`
* `pip check`: exit 1 because the pre-existing unrelated `tableshift==0.1`
  package declares missing dependencies and incompatible `numpy`/`ray`
  pins. It was not modified. Full output is retained in
  `artifacts/repository_repair/pip_check.txt`.

## Dataset validation hardening

* Artifact-hash validation: every processed manifest is checked against all
  stored artifact SHA-256 values.
* Ordered row alignment: feature and target row IDs must match exactly in
  order, with matching row counts and values.
* Label mapping: original-to-code and code-to-original mappings are checked
  as reversible contiguous mappings.
* Predictor grouping: `typed_predictor_sha256_v1` is used consistently.
* Cache recovery: processing uses temporary directories, atomic promotion,
  and quarantine records for invalid caches.
* Locking: each dataset uses a dedicated processing lock.
* Retry behavior: bounded attempts, timeout, and exponential backoff are
  configured and tested.

The offline registry report validates all 14 datasets. The preserved data
foundation remains 748 rows with split sizes 449/150/149, 502 predictor
groups, 69 duplicate groups, 31 conflicting-target groups, and zero groups
crossing partitions.

## GPU validation hardening

* Soft-limit enforcement: the 3,600 MiB limit is enforced; maximum observed
  reserved VRAM was 1,306 MiB.
* Fresh VRAM checks: performed before every primary and isolated launch.
* Continuous RAM monitoring: worker and descendant processes are sampled;
  the report records `monitoring_complete: true`.
* Strategy correction: strategies are named
  `repeated_inference_single_worker`, `fresh_worker_per_inference`, and
  `cpu_fallback`.
* Overall status logic: 12 requested primary profiles were accounted for and
  the GPU report status is `PASS`.

## Clean-checkout validation

* The default suite does not require ignored datasets or generated reports;
  the Windows CI workflow validates this on a clean checkout.
* Default non-network/non-GPU/non-foundation/non-evidence suite: 78 passed.
* Local populated-workspace evidence validation: `local_evidence_valid`.
* Offline model cache reuse: `PASS`, network disabled, zero network attempts.

## Preservation

* Data-foundation hashes: unchanged; comparison is retained in
  `artifacts/repository_repair/data_foundation_hash_comparison.json`.
* Checkpoint hashes: unchanged:
  * TabPFN: `d0d865d54dfbc524f5703104be90620182dca7e5fb2c16de72e9959ea18f3988`
  * TabICL: `bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0`
* Frozen configuration hashes: registry/model identities remain unchanged;
  operational hardening is documented in the repair evidence.

## Tests

| Test group | Passed | Failed | Skipped | Not verified |
| ---------- | -----: | -----: | ------: | -----------: |
| Default clean-checkout suite | 78 | 0 | 0 | 0 |
| Integration evidence suite | 5 | 0 | 0 | 0 |
| Evidence-only suite | 4 | 0 | 0 | 0 |
| Foundation-model marker suite | 0 | 0 | 2 | 0 |
| GPU marker suite | 0 | 0 | 1 | 0 |
| Ruff | 1 | 0 | 0 | 0 |
| Mypy (32 source files) | 1 | 0 | 0 | 0 |

The skipped marker tests point to the controlled P12 scripts; the actual
foundation CPU and GPU probes were executed and are recorded in the reports.

## Acceptance gates

The former R01–R40 local result is retained as historical evidence only and
is not an approval. The corrected structural validator and the final GitHub
Actions run are authoritative.

## Files renamed

The 11 history-preserving renames are listed in the naming migration table.

## Files created

* `.github/workflows/quality.yml`
* `requirements/p12_windows_constraints.txt`
* `schemas/condition_manifest.schema.json`
* `schemas/dataset_inventory.schema.json`
* `schemas/gpu_capacity_report.schema.json`
* `schemas/model_compatibility.schema.json`
* `schemas/repository_validation.schema.json`
* `scripts/generate_artifact_schemas.py`
* `scripts/validate_local_evidence.py`
* `scripts/validate_repository_naming.py`
* `scripts/validate_repository_repair.py`
* `src/schemaguard/artifact_contracts.py`
* `tests/integration/test_gpu_capacity_evidence.py`
* `tests/unit/test_artifact_schemas.py`
* `tests/unit/test_repository_naming.py`
* `artifacts/handoff/repository_repair_review.md`

Ignored generated evidence is retained locally under
`artifacts/repository_repair/` and `results/validation/`.

## Files modified

* `README.md`
* `SCHEMAGUARD_MASTER_PLAN.md`
* `.gitignore` policy tests and generated-path policy
* `configs/datasets/schemaorbit14.yaml`
* `pyproject.toml`
* `uv.lock`
* `artifacts/handoff/grouped_split_data_foundation_review.md`
* `scripts/audit_data_foundation.py`
* `src/schemaguard/compatibility/contracts.py`
* `src/schemaguard/compatibility/gpu_profiles.py`
* `src/schemaguard/compatibility/runner.py`
* `src/schemaguard/data/contracts.py`
* `src/schemaguard/data/pipeline.py`
* `src/schemaguard/data/schemaorbit.py`
* `src/schemaguard/manifest.py`
* `src/schemaguard/utils/logging.py`
* `tests/integration/test_foundation_cpu_probes.py`
* `tests/integration/test_foundation_gpu_probes.py`
* `tests/unit/test_gitignore_policy.py`
* `tests/unit/test_schemaorbit.py`

## Commands executed

```text
git rev-parse HEAD; git branch --show-current; git status --short; git log -1 --oneline — exit 0
conda run -n P12 python --version — exit 0 (Python 3.12.14)
conda run -n P12 python -m pip check — exit 1 (pre-existing tableshift conflict)
conda run -n P12 python scripts/generate_artifact_schemas.py — exit 0
conda run -n P12 python scripts/validate_repository_naming.py — exit 0
conda run -n P12 python -m ruff check . — exit 0
conda run -n P12 python -m mypy src/schemaguard — exit 0
conda run -n P12 python -m pytest -q -m "not network and not gpu and not foundation_model and not evidence" -rA — exit 0 (78 passed)
conda run -n P12 python -m pytest -q -m "integration and not network and not gpu and not foundation_model" -rA — exit 0 (5 passed)
conda run -n P12 python -m pytest -q -m "evidence and not network and not gpu and not foundation_model" -rA — exit 0 (4 passed)
conda run -n P12 python -m pytest -q -m "foundation_model and not network" -rA — exit 0 (2 controlled skips)
conda run -n P12 python -m pytest -q -m "gpu and not network" -rA — exit 0 (1 controlled skip)
conda run -n P12 python scripts/acquire_schemaorbit.py --config configs/datasets/schemaorbit14.yaml --offline — exit 0 (14 datasets PASS)
conda run -n P12 python scripts/check_model_compatibility.py --config configs/runtime/model_compatibility.yaml --device cpu --offline — exit 0 (5 CPU probes PASS)
conda run -n P12 python scripts/profile_gpu_capacity.py --inventory results/validation/dataset_registry_report.json — exit 0 (GPU report PASS)
conda run -n P12 python scripts/validate_local_evidence.py — exit 0
conda run -n P12 python scripts/validate_repository_repair.py — exit 0 (R01–R40: 40 PASS)
git diff --check — exit 0 (line-ending warnings only)
```

## Deviations

* `pip check` remains nonzero because of the pre-existing unrelated
  `tableshift==0.1` installation. No package was removed or upgraded to hide
  this conflict. The project’s five frozen model versions and P12 runtime
  evidence remain consistent.
* The clean-checkout claim is validated by the artifact-independent default
  suite and Windows CI configuration; no destructive checkout was performed
  in the user’s working tree.

## Failures

The former local pass claim is invalidated by the clean CI collection failure.
The `pip check` deviation is recorded above.

## Next permitted stage

`split_generation` may begin only after an independent review verifies this
repair, GitHub contains the five required schemas, P12 dependency
reproduction is accepted, clean-checkout tests pass, and data-foundation
hashes remain unchanged. This repair did not begin that stage.
