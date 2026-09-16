# Transformation Integrity Follow-up Handoff

## Status

`PASS_PENDING_REVIEW`

This follow-up repairs the five defects identified by the independent review of
the transformation engine. It is not an independent scientific approval and
must not be labelled `VERIFIED_PASS` until the review is accepted.

completed_stage: `transformation_integrity_followup`
next_stage: `independent_review`

## Starting state

* Starting commit: `34c2b8bb11671cae9f8196bcb3e0c1ea8d6e89d4`
* Branch: `main`
* Python executable: `D:\Conda\P12\python.exe`
* Python version: `3.12.14`
* Conda environment: `P12`
* Initial tracked worktree state: clean.
* Initial untracked state: `SchemaGuard_Complete_Research_and_Engineering_Plan.docx` only.
* The private root `.docx` was not opened, edited, moved, deleted, staged, committed, or pushed.
* Model adapters, model benchmarking, SCNF, COSA, pilot work, and the main experiment remain untouched and not started.

## Repository hygiene

* The repository naming validator passed with zero violations.
* No generated dataset, cache, prediction, or local review log is staged by this follow-up.
* The existing `.gitignore` policy remains active; generated data and reports stay local.
* `git check-ignore -v --no-index` returned no rule for `schemas/example.schema.json` or `tests/fixtures/example.csv`; it correctly ignored `data/raw/example.csv`, `data/processed/example.parquet`, and `results/example.csv`, while `artifacts/handoff/example.md` remained trackable.
* The reference `.docx` remains untracked and untouched.

## Repairs completed

### Cache identity

`TransformationCacheManifest` is now a strict Pydantic contract. Cache reads
and publication reconstruct one canonical key from dataset/version, source
feature hash, target hash, split logical hash, partition, view ID, view
configuration hash, implementation hash, fit-parameter hash, certificate
schema version, and Python major/minor version. The cache also rehashes feature
and certificate bytes, validates the certificate/output relationship, rejects
unexpected files, uses a per-key lock, and writes feature, certificate, and
manifest files atomically. Copied entries, changed manifest keys, identity
tampering, invalid hashes, and stale files are rejected.

### Protected artifacts

Protected snapshots now report changed, added, removed, unchanged, allowed,
and unexpected paths separately. A comparison passes only when every
unexpected category is empty. The final comparison is:

* Status: `PASS`
* Before snapshot: `97ee389225879d1da4db181c147b50c54d352f540e0a605c6d881ba29d238b9e`
* After snapshot: `e24ab902bc01065661a1996452873327c910645587b18e82cc6b6fb42edbe5a1`
* Files before/after: `296` / `298`
* Allowed changed files: the existing transformation handoff, inventory, and three transformation schemas.
* Allowed added files: `schemas/transformation_cache_manifest.schema.json` and `schemas/transformation_property_evidence.schema.json`.
* Unexpected changed/added/removed files: `0` / `0` / `0`.

### Property evidence

Each of the eleven views now fits a transformation through its training
interface for every example, creates an example-specific certificate, and
independently reconstructs and validates the output with the fitted
transformation. Validation checks exact row IDs, columns, dtypes, missing
masks, categorical value types, target hashes, repeat output identity, and
repeat certificate identity.

* Views: `V00` through `V10`
* Executed: `11,000` (`1,000` per view)
* Passed: `11,000`
* Failed: `0`
* Property runner hash:
  `1706786e19c7a2ebf332ebaa42f2a9a83b47d31734e0552df9278d271df519fb`
* Evidence SHA-256:
  `09b88e81df16f2babec53fabf9b150029f7a6378a4048187a95820d932ea403d`

### Inventory contract

The inventory contract enforces the exact frozen matrix of 14 dataset IDs,
five seeds, and views `V00`–`V10`: `770` unique tuples. Counts are derived
from records, all applicable records pass, every N/A record has a controlled
reason, the protected comparison must pass, and each view requires at least
`1,000` successful property examples with the current runner hash.

* Primary inventory: `PASS_PENDING_REVIEW`; `770` records; `545` PASS; `0` FAIL; `225` N/A.
* Repeat inventory: `PASS_PENDING_REVIEW`; `770` records; `545` PASS; `0` FAIL; `225` N/A.
* Logical comparison ignoring runtime/timestamp fields: `PASS`.
* Primary inventory SHA-256:
  `c0570d0d913656237420194429676decf52e1caf6935be1c596035e70cab7477`
* Split inventory hash: `368acbd6ecaffafbecf6b5bdbce301211da4902ceaea05fbfb0db933a3611482`

### Projection and certificate proofs

Projection relationships now require declared exact coverage, existing parent
and generated columns, identical missing masks, exact typed equality for
duplicates, finite affine values, and exact `3*x+7` relationships. Unknown,
missing, altered, or extra relationships fail closed. Certificates use closed
world missing/dtype policies, bind reserved view names and tolerances, reject
contradictory PASS evidence, and cannot claim configuration/implementation
identity without the fitted transformation used for validation.

## Phase 01 preservation

The protected comparison and split validator passed without changes to the
Phase 01 data foundation or split artifacts.

* Dataset: OpenML `1464`.
* Rows: `748`.
* Partition sizes: train `449`, calibration `150`, test `149`.
* Predictor groups: `502`.
* Duplicated predictor groups: `69`.
* Conflicting-target groups: `31`.
* Predictor groups crossing partitions: `0`.
* Strategy: `stratified_group_5fold_v1`.
* Split validator: `70`/`70` dataset-seed records passed.
* Protected-snapshot transformation inventory entry before: `4d9745443ca212a618fe413ca2369ee9b6341fa8f43c4d2b16d998112d048def`.
* Protected-snapshot transformation inventory entry after: `c0570d0d913656237420194429676decf52e1caf6935be1c596035e70cab7477`.
* The full protected snapshot reports no unexpected mutation.

## Tests and quality checks

| Test group | Passed | Failed | Skipped | Not verified |
| --- | ---: | ---: | ---: | ---: |
| Full relevant unit tests plus transformation roundtrip | 197 | 0 | 0 | 0 |
| Non-network, non-GPU, non-foundation marker suite | 214 | 0 | 0 | 0 |
| Non-network integration marker suite | 7 | 0 | 0 | 0 |
| Split validation | 70 | 0 | 0 | 0 |
| Repository repair, naming, and local evidence validators | 3 | 0 | 0 | 0 |
| Ruff | 1 | 0 | 0 | 0 |
| Mypy | 1 | 0 | 0 | 0 |

The full relevant pytest run includes cache identity, protected snapshot,
inventory coverage, property evidence, projection proof, certificate
consistency, fault-injection, and transformation roundtrip tests. The only
pytest warning is the existing scikit-learn warning for a deliberately small
split-selection fixture.

## Cache and offline verification

* First offline materialization run: invalid old-format cache was quarantined and the transformation was validated/published.
* Second offline materialization run: `validated_cache_hit`.
* Network attempts during both runs: `0`; `--offline` was used.
* No model checkpoint or dataset acquisition was performed.
* Logical primary/repeat inventory comparison: `PASS`.

## Generated evidence retained locally

| Artifact | SHA-256 | Retention |
| --- | --- | --- |
| `artifacts/handoff/transformation_inventory.json` | `c0570d0d913656237420194429676decf52e1caf6935be1c596035e70cab7477` | tracked handoff |
| `artifacts/handoff/transformation_property_evidence.json` | `09b88e81df16f2babec53fabf9b150029f7a6378a4048187a95820d932ea403d` | tracked handoff |
| `artifacts/transformation_engine/review/repair_hashes_after.json` | `e24ab902bc01065661a1996452873327c910645587b18e82cc6b6fb42edbe5a1` | ignored local evidence |
| `artifacts/transformation_engine/review/repair_hash_comparison.json` | `f0660ac0c8d9cd90934a9d7ce1ee8e375b29e339f7769f5916765fb680172c1a` | ignored local evidence |
| `artifacts/transformation_engine/review/followup_inventory_logical_comparison.json` | `bf5423d819add00a70631145dde127a70c033c6c4e4f7c5cf204ff8ed05ac56d` | ignored local evidence |
| `schemas/transformation_cache_manifest.schema.json` | `f5be6324eb2b3dbadc2474cf05c08eca7748b747084eddd42800b431a464b921` | tracked schema |

## Files created

* `artifacts/handoff/transformation_integrity_followup.md`
* `schemas/transformation_cache_manifest.schema.json`
* `tests/unit/test_transformation_cache_identity.py`
* `tests/unit/test_transformation_certificate_integrity.py`
* `tests/unit/test_transformation_inventory_contract.py`
* `tests/unit/test_transformation_inventory_integrity.py`
* `tests/unit/test_transformation_projection_proofs.py`

## Files modified

* `artifacts/handoff/transformation_inventory.json`
* `artifacts/handoff/transformation_property_evidence.json`
* `schemas/transformation_certificate.schema.json`
* `scripts/materialize_transformation.py`
* `scripts/run_transformation_properties.py`
* `scripts/validate_transformation_engine.py`
* `src/schemaguard/artifact_contracts.py`
* `src/schemaguard/transformations/caching.py`
* `src/schemaguard/transformations/certificates.py`
* `src/schemaguard/transformations/contracts.py`
* `src/schemaguard/transformations/inventory.py`
* `tests/integration/test_transformation_roundtrip.py`
* `tests/unit/test_artifact_schemas.py`
* `tests/unit/test_categorical_transformations.py`
* `tests/unit/test_composite_transformation.py`
* `tests/unit/test_identity_transformation.py`
* `tests/unit/test_numeric_transformations.py`
* `tests/unit/test_structural_transformations.py`
* `tests/unit/test_transformation_properties.py`

## Commands executed

```text
cd D:\DR2\SchemaGuard
git rev-parse HEAD
git branch --show-current
git status --short
git log -1 --oneline
conda run -n P12 python scripts/generate_artifact_schemas.py
conda run -n P12 python scripts/run_transformation_properties.py --examples 1000
conda run -n P12 python scripts/validate_transformation_engine.py --all --offline --inventory-output artifacts/handoff/transformation_inventory.json
conda run -n P12 python scripts/validate_transformation_engine.py --all --offline --inventory-output artifacts/transformation_engine/review/transformation_inventory_repeat.json
conda run -n P12 python scripts/validate_split_generation.py --all --offline
conda run -n P12 python -m ruff check .
conda run -n P12 python -m mypy src/schemaguard
conda run -n P12 python -m pytest -q tests/unit tests/integration/test_transformation_roundtrip.py -rA
conda run -n P12 python -m pytest -q -m "not network and not gpu and not foundation_model" -rA
conda run -n P12 python -m pytest -q -m "integration and not network and not gpu and not foundation_model" -rA
conda run -n P12 python scripts/validate_repository_repair.py
conda run -n P12 python scripts/validate_repository_naming.py
conda run -n P12 python scripts/validate_local_evidence.py
conda run -n P12 python scripts/materialize_transformation.py --dataset-id 1464 --seed 1729 --view numeric_affine_units --offline
conda run -n P12 python scripts/materialize_transformation.py --dataset-id 1464 --seed 1729 --view numeric_affine_units --offline
D:\Temp\SchemaGuardIntegrityCleanClone\ (fresh clone of the implementation commit)
conda run -n P12 python scripts/validate_repository_repair.py (run from the fresh clone)
git diff --check
```

Complete command output is retained under
`artifacts/transformation_engine/review/`, including the primary/repeat
inventory runs, property run, split validation, pytest runs, Ruff, mypy,
repository validators, schema generation, and both cache runs.

## Deviations

`None` from the final validated result. An intermediate inventory attempt
correctly failed because the snapshot scope omitted two existing handoff
files; the snapshot implementation was repaired, and both final inventory
runs passed with zero unexpected changes.

## Failures

`None` in the final evidence.

## Review and delivery state

* Implementation commit: `0b163c5f7d46848980a2144f56e6a6ca7f135437`.
* Fresh-clone repository validation at that commit: `PASS`.
* GitHub delivery: pending independent review and final push.
* Required review status remains `PASS_PENDING_REVIEW`; this handoff does not
  self-approve the work.

## Next permitted stage

Independent review may verify this handoff. No model-adapter implementation,
benchmarking, SCNF, COSA, pilot, or main experiment work is authorized by this
follow-up until the review accepts the evidence.
