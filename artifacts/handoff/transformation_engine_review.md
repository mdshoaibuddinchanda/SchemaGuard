# Transformation Engine Review

## Status

`PASS_PENDING_REVIEW`

The certified lossless transformation engine is implemented and locally verified. This status intentionally remains pending independent review; it is not a claim of scientific experiment completion.

## Starting state

* Starting commit: `49550b10fd1d8e652aae6f41094bb5aa20ecc313`
* Branch: `main`
* Implementation commit used for validation: `070564cc95038914d2027db1e6c17d1962f1f283`
* Python executable: `D:\Conda\P12\python.exe`
* Python version: `3.12`
* Conda environment: `P12`
* Initial worktree state: clean tracked worktree with one known untracked root reference document.
* The root `SchemaGuard_Complete_Research_and_Engineering_Plan.docx` was not opened, edited, moved, deleted, staged, or committed.

## Scope and naming

This workstream implements only the lossless transformation engine. It does not implement model adapters, benchmarking, SCNF, COSA, the pilot, or the main experiment. New paths use descriptive responsibility-based names; no new phase-numbered or execution-numbered path was introduced.

## Repository hygiene

`.gitignore` now ignores generated transformation materialization under `data/transformed/` and generated review material under `artifacts/transformation_engine/`. Existing generated data and result rules remain in force. No broad `schemas/`, `*.csv`, or `*.tsv` ignore rule was introduced.

The following `git check-ignore` probes were verified:

| Path | Result |
| --- | --- |
| `schemas/example.schema.json` | trackable |
| `tests/fixtures/example.csv` | trackable |
| `data/raw/example.csv` | ignored by `data/raw/` |
| `data/processed/example.parquet` | ignored by `data/processed/` |
| `results/example.csv` | ignored by `results/` |
| `artifacts/handoff/example.md` | trackable |
| `data/transformed/openml/example.parquet` | ignored by `data/transformed/` |

The protected root `.docx` remains untracked and untouched.

## Configuration and registry

Configuration: `configs/runtime/transformation_engine.yaml`

* Engine: `certified_lossless_transformations`
* Split strategy: `stratified_group_5fold_v1`
* Numeric column limit: `3`
* Category minimum: `2`
* Quotient modulus: `10`
* Numerical tolerance: `rtol=1.0e-10`, `atol=1.0e-12`
* Registry hash: `362fbd5d39f6a6e18703c15370e13e725ba96e71f752d127fe6244b05e150f51`
* CPU workers: `2`
* GPU enabled: `false`

| View ID | Meaningful name | Certificate type |
| --- | --- | --- |
| V00 | `identity` | CONTROL |
| V01 | `numeric_affine_units` | BIJECTION |
| V02 | `numeric_asinh` | BIJECTION |
| V03 | `category_permutation` | BIJECTION |
| V04 | `categorical_onehot` | BIJECTION |
| V05 | `duplicate_feature` | PROJECTION |
| V06 | `redundant_affine_feature` | PROJECTION |
| V07 | `integer_quotient_remainder` | BIJECTION |
| V08 | `column_permutation_control` | CONTROL |
| V09 | `row_permutation_control` | CONTROL |
| V10 | `composite_migration` | COMPOSITION |

## Inventory results

The complete local matrix covers 14 frozen datasets, 5 seeds, and 11 views: `770` records.

| View | PASS | NOT_APPLICABLE |
| --- | ---: | ---: |
| V00 | 70 | 0 |
| V01 | 55 | 15 |
| V02 | 55 | 15 |
| V03 | 35 | 35 |
| V04 | 35 | 35 |
| V05 | 70 | 0 |
| V06 | 55 | 15 |
| V07 | 10 | 60 |
| V08 | 70 | 0 |
| V09 | 70 | 0 |
| V10 | 20 | 50 |
| **Total** | **545** | **225** |

* Expected records: `770`
* Actual records: `770`
* Applicable: `545`
* Passed: `545`
* Not applicable: `225`, each with an explicit reason code
* Failed: `0`
* Missing: `0`
* Property examples: `1000`
* Maximum reconstruction absolute error: `1.1641532182693481e-10`
* Network access: disabled for validation

The inventory records source/output logical hashes, schema hashes, row identity hashes, configuration and implementation hashes, certificates, applicability reasons, validation status, and source commit. The tracked sanitized inventory is `artifacts/handoff/transformation_inventory.json`.

## Determinism and certificate identity

The complete inventory was executed twice at the same implementation commit. Both runs returned `PASS_PENDING_REVIEW`, `770` records, `545` passes, `225` explicit not-applicable records, and `0` failures. All `1,635` certificate IDs were identical in order. Certificate identity removes timestamps recursively, including nested V10 component certificates; timestamps remain present in the evidence records themselves.

The repeat inventory was retained locally at `artifacts/transformation_engine/review/transformation_inventory_repeat.json` and is ignored. The primary tracked inventory hash is:

`4d9745443ca212a618fe413ca2369ee9b6341fa8f43c4d2b16d998112d048def`

## Preservation of earlier artifacts

The pre-implementation protected snapshot contained `290` files and had snapshot hash:

`b94ca214fb837eaa67deaaf610a24c4fcda33f143d0b189c09d9f1e68f97cd9b`

After implementation, the protected scope contained `294` files. The four additional files are the newly introduced transformation contract schemas:

* `schemas/transformation_certificate.schema.json`
* `schemas/transformation_inventory.schema.json`
* `schemas/transformation_manifest.schema.json`
* `schemas/transformation_validation.schema.json`

Protected pre-existing files had `changed_files=[]`; comparison status was `PASS`. No Phase 01 raw, processed, split, baseline, or existing schema artifact was changed.

The preserved split inventory independently validated all `70` dataset/seed assignments offline with status `PASS`, and every record reported `cross_partition_group_count=0`. The earlier grouped-split foundation remains unchanged.

## Tests and quality

| Check | Result |
| --- | --- |
| Ruff | PASS: all checks passed |
| Mypy | PASS: no issues in 63 source files |
| Unit suite | PASS: all selected non-network/non-GPU/non-foundation/non-evidence tests |
| Transformation integration suite | PASS: 2 tests |
| Artifact schema generation and validation | PASS |
| Repository naming validation | PASS |
| Repository repair validation | PASS |
| Local evidence validation | PASS |
| Existing split validation | PASS: 70/70 offline |
| Transformation inventory | PASS_PENDING_REVIEW: 770/770 records, 0 failures |

The unit suite includes strict-contract, registry, applicability, fixture, numeric, categorical, structural, composite, cache, fault-injection, and certificate tests. Fault tests cover truncated manifests, invalid cache hashes, interrupted writes, stale/active locks, offline cache misses, unsupported configuration, and invalid results.

## Generated artifacts

Generated datasets, transformed materializations, caches, reports, and local review logs remain ignored. The following tracked or handoff artifacts were inspected:

| Artifact | Size | SHA-256 |
| --- | ---: | --- |
| `configs/runtime/transformation_engine.yaml` | 1121 bytes | `74bb1e9746a7f4761b970527fd7f6a42f76ec402ecb9b415f585c039fd48bd42` |
| `schemas/transformation_certificate.schema.json` | 4848 bytes | `4b9846d868fa1ec01b684295cef6f6f34c41f88f0b1ca8b97948fda5e8893be5` |
| `schemas/transformation_inventory.schema.json` | 6599 bytes | `d89de0cdffe8e52df2472bd0fc49e54d35525d334ba72e2aee35d1b1f305ff8b` |
| `schemas/transformation_manifest.schema.json` | 2262 bytes | `48610b8cb3da61ab6fd889f5c94c4ef94334927dd38237dfaf07551604b63975` |
| `schemas/transformation_validation.schema.json` | 4777 bytes | `7ad6d4ab0131ec2910e1e2a8394804cf80158db21c826c6acec94c35a63cb6cb` |
| `artifacts/handoff/transformation_inventory.json` | 636130 bytes | `4d9745443ca212a618fe413ca2369ee9b6341fa8f43c4d2b16d998112d048def` |

## Files created

* `configs/runtime/transformation_engine.yaml`
* `schemas/transformation_certificate.schema.json`
* `schemas/transformation_inventory.schema.json`
* `schemas/transformation_manifest.schema.json`
* `schemas/transformation_validation.schema.json`
* `scripts/materialize_transformation.py`
* `scripts/validate_transformation_engine.py`
* `src/schemaguard/transformations/__init__.py`
* `src/schemaguard/transformations/applicability.py`
* `src/schemaguard/transformations/base.py`
* `src/schemaguard/transformations/caching.py`
* `src/schemaguard/transformations/categorical_onehot.py`
* `src/schemaguard/transformations/category_permutation.py`
* `src/schemaguard/transformations/certificates.py`
* `src/schemaguard/transformations/column_permutation.py`
* `src/schemaguard/transformations/composite_migration.py`
* `src/schemaguard/transformations/contracts.py`
* `src/schemaguard/transformations/duplicate_feature.py`
* `src/schemaguard/transformations/identity.py`
* `src/schemaguard/transformations/inventory.py`
* `src/schemaguard/transformations/numeric_affine.py`
* `src/schemaguard/transformations/numeric_asinh.py`
* `src/schemaguard/transformations/quotient_remainder.py`
* `src/schemaguard/transformations/redundant_affine.py`
* `src/schemaguard/transformations/registry.py`
* `src/schemaguard/transformations/row_permutation.py`
* `src/schemaguard/transformations/serialization.py`
* `src/schemaguard/transformations/validation.py`
* `tests/integration/test_transformation_roundtrip.py`
* `tests/unit/test_categorical_transformations.py`
* `tests/unit/test_composite_transformation.py`
* `tests/unit/test_identity_transformation.py`
* `tests/unit/test_numeric_transformations.py`
* `tests/unit/test_structural_transformations.py`
* `tests/unit/test_transformation_cache.py`
* `tests/unit/test_transformation_certificates.py`
* `tests/unit/test_transformation_contracts.py`
* `tests/unit/test_transformation_faults.py`
* `tests/unit/test_transformation_properties.py`
* `tests/unit/test_transformation_registry.py`
* `tests/unit/transformation_helpers.py`
* `artifacts/handoff/transformation_engine_review.md`
* `artifacts/handoff/transformation_inventory.json`

## Files modified

* `.gitignore`
* `README.md`
* `src/schemaguard/artifact_contracts.py`
* `tests/unit/test_artifact_schemas.py`

## Commands executed

```text
git rev-parse HEAD
git branch --show-current
git status --short
git log -1 --oneline
conda run -n P12 python scripts/validate_repository_repair.py
conda run -n P12 python scripts/validate_local_evidence.py
conda run -n P12 python scripts/validate_split_generation.py --all --offline
conda run -n P12 python scripts/validate_repository_naming.py
conda run -n P12 python scripts/generate_artifact_schemas.py
conda run -n P12 python scripts/validate_transformation_engine.py --all --offline --inventory-output artifacts/handoff/transformation_inventory.json
conda run -n P12 python scripts/validate_transformation_engine.py --all --offline --inventory-output artifacts/transformation_engine/review/transformation_inventory_repeat.json
conda run -n P12 python -m ruff check .
conda run -n P12 python -m mypy src/schemaguard
conda run -n P12 python -m pytest -q tests/unit -m "not network and not gpu and not foundation_model and not evidence" -rA
conda run -n P12 python -m pytest -q tests/integration/test_transformation_roundtrip.py -rA
conda run -n P12 python scripts/materialize_transformation.py --dataset-id 1464 --seed 1729 --view numeric_affine_units --offline
git check-ignore -v schemas/example.schema.json tests/fixtures/example.csv data/raw/example.csv data/processed/example.parquet results/example.csv artifacts/handoff/example.md
```

## Cache and network behavior

The transformation cache uses content-addressed identities, atomic writes, validated certificates, and lock protection. Unit and fault-injection tests cover cache reuse, invalidation, partial/corrupt metadata, lock recovery, and offline cache misses. The full validation matrix ran with network access disabled. No dataset, model checkpoint, or external acquisition occurred in this workstream.

## Deviations

* No scientific model, dataset, split, metric, seed, or transformation registry identity was changed.
* The four transformation schemas are new protected-scope files and are explicitly recorded as additions in the protected comparison.
* Existing dependency declarations and `uv.lock` were not changed because this workstream has no new dependency requirement.

## Failures

`None` in the executed transformation matrix and required local checks.

## Next permitted stage

`Model adapters` may begin only after independent review accepts this handoff. The next stage must remain separate from experiments and must preserve the transformation certificates, protected Phase 01 artifacts, semantic naming policy, and the P12-only environment rule.
