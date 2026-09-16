# Transformation Evidence Finalization

## Status

`PASS_PENDING_REVIEW` — finalization checks pass locally. Independent review is still required; this is not `VERIFIED_PASS`.

## Commit sequence

- Required starting commit: `1e405f650878566f087fa5720f3ecda57aa47ea4` (`main`).
- Implementation commit used to generate final property evidence: `3e41b1a412ac7239334ca01e849b5c173a061039`.
- Final evidence commit: `5a8d59b53baaee8de26314c3803f8260dab3df16`.
- Final handoff commit: recorded in Git history and in the delivery message; a commit cannot include its own hash without changing that hash.
- Evidence and validation used the existing Conda `P12` environment with Python 3.12.

The intermediate evidence refresh at `c8e0b1c` was superseded after a post-commit repeat exposed a self-reference in the protected snapshot digest. Commit `3e41b1a` corrected the digest to hash the stable protected files while separately recording explicitly allowed path changes. The evidence was regenerated and committed afterward.

## Repairs and identity rules

- `source_file_hashes()` is the shared source-hash implementation. It canonicalizes CRLF to LF and returns the canonical LF and equivalent CRLF SHA-256 values. The inventory contract and `_load_property_evidence()` use this shared rule; semantic source changes remain rejected.
- Logical certificate identity recursively excludes only `created_at` and `source_commit`. Certificates retain both provenance fields, and raw certificate file hashes continue to preserve them. All scientific and implementation-relevant fields remain identity-bound, including configuration and implementation hashes, artifacts, targets, proof parameters, validation evidence, view, dataset, seed, partition, and certificate type.
- Property evidence binds both the property-runner source hash and a stable implementation hash over the transformation engine source tree and shared hashing helper. The strict Pydantic contract and generated schema require this binding.
- The protected snapshot digest excludes only the explicitly allowed changed/added/removed paths; the comparison still reports those path lists and fails on any other protected mutation.

## Evidence results

The final property evidence was generated at implementation commit `3e41b1a` before the evidence commit:

- 11 views; 1,000 examples per view; **11,000 passed, 0 failed**.
- Property runner SHA-256: `817cf14c86b3bbb33bcfe8ab9222a0da6a275cf9c975731353b1ecafe2949132`.
- Transformation implementation SHA-256: `f7473c6caec0b2600e50fb929b3bb18edce4583863caaa26ea8d3810c8b6a1ee`.
- Tracked property evidence SHA-256: `aff8f60bc804767f02547b4ac9e601a1bf2bf305a758772cd6e64d00ca808a4d`.
- Tracked inventory SHA-256: `6984ecef7342a0b7c6190f4636ccc69684e90a7a0343e88fc686362dd9ada104`.

The complete inventory was rerun after evidence commit `5a8d59b` to the ignored review path `artifacts/transformation_engine/review/transformation_inventory_finalization_repeat.json`. Result: `PASS_PENDING_REVIEW`, 770 records, 545 PASS, 225 controlled N/A, zero failures, zero missing records, and 11,000 property examples.

After excluding only volatile timestamps/runtime measurements and `source_commit` provenance, tracked and repeated inventories match exactly:

- 770/770 dataset-seed-view tuple identities;
- all 1,635 logical certificate identities;
- applicability, statuses/reasons, artifact hashes, reconstruction errors, and deterministic flags;
- all property-evidence identities;
- protected snapshot comparison, split inventory hash, and registry hash.

The final protected snapshot verdict is `PASS` with stable after-snapshot hash `7343dfee0a1f6373d226d0edfa418c6c6ee640798db38f1853de4d416919114e`. The split inventory hash is `368acbd6ecaffafbecf6b5bdbce301211da4902ceaea05fbfb0db933a3611482`; the registry hash is `91722aceba52724dc750377e9229e3411c7508c6c2482fc8ab89bd696cc02afa`.

## Portability and validation

- Clean LF checkout at the evidence commit: all 233 unit tests passed; tracked inventory/evidence contracts passed.
- Clean CRLF checkout: the 12 focused evidence/hash tests passed. `_load_property_evidence()` accepted all 11 tracked records against CRLF working-tree bytes, with the same runner and engine hashes above.
- Existing unit tests also prove line-ending-only changes pass and semantic source edits fail.
- Ruff: `conda run -n P12 python -m ruff check .` — passed.
- Mypy: `conda run -n P12 python -m mypy src/schemaguard` — passed for 65 source files.
- `conda run -n P12 python -m pytest -q tests/unit tests/integration/test_transformation_roundtrip.py -rA` — 234 passed.
- `conda run -n P12 python -m pytest -q -m "not network and not gpu and not foundation_model" -rA` — 251 passed.
- `conda run -n P12 python -m pytest -q -m "integration and not network and not gpu and not foundation_model" -rA` — 7 passed.
- Repository repair validator with local evidence: all 11 checks passed. Repository naming and local evidence validators passed.
- Split validation: 70/70 dataset-seed records passed; zero predictor groups crossed partitions.
- `git diff --check` passed.

Phase 01 preservation was rechecked across 11 raw, processed, and split files: all SHA-256 values match the baseline. The blood-transfusion split remains 748 rows (449 train, 150 calibration, 149 test), 502 predictor groups, 69 duplicated groups, 31 conflicting-target groups, and zero groups crossing partitions.

## Scope and exact repository files

Changed source, schema, evidence, and test files:

- `artifacts/handoff/transformation_inventory.json`
- `artifacts/handoff/transformation_property_evidence.json`
- `schemas/transformation_inventory.schema.json`
- `schemas/transformation_property_evidence.schema.json`
- `scripts/run_transformation_properties.py`
- `src/schemaguard/transformations/caching.py`
- `src/schemaguard/transformations/certificates.py`
- `src/schemaguard/transformations/contracts.py`
- `src/schemaguard/transformations/implementation.py`
- `src/schemaguard/transformations/inventory.py`
- `src/schemaguard/utils/hashing.py`
- `tests/unit/test_transformation_certificate_identity.py`
- `tests/unit/test_transformation_implementation_identity.py`
- `tests/unit/test_transformation_inventory_contract.py`
- `tests/unit/test_transformation_inventory_integrity.py`
- `artifacts/handoff/transformation_evidence_finalization.md`

No transformation mathematics, datasets, splits, model adapters, models, SCNF, COSA, benchmarks, or experiments were changed or started. No dataset was downloaded or regenerated. The private root `.docx` was not opened, modified, staged, deleted, or committed; it remains untracked.

## GitHub Actions

Finalization commits are to be pushed only after the local gates above pass. The finalization run URL and job conclusions will be recorded here after GitHub Actions completes. The starting-commit Actions run is not treated as evidence for these changes.
