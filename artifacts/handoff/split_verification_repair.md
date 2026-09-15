# Split Verification Repair Handoff

## Status

PASS

The independent audit's `REPAIR_REQUIRED` verdict was corrected for the
split-generation workstream only. Transformation code, model adapters,
training, experiments, SCNF, COSA, statistics, figures, and paper results
remain outside this workstream and were not started or modified.

## Starting state

* Starting repair commit: `d4fd7a10abcf8ae4ee4d364fb0a4e3d197eca6db`.
* Branch: `main`.
* Runtime: Conda `P12`, `D:\Conda\P12\python.exe`, Python 3.12.14.
* The only pre-existing worktree item was the untracked private root
  reference `.docx`; it was not opened, edited, moved, deleted, staged, or
  committed.
* Repair implementation commit: `0b82cdc3e33fa2c67c362ab0f6535c1fa42d0d55`.

## Defects repaired

* Persisted `fold_id` values now come from the row-level fold provenance
  mapping. The validator recomputes the fold assignment and compares every
  persisted row, the fold-to-partition mapping, and predictor-group fold
  consistency.
* Non-`PASS` manifests are rejected before any artifact is accepted.
* Protected-snapshot comparison now fails closed when local evidence exists
  without a baseline and raises on any comparison status other than `PASS`.
* Temporary and partial files, including empty transaction directories, are
  rejected across the active split tree.
* Every generated manifest records a deterministic
  `split_implementation_hash` and `cache_identity_hash`. Cache validation
  checks source hashes, configuration, grouping and implementation identity,
  schema version, assignment hash, logical hash, and fold provenance.
* The sanitized tracked inventory is generated from the authoritative strict
  contract and uses portable repository-relative paths.

## Split evidence

* Frozen configuration: 14 datasets × 5 seeds = 70 active tasks.
* Independent validation: 70/70 `PASS`.
* Cache rerun: 69 `CACHE_HIT` plus 1 protected baseline, exit code 0.
* All active manifests use implementation hash
  `f623df5c2054065695a8579f751f5c78fd38d30b5a64e823999978cdef5865fe` and
  source commit `0b82cdc3e33fa2c67c362ab0f6535c1fa42d0d55`.
* Every active record has zero crossing predictor groups.
* The corrected assignment artifact hash changed for all 69 regenerated
  non-protected tasks, while all 69 logical assignment hashes matched the
  pre-repair manifests retained in the local quarantine evidence.
* The protected OpenML 1464/seed 1729 assignment remains
  `e08b9d5c94578b317dcc844c2a6aa7a4f96ad8df699352d1e11d86a7dd452236`, with
  logical hash
  `f66b2ab092218813b82e49b6a3c19df98b06840d90b463bd9d09ba1bce385365`.

## Protected Phase 01 preservation

* Before snapshot SHA-256:
  `D23DB83257A924C1D775936E7C488F4115D9EC63C5137A371584590BAA4DDF1C`.
* After snapshot SHA-256:
  `C6CFFE6339A0CD7F611F5F6F4E972F7777258027E72B51FA9761F31652B42779`.
* Comparison SHA-256:
  `03AB0AC064A362E13E7D6A09836E2FE2A50849C0A8A87EBECF48428540C3F4`.
* Comparison status: `PASS`; 142 files before, 142 files after,
  `changed_files=[]`.
* Protected split remains 748 rows: train 449, calibration 150, test 149;
  502 predictor groups, 69 duplicate groups, 31 conflicting-target groups,
  and zero crossing groups.

## Tests and quality

* Ruff: `PASS`.
* Mypy: `PASS` on 42 source files.
* Required non-network unit suite: `PASS`.
* Split-focused unit and integration tests: `PASS` (29 focused tests and 1
  end-to-end generation/write/validate/promote/reload/cache test).
* Artifact schemas, tamper/fault tests, lock tests, and the tracked inventory
  contract: `PASS`.

## Tracked evidence

* `schemas/split_generation_inventory.schema.json` is generated from
  `SplitGenerationInventoryContract`.
* `artifacts/handoff/split_generation_inventory.json` contains 70 sanitized
  records with statuses, counts, artifact hashes, logical hashes, and
  implementation/cache hashes. It contains no row data, credentials, or
  private absolute paths.
* The earlier pre-audit review is explicitly superseded by this repair
  handoff.

## Clean clone and CI

* Clean-clone verification: `PASS` at commit
  `fc1fe8df70778811ce4f9bd5e872e1a5c286adda`. The structural validator
  passed, the required unit suite passed, the clone working tree was clean,
  and the private root `.docx` was not tracked. Local-data checks were
  explicitly `NOT_APPLICABLE_LOCAL_ARTIFACTS_ABSENT`.
* GitHub Actions verification: `PASS` for run
  `34991329468` on commit `d3fc21af6bf7c1f4e483f568045587af7ed9a85d`.
  Both `Core quality` and `Classical model validation` completed with
  conclusion `success`: https://github.com/mdshoaibuddinchanda/SchemaGuard/actions/runs/34991329468

## Commands

```text
conda run -n P12 python scripts/generate_splits.py --config configs/runtime/split_generation.yaml --offline --force-recompute
conda run -n P12 python scripts/validate_split_generation.py --config configs/runtime/split_generation.yaml --all --offline --inventory-output artifacts/handoff/split_generation_inventory.json
conda run -n P12 python scripts/generate_splits.py --config configs/runtime/split_generation.yaml --offline
conda run -n P12 python -m ruff check .
conda run -n P12 python -m mypy src/schemaguard
conda run -n P12 python -m pytest -q tests/unit -m "not network and not gpu and not foundation_model and not evidence"
conda run -n P12 python -m pytest -q tests/integration -m "integration and not network and not gpu and not foundation_model"
```

## Next permitted workstream

No later workstream is authorized by this repair. Independent review must
accept this handoff before any subsequent work begins.
