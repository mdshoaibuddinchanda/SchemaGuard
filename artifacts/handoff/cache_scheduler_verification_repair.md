# Cache and Scheduler Verification Repair

## Status

`PASS_PENDING_REVIEW`

`completed_stage`: Cache and Scheduler verification repair

`next_stage`: Independent review only; the smoke test has not been started

Implementation and local evidence gates pass. This is not independent approval. Do not start
the smoke test or a later research workstream until an independent reviewer accepts the evidence.

## Starting state and scope

- Starting branch: `main`
- Starting local and remote commit: `bec25bf6865acdde66f816aa5c103427657cc5af`
- Implementation commit: `a8c99f421af03d5662fe14880da87c4373004301`
- Refreshed schema/evidence commit: `8af18158f268c2df5fb3b5762d154f35ff6312dc`
- Runtime: existing Conda environment `P12`, Python 3.12, Windows
- The only pre-existing untracked root item was `SchemaGuard_Complete_Research_and_Engineering_Plan.docx`. It was not opened, read, hashed, copied, moved, edited, staged, or deleted; it remains untracked.
- No smoke experiment or later research workstream was started.

## Defects repaired

1. The transformation validator's protected-file allowlist rejected legitimate scheduler schema
   additions. It now permits only 22 exact scheduler schema paths; wildcard, directory, absolute,
   traversal, omitted-path, and unknown-schema allowances are rejected. Transformation source
   files and protected transformation schemas are pinned to the accepted transformation evidence
   commit `5a8d59b53baaee8de26314c3803f8260dab3df16`, with ancestry validation. Tests cover the
   permitted additions and prove unrelated transformation edits remain blocked. CI now runs these
   structural-policy tests.
2. Fault-case expected behavior, injection points, and identity rules had duplicate or insufficiently
   bound representations. `src/schemaguard/cache/fault_policy.py` is now the single immutable
   30-case policy used by fault generation, record contracts, and evidence validation. Every
   record binds its expected and observed outcome fields in its canonical hash; validation
   independently recomputes status and checks the canonical case order, offline-denial count,
   source commit, and evidence digest. Mutation tests cover each outcome field, contradictory
   rehashed status, malformed matrices, network-attempt counts, and commit/digest mismatches.

## Validation evidence

- Ordinary unit tests: 414 passed, 0 failed, 0 skipped.
- Scheduler integration tests: 9 passed, 0 failed.
- Fault-injection matrix: 30/30 passed. The one intentional offline network attempt was denied;
  no credentials or network access were used.
- Ruff: passed.
- Mypy: passed, 96 source files, no issues.
- Schema generation/freshness: passed; generated schemas match current contracts.
- Cache/scheduler evidence validator: `PASS_PENDING_REVIEW`; 4 cold probes executed, 4 validated
  results reused on resume, 0 failed probes, 0 duplicates, 0 offline network attempts outside
  the intentional denied fault case.
- Model-adapter evidence validator: passed, 37 records and 37 leakage proofs. Its evidence tests
  passed 2/2.
- Split-generation validator: 70/70 passed; no predictor group crossed partitions; all 142
  before/after artifact records matched.
- Protected tracked-file snapshot: 281 files, 0 changed and 0 missing. Split generation also
  independently verified 142 split artifact records unchanged.
- Repaired transformation validator: exit code 0, 770 records (545 `PASS`, 225 controlled `N/A`,
  0 failures or missing results). All 11,000 property examples and all 1,635 certificate IDs
  matched the accepted inventory. Protected comparison passed with no unexpected additions,
  changes, or removals.
- Resource observations for the synthetic scheduler probes: maximum CPU concurrency 2, mocked
  GPU queue concurrency 1, maximum process-tree RSS 99.734 MiB. VRAM was not measured because
  these probes do not run CUDA inference.

The pre-repair transformation command had returned `FAIL` because the old allowlist rejected the
scheduler schemas. That baseline is retained in the ignored review log; the repaired command and
its result are recorded below. It did not mutate transformation outputs or Phase 01 data.

## Key commands

All commands use the existing `P12` environment. Relevant successful commands:

```text
conda run -n P12 python -m pytest -q tests/unit -rA
conda run -n P12 python -m pytest -q tests/integration/test_scheduler_resume.py tests/integration/test_scheduler_concurrency.py tests/integration/test_scheduler_process_isolation.py -m "integration and not network and not gpu and not foundation_model" -rA
conda run -n P12 python -m ruff check .
conda run -n P12 python -m mypy src/schemaguard
conda run -n P12 python scripts/generate_artifact_schemas.py
conda run -n P12 python scripts/validate_cache_scheduler.py --config configs/runtime/cache_scheduler.yaml --inventory artifacts/handoff/cache_scheduler_inventory.json --fault-evidence artifacts/handoff/cache_scheduler_fault_evidence.json
conda run -n P12 python scripts/validate_transformation_engine.py --all --offline --inventory-output artifacts/transformation_engine/review/cache_scheduler_verification_repeat.json
conda run -n P12 python scripts/validate_split_generation.py --config configs/runtime/split_generation.yaml --all --offline
conda run -n P12 python scripts/validate_model_adapter_evidence.py
git diff --check
```

Detailed local command output is retained in ignored files under
`artifacts/cache_scheduler/runtime/` and `artifacts/transformation_engine/review/`. The
verification-repair unit log is `artifacts/cache_scheduler/runtime/verification_repair_unit_pytest.txt`;
the repaired transformation run is `artifacts/transformation_engine/review/cache_scheduler_transformation_after.txt`.

## Tracked evidence artifacts

Sizes and SHA-256 values are from the reviewed local files at handoff authoring time.

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| `schemas/cache_scheduler_fault_evidence.schema.json` | 3,027 | `fb13db63a23684e55e34d87f9c703e688c6a0d66f9fad10dc81cb49998ed59c2` |
| `schemas/cache_scheduler_fault_record.schema.json` | 1,606 | `6a7a2f8779073c736a9ebd24b331a7ddfc87af81014ced6680e6dea51b79eecd` |
| `artifacts/handoff/cache_scheduler_inventory.json` | 52,927 | `0b697c8f484b73c09413eaa7f266c5900a531ed3a896c3135abc5f6d5517b42f` |
| `artifacts/handoff/cache_scheduler_fault_evidence.json` | 17,814 | `b7f25cc9b6a12f7fadeab4742f3f58411a70b242490e336df28d16ab33049dd2` |
| `artifacts/handoff/cache_scheduler_probe_evidence.json` | 2,951 | `7908fde03dc6618ad37ee7ddece7acc256b82c7f9988465bdb34b09471aef743` |
| `artifacts/handoff/cache_scheduler_protected_hash_comparison.json` | 50,728 | `e3a1c059eeeebf0f5c0bcc298cd07e6c35a6c2ff29a95306d39fb08810aa6192` |

The evidence contains no dataset rows, predictions, secrets, credentials, checkpoint contents,
or private-document contents. Runtime logs, caches, and generated data remain ignored.

## Files changed for this repair

- Modified: `.github/workflows/quality.yml`, `scripts/validate_cache_scheduler.py`,
  `scripts/validate_transformation_engine.py`, `src/schemaguard/cache/contracts.py`,
  `src/schemaguard/runner/faults.py`, `src/schemaguard/runner/plan.py`,
  `tests/unit/test_cache_scheduler_evidence.py`, `tests/unit/test_scheduler_faults.py`.
- Added: `src/schemaguard/cache/fault_policy.py`,
  `tests/unit/test_transformation_validator_policy.py`,
  `artifacts/handoff/cache_scheduler_verification_repair.md`.
- Refreshed: the two schema files and four evidence JSON files listed above.
- No dependencies, model/dataset names, scientific parameters, or frozen configurations were
  changed.

## Delivery and review boundary

The implementation and evidence commits are listed above. Final handoff commit, push result,
clean-clone result, and the GitHub Actions run for the exact pushed SHA are to be appended to the
delivery record only after those checks complete. Until then, remote delivery/CI is not verified.
Only independent review may change this status to `VERIFIED_PASS`. The smoke test remains
prohibited until that approval.
