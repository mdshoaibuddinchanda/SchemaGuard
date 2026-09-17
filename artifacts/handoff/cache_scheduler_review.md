# Cache and Scheduler Review

## Status

`PASS_PENDING_REVIEW`

`completed_stage`: Cache and Scheduler workstream

`next_stage`: Ten-condition smoke test, permitted only after independent acceptance

The implementation and local evidence gates passed. This is not an independent approval, and no
smoke test or later research work was started.

## Starting state and provenance

- Required starting branch: `main`
- Required starting commit: `cc8e2a53ceb73e833d54917dd195eb99bf01719c`
- Source implementation commit: `6e1f90c0fc598dac70c1479ccae1f15aaed764c5`
- CI dependency correction commit: `428ff41fac0f7a1cf9c11f9c50b352c334a4e07a`
- Corrected evidence bundle commit: `f60e8a6895761e293f451f93591c2699e9774106`
- Superseded evidence draft: `8189320f788c1ee20ff0e04c8f7bcb202134d275`; the corrected evidence below
  is regenerated against the source implementation commit above.
- Runtime: existing Conda environment `P12`, Python 3.12, Windows
- The only pre-existing untracked root item was `SchemaGuard_Complete_Research_and_Engineering_Plan.docx`.
  It was not opened, read, hashed, copied, moved, edited, staged, or deleted; it remains untracked.
- No dependencies were added and no frozen scientific configuration was changed.

## Scope and architecture

Implemented content-addressed identity and artifact validation under `src/schemaguard/cache/`,
plus run planning, worker isolation, scheduling, state, resource enforcement, and fault injection
under `src/schemaguard/runner/`. The synthetic probe and evidence validator are
`scripts/run_cache_scheduler_probe.py` and `scripts/validate_cache_scheduler.py`.

The canonical cache identity binds dataset, grouped split, view certificate, model, parameters,
checkpoint, dependency lock, relevant source implementation, seed, device policy, artifact kind,
and schema version. Canonical JSON is hashed; missing/placeholder hashes, non-portable paths,
non-finite values, and unknown fields are rejected.

Publication holds a per-identity cross-process lock, writes to a same-filesystem transaction
directory, flushes payload and metadata, validates hashes/schema/identity, exposes the completion
marker last, and atomically publishes. Existing entries are immutable; a same-key byte conflict
fails closed. SQLite is a rebuildable derivative index; manifests remain authoritative and
read-only lookup does not create or mutate an index.

Plans are order-stable, state transitions are validated and persisted atomically, and resume skips
only fully validated cache entries. Failed work requires explicit retry; expired running leases
are recovered explicitly. CPU execution uses Windows spawn, two bounded workers, and two-thread
limits. GPU execution uses the accepted adapter GPU lock/resource policy and one exclusive queue;
the scheduler GPU test uses synthetic work and mocked telemetry only.

## Measured probes and resource use

The offline cold probe planned 4 tasks, executed 4, had 0 cache hits and 0 failures, and published
4 validated artifacts. The resume probe planned 4, executed 0, reused 4 validated artifacts, and
had 0 failures. No duplicate artifact was found. Four older, still-valid artifacts from the prior
implementation identity were retained separately; the cache contains 8 unique entries total and
uses 16,284 bytes. Ordinary cold/resume probes and the mocked GPU
policy probe recorded 0 network attempts. The fault suite intentionally attempted one offline
network request and verified that it was denied.

- Maximum observed CPU concurrency: 2
- Maximum observed GPU queue concurrency: 1 (mocked scheduler-policy probe; no CUDA inference)
- Peak worker/process-tree RSS: 91.6 MiB
- Aggregate cold worker time: 4.331 seconds; individual worker wall times were about 1.03–1.15 seconds
- Resume executed no workers; all four results were validated cache hits
- Peak VRAM: not measured; CUDA was not used in this synthetic scheduler workstream
- Four cache payloads occupied 8,142 bytes; dependency installation/storage change: none

## Validation results

- Unit tests: 379 passed, 0 failed
- Scheduler integration tests: 9 passed, 0 failed
- Fault injection: 30/30 passed, with explicit observed outcomes and lock-release checks
- Ruff: passed
- Mypy: passed (`95` source files)
- Schema generation and freshness: passed; all generated schemas matched the registered Pydantic contracts
- Repository naming, repository structure/evidence, and Model Adapter evidence validators: passed
- Cache artifact validation and derivative index validation: passed
- Private document status check: passed; only Git metadata was inspected
- Protected-file comparison: 93/93 unchanged, status `PASS`
- Existing raw/processed/split artifacts: 281 files rehashed against the last validated
  transformation repair snapshot, 0 mismatches
- Sanitized evidence validator: passed with status `PASS_PENDING_REVIEW`
- LF/CRLF configuration and dependency-lock identity tests: passed; clean-clone portability repair
  prevents Windows checkout line endings from invalidating evidence hashes
- Clean-clone validation at the corrected evidence bundle commit: passed, including 379 unit tests,
  9 integration tests, Ruff, mypy, schema regeneration without a diff, evidence validation,
  repository/naming validation, and Model Adapter evidence validation

The 93-record comparison binds protected local SHA-256/size records to the required baseline and
implementation Git trees. A separate read-only rehash compared all 281 existing raw, processed, and
split files in `repair_hashes_after.json`; all matched. Existing Phase 01 data/split/transformation/
model-adapter evidence remained untouched.
The already-validated transformation inventory remains 770 records (545 PASS, 225 controlled N/A,
0 failures), with 11,000/11,000 property examples and 1,635 certificate identities; the 70 grouped
split records retain zero cross-partition predictor groups; adapter evidence remains 37/37, including
35 CPU, 2 CUDA-policy cases, and 37 leakage checks. A post-change read-only representative
transformation check returned 11/11 PASS-or-N/A records, with 0 failures.

## Fault injection

All 30 required cases passed. Evidence records the fault name, injection point, expected and observed
category, artifact acceptance, lock release, resume behavior, per-case evidence hash, and status in
`cache_scheduler_fault_evidence.json`. Cases include interrupted writes, truncated/corrupt
manifests and payloads, identity changes, same-key conflicts, competing processes, lock-holder
crashes/stale files, resource limits, timeout/crash, restart recovery, index loss/corruption,
invalid completion state, failed-artifact candidates, and denied offline network access.

## Sanitized evidence and schemas

- `cache_scheduler_inventory.json` — source-bound gate statuses, counters, resource summaries,
  hashes, and protected comparison identity
- `cache_scheduler_probe_evidence.json` — sanitized cold/resume observations
- `cache_scheduler_fault_evidence.json` — 30 fault outcomes
- `cache_scheduler_protected_hash_comparison.json` — 93 protected file comparisons; no absolute paths
- 22 generated strict-contract schemas were added under `schemas/`

The generated evidence contains no dataset rows, predictions, credentials, checkpoint contents,
machine usernames, private paths, or private-document details. Runtime caches, receipts, logs,
temporary files, and SQLite state remain ignored.

## Files added or changed

Implementation and tests are in the source implementation commit:

- Modified: `.github/workflows/quality.yml`, `.gitignore`, `README.md`,
  `artifacts/handoff/README.md`, `src/schemaguard/artifact_contracts.py`,
  `tests/unit/test_artifact_schemas.py`, `tests/unit/test_gitignore_policy.py`
- Added: `configs/runtime/cache_scheduler.yaml`
- Added: `scripts/run_cache_scheduler_probe.py`, `scripts/validate_cache_scheduler.py`
- Added: `src/schemaguard/cache/` and `src/schemaguard/runner/`
- Added: `tests/unit/cache_scheduler_helpers.py`, cache/scheduler unit tests, and the three
  `tests/integration/test_scheduler_*.py` suites
- Added: 22 generated schemas and the four sanitized JSON evidence files listed above
- Added: `artifacts/handoff/cache_scheduler_review.md`

## Commands executed

All commands used the existing `P12` environment. The full unit/integration, lint/type/schema,
naming, repository, adapter-evidence, fault, probe, and protected-file checks were also rerun by
the evidence recorder. Key commands:

```text
conda run -n P12 python -m ruff check .
conda run -n P12 python -m mypy src/schemaguard
conda run -n P12 python -m pytest -q tests/unit -m "not network and not gpu and not foundation_model and not evidence" -rA
conda run -n P12 python -m pytest -q tests/integration/test_scheduler_resume.py tests/integration/test_scheduler_concurrency.py tests/integration/test_scheduler_process_isolation.py -m "integration and not network and not gpu and not foundation_model" -rA
conda run -n P12 python scripts/run_cache_scheduler_probe.py --config configs/runtime/cache_scheduler.yaml --offline
conda run -n P12 python scripts/run_cache_scheduler_probe.py --config configs/runtime/cache_scheduler.yaml --offline --resume
conda run -n P12 python scripts/validate_cache_scheduler.py --config configs/runtime/cache_scheduler.yaml --inventory artifacts/handoff/cache_scheduler_inventory.json --fault-evidence artifacts/handoff/cache_scheduler_fault_evidence.json --record
conda run -n P12 python scripts/validate_cache_scheduler.py --config configs/runtime/cache_scheduler.yaml --inventory artifacts/handoff/cache_scheduler_inventory.json --fault-evidence artifacts/handoff/cache_scheduler_fault_evidence.json
conda run -n P12 python scripts/generate_artifact_schemas.py
conda run -n P12 python scripts/validate_repository_naming.py
conda run -n P12 python scripts/validate_repository_repair.py
conda run -n P12 python scripts/validate_model_adapter_evidence.py
git diff --check
```

## Deviations and failures

- No implementation or required cache/scheduler gate failed.
- An initial clean clone of the superseded evidence draft exposed raw line-ending-sensitive hashes
  for YAML and lock files. The current implementation normalizes those identities, hashes text
  evidence consistently, and the new LF/CRLF tests pass. The superseded draft remains in Git history
  for provenance, not as current evidence; the corrected evidence set is in this handoff.
- The legacy transformation-engine subset CLI reports a nonzero protected-snapshot result when
  it sees the newly added scheduler schemas, because its historical allowlist does not include
  this workstream's schema additions. Its 11 computed representative view records were all PASS
  or controlled N/A; the same helper was run read-only with exit code 0. No transformation output,
  inventory, split, or dataset was written or modified. The dedicated 93-file protected comparison
  and all required protected-workstream evidence checks passed.
- GitHub Actions run `35226750306` passed for commit `c1eeec4aa0af0237a5b1ea403f9a2c5a2ea100c9`:
  [view the successful workflow](https://github.com/mdshoaibuddinchanda/SchemaGuard/actions/runs/35226750306).
  This handoff update records that verified result; the documentation-only commit carrying this
  update will receive its own exact-SHA CI check, reported in the delivery response.

## Review boundary

Only an independent reviewer may change the status to `VERIFIED_PASS`. The ten-condition smoke
test is the next permitted workstream only after that acceptance. Benchmarking, SCNF, COSA, pilot,
main experiment, statistical analysis, figures, and paper results were not started.
