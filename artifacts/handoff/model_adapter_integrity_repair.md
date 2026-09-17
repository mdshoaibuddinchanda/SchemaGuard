# SchemaGuard Model Adapter Integrity Repair

## Status

`PASS_PENDING_REVIEW`

This repair is complete locally and is awaiting independent acceptance. It does
not claim `VERIFIED_PASS`.

## Starting state

| Field | Result |
| --- | --- |
| Branch | `main` |
| Required starting commit | `e5f233fb3db64f23999463a6da1c85a044d42ecd` |
| Remote starting commit | Same as local starting commit |
| Required prior Actions run | `35167626660`, conclusion `success` |
| Python | `3.12.14` |
| Executable | P12 environment interpreter (`conda run -n P12 python`) |
| Conda environment | `P12` |
| Scope | Adapter-integrity repair only |

The private root `.docx` was not opened, read, hashed, edited, moved, staged,
or committed. It remains untracked. No smoke experiment, pilot, main
experiment, SCNF, COSA, statistics, figures, or paper-result generation was
started.

## Repair commits

The implementation was built in these descriptive commits:

| Commit | Purpose |
| --- | --- |
| `f36f9f4` | Repair model-adapter evidence integrity |
| `344873c` | Correct categorical-adapter leakage probing |
| `067dcf8` | Enforce CUDA cleanup baseline |
| `cad8be1` | Harden adapter resource telemetry |
| `0aadc04` | Harden worker telemetry race handling |

The regenerated evidence was produced from source commit
`0aadc047a5abc38bc90f971ae6062c423d095367`. The final evidence/handoff commit
is intentionally reported in the delivery message rather than embedded in its
own contents.

## Repaired defects

1. Leakage is now measured by executed evidence rather than asserted. Every
   passing record references model-, adapter-, preprocessing-, and
   source-commit-bound leakage evidence.
2. Inventory contracts now fail closed for invalid PASS records, incomplete
   matrices, incorrect identities, bad round trips, invalid hashes, network
   attempts, invalid probabilities, incomplete telemetry, and exceeded RAM or
   VRAM limits.
3. Prediction-cache writes accept identical identity-plus-payload bytes only.
   A different payload under the same identity raises `FAIL_CACHE_INTEGRITY`;
   the tested semantic-equivalence certificate is required for any deliberate
   nondeterministic serialization reuse.
4. Resource limits are enforced during execution and at acceptance. RAM,
   process-tree RSS, CUDA allocation/reservation, VRAM headroom, timeout, OOM,
   cleanup, and GPU-lock outcomes are recorded explicitly.
5. Portable parameter identity is calculated after checkpoint resolution and
   binds the resolved constructor policy without absolute Windows paths.
6. The existing transformation validator now recognizes the three required
   model-adapter schemas as legitimate additions to the protected repository
   surface. Transformation implementation modules and transformation evidence
   remain unchanged; this prevents unrelated adapter schemas from being
   misclassified as transformation mutations.

## Frozen identities

| Model ID | Package | Version | Checkpoint SHA-256 |
| --- | --- | --- | --- |
| `LR-1.9` | `scikit-learn` | `1.9.1` | `null` |
| `CAT-1.2` | `catboost` | `1.2.10` | `null` |
| `XGB-3.4` | `xgboost` | `3.4.1` | `null` |
| `TPFN3-8.5` | `tabpfn` | `8.5.0` | `d0d865d54dfbc524f5703104be90620182dca7e5fb2c16de72e9959ea18f3988` |
| `TICL2-2.2` | `tabicl` | `2.2.0` | `bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0` |

TabICL retained `kv_cache=False` and `allow_auto_download=False`. TabPFN did
not use `fit_with_cache`. Foundation models were run serially and never
concurrently.

## Exact evidence matrix

The sanitized tracked inventory contains exactly 37 unique logical cases:

* CPU: 35 cases = five frozen models × `binary_numerical`,
  `multiclass_numerical`, `missing_numerical`, `categorical_only`,
  `mixed_categorical`, `unseen_category`, and `missing_categorical`.
* CUDA: 2 cases = `TPFN3-8.5` × `binary_numerical`, and `TICL2-2.2` ×
  `binary_numerical`.

The validator recomputes logical case IDs; it does not trust stored IDs. The
tracked inventory and leakage evidence contain hashes and metadata only, with
no raw rows, raw targets, private paths, or prediction matrices.

## Leakage evidence

All 37 inventory cases have measured leakage evidence and
`leakage_evidence_sha256` references. The executed proof records:

* exactly one preprocessing fit call;
* training-only row, feature, and target hashes;
* disjoint inference row and feature hashes;
* excluded inference sentinels;
* rejection of calibration/test labels, target columns, split/group columns,
  duplicate IDs, empty inputs, single-class targets, and noncontiguous targets;
* unchanged preprocessing state before and after inference, including category
  vocabulary and feature ordering checks.

Leakage records: `37/37 passed`.

## Runtime and model results

| Measure | Result |
| --- | ---: |
| CPU cases | `35/35 PASS` |
| CUDA cases | `2/2 PASS` |
| Maximum probability-sum error | `7.450580596923828e-08` |
| Maximum within-case repeat difference | `0.0` |
| Maximum cross-run probability difference, CPU | `0.0` |
| Maximum cross-run probability difference, CUDA | `0.0` |
| Maximum process-tree RAM | `2109.296875 MiB` |
| Maximum recorded VRAM | `342.0 MiB` |
| Offline network attempts | `0` |
| Incomplete telemetry records | `0` |
| Failed cleanup records | `0` |
| Foundation worker overlap | `0` |

The environment report records Windows 11, an Intel 11th-generation CPU,
32,494.785 MiB installed RAM, an NVIDIA GeForce RTX 3050 Laptop GPU with
4,095.5 MiB total VRAM, driver `616.64`, CUDA visible, PyTorch `2.6.0+cu124`,
and CUDA build `12.4`. All five exact package versions were observed.

The enforced policy is a 28 GiB hard process-tree RAM limit, a 3,600 MiB
CUDA soft limit, at least 512 MiB launch headroom, two controlled CPU threads,
one GPU worker, and serial foundation execution.

## Repeat evidence

The complete adapter validation was rerun in separate CPU and CUDA output
directories. The repeat contained 35/35 CPU and 2/2 CUDA PASS cases. Exact
case sets, logical identities, model specification hashes, parameter hashes,
adapter/preprocessing hashes, checkpoint hashes, leakage references, class
orders, round-trip results, resource acceptance, and zero network attempts
matched. Cross-run maximum probability differences were `0.0` for both CPU
and CUDA.

## Cache and parameter identity

The cache tests cover identical-payload reuse, conflicting-payload rejection,
semantic-equivalence approval/rejection, concurrent identical and conflicting
writers, truncated or tampered envelopes, invalid Base64, changed size or
hash, stale and active locks, interrupted writes, offline cache misses, and
unexpected temporary files.

Portable parameter identity includes frozen parameters, seed, device policy,
checkpoint identifier and SHA-256, resolved download policy, TabICL KV-cache
policy, foundation fit/offload/precision policies, thread parameters,
class-dependent loss, package version, Python major/minor, and PyTorch version.
Absolute repository and checkpoint paths are excluded.

## Validation results

| Check | Result |
| --- | --- |
| Focused adapter unit suite | PASS |
| Required unit subset (`not ... evidence`) | `336 passed` |
| Full non-network/non-GPU/non-foundation suite | `362 passed` |
| Full non-foundation integration marker suite | `12 passed` |
| Isolated foundation CPU integration | `3 passed` |
| Isolated foundation CUDA integration | `2 passed` |
| Transformation/artifact contract follow-up | `22 passed` |
| Ruff | PASS — all checks passed |
| Mypy | PASS — 81 source files |
| Generated schema determinism | PASS — no schema diff |
| Strict adapter evidence validator | PASS |
| Repository naming validator | PASS |
| Repository repair validator | PASS |
| Local evidence validator | PASS |
| Grouped split validator | PASS — 70/70 |
| Transformation evidence validator | PASS_PENDING_REVIEW — 770 records, 545 PASS, 225 controlled N/A, 0 failures |
| Transformation property examples | `11,000/11,000` valid |
| Transformation certificate identities | `1,635` valid |

The only warnings were the known scikit-learn `penalty` deprecation warning
under the frozen `LogisticRegression` configuration and the expected warning
for an intentionally undersized split-selection fault fixture. Neither
changed a result.

## Protected Phase 01 preservation

The before/after comparison is `all_unchanged: true` with no changed paths.
The protected smoke foundation remains 748 rows split into train 449,
calibration 150, and test 149; it has 502 predictor groups, 69 duplicated
predictor groups, 31 conflicting-target groups, and zero predictor groups
crossing partitions. The strategy remains exactly
`stratified_group_5fold_v1`; the deprecated row-stratified strategy is not
selected.

The preserved Phase 01 SHA-256 values include the raw OpenML source manifest
and ARFF, processed features/targets/manifests/schema/quality/label mapping,
and the protected seed-1729 grouped assignment and split manifest. The
recorded comparison is retained locally under
`artifacts/phase_02a_model_compatibility/review/`.

## Tracked files changed

The repair changed only the following implementation, test, schema, workflow,
evidence, and handoff files:

```text
.github/workflows/quality.yml
schemas/model_adapter_inventory.schema.json
schemas/model_adapter_leakage_evidence.schema.json
schemas/model_adapter_result.schema.json
scripts/validate_model_adapter_evidence.py
scripts/validate_model_adapters.py
scripts/validate_transformation_engine.py
src/schemaguard/artifact_contracts.py
src/schemaguard/models/adapters/base.py
src/schemaguard/models/adapters/cache.py
src/schemaguard/models/adapters/contracts.py
src/schemaguard/models/adapters/evidence.py
src/schemaguard/models/adapters/preprocessing.py
src/schemaguard/models/adapters/resources.py
tests/integration/test_model_adapter_roundtrip.py
tests/unit/test_artifact_schemas.py
tests/unit/test_model_adapter_cache.py
tests/unit/test_model_adapter_contracts.py
tests/unit/test_model_adapter_evidence.py
tests/unit/test_model_adapter_parameter_identity.py
tests/unit/test_model_adapter_resource_limits.py
artifacts/handoff/model_adapter_inventory.json
artifacts/handoff/model_adapter_leakage_evidence.json
artifacts/handoff/model_adapter_integrity_repair.md
```

No frozen model, package version, checkpoint, preprocessing strategy,
scientific parameter, dataset, split, transformation implementation, or
generated model cache was changed.

## Evidence hashes

| Tracked evidence | SHA-256 |
| --- | --- |
| `artifacts/handoff/model_adapter_inventory.json` | `119ea901a3b81de726b154af1ac4b8601605139bc82b5593eaf7787427a42b56` |
| `artifacts/handoff/model_adapter_leakage_evidence.json` | `8992d9286e68da8c19dbe1bf6e2eec295495b725cf3b5406f785e8ea8212bf78` |

## Exact validation commands

```text
conda run -n P12 python -m pytest -q tests/unit -m "not network and not gpu and not foundation_model and not evidence" -rA
conda run -n P12 python -m pytest -q -m "not network and not gpu and not foundation_model" -rA
conda run -n P12 python -m pytest -q -m "integration and not network and not gpu and not foundation_model" -rA
conda run -n P12 python -m pytest -q tests/integration/test_model_adapter_isolation.py -m "integration and not gpu and not network" -rA
conda run -n P12 python -m pytest -q tests/integration/test_model_adapter_isolation.py -m "integration and gpu and not network" -rA
conda run -n P12 python scripts/generate_artifact_schemas.py
git diff --exit-code -- schemas
conda run -n P12 python -m ruff check .
conda run -n P12 python -m mypy src/schemaguard
conda run -n P12 python scripts/validate_model_adapter_evidence.py --inventory artifacts/handoff/model_adapter_inventory.json --leakage-evidence artifacts/handoff/model_adapter_leakage_evidence.json
conda run -n P12 python scripts/validate_repository_naming.py
conda run -n P12 python scripts/validate_repository_repair.py
conda run -n P12 python scripts/validate_local_evidence.py
conda run -n P12 python scripts/validate_split_generation.py --config configs/runtime/split_generation.yaml --all --offline
conda run -n P12 python scripts/validate_transformation_engine.py --all --offline --inventory-output artifacts/transformation_engine/review/model_adapter_repair_transformation_repeat.json
```

The fresh evidence commands were run offline and used isolated, ignored output
directories for CPU and sequential TabPFN/TabICL CUDA runs. Checkpoints,
datasets, caches, logs, tracebacks, and prediction arrays were not staged.

## Deviations

The requested repair expected an evidence-generation commit and a final
push. The evidence was generated at source commit `0aadc04`; the final commit
containing this handoff and sanitized evidence is created after this document
to avoid a self-referential SHA. The transformation validator allowlist was
extended for the three required adapter schemas, while transformation source
and scientific evidence stayed byte-identical. No other deviation is known.

## Failures

`None` in the final regenerated evidence and validation package.

## Next stage

No next workstream may begin from this handoff until an independent reviewer
accepts the repair and changes the status. Model Adapters remain
`PASS_PENDING_REVIEW` pending independent acceptance.
