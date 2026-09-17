# Model Adapter Review

## Review status

`PASS_PENDING_REVIEW`

- `completed_stage: model_adapters`
- `next_stage: independent_review_and_authorization`

This is an implementation-readiness result, not a scientific result. The next workstream remains
blocked until an independent reviewer accepts this evidence and project status is explicitly
advanced to `VERIFIED_PASS`.

## Scope and source identity

Implemented and tested the five adapters frozen in `configs/experiment_registry.yaml`, using the
existing `P12` Python 3.12 installation. All probes used deterministic synthetic data. No SchemaOrbit
dataset acquisition, full 770 dataset-seed-view matrix, ten-condition smoke experiment, benchmark,
pilot, main experiment, SCNF, COSA, statistical analysis, or paper result was run.

The required starting commit was `e6efd1183e932ecbbfa5e96a8161f674f6b13f92` on `main`; `origin/main`
matched it at preflight. The adapter implementation under test is source commit
`078d2e6559c67177f8397702aa6a1d1dd11a439e`. The source and evidence are kept in separate commits so
that all recorded probes identify the exact implementation commit. No commit or push occurred during
the validation runs themselves.

## Frozen model and preprocessing matrix

| Model ID | Installed package/version | Frozen preprocessing | CPU | CUDA |
| --- | --- | --- | --- | --- |
| `LR-1.9` | scikit-learn 1.9.1 | `common_onehot_standard` | 7/7 PASS | Not applicable |
| `CAT-1.2` | catboost 1.2.10 | `native_catboost` | 7/7 PASS | Not applicable |
| `XGB-3.4` | xgboost 3.4.1 | `common_onehot_no_scaling` | 7/7 PASS | Not applicable |
| `TPFN3-8.5` | tabpfn 8.5.0 | `native_tabpfn` | 7/7 PASS | PASS |
| `TICL2-2.2` | tabicl 2.2.0 | `native_tabicl` | 7/7 PASS | PASS |

No frozen model, package version, or preprocessing strategy was substituted. The adapter reads and
preserves the frozen model parameters; the tests additionally enforce `kv_cache=False` for TabICL,
two-thread CPU limits, canonical output row alignment, and train-only fitting. Test-label sentinels
did not enter fitting or preprocessing state.

## Checkpoint identity and access

Both checkpoints were already present locally and were hashed in full. Adapter probes were offline;
the inventory records zero network attempts for every case. No credentials or license acceptance
were needed or recorded.

| Model | Frozen checkpoint identifier | Local file size | SHA-256 | Access result |
| --- | --- | ---: | --- | --- |
| `TPFN3-8.5` | `tabpfn-v3-classifier-v3_default.ckpt` | 212,804,803 bytes | `d0d865d54dfbc524f5703104be90620182dca7e5fb2c16de72e9959ea18f3988` | Local identity verified; offline inference passed |
| `TICL2-2.2` | `tabicl-classifier-v2-20260212.ckpt` | 110,368,038 bytes | `bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0` | Local identity verified; offline inference passed |

The inventory is [model_adapter_inventory.json](model_adapter_inventory.json), SHA-256
`51afc7d0c81ddc97aefb6563a9112edcc4830f29b8a6cf9a0e90591fbf11d90e`. It contains 37 strict,
sanitized records tied to the source commit above. It stores identity hashes and validation/resource
summaries, not probability matrices or row-level predictions.

## Fixture and prediction results

Each CPU model passed these seven deterministic fixtures: `binary_numerical`, `multiclass_numerical`,
`missing_numerical`, `categorical_only`, `mixed_categorical`, `unseen_category`, and
`missing_categorical`. Each fixture uses separated train/inference rows and stable IDs. Results:

- CPU: 35/35 pass; zero failures; all five models were checked on all seven fixtures.
- CUDA: 2/2 pass; each foundation model used a separate sequential isolated worker.
- Cross-run prediction comparison against the initial CPU matrix: maximum absolute difference
  `0.0`.
- Maximum probability row-sum error across all recorded cases: `7.450580596923828e-08`, within the
  configured `1e-6` tolerance. All class orders were canonical `[0, 1]` for binary probes and were
  recorded for every case.
- Maximum repeated-prediction difference: `0.0`; this is within CPU tolerance `1e-10` and CUDA
  tolerance `1e-5`.
- All 37 records report passing train/test leakage sentinels and adapter-local cache or model
  serialization round trips.
- Offline network-attempt count: zero.

The strict inventory contains per-case hashes for the adapter, frozen specification, parameters,
fixture, preprocessing, row IDs, probabilities, and prediction identity. The raw synthetic
prediction artifacts remain local and ignored.

## Resource results

Runtime is the sum of per-case probe runtimes recorded in the inventory. Wall time is the validator
run duration. Resource telemetry is reported from the isolated workers rather than inferred from
host totals.

| Model | CPU cases / total probe seconds | Peak CPU-worker RAM | CUDA result / probe seconds | Peak CUDA-worker RAM | Peak VRAM |
| --- | ---: | ---: | ---: | ---: | ---: |
| `LR-1.9` | 7 / 2.286 s | 176.8 MiB | — | — | — |
| `CAT-1.2` | 7 / 67.930 s | 213.7 MiB | — | — | — |
| `XGB-3.4` | 7 / 3.375 s | 336.9 MiB | — | — | — |
| `TPFN3-8.5` | 7 / 26.127 s | 1,038.7 MiB | PASS / 4.700 s | 2,100.6 MiB | 342 MiB |
| `TICL2-2.2` | 7 / 13.154 s | 773.7 MiB | PASS / 2.211 s | 1,341.3 MiB | 158 MiB |

The final CPU matrix took 391.016 seconds wall time; the final CUDA checks took 52.547 seconds.
Observed environment: Windows 11, 11th Gen Intel Core i5-11260H (4 physical / 8 logical cores), 31.733 GiB RAM,
NVIDIA GeForce RTX 3050 Laptop GPU (4,096 MiB), driver 616.64, PyTorch 2.6.0+cu124 (CUDA build
12.4). CUDA was visible. GPU telemetry showed 2,535 MiB free before the final probes; the required
512 MiB headroom policy was met. Each model process exited before the next foundation model loaded.

Preprocessing cache timing is preserved per fixture in the inventory. Across CPU cases, mean
uncached-to-hit timings were: LR 7.168 to 1.119 ms; CatBoost 2.273 to 0.976 ms; XGBoost 7.337 to
1.180 ms; TabPFN 2.279 to 1.028 ms; TabICL 2.135 to 0.808 ms. These are fixture-local adapter
measurements, not claims about production workload speed.

## Tests and quality gates

Executed under `D:\Conda\P12\python.exe`:

| Gate | Result |
| --- | --- |
| Unit tests | `278 passed`, 4 warnings, 82.03 s |
| CPU integration tests | `6 passed`, 2 deselected, 1 warning, 48.98 s |
| GPU integration tests | `2 passed`, 3 deselected, 20.25 s |
| Ruff | PASS: `python -m ruff check .` |
| Mypy | PASS: `python -m mypy src/schemaguard`; 80 source files |
| Repository naming | PASS; zero violations |
| Repository repair validation | R01–R09 PASS; local-evidence R10/R11 not applicable |
| Strict inventory | PASS; 37 records, one source commit, all statuses PASS |
| Ignore policy | PASS; generated data/results ignored, schemas and CSV fixtures trackable, handoff JSON explicitly trackable |
| `git diff --check` | PASS |

The four unit-test warnings consist of three scikit-learn notices about the frozen explicit
`penalty="l2"` parameter and one existing low-class-count split-selection warning. No warning was
converted into a passing prediction or hidden.

Exact commands:

```powershell
D:\Conda\P12\python.exe scripts/validate_model_adapters.py --device cpu --compare-run results/validation/model_adapters_cpu_first/adapter_predictions.json --inventory artifacts/handoff/model_adapter_inventory.json --output-directory results/validation/model_adapters_cpu_final --estimated-case-seconds 10
D:\Conda\P12\python.exe scripts/validate_model_adapters.py --device cuda --merge-inventory artifacts/handoff/model_adapter_inventory.json --output-directory results/validation/model_adapters_cuda_final --estimated-case-seconds 15
D:\Conda\P12\python.exe -m pytest -q -o addopts='' tests/unit
D:\Conda\P12\python.exe -m pytest -q -o addopts='' -rA -m 'integration and not gpu and not network' tests/integration/test_model_adapter_roundtrip.py tests/integration/test_model_adapter_isolation.py
D:\Conda\P12\python.exe -m pytest -q -o addopts='' -rA -m 'integration and gpu and not network' tests/integration/test_model_adapter_isolation.py
D:\Conda\P12\python.exe -m ruff check .
D:\Conda\P12\python.exe -m mypy src/schemaguard
D:\Conda\P12\python.exe scripts/validate_repository_naming.py
D:\Conda\P12\python.exe scripts/validate_repository_repair.py
```

The local inventory reports CPU status `PASS`, 35 records, zero failures, no network access, and
cross-run maximum difference `0.0`. The CUDA report reports status `PASS`, two records, and zero
failures. Generated reports and predictions are under ignored `results/validation/`.

## Preservation, hygiene, and scope controls

The existing Phase 01 before/after comparison reports `all_unchanged: true` for all 11 recorded
raw-data, processed-data, and split artifacts; `changed_paths` is empty. The preserved split has
748 rows (449 train, 150 calibration, 149 test), 502 predictor groups, including 69 duplicate
predictor groups and 31 conflicting-target groups; no predictor group crosses partitions. The
selected strategy remains `stratified_group_5fold_v1`.

The private root reference `.docx` was not opened, edited, moved, deleted, staged, or committed. It
remains untracked. No checkpoint, generated dataset, prediction array, cache, log, or other large
generated output is included in the tracked handoff. `.gitignore` behavior was verified with
`git check-ignore -v --no-index` for generated data/results, schemas, CSV fixtures, and this handoff
JSON.

## Files added or changed

Implementation and tests are in the source commit. The evidence commit adds this review and the
sanitized inventory. Project-facing files updated are `README.md`, `docs/project_status.md`,
`docs/repository_guide.md`, and `artifacts/handoff/README.md`. The source implementation adds the
runtime config, adapters, strict contracts/schemas, validator CLI, and unit/integration tests; CI
also runs the classical adapter round-trip integration test.

## Decision

Local implementation, preservation, test, and quality gates passed. This is deliberately
`PASS_PENDING_REVIEW`, not `VERIFIED_PASS`: independent review remains required before any smoke or
later research experiment is authorized.
