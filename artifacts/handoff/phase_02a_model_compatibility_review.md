# Phase 02A Model Runtime Compatibility Handoff

## Status

`PASS`

## Starting state

* Starting commit: `6f7fc08597377187fdbc0c843953e5a044cf4599`
* Branch: `main`
* Python executable: `D:\Conda\P12\python.exe`
* Python version: `3.12.14`
* Conda environment: `P12`
* Existing worktree state: permitted untracked reference `.docx`; no unexpected tracked changes at precondition capture.

## Repository hygiene correction

* Removed `schemas/`, global `*.csv`, and global `*.tsv` ignore rules; retained directory-scoped generated output rules.
* `git check-ignore -v` evidence is in `artifacts/phase_02a_model_compatibility/review/gitignore_check.txt`.
* The reference `.docx` was not opened, edited, moved, deleted, staged, or committed.

## Environment

* OS: Windows 11
* CPU: Intel64 Family 6 Model 141 Stepping 1, GenuineIntel
* RAM: 32494.78515625 MiB installed / 8740.81640625 MiB available
* GPU: NVIDIA GeForce RTX 3050 Laptop GPU
* VRAM: 4095.5 MiB total / 3305.7000007629395 MiB free
* Driver: 616.64
* CUDA visibility: True
* PyTorch: 2.6.0+cu124; CUDA build: 12.4

## Dependency results

| Model ID | Expected package | Expected version | Observed version | Result |
| --- | --- | --- | --- | --- |
| LR-1.9 | scikit-learn | 1.9.1 | 1.9.1 | PASS |
| CAT-1.2 | catboost | 1.2.10 | 1.2.10 | PASS |
| XGB-3.4 | xgboost | 3.4.1 | 3.4.1 | PASS |
| TPFN3-8.5 | tabpfn | 8.5.0 | 8.5.0 | PASS |
| TICL2-2.2 | tabicl | 2.2.0 | 2.2.0 | PASS |

## Checkpoint results

| Model ID | Checkpoint | Resolved path | SHA-256 | Cache status | License/access status |
| --- | --- | --- | --- | --- | --- |
| LR-1.9 | None | None | None | not_executed | not_required/not_required |
| CAT-1.2 | None | None | None | not_executed | not_required/not_required |
| XGB-3.4 | None | None | None | not_executed | not_required/not_required |
| TPFN3-8.5 | tabpfn-v3-classifier-v3_default.ckpt | D:\DR2\SchemaGuard\tabpfn-v3-classifier-v3_default.ckpt | d0d865d54dfbc524f5703104be90620182dca7e5fb2c16de72e9959ea18f3988 | validated | resolved/package_default |
| TICL2-2.2 | tabicl-classifier-v2-20260212.ckpt | D:\DR2\SchemaGuard\tabicl-classifier-v2-20260212.ckpt | bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0 | validated | resolved/package_default |

## CPU probe results

| Model ID | Import | Construct | Infer | Probability | Determinism | Runtime | Peak RAM | Status |
| --- | --- | --- | --- | --- | --- | ---: | ---: | --- |
| LR-1.9 | PASS | PASS | PASS | 0.0 | 0.0 | 1.048 | 166.14453125 | PASS |
| CAT-1.2 | PASS | PASS | PASS | 2.220446049250313e-16 | 0.0 | 1.394 | 137.37890625 | PASS |
| XGB-3.4 | PASS | PASS | PASS | 7.171183824539185e-08 | 0.0 | 1.411 | 281.42578125 | PASS |
| TPFN3-8.5 | PASS | PASS | PASS | 5.960464477539063e-08 | 0.0 | 13.043 | 1036.16796875 | PASS |
| TICL2-2.2 | PASS | PASS | PASS | 7.450580596923828e-08 | 0.0 | 8.010 | 870.68359375 | PASS |

## GPU probe results

| Model ID | CUDA attempted | Result | Runtime | Peak VRAM | Fallback | Failure category |
| --- | --- | --- | ---: | ---: | --- | --- |
| TPFN3-8.5 | Yes | PASS | 13.079 | 413.7470703125 | No | PASS |
| TICL2-2.2 | Yes | PASS | 6.964 | 124.5146484375 | No | PASS |

## Tests

| Test group | Passed | Failed | Skipped | Not verified |
| --- | ---: | ---: | ---: | ---: |
| Non-network unit and integration suite | 65 | 0 | 0 | 0 |
| Foundation CPU probes | 2 | 0 | 0 | 0 |
| Foundation GPU probes | 2 | 0 | 0 | 0 |
| Ruff | 1 check | 0 | 0 | 0 |
| Mypy | 1 check | 0 | 0 | 0 |

Evidence is retained in `pytest_output.txt`, `pytest_integration_output.txt`, `ruff_output.txt`, and `mypy_output.txt`.

## Acceptance gates

* Passed count: 48
* Failed count: 0
* Not-verified count: 0

* C01 — Starting commit matches required commit: `PASS`
* C02 — No unexpected tracked worktree changes at precondition capture: `PASS`
* C03 — Reference docx is untouched and untracked: `PASS`
* C04 — Phase-numbering decision is recorded: `PASS`
* C05 — schemas/ is trackable: `PASS`
* C06 — Deliberate CSV/TSV fixtures are trackable: `PASS`
* C07 — Generated data and results remain ignored: `PASS`
* C08 — Python runtime is 3.12: `PASS`
* C09 — Runtime uses Conda environment P12: `PASS`
* C10 — All five frozen model IDs are present: `PASS`
* C11 — All exact package versions are present: `PASS`
* C12 — No frozen model was substituted: `PASS`
* C13 — LR import and construction pass: `PASS`
* C14 — CatBoost import and construction pass: `PASS`
* C15 — XGBoost import and construction pass: `PASS`
* C16 — TabPFN import and construction pass: `PASS`
* C17 — TabICL import and construction pass: `PASS`
* C18 — TabPFN checkpoint identity is recorded: `PASS`
* C19 — TabICL checkpoint identity is recorded: `PASS`
* C20 — License/access state is recorded: `PASS`
* C21 — LR CPU micro-inference passes: `PASS`
* C22 — CatBoost CPU micro-inference passes: `PASS`
* C23 — XGBoost CPU micro-inference passes: `PASS`
* C24 — TabPFN CPU micro-inference passes: `PASS`
* C25 — TabICL CPU micro-inference passes: `PASS`
* C26 — Binary probabilities pass validation: `PASS`
* C27 — Multiclass probabilities pass validation: `PASS`
* C28 — Class ordering is canonical and recorded: `PASS`
* C29 — CPU repeated inference meets tolerance: `PASS`
* C30 — CUDA capability is explicitly recorded: `PASS`
* C31 — CUDA OOM cannot crash the phase runner: `PASS`
* C32 — Foundation models are never loaded concurrently: `PASS`
* C33 — TabICL uses kv_cache=False: `PASS`
* C34 — TabPFN does not use fit_with_cache: `PASS`
* C35 — Offline checkpoint reuse passes: `PASS`
* C36 — Cache corruption is detected: `PASS`
* C37 — Atomic-write fault tests pass: `PASS`
* C38 — Lock-concurrency tests pass: `PASS`
* C39 — Resource records exist for every executed probe: `PASS`
* C40 — Every failed or unexecuted probe has an explicit reason: `PASS`
* C41 — Phase 01 hashes remain unchanged: `PASS`
* C42 — Grouped split still has zero crossing groups: `PASS`
* C43 — Ruff passes: `PASS`
* C44 — Mypy passes: `PASS`
* C45 — Required non-network tests pass: `PASS`
* C46 — Required integration tests pass: `PASS`
* C47 — Generated reports validate against strict contracts: `PASS`
* C48 — Handoff contains exact commands, hashes, and results: `PASS`

## Phase 01 preservation

* Hash comparison: `artifacts/phase_02a_model_compatibility/review/phase01_hash_comparison.json`.
* All unchanged: True
* Split sizes: train 449, calibration 150, test 149.
* Conflicting-target groups: 31.
* Predictor groups crossing splits: 0.
* Strategy: `stratified_group_5fold_v1`.
* Hashes before and after (SHA-256):
```text
data/processed/openml/1464/data_manifest.json | before=c1bfecd0545ce173e35d2123c6c082df38a21aff80bdfac3e15ba73f64d0dedf | after=c1bfecd0545ce173e35d2123c6c082df38a21aff80bdfac3e15ba73f64d0dedf
data/processed/openml/1464/features.parquet | before=9d3168b0bafc5af610caea42c96f95033f32bf5ae4dc011797f45d1809af458f | after=9d3168b0bafc5af610caea42c96f95033f32bf5ae4dc011797f45d1809af458f
data/processed/openml/1464/label_mapping.json | before=251d6c099788f9fb5850c3c80ddc1390021b5c480e4175a5f09d80813fbd94af | after=251d6c099788f9fb5850c3c80ddc1390021b5c480e4175a5f09d80813fbd94af
data/processed/openml/1464/quality_report.json | before=47e2c0bf256453278a0bec96099fc980f972fb9661c914a71818f920546f9e04 | after=47e2c0bf256453278a0bec96099fc980f972fb9661c914a71818f920546f9e04
data/processed/openml/1464/schema.json | before=cba51d40f1ecd2f34b4b83e4d55a6a134708d4d1a4a11d6ed263b161792d4e9c | after=cba51d40f1ecd2f34b4b83e4d55a6a134708d4d1a4a11d6ed263b161792d4e9c
data/processed/openml/1464/targets.parquet | before=9d4e0f665127dd2c41e990d4b129e08d8a0c55f83d42d2e57b758959a9fc23d0 | after=9d4e0f665127dd2c41e990d4b129e08d8a0c55f83d42d2e57b758959a9fc23d0
data/raw/openml/1464/blood-transfusion-service-center.arff | before=ee1304cac4a650ac31afe7395a100536a165f935bbe718a4643757c1a842316a | after=ee1304cac4a650ac31afe7395a100536a165f935bbe718a4643757c1a842316a
data/raw/openml/1464/openml_metadata.json | before=b80603f5f892337403d34bcd0542e79ed82b8b9559fab33fb67618542ba935a7 | after=b80603f5f892337403d34bcd0542e79ed82b8b9559fab33fb67618542ba935a7
data/raw/openml/1464/source_manifest.json | before=4fbaedd74510c93364ba31ec92cfc83999de0f8ee7bcb40315d3e01b40efb755 | after=4fbaedd74510c93364ba31ec92cfc83999de0f8ee7bcb40315d3e01b40efb755
data/splits/openml/1464/stratified_group_5fold_v1/seed_1729/assignments.parquet | before=e08b9d5c94578b317dcc844c2a6aa7a4f96ad8df699352d1e11d86a7dd452236 | after=e08b9d5c94578b317dcc844c2a6aa7a4f96ad8df699352d1e11d86a7dd452236
data/splits/openml/1464/stratified_group_5fold_v1/seed_1729/split_manifest.json | before=0a97a18238b209e1c933d7c2eb89135f477fa0abce55ba09840169aac0bae190 | after=0a97a18238b209e1c933d7c2eb89135f477fa0abce55ba09840169aac0bae190
```

## Files created

* `configs/runtime/model_compatibility.yaml`
* `src/schemaguard/compatibility/` and `src/schemaguard/models/` runtime modules
* `scripts/02_check_model_compatibility.py`
* Phase 02A unit and integration tests
* `artifacts/handoff/phase_numbering_decision.md` and this handoff.

## Files modified

* `.gitignore`
* `pyproject.toml`
* `uv.lock`

## Generated artifacts

| Artifact | Size | SHA-256 |
| --- | ---: | --- |
| results/validation/environment_report.json | 1939 | 800ee4459565600450afd278b8a86ef8f003f14a7750251d6681ac1b0c6d93b4 |
| results/validation/model_compatibility.parquet | 23794 | d4b1523b509a521e4d761f03c8f2daba8d2658f7e879b9f37b888093b0f0c365 |
| results/validation/checkpoint_inventory.json | 2022 | 32121d87f50c2d8ba840df877ad83ab75a64f87646dc289762f2052fb4cf6fd4 |
| results/validation/license_inventory.json | 1234 | 002620c4a447ace5deada41e68f9a5b3d1c4b53174bd7270b39a4f6881a1a3e2 |
| results/resources/model_probe_resources.parquet | 10044 | ee1470ff39b3b8e76de3d37f1d0930dcdcc9c46971cfe407e781daaab899dbf0 |
| artifacts/phase_02a_model_compatibility/review/acceptance_gates.md | 2704 | c6b6da88474578f851b103939d9e7ef1641a91c875925429537f76e7ffd1bc8a |
| artifacts/phase_02a_model_compatibility/review/acceptance_gates.json | 819 | 0ab9e4341ad89c477505e7107f897f30fea9fb07e73116229bb584d0599c35bb |
| artifacts/phase_02a_model_compatibility/review/commands.txt | 2868 | aaac2dee5245a389ba5b5d7326d18439eacec969d85c6d34d3a3c91e5f0da08e |
| artifacts/phase_02a_model_compatibility/review/offline_reuse.json | 910 | 03379882593ebfef814c4f1cd8276e8492a997ff126592024ab91240f6b67f94 |
| artifacts/phase_02a_model_compatibility/review/phase01_hash_comparison.json | 4226 | b4f0dc3f70c0c16436823b0b9cd099e8f4a7c7eb58e1352418cd3bce4c98e6fb |

## Commands executed

```text
# Phase 02A commands and evidence

cd D:\DR2\SchemaGuard

git rev-parse HEAD
git branch --show-current
git status --short
git log -1 --oneline

conda run -n P12 python --version

uv lock --python D:\Conda\P12\python.exe --no-managed-python
uv sync --extra dev --extra models-cpu --extra models-foundation --extra monitoring --python D:\Conda\P12\python.exe --no-managed-python
conda run -n P12 python -m pip install --dry-run --upgrade "scikit-learn==1.9.1" "catboost==1.2.10" "xgboost==3.4.1" "tabpfn==8.5.0" "tabicl==2.2.0" "psutil>=7,<8"
conda run -n P12 python -m pip install --upgrade "scikit-learn==1.9.1" "catboost==1.2.10" "xgboost==3.4.1" "tabpfn==8.5.0" "tabicl==2.2.0" "psutil>=7,<8"

uv run --python D:\Conda\P12\python.exe python -m ruff check .
uv run --python D:\Conda\P12\python.exe python -m mypy src/schemaguard
conda run -n P12 python -m pytest -q -m "not network and not gpu and not foundation_model" -rA
conda run -n P12 python -m pytest -q -m "integration and not network and not gpu and not foundation_model" -rA

conda run -n P12 python scripts/02_check_model_compatibility.py --config configs/runtime/model_compatibility.yaml --device cpu --offline --refresh
conda run -n P12 python scripts/02_check_model_compatibility.py --config configs/runtime/model_compatibility.yaml --model TPFN3-8.5 --device cuda --offline --refresh
conda run -n P12 python scripts/02_check_model_compatibility.py --config configs/runtime/model_compatibility.yaml --model TICL2-2.2 --device cuda --offline --refresh
conda run -n P12 python scripts/02_check_model_compatibility.py --config configs/runtime/model_compatibility.yaml --device auto --offline --refresh
conda run -n P12 python scripts/02_check_model_compatibility.py --config configs/runtime/model_compatibility.yaml --device auto --offline

# Not executed: the online foundation acquisition command was stopped at the first
# unresolved TabPFN v3 gated-checkpoint authorization requirement.
conda run -n P12 python scripts/02_check_model_compatibility.py --config configs/runtime/model_compatibility.yaml --device cpu --allow-network
conda run -n P12 python -c "import sys; sys.path.insert(0, 'src'); import json, pandas as pd; from schemaguard.compatibility.contracts import EnvironmentReport, PhaseResult, CheckpointRecord, LicenseRecord; root='results/validation'; EnvironmentReport.model_validate(json.load(open(root+'/environment_report.json'))); phase=PhaseResult.model_validate(json.load(open(root+'/phase_result.json'))); [CheckpointRecord.model_validate(x) for x in json.load(open(root+'/checkpoint_inventory.json'))]; [LicenseRecord.model_validate(x) for x in json.load(open(root+'/license_inventory.json'))]; assert len(pd.read_parquet(root+'/model_compatibility.parquet')) == 7; assert len(pd.read_parquet('results/resources/model_probe_resources.parquet')) == 7; print('strict_reports_ok')"

```

## Cache and offline verification

* Checkpoint and offline evidence is in `checkpoint_inventory.json`, `license_inventory.json`, and `offline_reuse.json`.
* No checkpoint files are committed.

## Deviations

* Exact deviations and blockers are recorded per probe in `model_compatibility.parquet` and traceback paths.
* The uv project environment resolved PyTorch 2.14.0, but all authoritative probes used P12's detected PyTorch 2.6.0+cu124; no probe used the uv environment.

## Failures

* None.
* No Phase 01 mutation was detected.

## Resource use

* Total recorded probe runtime: 44.950 seconds.
* Maximum observed RAM: 2103.988 MiB.
* Maximum observed VRAM allocation: 413.747 MiB; telemetry is null where not executed.
* Disk consumed by checkpoints: 323,172,841 bytes; checkpoint files remain ignored and untracked.

## Next permitted phase

Phase 02B may begin only after all five CPU paths, reproducible checkpoint identity, offline reuse, preservation, quality checks, and all C01–C48 gates pass.
