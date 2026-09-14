# Phase 02B Dataset Acquisition and GPU Optimization Handoff

## Status

`PASS`

Operational Phase 02B is complete. The work was limited to the DATA and GPU bounded workstreams. No new split expansion, pilot, schema migration, scientific result, or Phase 03 work was started.

## Starting and ending state

- Starting commit: `833127f72177788e18aa8e19ebebeeba523f207f`
- Starting branch: `main`
- Starting tracked worktree: clean
- Permitted untracked items: `SchemaGuard_Complete_Research_and_Engineering_Plan.docx` and pre-existing `schemas/`
- Ending commit: the descriptive commit containing this handoff; verify with `git rev-parse HEAD`
- Runtime: `D:\Conda\P12\python.exe`, Python `3.12.14`
- Conda execution: every command used `conda run -n P12`

## Dataset acquisition and validation

The authoritative OpenML metadata API was used for identity, version, default target, visibility, status, and provider MD5. All 14 records are active public ARFF datasets at version 1. Raw acquisition used per-dataset locks, streaming SHA-256/MD5, `.part` files, checksum validation, and atomic replacement. Processing used deterministic row IDs, contiguous target codes, source-bound manifests, and Zstandard Parquet. No split files were created for the 13 new datasets.

| OpenML ID | Dataset | Rows | Predictors | Classes | Provider MD5 | Raw SHA-256 |
| ---: | --- | ---: | ---: | ---: | --- | --- |
| 3 | kr-vs-kp | 3196 | 36 | 2 | `ad6eb32b7492524d4382a40e23cdbb8e` | `b22a8a12bd40648400b000bad0545683b8ecd33ac84f2c265dd5c309c832fd69` |
| 23 | cmc | 1473 | 9 | 3 | `3149646ecff276abac3e892d1556655f` | `9d8694d39e05d9d2963fccbe4f71e2af04c72c13de45f5dcbe775a586f214567` |
| 29 | credit-approval | 690 | 15 | 2 | `a948be968270a823fce432227c1aa36e` | `ae8e5eadb70e187885ddc6defb7f3e4afa726aa472ac63ebf5e387f0a8467891` |
| 31 | credit-g | 1000 | 20 | 2 | `9a475053fed0c26ee95cd4525e50074c` | `768be84ef749110d5fcac109a6541f2df190e8c6987e1dc432c0a4f29d0642f5` |
| 36 | segment | 2310 | 19 | 7 | `037621812203bfd85c8dab1d7f16ebc6` | `f54038600f6b8ab6cb6edf58887ba98c891781250b4140f5edeb74296e466854` |
| 37 | diabetes | 768 | 8 | 2 | `3cbaa3e54586aa88cf6aacb4033e4470` | `4eddd5b2b64679e8888348e306520a393d6a28e1ddc9643cfb76fc5d912d6d40` |
| 38 | sick | 3772 | 29 | 2 | `ee5cb4b7f41a5f44b0cd234f675ab492` | `85c39615380ca4c05c17a53c3faad9523924958d1e9bc5af0dcbefc6339e7800` |
| 44 | spambase | 4601 | 57 | 2 | `d9ace01aeac3461e326a8e1b2d53fd84` | `f0f6ccd8e8c6b1feeb09c0565ac691d7906cca5373959981512b9934ab4112a6` |
| 46 | splice | 3190 | 60 | 3 | `21a60c8d1b14bbf0f146b4afeda39287` | `4b22fd2fc0564f0b5af59a5a40ee0dd77476835ba48b717960d85d095be53036` |
| 50 | tic-tac-toe | 958 | 9 | 2 | `34b2992c41e5e23c42769817e96305ac` | `09cb4ceb7d410dc7663a5ae8a4d9da6d82270468bdfa5d73a4161204cbb90133` |
| 54 | vehicle | 846 | 18 | 4 | `fbba18157b188f309d772f9ca4e578f5` | `727ab4eb1bf76c0f01adccea696a07c690b5f1e25e15b3defc68c015360edf36` |
| 1067 | kc1 | 2109 | 21 | 2 | `bcc004464934075ecaf54d84e45c427a` | `fe0ea9ab3977bf3d2411dc72eefed2462ef4bf750a77668bdcf2acbb351c521d` |
| 1464 | blood-transfusion-service-center | 748 | 4 | 2 | `c3242468edab8c2e7a907674122dc851` | `ee1304cac4a650ac31afe7395a100536a165f935bbe718a4643757c1a842316a` |
| 1489 | phoneme | 5404 | 5 | 2 | `902ff27649dbfb512f3fb01fc05c6b24` | `8dcd1c5236cea66cdc4e75cb84fbb4da7a4b1e8b1fe5ad5cd3fd825a1ae4c48b` |

Observed quality records are retained in `results/validation/dataset_validation.parquet` and the complete metadata/feature inventory is in `results/validation/schemaorbit14_inventory.json`.

- Dataset count: `14/14 PASS`.
- The deliberate missing-value condition is retained in `credit-approval` and `sick`; no imputation or scientific transformation was applied.
- Duplicate predictor groups and conflicting-target groups are reported, not removed.
- `splice` excludes metadata-declared identifier `Instance_name`; its processed predictor count is exactly 60.
- Online acquisition passed. A second validation/reuse run passed. The complete offline rerun passed for all 14 datasets with network calls disabled (`0` network attempts).

## Phase 01 and Phase 02A preservation

- Phase 01 raw, processed, and active grouped-split hashes are unchanged; evidence: `artifacts/phase_02b_dataset_gpu/review/phase01_hash_comparison.json`.
- Phase 01 remains 748 rows with train/calibration/test `449/150/149`.
- It remains `stratified_group_5fold_v1`, with 502 predictor groups, 69 duplicated groups, 31 conflicting-target groups, and zero groups crossing partitions.
- The deprecated row-stratified split remains unselected.
- Existing Phase 02A source/config/handoff and generated validation reports were not modified.
- Existing root checkpoints were not modified.

## GPU environment and execution

- GPU: NVIDIA GeForce RTX 3050 Laptop GPU
- VRAM: 4095.5 MiB total; 3305.7 MiB free at profiling start
- Driver: `616.64`
- CUDA visibility: available; PyTorch `2.6.0+cu124`; CUDA build `12.4`
- RAM: 32494.8 MiB installed
- CPU: Intel64 Family 6 Model 141; 4 physical/8 logical CPUs

Six synthetic profiles were selected deterministically from acquired metadata: minimum rows, maximum rows, maximum predictors, maximum classes, maximum rows×predictors, and mid-range. Each used a 60/20 train/test approximation with seed 1729 and contained no dataset test observations.

| Model | CUDA profiles | Five-cycle repeat difference | Peak reserved VRAM | Peak RAM | Cleanup allocated VRAM |
| --- | ---: | ---: | ---: | ---: | ---: |
| TabPFN | 11/11 PASS | `0.0` | 1306.0 MiB | 2102.3 MiB | 8.1 MiB |
| TabICL | 11/11 PASS | `0.0` | 1100.0 MiB | 1548.2 MiB | 8.1 MiB |

All successful profiles stayed below the 3600 MiB soft VRAM limit. The two foundation models were never loaded concurrently. The GPU lock and isolated subprocess policy are retained. The isolated-process candidate was measured but not adopted as a scientific/model-configuration change; it remains the safety boundary for crash and memory containment. No CUDA OOM occurred; the deterministic policy remains CUDA safe check → controlled CUDA attempt → CPU fallback.

Checkpoint evidence:

- TabPFN `tabpfn-v3-classifier-v3_default.ckpt`, 212804803 bytes, SHA-256 `d0d865d54dfbc524f5703104be90620182dca7e5fb2c16de72e9959ea18f3988`.
- TabICL `tabicl-classifier-v2-20260212.ckpt`, 110368038 bytes, SHA-256 `bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0`.
- Both are exact root-provided frozen identifiers already validated by Phase 02A. No license/access term was accepted automatically and no credential was recorded.

## Tests and quality

| Test group | Passed | Failed | Skipped | Not verified |
| --- | ---: | ---: | ---: | ---: |
| Non-network, non-GPU, non-foundation suite | 76 | 0 | 0 | 0 |
| Non-network integration evidence suite | 5 | 0 | 0 | 0 |
| GPU profile executions | 22 | 0 | 0 | 0 |
| Ruff | pass | 0 | 0 | 0 |
| Mypy | pass | 0 | 0 | 0 |

The controlled Phase 02A foundation tests remain separately gated by their existing compatibility script; they were not converted into default tests. Fault-oriented coverage includes atomic writes, locks, cache corruption, OOM classification, probability validation, offline cache misses, and configuration mismatch handling.

## Acceptance gates

`PASS: 24/24; FAIL: 0; NOT_VERIFIED: 0`

1. Starting commit/branch/worktree hygiene: PASS.
2. Reference `.docx` and pre-existing `schemas/` untouched and unstaged: PASS.
3. All 14 exact OpenML identities, version 1, public active ARFF metadata: PASS.
4. Default targets, provider checksums, expected dimensions, and class constraints: PASS.
5. Raw streaming checksums and atomic acquisition: PASS.
6. Deterministic processed Parquet, row IDs, contiguous target codes, and source hashes: PASS.
7. `splice` metadata-declared identifier exclusion: PASS.
8. Duplicate/conflicting-group quality evidence retained: PASS.
9. Existing 1464 raw/processed/active split artifacts unchanged: PASS.
10. No new split expansion or scientific preprocessing: PASS.
11. Online acquisition and restart-safe reuse: PASS.
12. All 14 offline validations with zero network attempts: PASS.
13. Exact checkpoint names and SHA-256 values recorded: PASS.
14. RTX 3050/CUDA/PyTorch environment recorded: PASS.
15. Bounded profile selection from metadata and seed 1729: PASS.
16. Serialized GPU execution and exclusive lock: PASS.
17. Both foundation models have a safe CUDA envelope below soft VRAM limit: PASS.
18. Five-cycle cleanup/leak profile and repeated prediction equivalence: PASS.
19. Isolated-process comparison does not alter frozen scientific configuration: PASS.
20. Deterministic failure/fallback policy is implemented: PASS.
21. Generated outputs are ignored and are not staged: PASS.
22. Non-network tests and integration evidence tests pass: PASS.
23. Ruff and Mypy pass: PASS.
24. Handoff and locally retained evidence are complete: PASS.

## Files created or modified

Tracked source/config/test/handoff files created:

- `configs/datasets/schemaorbit14.yaml`
- `src/schemaguard/data/schemaorbit.py`
- `src/schemaguard/compatibility/gpu_profiles.py`
- `scripts/02b_acquire_schemaorbit.py`
- `scripts/02b_profile_gpu_capacity.py`
- `tests/unit/test_schemaorbit.py`
- `tests/unit/test_gpu_profiles.py`
- `tests/integration/test_phase02b_preservation.py`
- `artifacts/handoff/phase_02b_gpu_execution_decision.md`
- `artifacts/handoff/phase_02b_dataset_gpu_review.md`

No existing tracked file was modified. Generated raw/processed data, reports, resource Parquet, review logs, caches, and checkpoints remain ignored and local.

## Commands executed

```text
conda run -n P12 python scripts/02b_acquire_schemaorbit.py --config configs/datasets/schemaorbit14.yaml --allow-network
conda run -n P12 python scripts/02b_acquire_schemaorbit.py --config configs/datasets/schemaorbit14.yaml --offline
conda run -n P12 python scripts/02b_profile_gpu_capacity.py --inventory results/validation/schemaorbit14_inventory.json
conda run -n P12 python -m ruff check .
conda run -n P12 python -m mypy src/schemaguard
conda run -n P12 python -m pytest -q -m "not network and not gpu and not foundation_model" -rA
conda run -n P12 python -m pytest -q -m "integration and not network and not gpu and not foundation_model" -rA
git check-ignore -v schemas/example.schema.json tests/fixtures/example.csv data/raw/example.csv data/processed/example.parquet results/example.csv artifacts/handoff/example.md
```

## Generated evidence

| Artifact | Size | SHA-256 |
| --- | ---: | --- |
| `results/validation/dataset_validation.parquet` | 12928 | `480c054d7919bedba9ad04ee978c74818701e5c5d7b5d859225f3e440518e98f` |
| `results/validation/schemaorbit14_inventory.json` | 79971 | `fc74a52cfb3e9a08a6567ff2b98a5bddd1375e86d76065667501e589a157bd4b` |
| `results/validation/gpu_capacity_report.json` | 39848 | `6c320df3f8fd30f30ba339ddfd5e4051fc20672264eff58a7ca3782f4636d1dd` |
| `results/resources/gpu_capacity_profiles.parquet` | 26191 | `12a1c1e803253173e6b32736ac502329383f5da463f55e2863d0b519e931846d` |
| `results/resources/gpu_optimization_benchmarks.parquet` | 7853 | `dbc69f6444255b4d28b0991d05539b03a69d9696172273f835423561242689a5` |

Detailed logs, environment, checkpoint, phase-preservation, Ruff, Mypy, and pytest evidence are retained under the ignored local directory `artifacts/phase_02b_dataset_gpu/review/`.

## Dependencies and cache

`pyproject.toml` and `uv.lock` were unchanged because all required dependencies were already present. The `uv.lock` SHA-256 remained `e80be80bfc9334f62e3bd2285367752061960003bab0686549d00569ad139ead`. The raw data cache consumed approximately 3.14 MB; checkpoints remained local and ignored.

## Deviations and failures

`None` for scientific or acceptance deviations. A PyTorch telemetry quirk rejected `reset_peak_memory_stats(0)` on the otherwise valid sole GPU; the profiler uses the detected default device with identical semantics, records the decision in the GPU review, and all CUDA profiles pass.

## Next permitted phase

Do not start Phase 03 automatically. An independent review may authorize the next planned workstream after verifying this handoff, the ignored local evidence, the pushed commit, and the unchanged Phase 01/02A artifacts.
