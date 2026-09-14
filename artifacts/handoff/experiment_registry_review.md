# Experiment Registry Review

This is the historical Phase 00 registry review. Later Phase 01 work is documented in
`grouped_split_data_foundation_review.md`.

## Status

PASS

## Files created

- `SCHEMAGUARD_MASTER_PLAN.md`
- `AGENTS.md`
- `pyproject.toml`
- `configs/experiment_registry.yaml`
- `schemas/condition_manifest.schema.json`
- `src/schemaguard/config.py`
- `src/schemaguard/manifest.py`
- `scripts/generate_manifests.py`
- `tests/test_experiment_registry.py`
- `results/manifests/pilot_manifest.json`
- `results/manifests/main_manifest.json`
- `results/validation/validation_summary.json`

## Files modified

- None; this workspace had no pre-existing source tree.

## Commands executed

```text
uv sync
uv run ruff check .
uv run pytest -q
python scripts/generate_manifests.py
python scripts/generate_manifests.py
```

## Tests

| Test group | Passed | Failed | Skipped |
| ---------- | -----: | -----: | ------: |
| Experiment registry pytest suite | 7 | 0 | 0 |
| Ruff checks | 1 | 0 | 0 |
| Pilot manifest count | 990 | 0 | 0 |
| Main manifest count | 3850 | 0 | 0 |

## Artifacts

| Artifact | SHA-256 |
| -------- | ------- |
| `results/manifests/pilot_manifest.json` | `07D9A0CFC3D752E78F4FBFA04FD9A40A2682B9A98CF43B72D85B90605AD94B3F` |
| `results/manifests/main_manifest.json` | `55FC9D1E491B87458E8F12CD32D5A6EB30BEFB544283ADF1E5530AEB59982BBC` |
| `results/validation/validation_summary.json` | `D1F4F0724164FF2D94B3E178A8D40E22C4ED66FBA79B7E711ADC0E1415D06170` |
| `uv.lock` | `53E1A2A86483548D781AA68C04C0DDF6B486DDBED6FDF563F509B88DC16D6E24` |

## Refresh/retry verification

The second freeze invocation reported `manifest_changed: false` for both manifests. Existing state
is read and preserved when its canonical content is unchanged; writes use a temporary file and
atomic rename. This prevents duplicate records and protects the last valid manifest if a run is
interrupted during refresh/retry.

## Resource use

- Runtime: under 1 second for the freeze command
- Peak RAM: not measured; no model or dataset was loaded in the registry foundation
- Peak VRAM: not applicable

## Deviations

- Heavy model packages are deferred to the model compatibility work, as required by the stepwise plan.
- At the time of this Phase 00 review, the workspace had not yet been initialized as a Git
  repository; the later repository setup and Phase 01 correction are recorded separately.

## Blocking failures

- None.
