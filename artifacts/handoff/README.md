# Handoff index

Tracked handoffs contain concise, reviewable evidence. Detailed generated outputs, raw predictions,
checkpoints, logs, and caches remain ignored local artifacts.

## Current workstream

- [Independent ten-condition smoke review](smoke_independent_review.md) — status is
  `VERIFIED_PASS`; the smoke is a feasibility result, not a benchmark. Pilot-protocol freezing is
  the next permitted workstream.

- [Cache and scheduler review](cache_scheduler_review.md) — immutable content-addressed artifacts,
  atomic publication, worker scheduling, resume validation, resource controls, and fault evidence;
  status is `PASS_PENDING_REVIEW`.
- [Cache and scheduler inventory](cache_scheduler_inventory.json) — source-bound sanitized gates,
  counters, and protected-file comparison digest.
- [Cache and scheduler probe evidence](cache_scheduler_probe_evidence.json) — sanitized cold/resume
  summaries and measured resource bounds.
- [Cache and scheduler fault evidence](cache_scheduler_fault_evidence.json) — observed outcomes for
  all 30 required fault cases.
- [Protected-file comparison](cache_scheduler_protected_hash_comparison.json) — SHA-256 comparison
  of protected local files plus Git-tree identities from the starting and implementation commits.

## Accepted workstreams

- Model adapters — accepted as `VERIFIED_PASS` by the current workstream authorization. The existing
  [adapter review](model_adapters_review.md), [inventory](model_adapter_inventory.json), and
  [integrity handoff](model_adapter_integrity_repair.md) are retained as implementation evidence;
  their earlier pending-review wording is historical and superseded by that acceptance.
- [Sanitized model adapter inventory](model_adapter_inventory.json) — strict per-case hashes and
  validation summaries; no probability matrices or row-level predictions.

## Accepted foundations

- [Data foundation and grouped split](grouped_split_data_foundation_review.md)
- [Dataset registry](dataset_registry_review.md)
- [GPU capacity](gpu_capacity_review.md)
- [Split generation](split_generation_review.md)
- [Transformation evidence finalization](transformation_evidence_finalization.md)

Historical review files are retained for provenance. The independent smoke review verifies the
ten-condition outputs and cache/resume behavior without changing the separate cache/scheduler
workstream status. Pilot execution remains unstarted. Current project scope is documented in the
root README and project-status page.
