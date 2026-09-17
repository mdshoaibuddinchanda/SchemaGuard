# Handoff index

Tracked handoffs contain concise, reviewable evidence. Detailed generated outputs, raw predictions,
checkpoints, logs, and caches remain ignored local artifacts.

## Current workstream

- [Model adapter review](model_adapters_review.md) — five frozen adapters, fixture validation,
  resource checks, and evidence for independent review.
- [Sanitized model adapter inventory](model_adapter_inventory.json) — strict per-case hashes and
  validation summaries; no probability matrices or row-level predictions.

## Accepted foundations

- [Data foundation and grouped split](grouped_split_data_foundation_review.md)
- [Dataset registry](dataset_registry_review.md)
- [GPU capacity](gpu_capacity_review.md)
- [Split generation](split_generation_review.md)
- [Transformation evidence finalization](transformation_evidence_finalization.md)

Historical review files are retained for provenance. The current project status and the newest
accepted handoff identify which earlier findings have been superseded.
