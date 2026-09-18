# Project status

This page summarizes implementation readiness, not scientific findings. The five-model adapter
layer and ten-condition smoke review are complete. The smoke result is a feasibility check, not a
benchmark or evidence of generalization; the research experiment remains unstarted.

| Area | Status | Notes |
| --- | --- | --- |
| Experiment registry | `PASS` | Frozen identifiers and condition manifests are defined. |
| Data foundation | `VERIFIED_PASS` | Smoke dataset and predictor-group split were independently checked. |
| Model runtime compatibility | `PASS` | Exact package/checkpoint compatibility evidence is retained. |
| Dataset registry | `VERIFIED_PASS` | SchemaOrbit-14 identities and validation metadata are present. |
| GPU capacity policy | `VERIFIED_PASS` | Foundation models are serialized under the local VRAM policy. |
| Split generation | `VERIFIED_PASS` | Grouped split inventory and preservation checks are retained. |
| Transformation engine | `VERIFIED_PASS` | Lossless transformation evidence was independently accepted. |
| Model adapters | `VERIFIED_PASS` | Accepted under the current workstream authorization. |
| Cache and scheduler | `PASS_PENDING_REVIEW` | Separate workstream status; smoke cache/resume evidence is independently verified. |
| Smoke experiment | `VERIFIED_PASS` | [Independent review](../artifacts/handoff/smoke_independent_review.md); feasibility only. |
| Pilot and main experiments | `NOT_STARTED` | Not authorized by this workstream. |
| SCNF, COSA, statistics, figures, paper claims | `NOT_STARTED` | No scientific results are claimed. |

## Adapter gate scope

The adapter evidence covers five exact frozen models, seven synthetic fixtures per model, CPU
inference, strict probability/class-order validation, deterministic row alignment, and adapter-local
round trips. Separate controlled CUDA probes cover the two foundation models. That adapter gate did
not execute the 770 dataset-seed-view matrix; the later ten-condition smoke remains a separate
feasibility check.

Pilot-protocol freezing is the next permitted workstream. Pilot and main experiment execution remain
unstarted and are not authorized by this status update.
