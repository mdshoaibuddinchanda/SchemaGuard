# Project status

This page summarizes implementation readiness, not scientific findings. The current completed
workstream is the five-model adapter layer. The research experiment remains unstarted.

| Area | Status | Notes |
| --- | --- | --- |
| Experiment registry | `PASS` | Frozen identifiers and condition manifests are defined. |
| Data foundation | `VERIFIED_PASS` | Smoke dataset and predictor-group split were independently checked. |
| Model runtime compatibility | `PASS` | Exact package/checkpoint compatibility evidence is retained. |
| Dataset registry | `VERIFIED_PASS` | SchemaOrbit-14 identities and validation metadata are present. |
| GPU capacity policy | `VERIFIED_PASS` | Foundation models are serialized under the local VRAM policy. |
| Split generation | `VERIFIED_PASS` | Grouped split inventory and preservation checks are retained. |
| Transformation engine | `VERIFIED_PASS` | Lossless transformation evidence was independently accepted. |
| Model adapters | `PASS_PENDING_REVIEW` | Fixture-only local matrix and integration evidence are in the handoff. |
| Smoke experiment | `NOT_STARTED` | Requires independent acceptance of Model Adapters first. |
| Pilot and main experiments | `NOT_STARTED` | Not authorized by this workstream. |
| SCNF, COSA, statistics, figures, paper claims | `NOT_STARTED` | No scientific results are claimed. |

## Adapter gate scope

The adapter evidence covers five exact frozen models, seven synthetic fixtures per model, CPU
inference, strict probability/class-order validation, deterministic row alignment, and adapter-local
round trips. Separate controlled CUDA probes cover the two foundation models. It does not execute
the 770 dataset-seed-view matrix or the ten-condition smoke experiment.

The next workstream remains blocked until an independent reviewer accepts the adapter handoff and
the repository status is explicitly advanced from `PASS_PENDING_REVIEW` to `VERIFIED_PASS`.
