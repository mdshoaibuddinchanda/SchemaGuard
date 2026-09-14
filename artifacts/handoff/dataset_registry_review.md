# Dataset Registry Review

## Status

`PASS_PENDING_REPAIR_REVIEW`

## Completed stage

The SchemaOrbit-14 registry contains fourteen exact OpenML identities with
source-bound metadata, checksums, deterministic processed artifacts, and no new
split generation. The smoke data foundation remains read-only.

## Evidence

* Machine-readable report: `results/validation/dataset_registry_report.json`.
* Validation table: `results/validation/dataset_registry_validation.parquet`.
* Local evidence migration and data-foundation hash comparison are retained under
  the ignored semantic artifact directories.
* Processing uses temporary complete artifact sets, strict validation, per-dataset
  locks, timestamped quarantine, atomic promotion, and Zstandard Parquet.
* Duplicate and conflicting predictor groups use the type-aware SHA-256 grouping
  algorithm; no duplicate rows are removed.

## Restrictions

No split expansion, transformation implementation, model adapters, predictions,
SCNF, COSA, pilot, or main experiment was started by this workstream. The next
stage requires independent acceptance of the machine-readable repair gates.
