# SchemaGuard implementation rules

1. Read `SCHEMAGUARD_MASTER_PLAN.md` before changing code.
2. Implement exactly one numbered workstream at a time.
3. Do not rename datasets, models, views, metrics, or outputs.
4. Do not change frozen configurations without a decision record.
5. Add tests before completing a phase.
6. Run the entire relevant test group after every implementation.
7. Stop when a blocking test fails.
8. Do not replace failures with missing values.
9. Do not manually edit generated predictions, metrics, tables, or figures.
10. All expensive operations must use content-addressed caching.
11. CPU jobs may use two workers.
12. GPU jobs must be sequential and protected by a file lock.
13. Every output must record its source hashes.
14. Test labels must never enter preprocessing fitting, COSA selection, or parameter selection.
15. At the end of each workstream, create a clearly named review file in `artifacts/handoff/`.
16. Do not begin the next workstream until its review file reports PASS.
17. Use only the existing `P12` Conda environment; do not create or use another environment.
