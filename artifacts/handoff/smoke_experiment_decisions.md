# Smoke Experiment Configuration Interpretation Decisions

These decisions preserve the frozen scientific configuration. Neither registry, model identity,
checkpoint, dataset, view, seed, nor experiment parameter was changed.

## CatBoost `bootstrap_type`

`configs/experiment_registry.yaml` contains the existing unquoted YAML scalar `No`. The project's
PyYAML YAML 1.1-compatible loader parses that token as boolean `false`, while
`configs/runtime/model_compatibility.yaml` quotes it and therefore retains the intended string
`"No"`. The smoke planner maps only this exact `(model=CAT-1.2, observed=false,
frozen="No")` representation pair back to the frozen constructor string. Any other value is
rejected. The adapter receives the existing frozen `"No"` value.

## TabICL checkpoint-version field

The experiment registry records the exact TabICL checkpoint in its dedicated `checkpoint` field
and omits the duplicate constructor parameter `checkpoint_version`; the runtime model registry
contains `checkpoint_version` with that same exact identifier. The smoke planner supplies this
constructor parameter only when it exactly equals the dedicated frozen checkpoint name. A
mismatch or any other omitted parameter is rejected.

Both translations are covered by unit tests. They do not change the checkpoint bytes, adapter
parameters, or scientific treatment.

## Protected schema snapshot integration

The transformation evidence validator snapshots the complete `schemas/` directory. The smoke
contracts therefore register their fourteen exact generated schema paths in its existing
schema-addition allowlist. The transformation implementation, mathematics, runtime configuration,
datasets, splits, and accepted transformation inventory remain unchanged; unknown schema paths
remain rejected by the existing tests.

## Responsibility-based evidence names

New smoke evidence uses `protected_foundation` and `protected_split` names rather than execution
stage numbers. The existing local baseline was preserved and copied byte-for-byte to the semantic
smoke-runtime baseline path; no existing numbered repository path was renamed.
