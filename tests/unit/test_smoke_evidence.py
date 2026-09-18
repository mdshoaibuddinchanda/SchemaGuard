from __future__ import annotations

import pandas as pd

from schemaguard.experiments.evidence import _lineage_column_matches


def test_nullable_lineage_requires_null_only_when_plan_has_no_checkpoint() -> None:
    frame = pd.DataFrame({"checkpoint_sha256": [None, None]})
    assert _lineage_column_matches(frame, "checkpoint_sha256", None)
    assert not _lineage_column_matches(
        pd.DataFrame({"checkpoint_sha256": [None, "unexpected"]}),
        "checkpoint_sha256",
        None,
    )


def test_nonnullable_lineage_requires_exact_values_without_missing_entries() -> None:
    assert _lineage_column_matches(pd.DataFrame({"model_id": ["LR-1.9"]}), "model_id", "LR-1.9")
    assert not _lineage_column_matches(
        pd.DataFrame({"model_id": ["LR-1.9", None]}), "model_id", "LR-1.9"
    )
    assert not _lineage_column_matches(
        pd.DataFrame({"model_id": ["XGB-3.4"]}), "model_id", "LR-1.9"
    )
