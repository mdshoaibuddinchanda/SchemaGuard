from __future__ import annotations

import pandas as pd

from schemaguard.models.adapters.factory import create_adapter
from schemaguard.models.adapters.fixtures import AdapterFixture, adapter_fixture_catalog


def fit_predict(model_id: str, fixture: AdapterFixture, *, device: str = "cpu"):
    adapter = create_adapter(model_id, device=device)
    adapter.fit(
        fixture.train,
        fixture.target,
        fixture_id=fixture.name,
        fixture_sha256=fixture.sha256,
        split_identity=f"fixture:{fixture.name}:seed=1729",
    )
    prediction = adapter.predict_proba(
        fixture.test,
        partition="test",
        fixture_id=fixture.name,
        fixture_sha256=fixture.sha256,
        split_identity=f"fixture:{fixture.name}:seed=1729",
    )
    return adapter, prediction


def reordered(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.iloc[::-1].reset_index(drop=True)


FIXTURES = adapter_fixture_catalog()
