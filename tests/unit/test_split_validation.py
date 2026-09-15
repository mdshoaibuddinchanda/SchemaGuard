from pathlib import Path

import pytest

from schemaguard.splits.contracts import SplitGenerationConfig
from schemaguard.splits.validation import SplitValidationError, validate_split

from .test_split_contracts import valid_payload


def test_missing_split_is_rejected(tmp_path: Path) -> None:
    config = SplitGenerationConfig.model_validate(valid_payload())
    with pytest.raises(SplitValidationError):
        validate_split(tmp_path, config, 3, 1729)
