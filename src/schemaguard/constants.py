"""Frozen names used by the smoke-data foundation."""

from pathlib import Path

RESERVED_COLUMN_PREFIX = "__sg_"
ROW_ID_COLUMN = "__sg_row_id"
GROUP_ID_COLUMN = "__sg_group_id"
TARGET_LABEL_COLUMN = "target_label"
TARGET_CODE_COLUMN = "target_code"
SMOKE_DATASET_ID = "blood-transfusion-service-center"
OPENML_DATA_ID = 1464
OPENML_FILE_ID = 1586225
DEFAULT_TARGET_NAME = "Class"
MASTER_SEED = 1729
UTC = "UTC"


def repository_root() -> Path:
    """Return the repository root independent of the current working directory."""
    return Path(__file__).resolve().parents[2]
