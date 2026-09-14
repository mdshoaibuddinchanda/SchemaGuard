from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def is_ignored(path: str) -> bool:
    return (
        subprocess.run(
            ["git", "check-ignore", "-q", "--no-index", path], cwd=ROOT, check=False
        ).returncode
        == 0
    )


def test_generated_path_gitignore_policy() -> None:
    assert not is_ignored("schemas/example.schema.json")
    assert not is_ignored("tests/fixtures/example.csv")
    assert not is_ignored("tests/fixtures/example.tsv")
    assert is_ignored("data/raw/example.csv")
    assert is_ignored("data/processed/example.parquet")
    assert is_ignored("results/example.csv")
    assert not is_ignored("artifacts/handoff/example.md")
