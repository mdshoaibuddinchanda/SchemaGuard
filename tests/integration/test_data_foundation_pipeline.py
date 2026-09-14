from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from schemaguard.data.download import OfflineCacheUnavailable, SourceIntegrityError
from schemaguard.data.pipeline import run_pipeline

CONFIG = Path(__file__).parents[2] / "configs" / "datasets" / "smoke_blood_transfusion.yaml"


def smoke_arff() -> bytes:
    rows = [
        f"{index % 20},{(index % 10) + 1},{(index % 100) * 10},{index % 25},{1 if index % 2 else 2}"
        for index in range(748)
    ]
    text = "\n".join(
        [
            "@RELATION blood-transfusion-service-center",
            "",
            "@ATTRIBUTE V1 NUMERIC",
            "@ATTRIBUTE V2 NUMERIC",
            "@ATTRIBUTE V3 NUMERIC",
            "@ATTRIBUTE V4 NUMERIC",
            "@ATTRIBUTE Class {1,2}",
            "",
            "@DATA",
            *rows,
            "",
        ]
    )
    return text.encode("utf-8")


@dataclass
class FakeResponse:
    payload: Any = None
    body: bytes = b""
    url: str = "https://openml.org/data/v1/download/1586225/blood-transfusion-service-center.arff"
    headers: dict[str, str] | None = None
    status_code: int = 200

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self) -> Any:
        return self.payload

    def iter_bytes(self, chunk_size: int = 1024):
        for start in range(0, len(self.body), max(1, chunk_size)):
            yield self.body[start : start + max(1, chunk_size)]

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *_args: Any) -> None:
        return None


class FakeClient:
    def __init__(self, body: bytes, metadata: dict[str, Any]) -> None:
        self.body = body
        self.metadata = metadata
        self.get_calls = 0
        self.stream_calls = 0

    def get(self, _url: str) -> FakeResponse:
        self.get_calls += 1
        return FakeResponse(payload=self.metadata)

    def stream(self, _method: str, _url: str) -> FakeResponse:
        self.stream_calls += 1
        return FakeResponse(body=self.body, headers={"etag": "fixture-etag"})


def fake_metadata(body: bytes) -> dict[str, Any]:
    return {
        "data_set_description": {
            "id": "1464",
            "file_id": "1586225",
            "name": "blood-transfusion-service-center",
            "version": "1",
            "format": "ARFF",
            "default_target_attribute": "Class",
            "status": "active",
            "md5_checksum": hashlib.md5(body).hexdigest(),
        }
    }


def test_mocked_http_pipeline_and_cache_hit(tmp_path: Path) -> None:
    body = smoke_arff()
    client = FakeClient(body, fake_metadata(body))
    first = run_pipeline(CONFIG, root=tmp_path, client=client)
    assert first.summary.status == "PASS"
    assert first.parsed.frame.shape == (748, 6)
    assert first.splits.manifest.row_counts == {"train": 449, "calibration": 150, "test": 149}
    assert first.splits.manifest.strategy == "stratified_group_5fold_v1"
    assert first.splits.manifest.predictor_duplicate_groups_crossing_splits == 0
    assert client.get_calls == 1
    assert client.stream_calls == 1

    artifact_hashes = dict(first.processed.artifact_hashes)
    second = run_pipeline(CONFIG, root=tmp_path, offline=True)
    assert second.download.cache_status == "hit"
    assert second.processed.artifact_hashes == artifact_hashes
    assert (
        second.splits.manifest.assignment_file_sha256
        == first.splits.manifest.assignment_file_sha256
    )


def test_offline_without_cache_fails_with_specific_error(tmp_path: Path) -> None:
    with pytest.raises(OfflineCacheUnavailable):
        run_pipeline(CONFIG, root=tmp_path, offline=True)


def test_corrupted_raw_cache_is_rejected(tmp_path: Path) -> None:
    body = smoke_arff()
    first = run_pipeline(CONFIG, root=tmp_path, client=FakeClient(body, fake_metadata(body)))
    raw_path = first.download.raw_path
    corrupted = bytearray(raw_path.read_bytes())
    corrupted[-1] = (corrupted[-1] + 1) % 256
    raw_path.write_bytes(corrupted)
    with pytest.raises(SourceIntegrityError):
        run_pipeline(CONFIG, root=tmp_path, offline=True)


@pytest.mark.network
def test_real_openml_smoke_dataset(tmp_path: Path) -> None:
    outcome = run_pipeline(CONFIG, root=tmp_path)
    assert outcome.download.source_manifest.openml_data_id == 1464
    assert outcome.download.source_manifest.openml_file_id == 1586225
    assert outcome.download.source_manifest.dataset_name == "blood-transfusion-service-center"
    assert outcome.download.source_manifest.default_target_attribute == "Class"
    assert outcome.parsed.frame.shape == (748, 6)
    assert outcome.processed.features.shape == (748, 5)
    assert outcome.splits.manifest.row_counts == {"train": 449, "calibration": 150, "test": 149}
    assert outcome.splits.manifest.predictor_duplicate_groups_crossing_splits == 0
