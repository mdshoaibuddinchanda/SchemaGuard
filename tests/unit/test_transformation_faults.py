from __future__ import annotations

import json

from schemaguard.transformations.caching import TransformationCache


def test_truncated_manifest_is_rejected(tmp_path) -> None:
    cache = TransformationCache(tmp_path)
    key = "a" * 64
    features, certificate, manifest = cache.paths(key)
    manifest.parent.mkdir(parents=True)
    features.write_bytes(b"partial")
    certificate.write_bytes(b"{}")
    manifest.write_text("{", encoding="utf-8")
    assert cache.read_validated(key) is None


def test_invalid_cache_hash_is_rejected(tmp_path) -> None:
    cache = TransformationCache(tmp_path)
    key = "a" * 64
    features, certificate, manifest = cache.paths(key)
    manifest.parent.mkdir(parents=True)
    features.write_bytes(b"features")
    certificate.write_bytes(b"certificate")
    manifest.write_text(
        json.dumps({"cache_key": key, "feature_sha256": "bad", "certificate_sha256": "bad"}),
        encoding="utf-8",
    )
    assert cache.read_validated(key) is None
