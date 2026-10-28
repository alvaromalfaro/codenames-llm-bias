"""External resources are pinned by SHA-256: a missing or changed file is refused, never read.

Offline. The versioned resource (Glasgow) is checked for real; SWOW-EN18 is not in the repository,
so its own check runs only where the file has been downloaded.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from board_generator.external_resources import (
    EXTERNAL_RESOURCES,
    GLASGOW_NORMS,
    SWOW_EN18,
    ExternalResource,
    ResourceError,
    file_sha256,
    verify,
)


def _resource(tmp_path: Path, content: bytes, *, redistributable: bool) -> ExternalResource:
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "file.csv").write_bytes(content)
    return ExternalResource(
        name="toy",
        relpath="data/file.csv",
        sha256=hashlib.sha256(b"original").hexdigest(),
        license="CC-BY-4.0",
        redistributable=redistributable,
        source="https://example.org/toy",
    )


def test_verify_returns_the_path_of_an_unchanged_file(tmp_path: Path) -> None:
    resource = _resource(tmp_path, b"original", redistributable=True)
    assert verify(resource, tmp_path) == tmp_path / "data" / "file.csv"


def test_verify_refuses_a_changed_file(tmp_path: Path) -> None:
    resource = _resource(tmp_path, b"original\r\n", redistributable=True)
    with pytest.raises(ResourceError, match="expected"):
        verify(resource, tmp_path)


def test_a_missing_non_redistributable_file_says_where_to_get_it(tmp_path: Path) -> None:
    resource = _resource(tmp_path, b"original", redistributable=False)
    (tmp_path / "data" / "file.csv").unlink()
    with pytest.raises(ResourceError, match=r"missing\. It is not redistributable.*example\.org"):
        verify(resource, tmp_path)


def test_file_sha256_matches_hashlib(tmp_path: Path) -> None:
    content = bytes(range(256)) * 5000  # more than one read chunk
    (tmp_path / "blob").write_bytes(content)
    assert file_sha256(tmp_path / "blob") == hashlib.sha256(content).hexdigest()


def test_resource_names_are_unique() -> None:
    names = [resource.name for resource in EXTERNAL_RESOURCES]
    assert len(names) == len(set(names))


def test_the_versioned_glasgow_norms_match_their_pinned_hash() -> None:
    # Fails on a checkout that rewrote its line endings: .gitattributes keeps its bytes as they are.
    assert verify(GLASGOW_NORMS).name == "glasgow_norms.csv"


def test_swow_is_not_versioned_but_verified_where_downloaded() -> None:
    assert not SWOW_EN18.redistributable
    if not SWOW_EN18.path().is_file():
        pytest.skip("SWOW-EN18.zip is not downloaded (it cannot be in the repository)")
    verify(SWOW_EN18)
