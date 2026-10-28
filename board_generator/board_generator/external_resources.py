"""External data the generator reads, pinned by SHA-256.

Each resource names its file under ``resources/``, the hash of the exact bytes it was validated
with, its license, and whether the repository may carry it. ``verify`` refuses a missing or changed
file, so a bank is never built from data other than the data its manifest records. A resource that
cannot be redistributed (SWOW-EN18) is downloaded by hand into its directory, which git ignores; the
error for a missing file says where to get it.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

RESOURCES_DIR = Path(__file__).resolve().parent.parent / "resources"


class ResourceError(RuntimeError):
    """An external resource is missing, or its bytes are not the ones it is pinned to."""


@dataclass(frozen=True, slots=True)
class ExternalResource:
    """One external file: where it lives under ``resources/``, its pinned hash and its terms."""

    name: str
    relpath: str
    sha256: str
    license: str
    redistributable: bool
    source: str

    def path(self, resources_dir: Path = RESOURCES_DIR) -> Path:
        return resources_dir / self.relpath


GLASGOW_NORMS = ExternalResource(
    name="glasgow-norms",
    relpath="glasgow/glasgow_norms.csv",
    sha256="30d7776eed9219767f8a6aabdf95d7120cd050ddeab431acef13d435807bbc39",
    license="CC-BY-4.0",
    redistributable=True,
    source="https://doi.org/10.3758/s13428-018-1099-3 (ESM 2)",
)

SWOW_EN18 = ExternalResource(
    name="swow-en18",
    relpath="swow/SWOW-EN18.zip",
    sha256="69ee1f4f19c657bb2f6e1bc27abd2becbcd01d83d1fdade94a0455b4e66eb093",
    license="CC-BY-NC-ND-3.0",
    redistributable=False,
    source="https://smallworldofwords.org/en/project/research (SWOW-EN18, v1.0, 2018-10-18)",
)

EXTERNAL_RESOURCES: tuple[ExternalResource, ...] = (GLASGOW_NORMS, SWOW_EN18)


def file_sha256(path: Path) -> str:
    """SHA-256 of a file's bytes, read in chunks (SWOW-EN18 is 80 MB)."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(resource: ExternalResource, resources_dir: Path = RESOURCES_DIR) -> Path:
    """Return the resource's path once its bytes match the pinned hash.

    :raises ResourceError: the file is missing (for a resource the repository cannot carry, the
        message says where to download it) or its SHA-256 is not the pinned one.
    """
    path = resource.path(resources_dir)
    if not path.is_file():
        hint = (
            f" It is not redistributable: download it from {resource.source} and put it there, "
            f"unchanged (see {path.parent / 'README.md'})."
            if not resource.redistributable
            else ""
        )
        raise ResourceError(f"{resource.name}: {path} is missing.{hint}")
    actual = file_sha256(path)
    if actual != resource.sha256:
        raise ResourceError(
            f"{resource.name}: {path} has SHA-256 {actual}, expected {resource.sha256}. "
            "Restore the original file; another version would change the bank."
        )
    return path
