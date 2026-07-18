"""Seal each bank board artifact with its ``measurement_frame_id``.

Writes a single top-level key ``measurement_frame_id`` (the frozen frame's id) into every real bank
board (the files under ``data/boards/`` carrying an ``arbiters`` block) associating the bank with
the measurement frame.

This is inert with respect to identity: ``board_id`` is a manifest string and the per-board seed is
``sha256(f"{master_seed}:{board_id}")``, neither reads the serialized JSON, so adding a top-level
key cannot shift any board_id or derived seed. It touches no DB and enables no CHECK.

The binding safety guarantee is structural, not textual: after writing, the re-parsed file must
equal ``{**parsed_before, "measurement_frame_id": frame_id}`` exactly (deep equality on the parsed
dicts). The new key is appended and the rest is re-serialized with the board writer's own settings,
so in practice the textual diff is the one added key, but parsed-equality is what must hold. The
candidate payload is validated in memory before writing, so a failure never corrupts a file.

Idempotent: a board already sealed with the SAME frame_id is a no-op (not rewritten). A board
carrying a different ``measurement_frame_id`` raises ``StaleSealError`` (a stale seal from an
earlier frame must be reconciled, never silently overwritten).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from board_generator.board import DEFAULT_OUTPUT_DIR
from board_generator.emit_frame import FRAME_FILENAME

SEAL_KEY = "measurement_frame_id"


class StaleSealError(RuntimeError):
    """A board already carries a DIFFERENT measurement_frame_id: reconcile, do not overwrite."""


@dataclass
class SealResult:
    """Outcome of a sealing pass: filenames grouped by what happened to each."""

    sealed: list[str]
    already_sealed: list[str]
    skipped: list[str]


def read_frame_id(sidecar_path: Path) -> str:
    """Read the target ``frame_id`` from the measurement-frame sidecar."""
    return str(json.loads(sidecar_path.read_text(encoding="utf-8"))["frame_id"])


def seal_one(path: Path, frame_id: str) -> str:
    """Seal one artifact. Returns ``"sealed"`` | ``"already"`` | ``"skipped"``.

    Skips a file with no ``arbiters`` block (not a bank board). Raises ``StaleSealError`` if the
    file already carries a different frame_id. Enforces parsed-equality before and after the write.
    """
    before: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if "arbiters" not in before:
        return "skipped"

    existing = before.get(SEAL_KEY)
    if existing is not None:
        if existing != frame_id:
            raise StaleSealError(
                f"{path.name} carries {SEAL_KEY}={existing!r}, not the target {frame_id!r}"
            )
        return "already"

    after = {**before, SEAL_KEY: frame_id}
    payload = json.dumps(after, indent=2, ensure_ascii=False) + "\n"
    # Validate the candidate in memory before touching disk.
    if json.loads(payload) != after:
        raise AssertionError(
            f"pre-write parsed-equality failed for {path.name}")

    path.write_text(payload, encoding="utf-8")

    # Binding on-disk guarantee: re-parsed file == parsed_before + exactly the one new key.
    reparsed = json.loads(path.read_text(encoding="utf-8"))
    if reparsed != {**before, SEAL_KEY: frame_id}:
        raise AssertionError(
            f"parsed-equality failed after writing {path.name}")
    return "sealed"


def seal_board_dir(
    boards_dir: Path = DEFAULT_OUTPUT_DIR,
    frame_id: str | None = None,
    *,
    sidecar_name: str = FRAME_FILENAME,
) -> SealResult:
    """Seal every bank board in ``boards_dir``. The sidecar itself is always skipped by name.

    ``frame_id`` defaults to the id read from ``boards_dir/sidecar_name``. Stops on the first
    ``StaleSealError`` (does not seal the rest).
    """
    if frame_id is None:
        frame_id = read_frame_id(boards_dir / sidecar_name)

    result = SealResult(sealed=[], already_sealed=[], skipped=[])
    for path in sorted(boards_dir.glob("*.json")):
        if path.name == sidecar_name:
            result.skipped.append(path.name)
            continue
        outcome = seal_one(path, frame_id)
        if outcome == "sealed":
            result.sealed.append(path.name)
        elif outcome == "already":
            result.already_sealed.append(path.name)
        else:
            result.skipped.append(path.name)
    return result
