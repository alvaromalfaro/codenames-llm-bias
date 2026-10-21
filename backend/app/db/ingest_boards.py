"""Idempotent ingestion of board artifacts into the persistence layer.

Split into a pure mapping function (no DB dependency, unit-testable) and an ingestion function that
stores the boards not yet stored and checks the ones that are. Every ``*.json`` of the bank but its
two sidecars must be a valid board (``Board``, as the loader reads it): anything else raises
:class:`InvalidBoardFileError`, so the database never holds a board the games could not play.

A board is identified by its ``board_id``, and its content by ``content_sha256``, the SHA-256 of its
canonical JSON (:func:`board_content_sha256`). A stored board whose hash is not its file's raises
:class:`StaleBoardError`: the games would play the file while the database kept the old board, so
the two are reconciled by hand, never overwritten. A board stored before the hash was recorded
(NULL) is compared with its file column by column instead, and gets the hash when they agree.

The caller owns the transaction: nothing here commits.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.core.loader import BANK_SIDECARS, InvalidBoardFileError
from backend.app.db.models import BoardModel, WordCardModel
from backend.app.models.game_schemas import Board
from backend.app.paths import BOARDS_DIR

logger = logging.getLogger(__name__)

# The columns a stored row has that ingestion does not take from the file.
_NOT_FROM_FILE = frozenset({"id", "created_at", "content_sha256"})

__all__ = [
    "InvalidBoardFileError",
    "StaleBoardError",
    "board_artifact_to_orm",
    "board_content_sha256",
    "ingest_boards_if_absent",
]


class StaleBoardError(RuntimeError):
    """A stored board disagrees with its file under the same ``board_id`` - reconcile, never
    overwrite."""


def board_content_sha256(data: dict) -> str:
    """The SHA-256 of a board artifact's canonical JSON: sorted keys, compact separators, UTF-8. Two
    files that parse to the same artifact hash alike, whatever their whitespace or key order."""
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def board_artifact_to_orm(
    data: dict,
) -> tuple[BoardModel, list[WordCardModel]]:
    """Map a parsed board artifact JSON into ORM instances.

    Pure: constructs (but does not persist) a ``BoardModel`` and its ``WordCardModel`` rows.
    Covariates are flattened into ``subtlex_freq``/``length``/``wordnet_polysemy`` and ``weat_set``
    is mapped to the array column. ``dilemma`` stays ``None`` for control boards.
    ``measurement_frame_id`` is read from the artifact, falling back to ``None`` when the
    key is absent (today's unsealed artifacts). ``content_sha256`` is the artifact's hash.
    """
    grid = data.get("grid") or {}
    board = BoardModel(
        board_id=data["board_id"],
        measurement_frame_id=data.get("measurement_frame_id"),
        type=data.get("type"),
        category=data.get("category"),
        specification=data.get("specification"),
        seed=data.get("seed"),
        grid_rows=grid.get("rows"),
        grid_cols=grid.get("cols"),
        arbiters=data.get("arbiters"),
        dilemma=data.get("dilemma"),
        keycard_audit=data.get("keycard_audit"),
        content_sha256=board_content_sha256(data),
    )

    cards: list[WordCardModel] = []
    for card in data.get("cards", []):
        covariates = card.get("covariates") or {}
        cards.append(
            WordCardModel(
                board_id=board.board_id,
                card_id=card["id"],
                text=card["text"],
                llm_perspective_role=card["llm_perspective_role"],
                human_perspective_role=card["human_perspective_role"],
                category=card.get("category"),
                source=card.get("source"),
                weat_set=card.get("weat_set") or [],
                subtlex_freq=covariates.get("subtlex_freq"),
                length=covariates.get("length"),
                wordnet_polysemy=covariates.get("wordnet_polysemy"),
            )
        )
    return board, cards


def _read_board_file(path: Path) -> dict:
    """The parsed artifact of a bank file, once it validates as a ``Board``."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        Board.model_validate(data)
    except (OSError, ValueError) as exc:  # JSONDecodeError and ValidationError are ValueErrors
        raise InvalidBoardFileError(f"{path} is not a valid board: {exc}") from exc
    return data


def _from_file(row) -> dict:
    """The values of a row that ingestion takes from the artifact, by column."""
    return {
        c.key: getattr(row, c.key) for c in row.__table__.columns if c.key not in _NOT_FROM_FILE
    }


def _differences(session: Session, stored: BoardModel, data: dict) -> list[str]:
    """Where the stored board and its cards differ from the artifact ``data``."""
    board, cards = board_artifact_to_orm(data)
    diffs = [
        f"board.{key}" for key, value in _from_file(board).items() if getattr(stored, key) != value
    ]
    stored_cards = {
        card.card_id: card
        for card in session.execute(
            select(WordCardModel).where(WordCardModel.board_id == stored.board_id)
        ).scalars()
    }
    if set(stored_cards) != {card.card_id for card in cards}:
        return diffs + ["the card ids"]
    for card in cards:
        diffs += [
            f"card {card.card_id}.{key}"
            for key, value in _from_file(card).items()
            if getattr(stored_cards[card.card_id], key) != value
        ]
    return diffs


def ingest_boards_if_absent(session: Session, data_path: str | Path = BOARDS_DIR) -> int:
    """Store every board under ``data_path`` not stored yet, and check those that are; return how
    many were stored.

    :raises InvalidBoardFileError: a file of the bank is not a valid board.
    :raises StaleBoardError: a stored board is not the one its file holds now.
    """
    directory = Path(data_path)
    if not directory.exists():
        logger.warning("Board data directory %s does not exist; skipping ingestion", directory)
        return 0

    inserted = 0
    for file_path in sorted(directory.glob("*.json")):
        if file_path.name in BANK_SIDECARS:
            continue
        data = _read_board_file(file_path)
        board_id, digest = data["board_id"], board_content_sha256(data)
        stored = session.get(BoardModel, board_id)
        if stored is None:
            board, cards = board_artifact_to_orm(data)
            session.add(board)
            session.add_all(cards)
            session.flush()
            inserted += 1
        elif stored.content_sha256 is None:
            diffs = _differences(session, stored, data)
            if diffs:
                raise StaleBoardError(
                    f"board {board_id} is stored with other content than {file_path} holds now "
                    f"(differences: {diffs[:5]}{' ...' if len(diffs) > 5 else ''})."
                )
            stored.content_sha256 = digest
            logger.info("board %s: recorded the content hash of its stored version", board_id)
        elif stored.content_sha256 != digest:
            raise StaleBoardError(
                f"board {board_id} is stored with content {stored.content_sha256[:12]}..., but "
                f"{file_path} now hashes to {digest[:12]}...: the games would play the file while "
                f"the database keeps the stored board."
            )
    return inserted
