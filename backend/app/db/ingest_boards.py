"""Idempotent ingestion of board artifacts into the persistence layer.

Split into a pure mapping function (no DB dependency, unit-testable) and a commit function that 
skips boards already present. ``measurement_frame_id`` is left NULL for now - frame association 
belongs to the separate measurement frontend (deferred).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.db.models import BoardModel, WordCardModel

logger = logging.getLogger(__name__)


def board_artifact_to_orm(
    data: dict,
) -> tuple[BoardModel, list[WordCardModel]]:
    """Map a parsed board artifact JSON into ORM instances.

    Pure: constructs (but does not persist) a ``BoardModel`` and its ``WordCardModel`` rows. 
    Covariates are flattened into ``subtlex_freq``/``length``/``wordnet_polysemy`` and ``weat_set`` 
    is mapped to the array column. ``dilemma`` stays ``None`` for control boards. 
    ``measurement_frame_id`` is left ``None``.
    """
    grid = data.get("grid") or {}
    board = BoardModel(
        board_id=data["board_id"],
        measurement_frame_id=None,
        type=data.get("type"),
        category=data.get("category"),
        specification=data.get("specification"),
        seed=data.get("seed"),
        grid_rows=grid.get("rows"),
        grid_cols=grid.get("cols"),
        arbiters=data.get("arbiters"),
        dilemma=data.get("dilemma"),
        keycard_audit=data.get("keycard_audit"),
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


def ingest_boards_if_absent(
    session: Session, data_path: str | Path = "data/boards"
) -> int:
    """Ingest every ``*.json`` board under ``data_path`` that is not already stored."""
    directory = Path(data_path)
    if not directory.exists():
        logger.warning(
            "Board data directory %s does not exist; skipping ingestion", directory)
        return 0

    inserted = 0
    for file_path in sorted(directory.glob("*.json")):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "Skipping unreadable board file %s: %s", file_path, exc)
            continue

        board_id = data.get("board_id") if isinstance(data, dict) else None
        if not board_id:
            logger.warning(
                "Skipping non-board file %s (no board_id)", file_path)
            continue

        exists = session.execute(
            select(BoardModel.board_id).where(BoardModel.board_id == board_id)
        ).first()
        if exists is not None:
            continue

        try:
            board, cards = board_artifact_to_orm(data)
        except (KeyError, TypeError) as exc:
            logger.warning(
                "Skipping malformed board file %s: %s", file_path, exc)
            continue

        session.add(board)
        session.add_all(cards)
        session.commit()
        inserted += 1

    return inserted
