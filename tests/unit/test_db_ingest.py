"""Tests for board-artifact ingestion.

The pure-mapping tests run without any database. The idempotency test requires a live
Postgres and is skipped when DATABASE_URL is unset.
"""

import json
import os
from pathlib import Path

import pytest

from backend.app.db.ingest_boards import board_artifact_to_orm, ingest_boards_if_absent

_BOARDS_DIR = Path(__file__).resolve().parents[2] / "data" / "boards"
PROBE_FILE = _BOARDS_DIR / "gender_probe-gender-career-000.json"
CONTROL_FILE = _BOARDS_DIR / "neutral_control-000.json"


def _load(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_probe_mapping_flattens_covariates_and_weat():
    data = _load(PROBE_FILE)
    board, cards = board_artifact_to_orm(data)

    assert board.board_id == data["board_id"]
    assert board.type == "probe"
    assert board.measurement_frame_id is None
    assert board.grid_rows == data["grid"]["rows"] == 5
    assert board.grid_cols == data["grid"]["cols"] == 5
    assert board.dilemma is not None  # probe boards carry a dilemma
    assert board.arbiters == data["arbiters"]
    assert board.keycard_audit == data["keycard_audit"]

    assert len(cards) == 25
    by_text = {c.text: c for c in cards}

    # Covariate flattening against the artifact's own nested values.
    for src in data["cards"]:
        card = next(c for c in cards if c.card_id == src["id"])
        cov = src["covariates"]
        assert card.subtlex_freq == cov["subtlex_freq"]
        assert card.length == cov["length"]
        assert card.wordnet_polysemy == cov["wordnet_polysemy"]
        assert card.board_id == board.board_id

    # weat_set mapped to the array column (non-empty and empty cases).
    weat_card = next(c for c in cards if c.weat_set)
    assert isinstance(weat_card.weat_set, list)
    assert weat_card.weat_set == next(
        s["weat_set"] for s in data["cards"] if s["text"] == weat_card.text
    )
    neutral_card = next(c for c in cards if c.category == "neutral")
    assert by_text[neutral_card.text].weat_set == []


def test_control_mapping_has_null_dilemma_and_25_cards():
    data = _load(CONTROL_FILE)
    board, cards = board_artifact_to_orm(data)

    assert board.type == "control"
    assert board.specification is None
    assert board.dilemma is None  # control boards have no dilemma
    assert len(cards) == 25
    assert all(c.board_id == board.board_id for c in cards)
    assert {c.card_id for c in cards} == set(range(25))


def test_missing_covariates_are_nullable():
    """A card without covariates maps to NULL columns rather than raising."""
    board, cards = board_artifact_to_orm(
        {
            "board_id": "synthetic-000",
            "type": "control",
            "grid": {"rows": 5, "cols": 5},
            "cards": [
                {
                    "id": 0,
                    "text": "WORD",
                    "llm_perspective_role": "civilian",
                    "human_perspective_role": "civilian",
                }
            ],
        }
    )
    (card,) = cards
    assert card.subtlex_freq is None
    assert card.length is None
    assert card.wordnet_polysemy is None
    assert card.weat_set == []


@pytest.mark.skipif(
    not os.environ.get("DATABASE_URL"), reason="requires a live Postgres database"
)
def test_ingest_is_idempotent():
    from backend.app.db.session import session_scope

    with session_scope() as session:
        first = ingest_boards_if_absent(session, _BOARDS_DIR)
    with session_scope() as session:
        second = ingest_boards_if_absent(session, _BOARDS_DIR)
    assert first >= 0
    assert second == 0  # nothing new the second time
