"""Tests for board-artifact ingestion.

The pure-mapping tests run without any database. The tests marked ``db`` run on the test
session's Postgres (tests/conftest.py).
"""

import json
from pathlib import Path

import pytest

from backend.app.db.ingest_boards import (
    InvalidBoardFileError,
    StaleBoardError,
    board_artifact_to_orm,
    board_content_sha256,
    ingest_boards_if_absent,
)
from backend.app.db.models import BoardModel

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
    # The sealed artifact's measurement_frame_id (A.4) flows through unchanged onto the ORM model.
    assert board.measurement_frame_id == data.get("measurement_frame_id")
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


def _minimal_artifact(**overrides: object) -> dict:
    """A minimal control artifact for exercising the pure mapping in isolation."""
    return {
        "board_id": "synthetic-frame-000",
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
        **overrides,
    }


def test_measurement_frame_id_absent_maps_to_none():
    """An artifact without the key yields ``None`` on the ORM model (unchanged today)."""
    board, _ = board_artifact_to_orm(_minimal_artifact())

    assert board.measurement_frame_id is None


def test_measurement_frame_id_present_is_carried():
    """A sealed artifact's ``measurement_frame_id`` flows through onto the ORM model."""
    board, _ = board_artifact_to_orm(_minimal_artifact(measurement_frame_id="test-frame-abc"))

    assert board.measurement_frame_id == "test-frame-abc"


@pytest.mark.db
def test_ingest_is_idempotent():
    from backend.app.db.ingest_frame import ingest_frame_if_absent
    from backend.app.db.session import session_scope

    # The sealed boards (A.4) carry a measurement_frame_id whose FK is enforced, so the frame row
    # must exist first (the real startup order, A.5).
    with session_scope() as session:
        ingest_frame_if_absent(session, _BOARDS_DIR / "measurement_frame.json")
    with session_scope() as session:
        first = ingest_boards_if_absent(session, _BOARDS_DIR)
    with session_scope() as session:
        second = ingest_boards_if_absent(session, _BOARDS_DIR)
    assert first >= 0
    assert second == 0  # nothing new the second time


# content hash: a board whose file changed under the same board_id
def test_the_content_hash_ignores_layout_and_follows_content():
    data = _load(CONTROL_FILE)
    reordered = json.loads(json.dumps(data, indent=4, sort_keys=False))
    reordered = dict(reversed(list(reordered.items())))
    assert board_content_sha256(reordered) == board_content_sha256(data)

    changed = json.loads(json.dumps(data))
    changed["cards"][0]["text"] = "ELSEWHERE"
    assert board_content_sha256(changed) != board_content_sha256(data)


def test_the_mapped_board_carries_its_content_hash():
    data = _load(PROBE_FILE)
    board, _ = board_artifact_to_orm(data)
    assert board.content_sha256 == board_content_sha256(data)


class _FakeSession:
    """Just enough of a Session for the ingestion: one stored board (and its cards) by id."""

    def __init__(self, stored=None, cards=()):
        self.stored, self.cards, self.added = stored, list(cards), []

    def get(self, model, key):
        assert model is BoardModel
        return self.stored if self.stored is not None and self.stored.board_id == key else None

    def execute(self, statement):
        cards = self.cards

        class _Result:
            def scalars(self):
                return iter(cards)

        return _Result()

    def add(self, row):
        self.added.append(row)

    def add_all(self, rows):
        self.added.extend(rows)

    def flush(self):
        pass


def _bank(tmp_path, data, name="board.json"):
    (tmp_path / name).write_text(json.dumps(data), encoding="utf-8")
    return tmp_path


def test_a_new_board_is_stored_with_its_hash(tmp_path):
    data = _load(CONTROL_FILE)
    session = _FakeSession()
    assert ingest_boards_if_absent(session, _bank(tmp_path, data)) == 1
    board = next(row for row in session.added if isinstance(row, BoardModel))
    assert board.content_sha256 == board_content_sha256(data)


def test_a_stored_board_with_the_same_hash_is_left_alone(tmp_path):
    data = _load(CONTROL_FILE)
    stored, _ = board_artifact_to_orm(data)
    session = _FakeSession(stored)
    assert ingest_boards_if_absent(session, _bank(tmp_path, data)) == 0
    assert session.added == []


def test_a_stored_board_whose_file_changed_stops_the_ingestion(tmp_path):
    data = _load(CONTROL_FILE)
    stored, _ = board_artifact_to_orm(data)
    data["cards"][0]["text"] = "ELSEWHERE"
    with pytest.raises(StaleBoardError, match=data["board_id"]):
        ingest_boards_if_absent(_FakeSession(stored), _bank(tmp_path, data))


def test_a_board_stored_without_a_hash_gets_it_when_its_file_agrees(tmp_path):
    data = _load(PROBE_FILE)
    stored, cards = board_artifact_to_orm(data)
    stored.content_sha256 = None
    ingest_boards_if_absent(_FakeSession(stored, cards), _bank(tmp_path, data))
    assert stored.content_sha256 == board_content_sha256(data)


@pytest.mark.parametrize(
    "change, where",
    [
        (lambda d: d["cards"][4].update(text="ELSEWHERE"), "card 4.text"),  # not a dilemma word
        (lambda d: d["cards"][7]["covariates"].update(length=99), "card 7.length"),
        (lambda d: d.update(category="another"), "board.category"),
    ],
)
def test_a_board_stored_without_a_hash_that_differs_stops_the_ingestion(tmp_path, change, where):
    data = _load(PROBE_FILE)
    stored, cards = board_artifact_to_orm(data)
    stored.content_sha256 = None
    change(data)
    with pytest.raises(StaleBoardError, match=where):
        ingest_boards_if_absent(_FakeSession(stored, cards), _bank(tmp_path, data))
    assert stored.content_sha256 is None


@pytest.mark.parametrize(
    "content",
    [
        "not json",
        json.dumps({"no": "board_id"}),
        json.dumps({**_load(CONTROL_FILE), "cards": []}),
    ],
)
def test_a_file_of_the_bank_that_is_not_a_board_stops_the_ingestion(tmp_path, content):
    (tmp_path / "bad.json").write_text(content, encoding="utf-8")
    with pytest.raises(InvalidBoardFileError, match="bad.json"):
        ingest_boards_if_absent(_FakeSession(), tmp_path)


def test_the_sidecars_are_not_boards(tmp_path):
    for name in ("balance_report.json", "measurement_frame.json"):
        (tmp_path / name).write_text("{}", encoding="utf-8")
    assert ingest_boards_if_absent(_FakeSession(), tmp_path) == 0


@pytest.mark.db
def test_a_board_stored_before_the_hash_is_checked_against_its_file(tmp_path):
    """Against the real schema: a board stored with a NULL hash gets the hash of a file that agrees
    with it, and a changed file is refused."""
    import uuid

    from sqlalchemy import update

    from backend.app.db.ingest_frame import ingest_frame_if_absent
    from backend.app.db.session import session_scope

    data = {**_load(CONTROL_FILE), "board_id": f"test-hash-{uuid.uuid4()}"}
    bank = _bank(tmp_path, data)
    with session_scope() as session:
        ingest_frame_if_absent(session, _BOARDS_DIR / "measurement_frame.json")
    try:
        with session_scope() as session:
            assert ingest_boards_if_absent(session, bank) == 1
            session.execute(
                update(BoardModel)
                .where(BoardModel.board_id == data["board_id"])
                .values(content_sha256=None)
            )
        with session_scope() as session:
            assert ingest_boards_if_absent(session, bank) == 0
        with session_scope() as session:
            assert session.get(BoardModel, data["board_id"]).content_sha256 == board_content_sha256(
                data
            )

        data["cards"][0]["text"] = "ELSEWHERE"
        with pytest.raises(StaleBoardError):
            with session_scope() as session:
                ingest_boards_if_absent(session, _bank(tmp_path, data))
    finally:
        with session_scope() as session:
            session.query(BoardModel).filter_by(board_id=data["board_id"]).delete()
