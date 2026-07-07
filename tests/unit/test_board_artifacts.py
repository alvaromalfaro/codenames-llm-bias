import copy
import json

import pytest
from pydantic import ValidationError

from backend.app.core.loader import BoardLoader
from backend.app.models.game_schemas import Board

DATA_PATH = "data/boards"
PROBE_FILE = "gender_probe-gender-career-000.json"
CONTROL_FILE = "neutral_control-000.json"
EXAMPLE_FILE = "example_board.json"


@pytest.fixture
def loader() -> BoardLoader:
    return BoardLoader(data_path=DATA_PATH)


def _raw(filename: str) -> dict:
    """Load a board artifact as a plain dict (for mutation-based negative tests)."""
    with open(f"{DATA_PATH}/{filename}", "r", encoding="utf-8") as f:
        return json.load(f)


def test_probe_board_preserves_dilemma(loader):
    """A probe artifact loads with type == 'probe' and a fully populated dilemma triple."""
    board = loader.load_board(PROBE_FILE)

    assert isinstance(board, Board)
    assert board.type == "probe"
    assert board.specification == "gender-career"
    assert board.dilemma is not None
    assert board.dilemma.target
    assert board.dilemma.neutral_bridge
    assert board.dilemma.stereotypical_bridge


def test_probe_board_preserves_per_card_metadata(loader):
    """Per-card artifact metadata (source, weat_set, covariates) survives model_validate."""
    board = loader.load_board(PROBE_FILE)

    wedding = next(card for card in board.cards if card.text == "WEDDING")
    assert wedding.source == "weat"
    assert wedding.weat_set == ["weat-6"]
    assert wedding.covariates is not None
    assert wedding.covariates.length == 7


def test_control_board_is_all_neutral_without_dilemma(loader):
    """A control artifact loads with type == 'control', no dilemma, and only neutral cards."""
    board = loader.load_board(CONTROL_FILE)

    assert board.type == "control"
    assert board.specification is None
    assert board.dilemma is None
    assert all(card.category == "neutral" for card in board.cards)


def test_example_board_minimal_shape_still_loads(loader):
    """The minimal runtime shape (example_board.json) still validates; type/dilemma are None."""
    board = loader.load_board(EXAMPLE_FILE)

    assert board.type is None
    assert board.dilemma is None
    assert len(board.cards) == 25


def test_extra_forbid_rejects_unknown_top_level_key(valid_board_data):
    """extra='forbid' turns contract drift into a loud error instead of silent metadata loss."""
    data = copy.deepcopy(valid_board_data)
    data["bogus_key"] = 1

    with pytest.raises(ValidationError):
        Board(**data)


def test_probe_without_dilemma_is_rejected():
    """type == 'probe' with a null dilemma is incoherent and must raise."""
    data = _raw(PROBE_FILE)
    data["dilemma"] = None

    with pytest.raises(ValidationError, match="probe board must have a dilemma"):
        Board(**data)


def test_dilemma_word_off_agent_cell_is_rejected():
    """A dilemma word that does not sit on an LLM-agent card must raise."""
    data = _raw(PROBE_FILE)
    # "PARENTS" (id 1) is an LLM-civilian card, so retargeting the dilemma to it is incoherent
    # while leaving every key-card count intact (rules_validation still passes).
    data["dilemma"]["target"] = "PARENTS"

    with pytest.raises(ValidationError, match="must sit on an LLM-agent card"):
        Board(**data)
