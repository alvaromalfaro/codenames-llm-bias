from pydantic import ValidationError
import pytest
import re
from backend.app.models.game_schemas import Board, CardRole


def test_valid_board_passes(valid_board_data):
    """
    Validates that a correctly structured board configuration passes the validation without errors.

    :param valid_board_data: A fixture providing a valid board configuration as a dictionary.
    """
    board = Board(**valid_board_data)

    assert isinstance(board, Board)
    assert board.board_id == "test_board_001"
    assert board.category == "neutral"
    assert len(board.cards) == 25
    # Extract the card data from the valid_board_data for comparison
    valid_cards = valid_board_data["cards"]
    for i, card in enumerate(board.cards):
        assert card.id == i
        assert card.text == valid_cards[i]["text"]
        assert card.llm_perspective_role == valid_cards[i]["llm_perspective_role"]
        assert card.human_perspective_role == valid_cards[i]["human_perspective_role"]


def test_measurement_frame_id_absent_defaults_to_none(valid_board_data):
    """Today's unsealed artifacts carry no ``measurement_frame_id`` key and must still
    validate (regression guard), resolving the optional field to ``None``.
    """
    board = Board(**valid_board_data)

    assert board.measurement_frame_id is None


def test_measurement_frame_id_present_is_carried(valid_board_data):
    """A future sealed artifact carrying ``measurement_frame_id`` validates and keeps the
    value (the new capability).
    """
    data = {**valid_board_data, "measurement_frame_id": "test-frame-abc"}

    board = Board(**data)

    assert board.measurement_frame_id == "test-frame-abc"


@pytest.mark.parametrize("modification, expected_error", [
    ("less_llm_agents", "There must be exactly 9 agent cards for both LLM and human players (3 shared between them)."),
    ("less_human_agents", "There must be exactly 9 agent cards for both LLM and human players (3 shared between them)."),
    ("wrong_shared_count", "There must be exactly 9 agent cards for both LLM and human players (3 shared between them).")
])
def test_board_invalid_agent_rules(valid_board_data, modification, expected_error):
    """
    Validates that a board configuration with an incorrect number of agent cards or incorrect intersections fails validation.

    :param valid_board_data: A fixture providing a valid board configuration as a dictionary.
    :param modification: A string indicating the type of modification to apply to the valid board data to make it invalid.
    :param expected_error: A string indicating the expected error message to be raised during validation.
    """
    data = valid_board_data.copy()

    if modification == "less_llm_agents":
        # Modify the valid board data to have fewer LLM agent cards than required
        for card in data["cards"]:
            if card["llm_perspective_role"] == CardRole.AGENT:
                card["llm_perspective_role"] = CardRole.CIVILIAN
                break
    elif modification == "less_human_agents":
        # Modify the valid board data to have fewer human agent cards than required
        for card in data["cards"]:
            if card["human_perspective_role"] == CardRole.AGENT:
                card["human_perspective_role"] = CardRole.CIVILIAN
                break
    elif modification == "wrong_shared_count":
        # Modify the valid board data to have an incorrect number of shared agent cards

        # First, set one of the shared agent cards to be a civilian for the human player
        for card in data["cards"]:
            if card["llm_perspective_role"] == CardRole.AGENT and card["human_perspective_role"] == CardRole.AGENT:
                card["human_perspective_role"] = CardRole.CIVILIAN
                break

        # Then, set one of the civilian cards to be an agent for the human player to maintain the total count of 9 agents for the human player
        for card in data["cards"]:
            if card["llm_perspective_role"] == CardRole.CIVILIAN and card["human_perspective_role"] == CardRole.CIVILIAN:
                card["human_perspective_role"] = CardRole.AGENT
                break

    with pytest.raises(ValidationError, match=re.escape(expected_error)):
        Board(**data)


@pytest.mark.parametrize("modification, expected_error", [
    ("less_llm_assassins", "There must be exactly 3 assassin cards (1 shared between LLM and human players, 1 unique to LLM, 1 unique to human)."),
    ("less_human_assassins", "There must be exactly 3 assassin cards (1 shared between LLM and human players, 1 unique to LLM, 1 unique to human)."),
    ("wrong_shared_count", "There must be exactly 3 assassin cards (1 shared between LLM and human players, 1 unique to LLM, 1 unique to human)."),
    ("invalid_human_intersection",
     "One of the human's assassin cards must be one of the LLM's agent cards."),
    ("invalid_llm_intersection",
     "One of the LLM's assassin cards must be one of the human's agent cards.")
])
def test_board_invalid_assassin_rules(valid_board_data, modification, expected_error):
    """
    Validates that a board configuration with an incorrect number of assassin cards or incorrect intersections fails validation.

    :param valid_board_data: A fixture providing a valid board configuration as a dictionary.
    :param modification: A string indicating the type of modification to apply to the valid board data to make it invalid.
    :param expected_error: A string indicating the expected error message to be raised during validation.
    """
    data = valid_board_data.copy()

    if modification == "less_llm_assassins":
        # Modify the valid board data to have fewer LLM assassin cards than required
        for card in data["cards"]:
            if card["llm_perspective_role"] == CardRole.ASSASSIN:
                card["llm_perspective_role"] = CardRole.CIVILIAN
                break
    elif modification == "less_human_assassins":
        # Modify the valid board data to have fewer human assassin cards than required
        for card in data["cards"]:
            if card["human_perspective_role"] == CardRole.ASSASSIN:
                card["human_perspective_role"] = CardRole.CIVILIAN
                break
    elif modification == "wrong_shared_count":
        # Modify the valid board data to have an incorrect number of shared assassin cards

        # First, set the shared assassin card to be a civilian for the human player
        for card in data["cards"]:
            if card["llm_perspective_role"] == CardRole.ASSASSIN and card["human_perspective_role"] == CardRole.ASSASSIN:
                card["human_perspective_role"] = CardRole.CIVILIAN
                break

        # Then, set one of the civilian cards to be an assassin for the human player to maintain the total count of 3 assassins for the human player
        for card in data["cards"]:
            if card["llm_perspective_role"] == CardRole.CIVILIAN and card["human_perspective_role"] == CardRole.CIVILIAN:
                card["human_perspective_role"] = CardRole.ASSASSIN
                break
    elif modification == "invalid_human_intersection":
        # Modify the valid board data to have a human assassin card that is not one of the LLM's agent cards

        # First, set the human assassin card that intersects with the LLM's agent cards to be a civilian for the human player
        for card in data["cards"]:
            if card["human_perspective_role"] == CardRole.ASSASSIN and card["llm_perspective_role"] == CardRole.AGENT:
                card["human_perspective_role"] = CardRole.CIVILIAN
                break

        # Then, set one of the civilian cards to be an assassin for the human player to maintain the total count of 3 assassins for the human player
        for card in data["cards"]:
            if card["human_perspective_role"] == CardRole.CIVILIAN and card["llm_perspective_role"] == CardRole.CIVILIAN:
                card["human_perspective_role"] = CardRole.ASSASSIN
                break
    elif modification == "invalid_llm_intersection":
        # Modify the valid board data to have an LLM assassin card that is not one of the human's agent cards
        for card in data["cards"]:
            if card["llm_perspective_role"] == CardRole.ASSASSIN and card["human_perspective_role"] == CardRole.AGENT:
                card["llm_perspective_role"] = CardRole.CIVILIAN
                break

        # Then, set one of the civilian cards to be an assassin for the LLM player to maintain the total count of 3 assassins for the LLM player
        for card in data["cards"]:
            if card["llm_perspective_role"] == CardRole.CIVILIAN and card["human_perspective_role"] == CardRole.CIVILIAN:
                card["llm_perspective_role"] = CardRole.ASSASSIN
                break

    with pytest.raises(ValidationError, match=re.escape(expected_error)):
        Board(**data)
