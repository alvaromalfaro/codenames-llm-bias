import pytest
from backend.app.models.game_schemas import CardRole
from backend.app.models.llm_schemas import LLMRequest, LLMMessage


@pytest.fixture
def valid_board_data():
    """
    Provides a valid board configuration for testing the Board schema validation. It is based on the
    board configuration used in the official Codenames Duet rulesbook.
    """
    return {
        "board_id": "test_board_001",
        "category": "neutral",
        "cards": _get_cards()
    }


def _get_cards():
    return [
        {
            "id": 0,
            "text": "BUCKET",
            "human_perspective_role": CardRole.CIVILIAN,
            "llm_perspective_role": CardRole.CIVILIAN,
            "category": "neutral"
        },
        {
            "id": 1,
            "text": "BRICK",
            "human_perspective_role": CardRole.AGENT,
            "llm_perspective_role": CardRole.AGENT,
            "category": "neutral"
        },
        {
            "id": 2,
            "text": "ANT",
            "human_perspective_role": CardRole.AGENT,
            "llm_perspective_role": CardRole.ASSASSIN,
            "category": "neutral"
        },
        {
            "id": 3,
            "text": "LEMONADE",
            "human_perspective_role": CardRole.ASSASSIN,
            "llm_perspective_role": CardRole.ASSASSIN,
            "category": "neutral"
        },
        {
            "id": 4,
            "text": "RUSSIA",
            "human_perspective_role": CardRole.CIVILIAN,
            "llm_perspective_role": CardRole.AGENT,
            "category": "neutral"
        },
        {
            "id": 5,
            "text": "CAVE",
            "human_perspective_role": CardRole.AGENT,
            "llm_perspective_role": CardRole.CIVILIAN,
            "category": "neutral"
        },
        {
            "id": 6,
            "text": "FIDDLE",
            "human_perspective_role": CardRole.CIVILIAN,
            "llm_perspective_role": CardRole.CIVILIAN,
            "category": "neutral"
        },
        {
            "id": 7,
            "text": "VAMPIRE",
            "human_perspective_role": CardRole.CIVILIAN,
            "llm_perspective_role": CardRole.CIVILIAN,
            "category": "neutral"
        },
        {
            "id": 8,
            "text": "TATTOO",
            "human_perspective_role": CardRole.AGENT,
            "llm_perspective_role": CardRole.AGENT,
            "category": "neutral"
        },
        {
            "id": 9,
            "text": "RANCH",
            "human_perspective_role": CardRole.AGENT,
            "llm_perspective_role": CardRole.CIVILIAN,
            "category": "neutral"
        },
        {
            "id": 10,
            "text": "LOCUST",
            "human_perspective_role": CardRole.ASSASSIN,
            "llm_perspective_role": CardRole.CIVILIAN,
            "category": "neutral"
        },
        {
            "id": 11,
            "text": "RIFLE",
            "human_perspective_role": CardRole.CIVILIAN,
            "llm_perspective_role": CardRole.AGENT,
            "category": "neutral"
        },
        {
            "id": 12,
            "text": "VIRUS",
            "human_perspective_role": CardRole.CIVILIAN,
            "llm_perspective_role": CardRole.AGENT,
            "category": "neutral"
        },
        {
            "id": 13,
            "text": "IGLOO",
            "human_perspective_role": CardRole.CIVILIAN,
            "llm_perspective_role": CardRole.CIVILIAN,
            "category": "neutral"
        },
        {
            "id": 14,
            "text": "MAKEUP",
            "human_perspective_role": CardRole.CIVILIAN,
            "llm_perspective_role": CardRole.ASSASSIN,
            "category": "neutral"
        },
        {
            "id": 15,
            "text": "POTTER",
            "human_perspective_role": CardRole.CIVILIAN,
            "llm_perspective_role": CardRole.AGENT,
            "category": "neutral"
        },
        {
            "id": 16,
            "text": "CAESAR",
            "human_perspective_role": CardRole.AGENT,
            "llm_perspective_role": CardRole.CIVILIAN,
            "category": "neutral"
        },
        {
            "id": 17,
            "text": "NAPOLEON",
            "human_perspective_role": CardRole.AGENT,
            "llm_perspective_role": CardRole.AGENT,
            "category": "neutral"
        },
        {
            "id": 18,
            "text": "GOLF",
            "human_perspective_role": CardRole.CIVILIAN,
            "llm_perspective_role": CardRole.CIVILIAN,
            "category": "neutral"
        },
        {
            "id": 19,
            "text": "PINE",
            "human_perspective_role": CardRole.ASSASSIN,
            "llm_perspective_role": CardRole.AGENT,
            "category": "neutral"
        },
        {
            "id": 20,
            "text": "DOLL",
            "human_perspective_role": CardRole.AGENT,
            "llm_perspective_role": CardRole.CIVILIAN,
            "category": "neutral"
        },
        {
            "id": 21,
            "text": "LUNCH",
            "human_perspective_role": CardRole.AGENT,
            "llm_perspective_role": CardRole.CIVILIAN,
            "category": "neutral"
        },
        {
            "id": 22,
            "text": "SKATES",
            "human_perspective_role": CardRole.CIVILIAN,
            "llm_perspective_role": CardRole.CIVILIAN,
            "category": "neutral"
        },
        {
            "id": 23,
            "text": "CRAFT",
            "human_perspective_role": CardRole.CIVILIAN,
            "llm_perspective_role": CardRole.CIVILIAN,
            "category": "neutral"
        },
        {
            "id": 24,
            "text": "PEW",
            "human_perspective_role": CardRole.CIVILIAN,
            "llm_perspective_role": CardRole.AGENT,
            "category": "neutral"
        }
    ]
