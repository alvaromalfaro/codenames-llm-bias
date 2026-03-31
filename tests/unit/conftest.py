import pytest
from backend.app.models.game_schemas import CardRole, GameState, Board, WordCard, GamePhase, ClueEntry
from backend.app.models.llm_schemas import LLMRequest, LLMMessage


@pytest.fixture
def valid_board_data() -> dict:
    """
    Provides a valid board configuration for testing the Board schema validation. It is based on the
    board configuration used in the official Codenames Duet rulesbook.
    """
    return {
        "board_id": "test_board_001",
        "category": "neutral",
        "cards": _get_cards()
    }


@pytest.fixture
def llm_request_cg() -> LLMRequest:
    """
    Provides a valid LLM request for testing the LLM client when generating a response for a clue
    generation task.
    """
    with open("data/prompt_templates/SYSTEM_TEMPLATE_CLUE_GIVER.txt", "r") as f:
        system_prompt = f.read()

    with open("data/prompt_templates/USER_TEMPLATE_CLUE_GIVER.txt", "r") as f:
        user_prompt = f.read()

    turn_number = 1
    cards = _get_cards()
    agent_words = [i["text"]
                   for i in cards if i["llm_perspective_role"] == CardRole.AGENT]
    danger_words = [i["text"]
                    for i in cards if i["llm_perspective_role"] != CardRole.AGENT]
    revealed_words = []

    user_prompt = user_prompt.format(
        turn_number=turn_number,
        agent_words=agent_words,
        danger_words=danger_words,
        revealed_words=revealed_words
    )

    return LLMRequest(
        messages=[
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=user_prompt)
        ],
        model="ollama3.2:latest"
    )


@pytest.fixture
def game_state_cg() -> GameState:
    """
    Provides a valid game state in the clue-giving phase for testing the LLM service when proposing
    a clue.
    """
    return GameState(
        game_id="test_game_001",
        board=_get_board(),
        clue_giver=0,
        guesser=1
    )


@pytest.fixture
def game_state_guessing() -> GameState:
    """
    Provides a valid game state in the guessing phase for testing the LLM service when proposing
    a clue during an invalid phase of the game.
    """
    game_state = GameState(
        game_id="test_game_001",
        board=_get_board(),
        clue_giver=1,
        guesser=0
    )
    game_state.current_phase = GamePhase.GUESSING
    clue = ClueEntry(
        clue="battle",
        count=3,
        clue_giver=1,
        turn_number=1
    )
    game_state.current_clue = clue
    game_state.clue_history.append(clue)

    return game_state


def _get_board():
    return Board(
        board_id="test_board_001",
        category="neutral",
        cards=[
            WordCard(id=i["id"], text=i["text"],
                     human_perspective_role=i["human_perspective_role"],
                     llm_perspective_role=i["llm_perspective_role"],
                     category=i["category"]) for i in _get_cards()
        ]
    )


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
