import pytest
from backend.app.models.game_schemas import CardRole


@pytest.fixture
def valid_board_data():
    """
    Provides a valid board configuration for testing the Board schema validation.
    """
    cards = []

    for i in range(25):
        # 0-2: SHARED AGENTS (3)
        # LLM: Agent | Human: Agent
        if i < 3:
            llm_perspective_role, human_perspective_role = CardRole.AGENT, CardRole.AGENT

        # 3-7: UNIQUE LLM AGENTS (5)
        # LLM: Agent | Human: Civilian
        elif i < 8:
            llm_perspective_role, human_perspective_role = CardRole.AGENT, CardRole.CIVILIAN

        # 8: LLM AGENT / HUMAN ASSASSIN INTERSECTION (1)
        # LLM: Agent | Human: Assassin -> Total LLM Agents: 3+5+1 = 9
        elif i == 8:
            llm_perspective_role, human_perspective_role = CardRole.AGENT, CardRole.ASSASSIN

        # 9-13: UNIQUE HUMAN AGENTS (5)
        # LLM: Civilian | Human: Agent
        elif i < 14:
            llm_perspective_role, human_perspective_role = CardRole.CIVILIAN, CardRole.AGENT

        # 14: HUMAN AGENT / LLM ASSASSIN INTERSECTION (1)
        # LLM: Assassin | Human: Agent -> Total Human Agents: 3+5+1 = 9
        elif i == 14:
            llm_perspective_role, human_perspective_role = CardRole.ASSASSIN, CardRole.AGENT

        # 15: SHARED ASSASSIN (1)
        # LLM: Assassin | Human: Assassin
        elif i == 15:
            llm_perspective_role, human_perspective_role = CardRole.ASSASSIN, CardRole.ASSASSIN

        # 16: UNIQUE LLM ASSASSIN (1)
        # LLM: Assassin | Human: Civilian -> Total LLM Assassins: 1+1+1 = 3
        elif i == 16:
            llm_perspective_role, human_perspective_role = CardRole.ASSASSIN, CardRole.CIVILIAN

        # 17: UNIQUE HUMAN ASSASSIN (1)
        # LLM: Civilian | Human: Assassin -> Total Human Assassins: 1+1+1 = 3
        elif i == 17:
            llm_perspective_role, human_perspective_role = CardRole.CIVILIAN, CardRole.ASSASSIN

        # 18-24: PURE CIVILIANS (7)
        # LLM: Civilian | Human: Civilian
        else:
            llm_perspective_role, human_perspective_role = CardRole.CIVILIAN, CardRole.CIVILIAN

        cards.append({
            "id": i,
            "text": f"Word_{i}",
            "llm_perspective_role": llm_perspective_role,
            "human_perspective_role": human_perspective_role,
            "category": "neutral"
        })

    return {
        "board_id": "test_board_001",
        "category": "neutral",
        "cards": cards
    }
