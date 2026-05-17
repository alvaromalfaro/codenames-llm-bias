import pytest
import json
from unittest.mock import MagicMock, AsyncMock
from backend.app.core.llm_service import LLMService
from backend.app.core.llm.client import LLMClient
from backend.app.core.clue_validator import ClueValidator
from backend.app.models.llm_schemas import ClueProposal, GuessProposal
from backend.app.models.game_schemas import GamePhase


@pytest.mark.asyncio
async def test_llm_service_propose_clue_success(game_state_cg):
    """
    Tests that the LLMService correctly processes a valid response from the LLM client and returns
    the expected clue proposal.
    """
    # Create a mock LLMResponse with the expected structure
    mock_response = MagicMock()
    mock_response.text = (
        "{\"clue\": \"battle\", "
        "\"count\": 3, "
        "\"reasoning\": \"The word 'battle' captures a military commander ('NAPOLEON'),"
        "the hardware ('RIFLE') and a primary theater of conflict ('RUSSIA')\""
        "}"
    )
    mock_response.model_used = "test_model"
    mock_response.latency_ms = 3200
    mock_response.raw_payload = json.loads(mock_response.text)

    # Create a mock LLMClient that returns the mock response when generate is called
    mock_client = MagicMock(spec=LLMClient)
    mock_client.local_model = "test_model"
    mock_client.generate = AsyncMock(return_value=mock_response)
    service = LLMService()

    # Call the propose_clue method, capture the result, and assert that it matches the expected
    # ClueProposal based on the mock response
    result = await service.propose_clue(mock_client, game_state_cg, ClueValidator(game_state_cg.board.cards))

    assert isinstance(result, ClueProposal)
    assert result.clue == "battle"  # "battle" is not a board word so validation passes
    assert result.count == 3
    assert result.reasoning == "The word 'battle' captures a military commander ('NAPOLEON'),the " \
        "hardware ('RIFLE') and a primary theater of conflict ('RUSSIA')"
    assert result.raw_payload == json.loads(mock_response.text)
    # Verify that the LLM client's generate method was called once with the expected LLMRequest
    mock_client.generate.assert_awaited_once()


@pytest.mark.asyncio
async def test_llm_service_propose_clue_wrong_phase(game_state_cg):
    """
    Tests that the LLMService raises a ValueError when propose_clue is called during a phase of the
    game where clue proposals are not allowed.
    """
    # Create a mock LLMClient (the specific behavior of the client is not relevant for this test)
    mock_client = MagicMock(spec=LLMClient)
    service = LLMService()

    # Modify the game state to be in GUESSING phase
    game_state_cg.current_phase = GamePhase.GUESSING

    with pytest.raises(ValueError,
                       match="Cannot propose a clue when the game is not in the GIVING_CLUE phase."):
        await service.propose_clue(mock_client, game_state_cg, MagicMock())


@pytest.mark.asyncio
async def test_llm_service_propose_clue_wrong_player(game_state_cg):
    """
    Tests that the LLMService raises a ValueError when propose_clue is called by a player who is not
    the clue giver.
    """
    # Create a mock LLMClient (the specific behavior of the client is not relevant for this test)
    mock_client = MagicMock(spec=LLMClient)
    service = LLMService()

    # Modify the game state to have a different clue giver
    game_state_cg.clue_giver = 1  # Set clue giver to player 1 instead of player 0

    with pytest.raises(ValueError, match="The player must be the clue giver to propose a clue."):
        await service.propose_clue(mock_client, game_state_cg, MagicMock())


@pytest.mark.asyncio
async def test_llm_service_propose_guess_success(game_state_guessing):
    """
    Tests that the LLMService correctly processes a valid response from the LLM client and returns
    the expected guess proposal.
    """
    # Create a mock LLMResponse with the expected structure
    mock_response = MagicMock()
    mock_response.text = (
        "{"
        "\"proposals\": ["
        "{\"word\": \"NAPOLEON\", \"confidence\": 0.9}, "
        "{\"word\": \"RIFLE\", \"confidence\": 0.8} "
        "],"
        "\"reasoning\": \"The word 'battle' captures a military commander ('NAPOLEON'), and the "
        "hardware ('RIFLE')\","
        "\"stop_reason\": \"Cannot determine other words\""
        "}"
    )
    mock_response.model_used = "test_model"
    mock_response.latency_ms = 3200
    mock_response.raw_payload = json.loads(mock_response.text)

    # Create a mock LLMClient that returns the mock response when generate is called
    mock_client = MagicMock(spec=LLMClient)
    mock_client.local_model = "test_model"
    mock_client.generate = AsyncMock(return_value=mock_response)
    service = LLMService()

    # Call the propose_guess method, capture the result, and assert that it matches the expected
    # GuessProposal based on the mock response
    result = await service.propose_guess(mock_client, game_state_guessing)

    assert isinstance(result, GuessProposal)
    assert result.proposals == ["NAPOLEON", "RIFLE"]
    assert result.confidence == [0.9, 0.8]
    assert result.reasoning == ("The word 'battle' captures a military commander ('NAPOLEON'), and "
                                "the hardware ('RIFLE')")
    assert result.stop_reason == "Cannot determine other words"
    assert result.raw_payload == json.loads(mock_response.text)
    mock_client.generate.assert_awaited_once()


@pytest.mark.asyncio
async def test_llm_service_propose_guess_wrong_phase(game_state_guessing):
    """
    Tests that the LLMService raises a ValueError when propose_guess is called during a phase of the
    game where guess proposals are not allowed.
    """
    # Create a mock LLMClient (the specific behavior of the client is not relevant for this test)
    mock_client = MagicMock(spec=LLMClient)
    service = LLMService()

    # Modify the game state to be in GIVING_CLUE phase
    game_state_guessing.current_phase = GamePhase.GIVING_CLUE

    with pytest.raises(ValueError,
                       match="Cannot propose a guess when the game is not in the GUESSING phase."):
        await service.propose_guess(mock_client, game_state_guessing)


@pytest.mark.asyncio
async def test_llm_service_propose_guess_wrong_player(game_state_guessing):
    """
    Tests that the LLMService raises a ValueError when propose_guess is called by a player who is 
    not the guesser.
    """
    # Create a mock LLMClient (the specific behavior of the client is not relevant for this test)
    mock_client = MagicMock(spec=LLMClient)
    service = LLMService()

    # Modify the game state to have a different guesser
    game_state_guessing.guesser = 1

    with pytest.raises(ValueError, match="The player must be the guesser to propose a guess."):
        await service.propose_guess(mock_client, game_state_guessing)


@pytest.mark.asyncio
async def test_llm_service_propose_guess_no_clue(game_state_guessing):
    """
    Tests that the LLMService raises a ValueError when propose_guess is called but there is no clue
    available in the game state for the guesser to base their guesses on.
    """
    # Create a mock LLMClient (the specific behavior of the client is not relevant for this test)
    mock_client = MagicMock(spec=LLMClient)
    service = LLMService()

    # Ensure there is no current clue in the game state
    game_state_guessing.current_clue.turn_number = game_state_guessing.current_clue.turn_number - 1

    with pytest.raises(ValueError, match="Cannot propose a guess when there is no clue available."):
        await service.propose_guess(mock_client, game_state_guessing)


def test_llm_service_build_clue_request_player1(game_state_cg):
    """
    Tests that the LLMService correctly builds an LLMRequest for proposing a clue when the current
    player is Player 1 (human [an LLM in fact, to automate future games]).
    """
    mock_client = MagicMock(spec=LLMClient)
    service = LLMService()

    request = service._build_clue_request(
        game_state_cg, 'test_client', player_id=1)

    agent_words = [
        card.text for card in game_state_cg.board.cards if card.human_perspective_role == "agent"]
    danger_words = [
        card.text for card in game_state_cg.board.cards if card.human_perspective_role != "agent"]

    assert request is not None
    assert len(request.messages) == 4
    assert request.messages[0].role == "system"
    assert request.messages[1].role == "user"
    assert f"agent_words={{", ".join(agent_words)}}" in request.messages[1].content
    assert f"danger_words={{", ".join(danger_words)}}" in request.messages[1].content


def test_llm_service_build_clue_request_player0(game_state_cg):
    """
    Tests that _build_clue_request for player 0 uses the LLM perspective words, not the human
    perspective. RIFLE is an LLM agent but a human civilian, so it should appear in the agent
    section. CAESAR is a human agent but an LLM civilian, so it must not appear there.
    """
    service = LLMService()

    request = service._build_clue_request(
        game_state_cg, "test_model", player_id=0)

    llm_agent_words = [
        card.text for card in game_state_cg.board.cards
        if card.llm_perspective_role == "agent"
    ]
    human_only_agent_words = [
        card.text for card in game_state_cg.board.cards
        if card.human_perspective_role == "agent" and card.llm_perspective_role != "agent"
    ]

    user_prompt = request.messages[-1].content
    assert request.messages[0].role == "system"
    assert request.messages[1].role == "user"
    assert request.messages[2].role == "assistant"
    assert request.messages[3].role == "user"
    for word in llm_agent_words:
        assert word in user_prompt
    for word in human_only_agent_words:
        # Human-only agents must not appear in the LLM's agent section
        assert f"AGENTS" not in user_prompt or word not in user_prompt.split("ASSASSINS")[
            0]


def test_llm_service_build_guess_request(game_state_guessing):
    """
    Tests that _build_guess_request correctly formats the clue, count, and unrevealed board
    words into the user prompt.
    """
    service = LLMService()

    request = service._build_guess_request(
        game_state_guessing, "test_model", player_id=0)

    user_prompt = request.messages[-1].content
    assert len(request.messages) == 4
    assert request.messages[0].role == "system"
    assert request.messages[1].role == "user"
    assert request.messages[2].role == "assistant"
    assert request.messages[3].role == "user"
    assert game_state_guessing.current_clue.clue in user_prompt
    assert str(game_state_guessing.current_clue.count) in user_prompt
    for card in game_state_guessing.board.cards:
        assert card.text in user_prompt


def test_llm_service_build_clue_request_no_one_shot(game_state_cg):
    """
    Tests that when the one-shot examples are disabled (empty strings), _build_clue_request
    falls back to only 2 messages: system and user.
    """
    service = LLMService()
    service._one_shot_user_cg = ""
    service._one_shot_assistant_cg = ""

    request = service._build_clue_request(
        game_state_cg, "test_model", player_id=0)

    assert len(request.messages) == 2
    assert request.messages[0].role == "system"
    assert request.messages[1].role == "user"


def test_llm_service_build_clue_proposal_json_error(llm_response):
    """
    Tests that the LLMService raises a ValueError when the LLMResponse text cannot be parsed as JSON
    when building a clue proposal.
    """
    service = LLMService()

    # Modify the LLMResponse text to be invalid JSON
    llm_response.text = "This is not valid JSON"

    with pytest.raises(ValueError, match="LLM response is not valid JSON. Response content: " +
                       llm_response.text):
        service._build_clue_proposal(llm_response)


def test_llm_service_build_guess_proposal_json_error(llm_response):
    """
    Tests that the LLMService raises a ValueError when the LLMResponse text cannot be parsed as JSON
    when building a guess proposal.
    """
    service = LLMService()

    # Modify the LLMResponse text to be invalid JSON
    llm_response.text = "This is not valid JSON"

    with pytest.raises(ValueError, match="LLM response is not valid JSON. Response content: " +
                       llm_response.text):
        service._build_guess_proposal(llm_response)


@pytest.mark.asyncio
async def test_propose_clue_retries_on_invalid_clue(game_state_cg):
    """
    Tests that propose_clue retries when the LLM returns a clue that fails ClueValidator
    (a direct board-word match), and returns the valid proposal from the second attempt.
    The retry request must include the failed attempt as an assistant message followed by
    a correction user message.
    """
    invalid_text = '{"clue": "BUCKET", "count": 2, "reasoning": "bucket reasoning"}'
    valid_text = '{"clue": "battle", "count": 2, "reasoning": "battle reasoning"}'

    def make_response(text):
        r = MagicMock()
        r.text = text
        r.raw_payload = json.loads(text)
        return r

    mock_client = MagicMock(spec=LLMClient)
    mock_client.local_model = "test_model"
    mock_client.generate = AsyncMock(side_effect=[
        make_response(invalid_text),
        make_response(valid_text),
    ])

    service = LLMService()
    result = await service.propose_clue(mock_client, game_state_cg, ClueValidator(game_state_cg.board.cards))

    assert isinstance(result, ClueProposal)
    assert result.clue == "battle"
    assert mock_client.generate.await_count == 2

    # The retry request's last two messages must be the failed assistant response and the
    # correction user message annotating the rejection.
    retry_request = mock_client.generate.await_args_list[1][0][0]
    messages = retry_request.messages
    assert messages[-2].role == "assistant"
    assert "BUCKET" in messages[-2].content
    assert messages[-1].role == "user"
    assert "BUCKET" in messages[-1].content
    assert "rejected" in messages[-1].content.lower()


@pytest.mark.asyncio
async def test_propose_clue_raises_after_max_retries(game_state_cg):
    """
    Tests that propose_clue raises a ValueError after exhausting all retry attempts
    without receiving a valid clue from the LLM.
    """
    invalid_text = '{"clue": "BUCKET", "count": 2, "reasoning": "bucket reasoning"}'

    def make_response(text):
        r = MagicMock()
        r.text = text
        r.raw_payload = json.loads(text)
        return r

    mock_client = MagicMock(spec=LLMClient)
    mock_client.local_model = "test_model"
    mock_client.generate = AsyncMock(return_value=make_response(invalid_text))

    service = LLMService()

    with pytest.raises(ValueError, match="LLM failed to produce a valid clue after"):
        await service.propose_clue(mock_client, game_state_cg, ClueValidator(game_state_cg.board.cards))

    assert mock_client.generate.await_count == 3
