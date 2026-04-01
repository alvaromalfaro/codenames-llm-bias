import pytest
import json
from unittest.mock import MagicMock, AsyncMock
from backend.app.core.llm_service import LLMService
from backend.app.core.lm.llm_client import LLMClient
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
    mock_client.generate = AsyncMock(return_value=mock_response)
    service = LLMService(llm_client=mock_client)

    # Call the propose_clue method, capture the result, and assert that it matches the expected
    # ClueProposal based on the mock response
    result = await service.propose_clue(game_state_cg)

    assert isinstance(result, ClueProposal)
    assert result.clue == "battle"
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
    service = LLMService(llm_client=mock_client)

    # Modify the game state to be in GUESSING phase
    game_state_cg.current_phase = GamePhase.GUESSING

    with pytest.raises(ValueError,
                       match="Cannot propose a clue when the game is not in the GIVING_CLUE phase."):
        await service.propose_clue(game_state_cg)


@pytest.mark.asyncio
async def test_llm_service_propose_clue_wrong_player(game_state_cg):
    """
    Tests that the LLMService raises a ValueError when propose_clue is called by a player who is not
    the clue giver.
    """
    # Create a mock LLMClient (the specific behavior of the client is not relevant for this test)
    mock_client = MagicMock(spec=LLMClient)
    service = LLMService(llm_client=mock_client)

    # Modify the game state to have a different clue giver
    game_state_cg.clue_giver = 1  # Set clue giver to player 1 instead of player 0

    with pytest.raises(ValueError, match="The player must be the clue giver to propose a clue."):
        await service.propose_clue(game_state_cg)


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
    mock_client.generate = AsyncMock(return_value=mock_response)
    service = LLMService(llm_client=mock_client)

    # Call the propose_guess method, capture the result, and assert that it matches the expected
    # GuessProposal based on the mock response
    result = await service.propose_guess(game_state_guessing)

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
    service = LLMService(llm_client=mock_client)

    # Modify the game state to be in GIVING_CLUE phase
    game_state_guessing.current_phase = GamePhase.GIVING_CLUE

    with pytest.raises(ValueError,
                       match="Cannot propose a guess when the game is not in the GUESSING phase."):
        await service.propose_guess(game_state_guessing)


@pytest.mark.asyncio
async def test_llm_service_propose_guess_wrong_player(game_state_guessing):
    """
    Tests that the LLMService raises a ValueError when propose_guess is called by a player who is 
    not the guesser.
    """
    # Create a mock LLMClient (the specific behavior of the client is not relevant for this test)
    mock_client = MagicMock(spec=LLMClient)
    service = LLMService(llm_client=mock_client)

    # Modify the game state to have a different guesser
    game_state_guessing.guesser = 1

    with pytest.raises(ValueError, match="The player must be the guesser to propose a guess."):
        await service.propose_guess(game_state_guessing)


@pytest.mark.asyncio
async def test_llm_service_propose_guess_no_clue(game_state_guessing):
    """
    Tests that the LLMService raises a ValueError when propose_guess is called but there is no clue
    available in the game state for the guesser to base their guesses on.
    """
    # Create a mock LLMClient (the specific behavior of the client is not relevant for this test)
    mock_client = MagicMock(spec=LLMClient)
    service = LLMService(llm_client=mock_client)

    # Ensure there is no current clue in the game state
    game_state_guessing.current_clue.turn_number = game_state_guessing.current_clue.turn_number - 1

    with pytest.raises(ValueError, match="Cannot propose a guess when there is no clue available."):
        await service.propose_guess(game_state_guessing)


def test_llm_service_build_clue_request_player1(game_state_cg):
    """
    Tests that the LLMService correctly builds an LLMRequest for proposing a clue when the current
    player is Player 1 (human [an LLM in fact, to automate future games]).
    """
    mock_client = MagicMock(spec=LLMClient)
    service = LLMService(llm_client=mock_client)

    request = service._build_clue_request(game_state_cg, player_id=1)

    agent_words = [
        card.text for card in game_state_cg.board.cards if card.human_perspective_role == "agent"]
    danger_words = [
        card.text for card in game_state_cg.board.cards if card.human_perspective_role != "agent"]

    assert request is not None
    assert len(request.messages) == 2
    assert request.messages[0].role == "system"
    assert request.messages[1].role == "user"
    assert f"agent_words={{", ".join(agent_words)}}" in request.messages[1].content
    assert f"danger_words={{", ".join(danger_words)}}" in request.messages[1].content


def test_llm_service_build_clue_proposal_json_error(llm_response):
    """
    Tests that the LLMService raises a ValueError when the LLMResponse text cannot be parsed as JSON
    when building a clue proposal.
    """
    service = LLMService(llm_client=MagicMock(spec=LLMClient))

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
    service = LLMService(llm_client=MagicMock(spec=LLMClient))

    # Modify the LLMResponse text to be invalid JSON
    llm_response.text = "This is not valid JSON"

    with pytest.raises(ValueError, match="LLM response is not valid JSON. Response content: " +
                       llm_response.text):
        service._build_guess_proposal(llm_response)
