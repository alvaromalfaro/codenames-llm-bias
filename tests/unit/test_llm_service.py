import pytest
import json
from unittest.mock import MagicMock, AsyncMock, patch
from backend.app.core.llm_service import LLMService
from backend.app.core.llm import client as client_module
from backend.app.core.llm.client import LLMClient
from backend.app.core.llm.client_local import LLMClientLocal
from backend.app.models.llm_errors import LLMEmptyResponseError
from backend.app.models.llm_schemas import ClueProposal, GuessProposal
from backend.app.models.game_schemas import CardRole, GamePhase, ClueEntry, ResolvedTarget


def _mock_response(text, raw_payload=None):
    """A mock LLM response with realistic telemetry.

    The service now reads sampling telemetry off the response to build the in-memory audit carrier,
    so these fields must be typed values (None), not the auto-created child mocks a bare MagicMock
    would return.
    """
    r = MagicMock()
    r.text = text
    r.model_used = "test_model"
    r.latency_ms = 3200
    r.raw_payload = raw_payload if raw_payload is not None else json.loads(
        text)
    r.usage = None
    r.finish_reason = None
    r.provider = None
    r.request_id = None
    r.resolved_model = None
    r.system_fingerprint = None
    r.requested_temperature = None
    r.requested_seed = None
    return r


@pytest.mark.asyncio
async def test_llm_service_propose_clue_success(game_state_cg):
    """
    Tests that the LLMService correctly processes a valid response from the LLM client and returns
    the expected clue proposal.
    """
    # Create a mock LLMResponse with the expected structure
    mock_response = _mock_response(
        "{\"clue\": \"battle\", "
        "\"count\": 3, "
        "\"reasoning\": \"The word 'battle' captures a military commander ('NAPOLEON'),"
        "the hardware ('RIFLE') and a primary theater of conflict ('RUSSIA')\""
        "}"
    )

    # Create a mock LLMClient that returns the mock response when generate is called
    mock_client = MagicMock(spec=LLMClient)
    mock_client.model_name = "test_model"
    mock_client.generate = AsyncMock(return_value=mock_response)
    service = LLMService()

    # Call the propose_clue method, capture the result, and assert that it matches the expected
    # ClueProposal based on the mock response
    result = await service.propose_clue(mock_client, game_state_cg)

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
        await service.propose_clue(mock_client, game_state_cg)


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
        await service.propose_clue(mock_client, game_state_cg)


@pytest.mark.asyncio
async def test_llm_service_propose_guess_success(game_state_guessing):
    """
    Tests that the LLMService correctly processes a valid response from the LLM client and returns
    the expected guess proposal.
    """
    # Create a mock LLMResponse with the expected structure
    mock_response = _mock_response(
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

    # Create a mock LLMClient that returns the mock response when generate is called
    mock_client = MagicMock(spec=LLMClient)
    mock_client.model_name = "test_model"
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


# The clue giver sees the board as it is on the table (§8.2): covered words are out of play, and a
# word its partner can no longer touch is still visible.

def _clue_giver_lists(game_state, seat) -> dict[str, list[str]]:
    """The word lists of a clue-giver prompt, by list name, as '- WORD' lines."""
    prompt = LLMService()._build_clue_request(
        game_state, "m", player_id=seat).messages[-1].content
    blocks = prompt.split("### BOARD STATUS ###\n", 1)[1].split("\n\n")
    return {block.splitlines()[0].split(" (")[0]: block.splitlines()[1:] for block in blocks}


# For each seat: one of its black words and one of its beige words that are green for the partner,
# so this seat covers them as the guesser.
@pytest.mark.parametrize("seat, black, beige", [(0, "ANT", "CAVE"), (1, "PINE", "RUSSIA")])
def test_clue_giver_prompt_shows_the_board_as_it_is_on_the_table(game_state_cg, seat, black, beige):
    """A covered word is out of play and no longer visible, so it is no word to avoid: a covered
    green word is listed as revealed, a covered black or beige word nowhere. A beige word with the
    partner's time token is still visible (§8.2), so it is listed apart, not as a civilian. A word
    with the giver's own token stays where it was: the partner can still touch it."""
    cards = {c.text: c for c in game_state_cg.board.cards}
    for covered in (black, beige, "BRICK"):  # BRICK is green on both sides
        cards[covered].revealed = True
    cards["BUCKET"].time_marker_by = [1 - seat]  # the partner touched it
    cards["FIDDLE"].time_marker_by = [seat]  # the giver touched it

    lists = _clue_giver_lists(game_state_cg, seat)

    listed = [line for lines in lists.values() for line in lines]
    assert f"- {black}" not in listed and f"- {beige}" not in listed
    assert "- BRICK" not in lists["AGENTS"]
    assert lists["REVEALED WORDS"] == ["- BRICK"]
    assert lists["OTHER WORDS STILL ON THE BOARD"] == ["- BUCKET"]
    assert "- BUCKET" not in lists["CIVILIANS"]
    assert "- FIDDLE" in lists["CIVILIANS"]


@pytest.mark.parametrize("seat", [0, 1])
def test_clue_giver_prompt_drops_a_word_once_two_time_tokens_cover_it(game_state_cg, seat):
    """A word under the partner's time token is still on the board. Once the second token covers it
    (§6.6), it is out of play and no longer visible, and drops out of every list."""
    bucket = next(c for c in game_state_cg.board.cards if c.text == "BUCKET")
    bucket.time_marker_by = [1 - seat]
    assert _clue_giver_lists(game_state_cg, seat)[
        "OTHER WORDS STILL ON THE BOARD"] == ["- BUCKET"]

    bucket.time_marker_by = [1 - seat, seat]
    lists = _clue_giver_lists(game_state_cg, seat)

    assert lists["OTHER WORDS STILL ON THE BOARD"] == ["None."]
    assert all("- BUCKET" not in lines for lines in lists.values())


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
async def test_propose_clue_returns_an_invalid_clue_without_retrying(game_state_cg):
    """
    An invalid clue (here a visible board word) is returned as it is, from a single call: the
    engine plays it with a penalty token (§8.4), so the model is not asked for another one.

    It used to be regenerated, up to three attempts, which meant a model was never penalised.
    """
    mock_client = MagicMock(spec=LLMClient)
    mock_client.model_name = "test_model"
    mock_client.generate = AsyncMock(return_value=_mock_response(
        '{"clue": "BUCKET", "count": 2, "reasoning": "bucket reasoning"}'))

    result = await LLMService().propose_clue(mock_client, game_state_cg)

    assert result.clue == "BUCKET"
    assert result.count == 2
    assert mock_client.generate.await_count == 1
    assert [c.retry_index for c in result.llm_calls] == [0]
    assert result.llm_calls[0].role == "clue_giver"


@pytest.mark.asyncio
async def test_propose_clue_success_records_single_accepted_call(game_state_cg):
    """A clue accepted on the first attempt carries exactly one llm_call (retry_index 0)."""
    mock_client = MagicMock(spec=LLMClient)
    mock_client.model_name = "test_model"
    mock_client.generate = AsyncMock(return_value=_mock_response(
        '{"clue": "battle", "count": 2, "reasoning": "r"}'))

    result = await LLMService().propose_clue(mock_client, game_state_cg)

    assert len(result.llm_calls) == 1
    assert result.llm_calls[0].role == "clue_giver"
    assert result.llm_calls[0].retry_index == 0
    assert len(result.llm_calls[0].rendered_prompt) > 0


@pytest.mark.asyncio
async def test_propose_clue_accepts_a_repeated_clue(game_state_cg):
    """
    Repeating a clue already given in the game is legal (§8.1): a clue that is already in the clue
    history is accepted on the first attempt, with no retry.
    """
    game_state_cg.clue_history.append(
        ClueEntry(clue="battle", count=2, clue_giver=1, turn_number=0))
    mock_client = MagicMock(spec=LLMClient)
    mock_client.model_name = "test_model"
    mock_client.generate = AsyncMock(return_value=_mock_response(
        '{"clue": "battle", "count": 2, "reasoning": "r"}'))

    result = await LLMService().propose_clue(mock_client, game_state_cg)

    assert result.clue == "battle"
    assert mock_client.generate.await_count == 1
    assert [c.retry_index for c in result.llm_calls] == [0]


@pytest.mark.asyncio
async def test_propose_guess_records_llm_call(game_state_guessing):
    """propose_guess attaches a single 'guesser' llm_call carrying the messages as sent."""
    mock_client = MagicMock(spec=LLMClient)
    mock_client.model_name = "test_model"
    mock_client.generate = AsyncMock(return_value=_mock_response(
        '{"proposals": [{"word": "NAPOLEON", "confidence": 0.9}], '
        '"reasoning": "r", "stop_reason": "s"}'))

    result = await LLMService().propose_guess(mock_client, game_state_guessing)

    assert result.llm_call is not None
    assert result.llm_call.role == "guesser"
    # The rendered prompt is exactly the request's messages that were sent.
    sent_request = mock_client.generate.await_args_list[0][0][0]
    assert result.llm_call.rendered_prompt == sent_request.messages


@pytest.mark.asyncio
async def test_propose_clue_parses_targets_onto_proposal(game_state_cg):
    """
    propose_clue extracts the intended target set S from the clue JSON and places it, verbatim,
    onto the returned ClueProposal.
    """
    mock_response = _mock_response(
        "{\"clue\": \"battle\", \"count\": 3, "
        "\"reasoning\": \"military theme\", "
        "\"targets\": [\"NAPOLEON\", \"RIFLE\", \"RUSSIA\"]}"
    )

    mock_client = MagicMock(spec=LLMClient)
    mock_client.model_name = "test_model"
    mock_client.generate = AsyncMock(return_value=mock_response)

    result = await LLMService().propose_clue(mock_client, game_state_cg)

    assert isinstance(result, ClueProposal)
    assert result.targets == ["NAPOLEON", "RIFLE", "RUSSIA"]


@pytest.mark.asyncio
async def test_propose_clue_missing_targets_defaults_empty(game_state_cg):
    """
    A clue JSON without a targets field is accepted; the proposal simply carries an empty S.
    """
    mock_response = _mock_response(
        '{"clue": "battle", "count": 3, "reasoning": "r"}')

    mock_client = MagicMock(spec=LLMClient)
    mock_client.model_name = "test_model"
    mock_client.generate = AsyncMock(return_value=mock_response)

    result = await LLMService().propose_clue(mock_client, game_state_cg)

    assert result.targets == []


@pytest.mark.asyncio
async def test_propose_clue_records_malformed_targets_without_retry(game_state_cg):
    """
    Record-only guarantee: a malformed target set S (here |targets| != count, a non-agent target,
    and an off-board word all at once) is captured as-is on a valid clue. It must not raise on
    account of S and must not trigger any extra generation (S is never a retry trigger).
    """
    # Clue 'battle' is legal (not a board word); count is 3 but only 1 target is listed, that
    # target is a civilian (from the LLM perspective) rather than an agent, plus an off-board word.
    mock_response = _mock_response(
        "{\"clue\": \"battle\", \"count\": 3, \"reasoning\": \"r\", "
        "\"targets\": [\"BUCKET\", \"ZEBRA\"]}"
    )

    mock_client = MagicMock(spec=LLMClient)
    mock_client.model_name = "test_model"
    mock_client.generate = AsyncMock(return_value=mock_response)

    result = await LLMService().propose_clue(mock_client, game_state_cg)

    # Accepted and stored despite the malformation, and generated exactly once (no S-driven retry).
    assert result.clue == "battle"
    assert result.targets == ["BUCKET", "ZEBRA"]
    assert mock_client.generate.await_count == 1


@pytest.mark.asyncio
async def test_targets_never_reach_guesser_prompt(game_state_guessing):
    """
    The intended target set S must never enter any guesser-side code path. A sentinel target word 
    placed on the current clue (and on a historical clue) must appear nowhere in the guesser request 
    built for that turn.
    """
    sentinel = "ZZ_SENTINEL_TARGET_ZZ"

    # Stamp the sentinel onto S for both the current clue and a prior clue in history.
    current_clue = game_state_guessing.current_clue
    current_clue.targets = [sentinel]
    current_clue.targets_resolved = [ResolvedTarget(word=sentinel)]

    history_clue = ClueEntry(
        clue="ocean", count=2, clue_giver=1, turn_number=0,
        targets=[sentinel], targets_resolved=[ResolvedTarget(word=sentinel)],
    )
    game_state_guessing.clue_history.insert(0, history_clue)

    # game_state_guessing has guesser == 0.
    request = LLMService()._build_guess_request(
        game_state_guessing, "test_model", player_id=0)

    for message in request.messages:
        assert sentinel not in message.content


# The guesser sees its own side of the key card (§3, §4.3), in the standard and the SD guess
# prompts alike.

def _own_key_section(user_prompt: str) -> list[list[str]]:
    """The black, green and beige blocks of a guess prompt's key-card section, as lists of lines."""
    section = user_prompt.split("### YOUR SIDE OF THE KEY CARD ###\n", 1)[1]
    section = section.split("\n\n### UNREVEALED BOARD WORDS ###", 1)[0]
    return [block.splitlines() for block in section.split("\n\n")]


@pytest.mark.parametrize("seat", [0, 1])
def test_guess_prompt_shows_the_guesser_own_key_side(game_state_guessing, seat):
    """The guesser gets its own side, not its partner's: its 3 black words, and its green and beige
    words, in board order."""
    def own_words(role):
        return [f"- {c.text}" for c in game_state_guessing.board.cards
                if (c.llm_perspective_role if seat == 0 else c.human_perspective_role) == role]

    prompt = LLMService()._build_guess_request(
        game_state_guessing, "m", player_id=seat).messages[-1].content
    black, green, beige = _own_key_section(prompt)

    expected_black = ["- ANT", "- LEMONADE", "- MAKEUP"] if seat == 0 else [
        "- LEMONADE", "- LOCUST", "- PINE"]
    assert black == ["Black on your side (all 3):", *expected_black]
    assert green == ["Green on your side (only the words you can still guess):",
                     *own_words(CardRole.AGENT)]
    assert beige == ["Beige on your side (only the words you can still guess):",
                     *own_words(CardRole.CIVILIAN)]


def test_own_key_side_states_what_the_guesser_found_on_its_black_words(game_state_guessing):
    """A black word the guesser can no longer touch says what its guess showed: an agent card means
    it was the guesser's agent, its own time token means it was a civilian. The green and beige lists
    keep only the words the guesser can still guess."""
    cards = {c.text: c for c in game_state_guessing.board.cards}
    cards["ANT"].revealed = True  # black for seat 0, an agent for it
    cards["ANT"].revealed_by = [0, 1]
    cards["MAKEUP"].time_marker_by = [0]  # black for seat 0, a civilian for it
    cards["BRICK"].revealed = True  # green for seat 0, covered
    cards["RUSSIA"].time_marker_by = [0]  # green for seat 0, seat 0's own token
    cards["BUCKET"].time_marker_by = [1]  # beige for seat 0, the partner's token only

    prompt = LLMService()._build_guess_request(
        game_state_guessing, "m", player_id=0).messages[-1].content
    black, green, beige = _own_key_section(prompt)

    assert black == [
        "Black on your side (all 3):",
        "- ANT (covered by an agent card: you guessed it, and it was an Agent for you)",
        "- LEMONADE",
        "- MAKEUP (your time token is on it: you guessed it, and it was a Civilian for you)",
    ]
    assert "- BRICK" not in green and "- RUSSIA" not in green
    assert "- BUCKET" in beige


def test_sd_guess_prompt_shows_the_same_own_key_side(game_state_guessing):
    """Sudden death keeps the key side: the SD guess prompt renders the same section."""
    service = LLMService()
    game_state_guessing.board.cards[2].revealed = True  # ANT

    standard = service._build_guess_request(
        game_state_guessing, "m", player_id=0).messages[-1].content
    sudden_death = service._build_guess_sd_request(
        _sd_llm_state(game_state_guessing), "m", player_id=0).messages[-1].content

    assert _own_key_section(sudden_death) == _own_key_section(standard)


# An optional per-call seed flows method -> builder -> LLMRequest.seed.

def _mock_client_seq(texts):
    """A mock LLMClient whose ``generate`` returns the given canned responses in order."""
    client = MagicMock(spec=LLMClient)
    client.model_name = "test_model"
    client.generate = AsyncMock(side_effect=[_mock_response(t) for t in texts])
    return client


def _sd_llm_state(game_state_guessing):
    """Reshape the guessing fixture (guesser 0) into a valid SUDDEN_DEATH_LLM state for seat 0."""
    game_state_guessing.current_phase = GamePhase.SUDDEN_DEATH_LLM
    return game_state_guessing


@pytest.mark.asyncio
async def test_propose_guess_seed_reaches_request(game_state_guessing):
    client = _mock_client_seq(
        ['{"proposals": [{"word": "NAPOLEON", "confidence": 0.9}], '
         '"reasoning": "r", "stop_reason": "s"}'])

    await LLMService().propose_guess(client, game_state_guessing, player_id=0, seed=12345)

    assert client.generate.await_args_list[0][0][0].seed == 12345


@pytest.mark.asyncio
async def test_propose_guess_sd_seed_reaches_request(game_state_guessing):
    state = _sd_llm_state(game_state_guessing)
    client = _mock_client_seq(
        ['{"proposals": [{"word": "NAPOLEON", "confidence": 0.9}], '
         '"reasoning": "r", "stop_reason": "s"}'])

    await LLMService().propose_guess_sd(client, state, player_id=0, seed=777)

    assert client.generate.await_args_list[0][0][0].seed == 777


@pytest.mark.asyncio
async def test_elicit_confidence_ranking_seed_reaches_request(game_state_guessing):
    client = _mock_client_seq(
        ['{"reasoning": "r", "rankings": [{"word": "NAPOLEON", "confidence": 0.5}]}'])

    await LLMService().elicit_confidence_ranking(client, game_state_guessing, player_id=0, seed=42)

    assert client.generate.await_args_list[0][0][0].seed == 42


@pytest.mark.asyncio
async def test_elicit_confidence_ranking_sd_seed_reaches_request(game_state_guessing):
    state = _sd_llm_state(game_state_guessing)
    client = _mock_client_seq(
        ['{"reasoning": "r", "rankings": [{"word": "NAPOLEON", "confidence": 0.5}]}'])

    await LLMService().elicit_confidence_ranking_sd(client, state, player_id=0, seed=99)

    assert client.generate.await_args_list[0][0][0].seed == 99


@pytest.mark.asyncio
async def test_propose_clue_seed_reaches_request(game_state_cg):
    client = _mock_client_seq(
        ['{"clue": "battle", "count": 2, "reasoning": "r"}'])

    await LLMService().propose_clue(client, game_state_cg, seed=555)

    assert client.generate.await_args_list[0][0][0].seed == 555


@pytest.mark.asyncio
async def test_no_seed_leaves_request_seed_none(game_state_guessing):
    """Default-None equivalence: with no seed argument, the request seed is None."""
    client = _mock_client_seq(
        ['{"proposals": [{"word": "NAPOLEON", "confidence": 0.9}], '
         '"reasoning": "r", "stop_reason": "s"}'])

    await LLMService().propose_guess(client, game_state_guessing, player_id=0)

    assert client.generate.await_args_list[0][0][0].seed is None


# empty-response re-sample, end to end through the service
def _mock_ollama_chat_response(content: str) -> MagicMock:
    """A MagicMock mimicking a raw ollama chat() return carrying exactly ``content``."""
    r = MagicMock()
    r.message.content = content
    r.total_duration = 3200
    r.prompt_eval_count = 10
    r.eval_count = 20
    r.done_reason = "stop"
    r.model_dump_json.return_value = json.dumps(
        {"message": {"content": content}})
    return r


_VALID_CLUE_JSON = json.dumps(
    {"reasoning": "military theme", "clue": "battle", "count": 3})


@pytest.mark.asyncio
async def test_propose_clue_survives_empty_draw_and_audits_the_resample(game_state_cg):
    """An intermittent empty draw must not kill the game.

    Driving the real LLMClientLocal (only the ollama transport mocked): attempt 1 returns '',
    attempt 2 returns valid JSON. propose_clue must return a usable clue, and the audit carrier for
    that call must record exactly 1 empty-response re-sample.
    """
    with patch("backend.app.core.llm.client_local.Client") as MockClient:
        mock_chat = MockClient.return_value.chat
        mock_chat.side_effect = [
            _mock_ollama_chat_response(""),
            _mock_ollama_chat_response(_VALID_CLUE_JSON),
        ]
        client = LLMClientLocal("ollama3.2:latest", max_retries=3)
        service = LLMService()

        result = await service.propose_clue(client, game_state_cg)

        assert isinstance(result, ClueProposal)
        assert result.clue == "battle"
        assert mock_chat.call_count == 2
        # One call (retry_index 0) that internally absorbed one empty draw: the re-sample is not a
        # separate call.
        assert len(result.llm_calls) == 1
        assert result.llm_calls[0].retry_index == 0
        assert result.llm_calls[0].raw_payload[client_module.EMPTY_RESAMPLE_KEY] == 1


@pytest.mark.asyncio
async def test_propose_clue_still_fails_when_every_draw_is_empty(game_state_cg):
    """The re-sample is bounded: when every draw is empty the budget is exhausted and the error
    surfaces rather than looping forever."""
    with patch("backend.app.core.llm.client_local.Client") as MockClient:
        mock_chat = MockClient.return_value.chat
        mock_chat.return_value = _mock_ollama_chat_response("")
        client = LLMClientLocal("ollama3.2:latest", max_retries=3)
        service = LLMService()

        with pytest.raises(LLMEmptyResponseError):
            await service.propose_clue(client, game_state_cg)
        assert mock_chat.call_count == 4  # max_retries=3 -> 4 attempts, then raise


# Every hardcoded _default_* fallback is a verbatim copy of its template file, so a missing file
# never changes the prompt that is sent.

@pytest.mark.parametrize("path_attr, default_name", [
    ("SYSTEM_TEMP_CG_PATH", "_default_system_prompt_cg"),
    ("USER_TEMP_CG_PATH", "_default_user_prompt_cg"),
    ("SYSTEM_TEMP_GG_PATH", "_default_system_prompt_gg"),
    ("USER_TEMP_GG_PATH", "_default_user_prompt_gg"),
    ("SYSTEM_TEMP_SD_GG_PATH", "_default_system_prompt_sd_gg"),
    ("USER_TEMP_SD_GG_PATH", "_default_user_prompt_sd_gg"),
    ("SYSTEM_TEMP_MEAS_GG_PATH", "_default_system_prompt_meas_gg"),
    ("USER_TEMP_MEAS_GG_PATH", "_default_user_prompt_meas_gg"),
    ("SYSTEM_TEMP_MEAS_SD_PATH", "_default_system_prompt_meas_sd"),
    ("USER_TEMP_MEAS_SD_PATH", "_default_user_prompt_meas_sd"),
    ("ONE_SHOT_USER_CG_PATH", "_default_os_user_cg"),
    ("ONE_SHOT_ASSISTANT_CG_PATH", "_default_os_assistant_cg"),
    ("ONE_SHOT_USER_GG_PATH", "_default_os_user_gg"),
    ("ONE_SHOT_ASSISTANT_GG_PATH", "_default_os_assistant_gg"),
])
def test_default_template_matches_its_file(path_attr, default_name):
    # Read the way the service reads it: text mode, so a CRLF checkout compares as LF.
    with open(getattr(LLMService, path_attr), "r", encoding="utf-8") as f:
        on_disk = f.read()

    assert getattr(LLMService(), default_name)() == on_disk
