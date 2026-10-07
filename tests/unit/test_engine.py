import random
import pytest
from backend.app.core.engine import CodenamesDuetEngine
from backend.app.models.game_schemas import Board, GamePhase, CardRole, ClueEntry, ResolvedTarget


def test_engine_initialization(valid_board_data: dict):
    """
    Validates that the CodenamesDuetEngine initializes correctly with a valid board configuration.

    :param valid_board_data: A fixture providing a valid board configuration as a dictionary.
    """
    board = Board(**valid_board_data)
    engine = CodenamesDuetEngine(board=board)

    # Validate that the engine's state is initialized correctly
    assert engine.state.board == board
    assert engine.state.current_phase == GamePhase.GIVING_CLUE
    assert engine.state.timer_tokens == 9
    assert engine.state.current_clue is None
    assert engine.state.guesses_made_this_turn == 0
    assert engine.state.clue_giver in [0, 1]
    assert engine.state.guesser in [0, 1]
    assert engine.state.clue_giver != engine.state.guesser
    assert engine.state.agents_remaining == [9, 9]


def test_engine_receive_clue(valid_board_data: dict):
    """
    Validates that the receive_clue method correctly processes a valid clue input and updates the
    game state accordingly.

    :param valid_board_data: A fixture providing a valid board configuration as a dictionary.
    """
    board = Board(**valid_board_data)
    engine = CodenamesDuetEngine(board=board)

    # Force the clue giver to be player 0 for testing
    engine.state.clue_giver = 0
    engine.state.guesser = 1
    engine.state.current_phase = GamePhase.GIVING_CLUE
    engine.state.turn_number = 1

    engine.receive_clue(clue="TestClue", count=2, player_id=0)

    assert engine.state.current_phase == GamePhase.GUESSING
    assert engine.state.guesses_made_this_turn == 0
    assert engine.state.current_clue is not None
    assert engine.state.current_clue.clue == "TestClue"
    assert engine.state.current_clue.count == 2
    assert engine.state.current_clue.clue_giver == 0
    assert engine.state.current_clue.turn_number == 1
    assert engine.state.clue_history == []
    assert engine.state.current_clue.raw_payload is None
    # Absent an intended target set, S is captured as empty (never None / never rejected).
    assert engine.state.current_clue.targets == []
    assert engine.state.current_clue.targets_resolved == []


def test_engine_receive_clue_resolves_target_snapshot(valid_board_data: dict):
    """
    receive_clue records a clue-time resolved snapshot of the intended target set S: each target
    maps to its card_id, the giver-perspective role, and the reveal state at clue time. An
    off-board word yields an all-None snapshot (the malformation diagnostic).
    """
    board = Board(**valid_board_data)
    engine = CodenamesDuetEngine(board=board)
    engine.state.clue_giver = 0
    engine.state.guesser = 1

    # BRICK -> LLM-agent (id 1); BUCKET -> LLM-civilian (id 0); ZEBRA -> not on the board.
    engine.receive_clue(clue="battle", count=2, player_id=0,
                        targets=["BRICK", "BUCKET", "ZEBRA"])

    resolved = engine.state.current_clue.targets_resolved
    assert engine.state.current_clue.targets == ["BRICK", "BUCKET", "ZEBRA"]
    assert resolved[0] == ResolvedTarget(
        word="BRICK", card_id=1, giver_role=CardRole.AGENT, revealed_at_clue=False)
    assert resolved[1] == ResolvedTarget(
        word="BUCKET", card_id=0, giver_role=CardRole.CIVILIAN, revealed_at_clue=False)
    assert resolved[2] == ResolvedTarget(
        word="ZEBRA", card_id=None, giver_role=None, revealed_at_clue=None)


def test_engine_resolve_target_snapshot_is_perspective_aware(valid_board_data: dict):
    """
    The resolved snapshot uses the CLUE-GIVER's perspective: the same word (RUSSIA, id 4, which is
    an LLM-agent but a human-civilian) resolves to different giver roles depending on the seat.
    """
    # Player 0 (LLM) seat: RUSSIA is an agent.
    engine0 = CodenamesDuetEngine(board=Board(**valid_board_data))
    engine0.state.clue_giver = 0
    engine0.state.guesser = 1
    engine0.receive_clue(clue="battle", count=1,
                         player_id=0, targets=["RUSSIA"])
    assert engine0.state.current_clue.targets_resolved[0].giver_role == CardRole.AGENT

    # Player 1 (human) seat: the very same card is a civilian.
    engine1 = CodenamesDuetEngine(board=Board(**valid_board_data))
    engine1.state.clue_giver = 1
    engine1.state.guesser = 0
    engine1.receive_clue(clue="battle", count=1,
                         player_id=1, targets=["RUSSIA"])
    assert engine1.state.current_clue.targets_resolved[0].giver_role == CardRole.CIVILIAN


@pytest.mark.parametrize("modification, expected_error", [
    ("invalid_phase", "Clues can only be given during the GIVING_CLUE phase."),
    ("invalid_player", "Only the clue giver can provide a clue."),
    ("exact_match", "Invalid clue:")
])
def test_engine_receive_clue_invalid_inputs(valid_board_data: dict, modification: str, expected_error: str):
    """
    Validates that the receive_clue method raises appropriate exceptions when given invalid inputs.

    :param valid_board_data: A fixture providing a valid board configuration as a dictionary.
    :param modification: A string indicating the type of invalid input to test.
    :param expected_error: The expected error message to be raised for the given modification.
    """
    board = Board(**valid_board_data)
    engine = CodenamesDuetEngine(board=board)

    engine.state.clue_giver = 0
    engine.state.guesser = 1
    engine.state.current_phase = GamePhase.GIVING_CLUE

    if modification == "invalid_phase":
        engine.state.current_phase = GamePhase.GUESSING
        with pytest.raises(ValueError, match=expected_error):
            engine.receive_clue(clue="TestClue", count=2, player_id=0)
    elif modification == "invalid_player":
        with pytest.raises(PermissionError, match=expected_error):
            engine.receive_clue(clue="TestClue", count=2, player_id=1)
    elif modification == "exact_match":
        with pytest.raises(ValueError, match=expected_error):
            engine.receive_clue(
                clue=valid_board_data["cards"][0]["text"], count=2, player_id=0)


@pytest.mark.parametrize("clue, count", [
    ("", 2),
    ("TestClue", 0),
])
def test_engine_receive_clue_pydantic_validation(valid_board_data: dict, clue: str, count: int):
    """
    Validates that receive_clue raises ValueError (wrapping Pydantic ValidationError) for
    structurally invalid clue entries — empty clue or non-positive count.
    """
    board = Board(**valid_board_data)
    engine = CodenamesDuetEngine(board=board)
    engine.state.clue_giver = 0
    engine.state.guesser = 1
    engine.state.current_phase = GamePhase.GIVING_CLUE

    with pytest.raises(ValueError):
        engine.receive_clue(clue=clue, count=count, player_id=0)


def test_engine_receive_clue_accepts_a_repeated_clue(valid_board_data: dict):
    """
    Repeating a clue already given in the game is legal (§8.1), whichever seat gave it first: the
    clue is validated against the board only, never against the clue history.

    :param valid_board_data: A fixture providing a valid board configuration as a dictionary.
    """
    LLM, HUMAN = 0, 1
    BUCKET, FIDDLE = 0, 6  # beige on both faces

    board = Board(**valid_board_data)
    engine = CodenamesDuetEngine(board=board)
    engine.state.clue_giver = LLM
    engine.state.guesser = HUMAN

    engine.receive_clue("battle", 1, LLM)
    assert engine.resolve_guess(card_id=BUCKET, player_id=HUMAN) == "civilian"

    # The roles alternate: the human repeats the LLM's clue, and then the LLM repeats it again.
    engine.receive_clue("battle", 1, HUMAN)
    assert engine.resolve_guess(card_id=FIDDLE, player_id=LLM) == "civilian"
    engine.receive_clue("battle", 1, LLM)

    assert [entry.clue for entry in engine.state.clue_history] == ["battle", "battle"]
    assert engine.state.current_clue.clue == "battle"
    assert engine.state.current_phase == GamePhase.GUESSING


def test_engine_resolve_guess_normal_agent(valid_board_data: dict):
    """
    Validates that the resolve_guess method correctly processes a valid guess of an agent card and
    updates the game state accordingly.

    :param valid_board_data: A fixture providing a valid board configuration as a dictionary.
    """
    board = Board(**valid_board_data)
    engine = CodenamesDuetEngine(board=board)

    # Force the guesser to be player 1 for testing
    engine.state.clue_giver = 0
    engine.state.guesser = 1
    engine.state.current_phase = GamePhase.GUESSING

    result = engine.resolve_guess(card_id=4, player_id=1)

    assert result == "agent"
    assert engine.state.board.cards[4].revealed is True
    assert 1 in engine.state.board.cards[4].revealed_by
    # Agent count for player 1 should decrease by 1
    assert engine.state.agents_remaining[1] == 8
    assert engine.state.agents_remaining[0] == 9
    # Should still be guessing phase after a correct guess
    assert engine.state.current_phase == GamePhase.GUESSING


def test_engine_resolve_guess_shared_agent(valid_board_data: dict):
    """
    Validates that the resolve_guess method correctly processes a valid guess of a shared agent card
    and updates the game state accordingly, including decreasing the agent count for both players.

    :param valid_board_data: A fixture providing a valid board configuration as a dictionary.
    """
    board = Board(**valid_board_data)
    engine = CodenamesDuetEngine(board=board)

    # Force the guesser to be player 1 for testing
    engine.state.clue_giver = 0
    engine.state.guesser = 1
    engine.state.current_phase = GamePhase.GUESSING

    result = engine.resolve_guess(card_id=1, player_id=1)

    assert result == "agent"
    assert engine.state.board.cards[1].revealed is True
    assert 1 in engine.state.board.cards[1].revealed_by
    # Agent count for both players should decrease by 1 since it's a shared agent
    assert engine.state.agents_remaining[0] == 8
    assert engine.state.agents_remaining[1] == 8
    # Should still be guessing phase after a correct guess
    assert engine.state.current_phase == GamePhase.GUESSING


def test_engine_resolve_guess_victory(valid_board_data: dict):
    """
    Validates that the resolve_guess method correctly identifies a victory condition when the last
    agent card is guessed and updates the game state to reflect the victory.

    :param valid_board_data: A fixture providing a valid board configuration as a dictionary.
    """
    board = Board(**valid_board_data)
    engine = CodenamesDuetEngine(board=board)

    # Force the guesser to be player 1 for testing
    engine.state.clue_giver = 0
    engine.state.guesser = 1
    engine.state.current_phase = GamePhase.GUESSING

    # Manually reveal all agent cards
    for card in board.cards:
        if CardRole.AGENT in [card.llm_perspective_role, card.human_perspective_role]:
            card.revealed = True
            card.revealed_by.append(1)

    # Ensure the last agent card is not revealed
    engine.state.board.cards[1].revealed = False
    engine.state.board.cards[1].revealed_by = []

    # Set remaining agents to 1 for testing victory condition
    engine.state.agents_remaining[1] = 1
    engine.state.agents_remaining[0] = 1

    result = engine.resolve_guess(card_id=1, player_id=1)

    assert result == "victory"
    assert engine.state.current_phase == GamePhase.GAME_OVER
    assert engine.state.is_game_over is True
    assert engine.state.result == "victory"


def test_engine_resolve_guess_assassin(valid_board_data: dict):
    """
    Validates that the resolve_guess method correctly processes a guess of an assassin card, updates
    the game state to reflect the loss, and ends the game.

    :param valid_board_data: A fixture providing a valid board configuration as a dictionary.
    """
    board = Board(**valid_board_data)
    engine = CodenamesDuetEngine(board=board)

    # Force the guesser to be player 1 for testing
    engine.state.clue_giver = 0
    engine.state.guesser = 1
    engine.state.current_phase = GamePhase.GUESSING

    result = engine.resolve_guess(card_id=2, player_id=1)

    assert result == "assassin"
    assert engine.state.current_phase == GamePhase.GAME_OVER
    assert engine.state.is_game_over is True
    assert engine.state.result == "loss_assassin"


def test_engine_resolve_guess_civilian(valid_board_data: dict):
    """
    Validates that the resolve_guess method correctly processes a guess of a civilian card, updates
    the game state to reflect the loss of a timer token, and switches roles appropriately.

    :param valid_board_data: A fixture providing a valid board configuration as a dictionary.
    """
    board = Board(**valid_board_data)
    engine = CodenamesDuetEngine(board=board)

    # Force the guesser to be player 1 for testing
    engine.state.clue_giver = 0
    engine.state.guesser = 1
    engine.state.current_phase = GamePhase.GUESSING

    result = engine.resolve_guess(card_id=5, player_id=1)

    assert result == "civilian"
    assert engine.state.board.cards[5].revealed is False
    assert engine.state.board.cards[5].revealed_by == []
    assert 1 in engine.state.board.cards[5].time_marker_by
    assert engine.state.timer_tokens == 8
    # Should switch roles after guessing a civilian
    assert engine.state.clue_giver == 1
    assert engine.state.guesser == 0
    assert engine.state.current_phase == GamePhase.GIVING_CLUE


def test_engine_resolve_guess_time_marker_from_other_player(valid_board_data: dict):
    """
    Validates that the resolve_guess method correctly processes a guess of a civilian card that has
    been marked by a time token from the other player, updates the game state to reflect the loss of
    a timer token, and switches roles appropriately.

    :param valid_board_data: A fixture providing a valid board configuration as a dictionary.
    """
    board = Board(**valid_board_data)
    engine = CodenamesDuetEngine(board=board)

    # Force the guesser to be player 1 for testing
    engine.state.clue_giver = 0
    engine.state.guesser = 1
    engine.state.current_phase = GamePhase.GUESSING

    # Manually mark a card with a time token by the other player for testing
    engine.state.board.cards[18].time_marker_by.append(0)

    result = engine.resolve_guess(card_id=18, player_id=1)

    assert result == "civilian"
    assert 1 in engine.state.board.cards[18].time_marker_by
    assert engine.state.timer_tokens == 8
    # Should switch roles after guessing a civilian
    assert engine.state.clue_giver == 1
    assert engine.state.guesser == 0
    assert engine.state.current_phase == GamePhase.GIVING_CLUE


@pytest.mark.parametrize("modification, expected_error", [
    ("invalid_phase", "Guesses can only be made during the GUESSING, SUDDEN_DEATH_HUMAN, or SUDDEN_DEATH_LLM phase."),
    ("invalid_player", "Only the guesser can make guesses."),
    ("unknown_card_id", "There is no card with id 99 on this board."),
    ("covered_by_own_agent_card",
     "This card is covered by an agent card and cannot be guessed."),
    ("covered_by_other_seats_agent_card",
     "This card is covered by an agent card and cannot be guessed."),
    ("covered_by_two_time_tokens",
     "This card is covered by two time tokens and cannot be guessed."),
    ("time_token", "This card is currently marked by a time token and cannot be guessed."),
    ("sudden_death_no_agents_left",
     "The player has already revealed all of their agents and cannot make more guesses.")
])
def test_engine_resolve_guess_invalid_inputs(valid_board_data: dict, modification: str, expected_error: str):
    """
    Validates that the resolve_guess method raises appropriate exceptions when given invalid inputs
    during the GUESSING or SUDDEN_DEATH phases.

    :param valid_board_data: A fixture providing a valid board configuration as a dictionary.
    :param modification: A string indicating the type of invalid input to test.
    :param expected_error: The expected error message to be raised for the given modification.
    """
    board = Board(**valid_board_data)
    engine = CodenamesDuetEngine(board=board)

    # Force the guesser to be player 1 for testing
    engine.state.clue_giver = 0
    engine.state.guesser = 1
    engine.state.current_phase = GamePhase.GUESSING

    if modification == "invalid_phase":
        engine.state.current_phase = GamePhase.GIVING_CLUE
        with pytest.raises(PermissionError, match=expected_error):
            engine.resolve_guess(card_id=0, player_id=1)
    elif modification == "invalid_player":
        with pytest.raises(PermissionError, match=expected_error):
            engine.resolve_guess(card_id=0, player_id=0)
    elif modification == "unknown_card_id":
        with pytest.raises(ValueError, match=expected_error):
            engine.resolve_guess(card_id=99, player_id=1)
    elif modification == "covered_by_own_agent_card":
        engine.state.board.cards[0].revealed = True
        engine.state.board.cards[0].revealed_by.append(1)
        with pytest.raises(ValueError, match=expected_error):
            engine.resolve_guess(card_id=0, player_id=1)
    elif modification == "covered_by_other_seats_agent_card":
        # Coverage is a property of the word, not of the seat asking: the guesser is absent from
        # revealed_by, yet an agent card still sits on the word.
        engine.state.board.cards[0].revealed = True
        engine.state.board.cards[0].revealed_by.append(0)
        with pytest.raises(ValueError, match=expected_error):
            engine.resolve_guess(card_id=0, player_id=1)
    elif modification == "covered_by_two_time_tokens":
        engine.state.board.cards[0].time_marker_by.extend([0, 1])
        with pytest.raises(ValueError, match=expected_error):
            engine.resolve_guess(card_id=0, player_id=1)
    elif modification == "time_token":
        engine.state.board.cards[0].time_marker_by.append(
            1)  # Marked by the guesser
        with pytest.raises(ValueError, match=expected_error):
            engine.resolve_guess(card_id=0, player_id=1)
    elif modification == "sudden_death_no_agents_left":
        engine.state.current_phase = GamePhase.SUDDEN_DEATH_HUMAN
        engine.state.agents_remaining[1] = 0  # No agents left for the guesser
        with pytest.raises(PermissionError, match=expected_error):
            engine.resolve_guess(card_id=0, player_id=1)


def test_engine_covered_agent_card_cannot_be_retouched_by_the_other_seat(valid_board_data: dict):
    """
    A word covered by an agent card is out of play for BOTH players, including the seat that did not
    cover it and is therefore absent from ``revealed_by``.

    This is the exact sequence that used to end games: ANT is an agent on the human's face and an
    ASSASSIN on the LLM's. The LLM covers it with an agent card, then on the next turn - with the
    LLM now giving the clue - the human touches it again and the guess is resolved against the
    LLM's face, losing the game on a word that was already covered.

    :param valid_board_data: A fixture providing a valid board configuration as a dictionary.
    """
    ANT = 2  # human_perspective_role=agent, llm_perspective_role=assassin
    LLM, HUMAN = 0, 1

    board = Board(**valid_board_data)
    engine = CodenamesDuetEngine(board=board)
    engine.state.clue_giver = HUMAN
    engine.state.guesser = LLM

    # Turn 1: the human gives the clue and the LLM covers ANT with an agent card.
    engine.receive_clue("insect", 1, HUMAN)
    assert engine.resolve_guess(card_id=ANT, player_id=LLM) == "agent"
    assert engine.state.board.cards[ANT].revealed is True
    assert engine.state.board.cards[ANT].revealed_by == [LLM]
    engine.pass_turn(LLM)

    # Turn 2: roles have switched, so ANT would now resolve against the LLM's face (ASSASSIN).
    engine.receive_clue("tiny", 1, LLM)
    assert engine.state.guesser == HUMAN
    tokens_before = engine.state.timer_tokens

    with pytest.raises(ValueError, match="covered by an agent card"):
        engine.resolve_guess(card_id=ANT, player_id=HUMAN)

    # The refused guess costs nothing: no loss, no token, and it does not count as an attempt.
    assert engine.state.is_game_over is False
    assert engine.state.result is None
    assert engine.state.timer_tokens == tokens_before
    assert engine.state.guesses_made_this_turn == 0
    assert engine.state.current_phase == GamePhase.GUESSING


def test_engine_shared_agent_is_covered_for_both_seats(valid_board_data: dict):
    """
    Covering a shared agent (green on both faces) puts it out of reach for both seats, and the
    engine reports it as covered rather than as "already revealed for you".

    :param valid_board_data: A fixture providing a valid board configuration as a dictionary.
    """
    BRICK = 1  # agent on both faces

    board = Board(**valid_board_data)
    engine = CodenamesDuetEngine(board=board)
    engine.state.clue_giver = 0
    engine.state.guesser = 1
    engine.state.current_phase = GamePhase.GUESSING

    assert engine.resolve_guess(card_id=BRICK, player_id=1) == "agent"

    card = engine.state.board.cards[BRICK]
    assert card.is_covered is True
    assert card.is_guessable_by(0) is False
    assert card.is_guessable_by(1) is False


def test_engine_single_time_token_only_blocks_the_seat_that_left_it(valid_board_data: dict):
    """
    A single time token blocks only the seat that touched the word; the other seat may still try it.
    A second token, one per direction, covers the word for both.

    :param valid_board_data: A fixture providing a valid board configuration as a dictionary.
    """
    board = Board(**valid_board_data)
    engine = CodenamesDuetEngine(board=board)
    card = engine.state.board.cards[0]

    card.time_marker_by = [1]
    assert card.is_covered is False
    assert card.is_guessable_by(0) is True
    assert card.is_guessable_by(1) is False

    card.time_marker_by = [1, 0]
    assert card.is_covered is True
    assert card.is_guessable_by(0) is False
    assert card.is_guessable_by(1) is False


def test_engine_resolve_guess_change_to_sudden_death(valid_board_data: dict):
    """
    Validates that the resolve_guess method correctly transitions the game to the SUDDEN_DEATH phase
    when the timer tokens run out, and that the game state is updated accordingly.

    :param valid_board_data: A fixture providing a valid board configuration as a dictionary.
    """
    board = Board(**valid_board_data)
    engine = CodenamesDuetEngine(board=board)

    # Force the guesser to be player 1 for testing
    engine.state.clue_giver = 0
    engine.state.guesser = 1
    engine.state.current_phase = GamePhase.GUESSING

    # Force the timer tokens to run out
    engine.state.timer_tokens = 1

    result = engine.resolve_guess(card_id=16, player_id=1)
    assert result == "civilian"
    assert engine.state.current_phase == GamePhase.SUDDEN_DEATH_LLM
    assert engine.state.timer_tokens == 0


def test_engine_resolve_guess_sudden_death(valid_board_data: dict):
    """
    Validates that the resolve_guess method correctly identifies a victory condition in the 
    SUDDEN_DEATH phase when the last agent card is guessed, and updates the game state to reflect
    the victory.

    :param valid_board_data: A fixture providing a valid board configuration as a dictionary.
    """
    board = Board(**valid_board_data)
    engine = CodenamesDuetEngine(board=board)

    engine.state.current_phase = GamePhase.SUDDEN_DEATH_HUMAN
    # Only one agent left for the guesser
    engine.state.agents_remaining[1] = 1
    # Only one agent left for the LLM
    engine.state.agents_remaining[0] = 1

    # Guess the last agent card correctly (card 1 is a shared agent, so both drop to 0 → victory)
    result = engine.resolve_guess(card_id=1, player_id=1)

    assert result == "victory_sd"
    assert engine.state.current_phase == GamePhase.GAME_OVER
    assert engine.state.is_game_over is True
    assert engine.state.result == "victory_sd"


@pytest.mark.parametrize("modification, result", [
    ("guess_civilian", "loss_civilian_sd"),
    ("guess_assassin", "loss_assassin_sd")
])
def test_engine_resolve_guess_sudden_death_loss_civilian(valid_board_data: dict, modification: str, result: str):
    """
    Validates that the resolve_guess method correctly identifies a loss condition in the
    SUDDEN_DEATH phase when a civilian or assassin card is guessed, and updates the game state to
    reflect the loss.

    :param valid_board_data: A fixture providing a valid board configuration as a dictionary.
    :param modification: A string indicating whether to test guessing a civilian or an assassin card.
    :param result: The expected result string to be set in the game state for the loss
    """
    board = Board(**valid_board_data)
    engine = CodenamesDuetEngine(board=board)

    engine.state.current_phase = GamePhase.SUDDEN_DEATH_HUMAN
    # Only one agent left for the guesser
    engine.state.agents_remaining[1] = 1
    # Only one agent left for the LLM
    engine.state.agents_remaining[0] = 1

    if modification == "guess_assassin":
        # Guess the assassin card
        result = engine.resolve_guess(card_id=2, player_id=1)
        assert result == "loss_assassin_sd"
    else:
        # Guess a civilian card
        result = engine.resolve_guess(card_id=5, player_id=1)
        assert result == "loss_civilian_sd"

    assert engine.state.current_phase == GamePhase.GAME_OVER
    assert engine.state.is_game_over is True
    assert engine.state.result == result


def test_engine_sudden_death_llm_to_human_transition(valid_board_data: dict):
    """
    Validates that when the LLM guesses their last agent in SUDDEN_DEATH_LLM, the phase
    transitions to SUDDEN_DEATH_HUMAN (not victory) when the human still has agents remaining.
    """
    board = Board(**valid_board_data)
    engine = CodenamesDuetEngine(board=board)

    engine.state.current_phase = GamePhase.SUDDEN_DEATH_LLM
    engine.state.agents_remaining[0] = 1   # LLM has one agent left
    engine.state.agents_remaining[1] = 3   # human still has agents

    result = engine.resolve_guess(card_id=2, player_id=0)

    assert result == "agent"
    assert engine.state.agents_remaining[0] == 0
    assert engine.state.current_phase == GamePhase.SUDDEN_DEATH_HUMAN


def test_engine_sudden_death_skip_human_if_done(valid_board_data: dict):
    """
    Validates that when the human has no agents remaining at the time sudden death triggers
    (timer hits 0), the engine goes directly to SUDDEN_DEATH_LLM instead of SUDDEN_DEATH_HUMAN.
    """
    board = Board(**valid_board_data)
    engine = CodenamesDuetEngine(board=board)

    engine.state.clue_giver = 0
    engine.state.guesser = 1
    engine.state.current_phase = GamePhase.GUESSING
    engine.state.timer_tokens = 1
    engine.state.agents_remaining[1] = 0   # human already done
    engine.state.agents_remaining[0] = 3   # LLM still has agents

    # Guess a civilian to drain the last timer token and trigger _switch_roles
    # ensure card 16 is clean
    engine.state.board.cards[16].time_marker_by = []
    result = engine.resolve_guess(card_id=16, player_id=1)

    assert result == "civilian"
    assert engine.state.timer_tokens == 0
    assert engine.state.current_phase == GamePhase.SUDDEN_DEATH_LLM


@pytest.mark.parametrize("player_id, phase, agents, card_id, agents_after", [
    # CAVE is green only on the human's face: one of the LLM's words
    (0, GamePhase.SUDDEN_DEATH_LLM, [2, 3], 5, [1, 3]),
    # RUSSIA is green only on the LLM's face: one of the human's words
    (1, GamePhase.SUDDEN_DEATH_HUMAN, [0, 3], 4, [0, 2]),
])
def test_engine_concede_sudden_death_ends_the_game_as_a_loss(
        valid_board_data: dict, player_id: int, phase: GamePhase, agents: list[int],
        card_id: int, agents_after: list[int]):
    """
    Validates that the seat guessing in sudden death can stop with words still pending, and that
    stopping ends the game as a loss for both without touching a card (§7.3, §14). Before this
    action existed, a seat that proposed nothing playable left the game stuck in its sudden-death
    phase.

    :param valid_board_data: A fixture providing a valid board configuration as a dictionary.
    :param player_id: The seat guessing in sudden death.
    :param phase: That seat's sudden-death phase.
    :param agents: agents_remaining when sudden death starts.
    :param card_id: One of this seat's words, found before it stops.
    :param agents_after: agents_remaining after that hit.
    """
    board = Board(**valid_board_data)
    engine = CodenamesDuetEngine(board=board)

    engine.state.current_phase = phase
    engine.state.timer_tokens = 0
    engine.state.agents_remaining = agents

    assert engine.resolve_guess(card_id=card_id, player_id=player_id) == "agent"
    assert engine.state.current_phase == phase

    assert engine.concede_sudden_death(player_id) == "loss_stopped_sd"
    assert engine.state.current_phase == GamePhase.GAME_OVER
    assert engine.state.is_game_over is True
    assert engine.state.result == "loss_stopped_sd"
    # Conceding touches no card and spends nothing.
    assert engine.state.agents_remaining == agents_after
    assert engine.state.timer_tokens == 0
    assert [c.id for c in engine.state.board.cards if c.revealed] == [card_id]


@pytest.mark.parametrize("phase, player_id", [
    (GamePhase.GUESSING, 1),            # normal play: the turn ends with pass_turn instead
    (GamePhase.GIVING_CLUE, 1),
    (GamePhase.GAME_OVER, 0),
    (GamePhase.SUDDEN_DEATH_LLM, 1),    # not this seat's sudden-death turn
    (GamePhase.SUDDEN_DEATH_HUMAN, 0),
    (GamePhase.SUDDEN_DEATH_LLM, 2),    # not a seat
])
def test_engine_concede_sudden_death_rejected(
        valid_board_data: dict, phase: GamePhase, player_id: int):
    """
    Validates that sudden death can only be conceded during a sudden-death phase, and only by the
    seat guessing in it. A rejected concession leaves the game untouched.

    :param valid_board_data: A fixture providing a valid board configuration as a dictionary.
    :param phase: The phase the game is in.
    :param player_id: The seat attempting to concede.
    """
    board = Board(**valid_board_data)
    engine = CodenamesDuetEngine(board=board)

    engine.state.current_phase = phase
    engine.state.clue_giver = 0
    engine.state.guesser = 1

    with pytest.raises(PermissionError):
        engine.concede_sudden_death(player_id)

    assert engine.state.current_phase == phase
    assert engine.state.result is None


def test_engine_pass_turn(valid_board_data: dict):
    """
    Validates that the pass_turn method correctly allows the guesser to pass their turn during the
    GUESSING phase, updates the game state to switch roles, and decreases the timer tokens.

    :param valid_board_data: A fixture providing a valid board configuration as a dictionary.
    """
    board = Board(**valid_board_data)
    engine = CodenamesDuetEngine(board=board)

    # Force the clue giver to be player 0 and guesser to be player 1 for testing
    engine.state.clue_giver = 0
    engine.state.guesser = 1
    engine.state.current_phase = GamePhase.GUESSING
    engine.state.guesses_made_this_turn = 1  # Simulate that a guess has been made

    engine.pass_turn(player_id=1)

    # Should switch roles after passing the turn
    assert engine.state.clue_giver == 1
    assert engine.state.guesser == 0
    assert engine.state.current_phase == GamePhase.GIVING_CLUE
    assert engine.state.timer_tokens == 8


def test_engine_pass_turn_save_clue(valid_board_data: dict):
    """
    Validates that the clue provided for the current turn is correctly saved in the clue history
    when the guesser passes their turn, and that the clue data is accurate.

    :param valid_board_data: A fixture providing a valid board configuration as a dictionary.
    """
    board = Board(**valid_board_data)
    engine = CodenamesDuetEngine(board=board)

    # Force the clue giver to be player 0 and guesser to be player 1 for testing
    engine.state.clue_giver = 0
    engine.state.guesser = 1
    engine.state.current_phase = GamePhase.GUESSING
    # Simulate that a guess has been made and a clue is active
    engine.state.guesses_made_this_turn = 1
    engine.state.current_clue = ClueEntry(
        clue="TestClue", count=2, clue_giver=0, turn_number=1)

    engine.pass_turn(player_id=1)

    # The current clue should be saved in the clue history after passing the turn
    assert len(engine.state.clue_history) == 1
    assert engine.state.clue_history[0].clue == "TestClue"
    # Count should be 0 since the turn is passed without using up guesses
    assert engine.state.clue_history[0].count == 2
    assert engine.state.clue_history[0].clue_giver == 0
    assert engine.state.clue_history[0].turn_number == 1


@pytest.mark.parametrize("modification, expected_error", [
    ("invalid_phase", "Turns can only be passed during the GUESSING phase."),
    ("invalid_player", "Only the guesser can pass the turn."),
    ("no_guesses_made", "The guesser must make at least one guess before passing.")
])
def test_engine_pass_turn_invalid_inputs(valid_board_data: dict, modification: str, expected_error: str):
    """
    Validates that the pass_turn method raises appropriate exceptions when given invalid inputs,
    such as being called during the wrong phase, by the wrong player, or without any guesses made.

    :param valid_board_data: A fixture providing a valid board configuration as a dictionary.
    :param modification: A string indicating the type of invalid input to test.
    :param expected_error: The expected error message to be raised for the given modification.
    """
    board = Board(**valid_board_data)
    engine = CodenamesDuetEngine(board=board)

    # Force the clue giver to be player 0 and guesser to be player 1 for testing
    engine.state.clue_giver = 0
    engine.state.guesser = 1
    engine.state.current_phase = GamePhase.GUESSING

    if modification == "invalid_phase":
        engine.state.current_phase = GamePhase.GIVING_CLUE
        with pytest.raises(PermissionError, match=expected_error):
            engine.pass_turn(player_id=1)
    elif modification == "invalid_player":
        with pytest.raises(PermissionError, match=expected_error):
            engine.pass_turn(player_id=0)
    elif modification == "no_guesses_made":
        engine.state.guesses_made_this_turn = 0  # No guesses made
        with pytest.raises(ValueError, match=expected_error):
            engine.pass_turn(player_id=1)


def test_engine_seat_without_pending_words_gives_every_remaining_clue(valid_board_data: dict):
    """
    Once every green on the LLM's face is covered, the human has nothing left to guess, so from then
    on the human gives every clue and the LLM is always the guesser (§6.1, §6.8) - whether the turn
    ends with a stop or a miss, and up to the sudden-death transition.

    With strict alternation the human was handed the guesser role on alternate turns: it cannot pass
    without a guess, and every word it can touch is beige or black on the LLM's face, so the turn
    always cost a token - or the game, on an assassin.

    :param valid_board_data: A fixture providing a valid board configuration as a dictionary.
    """
    LLM, HUMAN = 0, 1
    LLM_GREENS = [1, 4, 8, 11, 12, 15, 17, 19, 24]  # the human's pending words
    ANT = 2  # agent on the human's face
    HUMAN_CIVILIANS = [0, 6, 7, 13, 14, 18, 22]  # beige on the human's face, never touched here

    board = Board(**valid_board_data)
    engine = CodenamesDuetEngine(board=board)
    engine.state.clue_giver = LLM
    engine.state.guesser = HUMAN

    # Turn 1: the human covers every green on the LLM's face; the last one ends the turn (§6.8).
    engine.receive_clue("battle", 9, LLM)
    for card_id in LLM_GREENS[:-1]:
        assert engine.resolve_guess(card_id=card_id, player_id=HUMAN) == "agent"
    assert engine.resolve_guess(card_id=LLM_GREENS[-1], player_id=HUMAN) == "agent_turn_end"
    assert engine.state.agents_remaining == [6, 0]

    # Turn 2: the LLM still has pending words, so the roles alternate as usual.
    assert (engine.state.clue_giver, engine.state.guesser) == (HUMAN, LLM)
    engine.receive_clue("insect", 1, HUMAN)
    assert engine.resolve_guess(card_id=ANT, player_id=LLM) == "agent"
    engine.pass_turn(LLM)

    # Every remaining turn - here all ending on a miss - keeps the human as the clue giver.
    for card_id in HUMAN_CIVILIANS:
        assert (engine.state.clue_giver, engine.state.guesser) == (HUMAN, LLM)
        assert engine.state.current_phase == GamePhase.GIVING_CLUE
        engine.receive_clue("ocean", 1, HUMAN)
        assert engine.resolve_guess(card_id=card_id, player_id=LLM) == "civilian"

    # The last token is gone: sudden death for the only seat with pending words.
    assert engine.state.timer_tokens == 0
    assert engine.state.agents_remaining == [5, 0]
    assert engine.state.current_phase == GamePhase.SUDDEN_DEATH_LLM


@pytest.mark.parametrize("exhausted", [0, 1])
def test_engine_seat_can_run_out_of_pending_words_on_the_partners_turn(valid_board_data: dict,
                                                                       exhausted: int):
    """
    A seat can run out of pending words while it is the clue giver: the partner covering a shared
    green also covers it for that seat. The seat that just gave the clue then gives the next one too,
    so it gives two clues in a row (§6.8).

    :param valid_board_data: A fixture providing a valid board configuration as a dictionary.
    :param exhausted: The seat that ends up with no pending words.
    """
    other = 1 - exhausted
    SHARED_GREENS = [1, 8, 17]  # BRICK, TATTOO, NAPOLEON
    # Greens only on the partner's face: what the exhausted seat has to find on its own.
    OWN_PENDING = {0: [2, 5, 9, 16, 20, 21], 1: [4, 11, 12, 15, 19, 24]}[exhausted]
    BUCKET = 0  # beige on both faces

    board = Board(**valid_board_data)
    engine = CodenamesDuetEngine(board=board)
    engine.state.clue_giver = other
    engine.state.guesser = exhausted

    # Turn 1: the exhausted seat finds every pending word except the shared ones.
    engine.receive_clue("battle", 6, other)
    for card_id in OWN_PENDING:
        assert engine.resolve_guess(card_id=card_id, player_id=exhausted) == "agent"
    engine.pass_turn(exhausted)

    # Turn 2: plain alternation. The partner covers the shared greens, which empties the giver.
    assert (engine.state.clue_giver, engine.state.guesser) == (exhausted, other)
    engine.receive_clue("ocean", 3, exhausted)
    for card_id in SHARED_GREENS:
        assert engine.resolve_guess(card_id=card_id, player_id=other) == "agent"
    assert engine.state.agents_remaining[exhausted] == 0
    assert engine.state.agents_remaining[other] == 6
    engine.pass_turn(other)

    # Turn 3: the seat with nothing pending gives the clue again instead of guessing.
    assert (engine.state.clue_giver, engine.state.guesser) == (exhausted, other)
    engine.receive_clue("winter", 1, exhausted)
    assert engine.resolve_guess(card_id=BUCKET, player_id=other) == "civilian"

    # Turn 4: a miss does not hand the guess back either.
    assert (engine.state.clue_giver, engine.state.guesser) == (exhausted, other)
    assert engine.state.current_phase == GamePhase.GIVING_CLUE


def test_engine_turn_ends_when_the_guesser_covers_its_last_pending_word(valid_board_data: dict):
    """
    The hit that covers the guesser's last pending word ends the turn by itself (§6.8): the giver
    must tell it there is nothing left to guess, and any word it could still touch is beige or black
    on the giver's face. The turn closes as a voluntary stop - one token - and the partner guesses.

    This is the sequence that used to cost games: the human covers RUSSIA, its last pending word,
    the turn stays in GUESSING, and the human goes on to touch MAKEUP - an assassin on the LLM's face.

    :param valid_board_data: A fixture providing a valid board configuration as a dictionary.
    """
    LLM, HUMAN = 0, 1
    RUSSIA = 4  # agent on the LLM's face
    MAKEUP = 14  # assassin on the LLM's face
    LLM_GREENS = [1, 8, 11, 12, 15, 17, 19, 24, RUSSIA]  # the human's pending words, RUSSIA last

    board = Board(**valid_board_data)
    engine = CodenamesDuetEngine(board=board)
    engine.state.clue_giver = LLM
    engine.state.guesser = HUMAN
    turn_before = engine.state.turn_number

    engine.receive_clue("battle", 9, LLM)
    for card_id in LLM_GREENS[:-1]:
        assert engine.resolve_guess(card_id=card_id, player_id=HUMAN) == "agent"
    assert engine.state.current_phase == GamePhase.GUESSING

    assert engine.resolve_guess(card_id=RUSSIA, player_id=HUMAN) == "agent_turn_end"
    assert engine.state.agents_remaining == [6, 0]

    # Closed exactly like pass_turn: one token, clue archived, next turn with the LLM guessing.
    assert engine.state.timer_tokens == 8
    assert engine.state.current_phase == GamePhase.GIVING_CLUE
    assert (engine.state.clue_giver, engine.state.guesser) == (HUMAN, LLM)
    assert engine.state.turn_number == turn_before + 1
    assert engine.state.guesses_made_this_turn == 0
    assert engine.state.current_clue is None
    assert [entry.clue for entry in engine.state.clue_history] == ["battle"]

    # The turn is over, so the human can no longer touch MAKEUP and lose the game on it.
    with pytest.raises(PermissionError):
        engine.resolve_guess(card_id=MAKEUP, player_id=HUMAN)
    assert engine.state.is_game_over is False


@pytest.mark.parametrize("exhausted, sudden_death_phase", [
    (0, GamePhase.SUDDEN_DEATH_HUMAN),
    (1, GamePhase.SUDDEN_DEATH_LLM),
])
def test_engine_last_pending_word_on_the_last_token_starts_sudden_death(
        valid_board_data: dict, exhausted: int, sudden_death_phase: GamePhase):
    """
    The stop forced by the last pending word spends a token like any other stop, so on the last
    token it starts sudden death for the partner, the only seat with words left.

    :param valid_board_data: A fixture providing a valid board configuration as a dictionary.
    :param exhausted: The guesser that covers its last pending word.
    :param sudden_death_phase: The sudden-death phase of the partner.
    """
    other = 1 - exhausted
    # A green only on the partner's face: CAVE is green for the human, RUSSIA for the LLM.
    last_word = {0: 5, 1: 4}[exhausted]

    board = Board(**valid_board_data)
    engine = CodenamesDuetEngine(board=board)
    engine.state.clue_giver = other
    engine.state.guesser = exhausted
    engine.state.current_phase = GamePhase.GUESSING
    engine.state.timer_tokens = 1
    engine.state.agents_remaining[exhausted] = 1
    engine.state.agents_remaining[other] = 3

    assert engine.resolve_guess(card_id=last_word, player_id=exhausted) == "agent_turn_end"
    assert engine.state.timer_tokens == 0
    assert engine.state.current_phase == sudden_death_phase
    assert engine.state.sd_measurement_pending is True


def test_engine_seeded_rng_is_deterministic(valid_board_data: dict):
    """
    Validates that injecting a seeded random.Random produces a deterministic start player, and that
    two engines seeded identically agree on the start player (i.e. the engine's randomness is
    reproducible and per-instance).

    :param valid_board_data: A fixture providing a valid board configuration as a dictionary.
    """
    board = Board(**valid_board_data)

    engine_a = CodenamesDuetEngine(board=board, rng=random.Random(42))
    engine_b = CodenamesDuetEngine(board=board, rng=random.Random(42))

    # A seeded RNG makes the start player deterministic and reproducible across instances.
    assert engine_a.state.clue_giver == engine_b.state.clue_giver
    assert engine_a.state.guesser == engine_b.state.guesser
    assert engine_a.state.clue_giver != engine_a.state.guesser
    # Independently recompute what random.Random(42) yields for the start-player draw.
    expected_start = random.Random(42).choice([0, 1])
    assert engine_a.state.clue_giver == expected_start


def test_engine_default_rng_still_works(valid_board_data: dict):
    """
    Validates that omitting the rng argument still produces a valid game (a fresh unseeded
    random.Random is used), preserving the previous non-breaking behaviour.

    :param valid_board_data: A fixture providing a valid board configuration as a dictionary.
    """
    board = Board(**valid_board_data)
    engine = CodenamesDuetEngine(board=board)

    assert engine.state.clue_giver in [0, 1]
    assert engine.state.guesser in [0, 1]
    assert engine.state.clue_giver != engine.state.guesser
