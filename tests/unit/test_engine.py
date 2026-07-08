import random
import pytest
from backend.app.core.engine import CodenamesDuetEngine
from backend.app.models.game_schemas import Board, GamePhase, CardRole, ClueEntry


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
    ("already_revealed", "This card has already been revealed."),
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
    elif modification == "already_revealed":
        engine.state.board.cards[0].revealed = True
        engine.state.board.cards[0].revealed_by.append(1)
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
    assert engine.state.current_phase == GamePhase.SUDDEN_DEATH_HUMAN
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


def test_engine_sudden_death_human_to_llm_transition(valid_board_data: dict):
    """
    Validates that when the human guesses their last agent in SUDDEN_DEATH_HUMAN, the phase
    transitions to SUDDEN_DEATH_LLM (not victory) when the LLM still has agents remaining.
    """
    board = Board(**valid_board_data)
    engine = CodenamesDuetEngine(board=board)

    engine.state.current_phase = GamePhase.SUDDEN_DEATH_HUMAN
    engine.state.agents_remaining[1] = 1   # human has one agent left
    engine.state.agents_remaining[0] = 3   # LLM still has agents

    # Card 4 is an LLM-only agent (llm=AGENT, human=CIVILIAN), so only agents_remaining[1] drops
    result = engine.resolve_guess(card_id=4, player_id=1)

    assert result == "agent"
    assert engine.state.agents_remaining[1] == 0
    assert engine.state.current_phase == GamePhase.SUDDEN_DEATH_LLM


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
    engine.state.board.cards[16].time_marker_by = []   # ensure card 16 is clean
    result = engine.resolve_guess(card_id=16, player_id=1)

    assert result == "civilian"
    assert engine.state.timer_tokens == 0
    assert engine.state.current_phase == GamePhase.SUDDEN_DEATH_LLM


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
