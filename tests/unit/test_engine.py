import pytest
from backend.app.core.engine import CodenamesDuetEngine
from backend.app.models.game_schemas import Board, GamePhase, CardRole


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
    ("empty_clue", "Clue cannot be empty."),
    ("invalid_count", "Clue count must be at least 1."),
    ("exact_match", "Clue cannot be the same as any word on the board.")
])
def test_engine_receive_clue_invalid_inputs(valid_board_data: dict, modification, expected_error):
    board = Board(**valid_board_data)
    engine = CodenamesDuetEngine(board=board)

    # Force the clue giver to be player 0 for testing
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
    elif modification == "empty_clue":
        with pytest.raises(ValueError, match=expected_error):
            engine.receive_clue(clue="", count=2, player_id=0)
    elif modification == "invalid_count":
        with pytest.raises(ValueError, match=expected_error):
            engine.receive_clue(clue="TestClue", count=0, player_id=0)
    elif modification == "exact_match":
        with pytest.raises(ValueError, match=expected_error):
            engine.receive_clue(
                clue=valid_board_data["cards"][0]["text"], count=2, player_id=0)


def test_engine_resolve_guess_normal_agent(valid_board_data: dict):
    board = Board(**valid_board_data)
    engine = CodenamesDuetEngine(board=board)

    # Force the guesser to be player 1 for testing
    engine.state.clue_giver = 0
    engine.state.guesser = 1
    engine.state.current_phase = GamePhase.GUESSING

    result = engine.resolve_guess(card_id=3, player_id=1)

    assert result == "agent"
    assert engine.state.board.cards[3].revealed is True
    assert engine.state.board.cards[3].revealed_by == 1
    # Agent count for player 1 should decrease by 1
    assert engine.state.agents_remaining[1] == 8
    assert engine.state.agents_remaining[0] == 9
    # Should still be guessing phase after a correct guess
    assert engine.state.current_phase == GamePhase.GUESSING


def test_engine_resolve_guess_shared_agent(valid_board_data: dict):
    board = Board(**valid_board_data)
    engine = CodenamesDuetEngine(board=board)

    # Force the guesser to be player 1 for testing
    engine.state.clue_giver = 0
    engine.state.guesser = 1
    engine.state.current_phase = GamePhase.GUESSING

    result = engine.resolve_guess(card_id=0, player_id=1)

    assert result == "agent"
    assert engine.state.board.cards[0].revealed is True
    assert engine.state.board.cards[0].revealed_by == 1
    # Agent count for both players should decrease by 1 since it's a shared agent
    assert engine.state.agents_remaining[0] == 8
    assert engine.state.agents_remaining[1] == 8
    # Should still be guessing phase after a correct guess
    assert engine.state.current_phase == GamePhase.GUESSING


def test_engine_resolve_guess_victory(valid_board_data: dict):
    board = Board(**valid_board_data)
    engine = CodenamesDuetEngine(board=board)

    # Force the guesser to be player 1 for testing
    engine.state.clue_giver = 0
    engine.state.guesser = 1
    engine.state.current_phase = GamePhase.GUESSING

    # Manually reveal all agent cards
    for card in board.cards:
        if CardRole.AGENT in [card.llm_role, card.human_role]:
            card.revealed = True
            card.revealed_by = 1

    # Ensure the last agent card is not revealed
    engine.state.board.cards[0].revealed = False
    engine.state.board.cards[0].revealed_by = None

    # Set remaining agents to 1 for testing victory condition
    engine.state.agents_remaining[1] = 1
    engine.state.agents_remaining[0] = 1

    result = engine.resolve_guess(card_id=0, player_id=1)

    assert result == "victory"
    assert engine.state.current_phase == GamePhase.GAME_OVER
    assert engine.state.is_game_over is True
    assert engine.state.result == "victory"


def test_engine_resolve_guess_assassin(valid_board_data: dict):
    board = Board(**valid_board_data)
    engine = CodenamesDuetEngine(board=board)

    # Force the guesser to be player 1 for testing
    engine.state.clue_giver = 0
    engine.state.guesser = 1
    engine.state.current_phase = GamePhase.GUESSING

    result = engine.resolve_guess(card_id=15, player_id=1)

    assert result == "assassin"
    assert engine.state.current_phase == GamePhase.GAME_OVER
    assert engine.state.is_game_over is True
    assert engine.state.result == "loss_assassin"


def test_engine_resolve_guess_civilian(valid_board_data: dict):
    board = Board(**valid_board_data)
    engine = CodenamesDuetEngine(board=board)

    # Force the guesser to be player 1 for testing
    engine.state.clue_giver = 0
    engine.state.guesser = 1
    engine.state.current_phase = GamePhase.GUESSING

    result = engine.resolve_guess(card_id=17, player_id=1)

    assert result == "civilian"
    assert engine.state.board.cards[17].revealed is False
    assert engine.state.board.cards[17].revealed_by is None
    assert 1 in engine.state.board.cards[17].time_marker_by
    assert engine.state.timer_tokens == 8
    # Should switch roles after guessing a civilian
    assert engine.state.clue_giver == 1
    assert engine.state.guesser == 0
    assert engine.state.current_phase == GamePhase.GIVING_CLUE


def test_engine_resolve_guess_time_marker_from_other_player(valid_board_data: dict):
    board = Board(**valid_board_data)
    engine = CodenamesDuetEngine(board=board)

    # Force the guesser to be player 1 for testing
    engine.state.clue_giver = 0
    engine.state.guesser = 1
    engine.state.current_phase = GamePhase.GUESSING

    # Manually mark a card with a time token by the other player for testing
    engine.state.board.cards[18].time_marker_by.append(
        0)  # Marked by the clue giver

    result = engine.resolve_guess(card_id=18, player_id=1)

    assert result == "civilian"
    assert 1 in engine.state.board.cards[18].time_marker_by
    assert engine.state.timer_tokens == 8
    # Should switch roles after guessing a civilian
    assert engine.state.clue_giver == 1
    assert engine.state.guesser == 0
    assert engine.state.current_phase == GamePhase.GIVING_CLUE


@pytest.mark.parametrize("modification, expected_error", [
    ("invalid_phase", "Guesses can only be made during the GUESSING or SUDDEN_DEATH phase."),
    ("invalid_player", "Only the guesser can make guesses."),
    ("already_revealed", "This card has already been revealed."),
    ("time_token", "This card is currently marked by a time token and cannot be guessed.")
])
def test_engine_resolve_guess_invalid_inputs(valid_board_data: dict, modification, expected_error):
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
        with pytest.raises(ValueError, match=expected_error):
            engine.resolve_guess(card_id=0, player_id=1)
    elif modification == "time_token":
        engine.state.board.cards[0].time_marker_by.append(
            1)  # Marked by the guesser
        with pytest.raises(ValueError, match=expected_error):
            engine.resolve_guess(card_id=0, player_id=1)


def test_engine_pass_turn(valid_board_data: dict):
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


@pytest.mark.parametrize("modification, expected_error", [
    ("invalid_player", "Only the guesser can pass the turn."),
    ("no_guesses_made", "The guesser must make at least one guess before passing.")
])
def test_engine_pass_turn_invalid_inputs(valid_board_data: dict, modification, expected_error):
    board = Board(**valid_board_data)
    engine = CodenamesDuetEngine(board=board)

    # Force the clue giver to be player 0 and guesser to be player 1 for testing
    engine.state.clue_giver = 0
    engine.state.guesser = 1
    engine.state.current_phase = GamePhase.GUESSING

    if modification == "invalid_player":
        with pytest.raises(PermissionError, match=expected_error):
            engine.pass_turn(player_id=0)
    elif modification == "no_guesses_made":
        engine.state.guesses_made_this_turn = 0  # No guesses made
        with pytest.raises(ValueError, match=expected_error):
            engine.pass_turn(player_id=1)
