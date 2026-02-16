import pytest
from backend.app.core.engine import CodenamesDuetEngine
from backend.app.models.game_schemas import Board, GamePhase


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


def test_engine_receive_clue_invalid_phase(valid_board_data: dict):
    board = Board(**valid_board_data)
    engine = CodenamesDuetEngine(board=board)

    # Force the clue giver to be player 0 for testing
    engine.state.clue_giver = 0
    engine.state.guesser = 1
    engine.state.current_phase = GamePhase.GUESSING  # Invalid phase for giving a clue

    with pytest.raises(ValueError, match="Clues can only be given during the GIVING_CLUE phase."):
        engine.receive_clue(clue="TestClue", count=2, player_id=0)


def test_engine_receive_clue_invalid_player(valid_board_data: dict):
    board = Board(**valid_board_data)
    engine = CodenamesDuetEngine(board=board)

    # Force the clue giver to be player 0 for testing
    engine.state.clue_giver = 0
    engine.state.guesser = 1
    engine.state.current_phase = GamePhase.GIVING_CLUE

    with pytest.raises(PermissionError, match="Only the clue giver can provide a clue."):
        engine.receive_clue(clue="TestClue", count=2, player_id=1)


def test_engine_receive_clue_empty_clue(valid_board_data: dict):
    board = Board(**valid_board_data)
    engine = CodenamesDuetEngine(board=board)

    # Force the clue giver to be player 0 for testing
    engine.state.clue_giver = 0
    engine.state.guesser = 1
    engine.state.current_phase = GamePhase.GIVING_CLUE

    with pytest.raises(ValueError, match="Clue cannot be empty."):
        engine.receive_clue(clue="", count=2, player_id=0)


def test_engine_receive_clue_invalid_count(valid_board_data: dict):
    board = Board(**valid_board_data)
    engine = CodenamesDuetEngine(board=board)

    # Force the clue giver to be player 0 for testing
    engine.state.clue_giver = 0
    engine.state.guesser = 1
    engine.state.current_phase = GamePhase.GIVING_CLUE

    with pytest.raises(ValueError, match="Clue count must be at least 1."):
        engine.receive_clue(clue="TestClue", count=0, player_id=0)


def test_engine_receive_clue_exact_match(valid_board_data: dict):
    board = Board(**valid_board_data)
    engine = CodenamesDuetEngine(board=board)

    # Force the clue giver to be player 0 for testing
    engine.state.clue_giver = 0
    engine.state.guesser = 1
    engine.state.current_phase = GamePhase.GIVING_CLUE

    with pytest.raises(ValueError, match="Clue cannot be the same as any word on the board."):
        engine.receive_clue(
            clue=valid_board_data["cards"][0]["text"], count=2, player_id=0)
