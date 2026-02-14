import json
import pytest
from backend.app.core.loader import BoardLoader
from backend.app.models.game_schemas import Board


def test_load_board_success(tmp_path, valid_board_data):
    """
    Validates that the BoardLoader can successfully load and validate a board configuration from a JSON file.
    :param tmp_path: A pytest fixture providing a temporary directory for file operations during the test.
    :param valid_board_data: A fixture providing a valid board configuration as a dictionary.
    """
    # Create a temporary data directory and write the valid board configuration to a JSON file
    d = tmp_path / "data"
    d.mkdir()
    board_file = d / "test_board.json"
    board_file.write_text(json.dumps(valid_board_data))

    # Initialize the BoardLoader with the path to the temporary data directory and load the board configuration
    loader = BoardLoader(data_path=str(d))
    board = loader.load_board("test_board.json")

    # Validate the loaded board against the expected values from the valid_board_data fixture
    assert isinstance(board, Board)
    assert board.board_id == "test_board_001"
    assert board.category == "neutral"
    assert len(board.cards) == 25
    for i, card in enumerate(board.cards):
        assert card.id == i
        assert card.text == f"Word_{i}"
        assert card.llm_role == valid_board_data["cards"][i]["llm_role"]
        assert card.human_role == valid_board_data["cards"][i]["human_role"]


def test_load_board_file_not_found(tmp_path):
    """
    Validates that the BoardLoader raises a FileNotFoundError when attempting to load a non-existent board configuration file.
    :param tmp_path: A pytest fixture providing a temporary directory for file operations during the test.
    """
    # Initialize the BoardLoader with the path to the temporary data directory (which is empty) and attempt to load a non-existent board configuration file
    loader = BoardLoader(data_path=str(tmp_path))

    # Assert that a FileNotFoundError is raised when trying to load a board configuration file that does not exist in the temporary data directory
    with pytest.raises(FileNotFoundError):
        loader.load_board("non_existent_board.json")


def test_list_available_boards(tmp_path):
    """
    Validates that the BoardLoader can list all available board configuration files in the data directory.
    :param tmp_path: A pytest fixture providing a temporary directory for file operations during the test.
    """
    # Create a temporary data directory and write multiple valid board configurations to JSON files
    d = tmp_path / "data"
    d.mkdir()
    for i in range(3):
        board_file = d / f"board_{i}.json"
        board_file.write_text(json.dumps(
            {"board_id": f"board_{i}", "category": "neutral", "cards": []}))

    # Create a non-JSON file in the data directory to ensure that it is not included in the list of available boards
    (d / "readme.txt").write_text("This is a readme file and should not be listed as an available board configuration.")

    # Initialize the BoardLoader with the path to the temporary data directory and list the available board configuration files
    loader = BoardLoader(data_path=str(d))
    available_boards = loader.list_available_boards()

    # Validate that the list of available boards contains the expected filenames
    assert len(available_boards) == 3
    assert set(available_boards) == {f"board_{i}.json" for i in range(3)}
    assert "readme.txt" not in available_boards
