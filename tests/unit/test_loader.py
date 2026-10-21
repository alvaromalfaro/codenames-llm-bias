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
        assert card.text == valid_board_data["cards"][i]["text"]
        assert card.llm_perspective_role == valid_board_data["cards"][i]["llm_perspective_role"]
        assert card.human_perspective_role == valid_board_data["cards"][i]["human_perspective_role"]


def _bank_with(tmp_path, valid_board_data, bad_content):
    import json

    (tmp_path / "good.json").write_text(json.dumps(valid_board_data), encoding="utf-8")
    (tmp_path / "bad.json").write_text(bad_content, encoding="utf-8")
    for sidecar in ("balance_report.json", "measurement_frame.json"):
        (tmp_path / sidecar).write_text("{}", encoding="utf-8")
    return tmp_path


def test_a_lenient_loader_leaves_out_a_file_that_is_not_a_board(tmp_path, valid_board_data):
    loader = BoardLoader(_bank_with(tmp_path, valid_board_data, "not json"))
    assert sum(len(group) for group in loader.boards.values()) == 1


@pytest.mark.parametrize("bad_content", ["not json", '{"board_id": "half-a-board"}'])
def test_a_strict_loader_stops_at_a_file_that_is_not_a_board(
    tmp_path, valid_board_data, bad_content
):
    """The batch loads strictly: a board left out would be replaced by the next one in the
    schedule, under the same game ids."""
    from backend.app.core.loader import InvalidBoardFileError

    with pytest.raises(InvalidBoardFileError, match="bad.json"):
        BoardLoader(_bank_with(tmp_path, valid_board_data, bad_content), strict=True)


def test_a_strict_loader_skips_the_sidecars(tmp_path, valid_board_data):
    import json

    (tmp_path / "good.json").write_text(json.dumps(valid_board_data), encoding="utf-8")
    for sidecar in ("balance_report.json", "measurement_frame.json"):
        (tmp_path / sidecar).write_text("{}", encoding="utf-8")
    loader = BoardLoader(tmp_path, strict=True)
    assert sum(len(group) for group in loader.boards.values()) == 1


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
        board_file.write_text(
            json.dumps({"board_id": f"board_{i}", "category": "neutral", "cards": []})
        )

    # Create a non-JSON file in the data directory to ensure that it is not included in the list of available boards
    (d / "readme.txt").write_text(
        "This is a readme file and should not be listed as an available board configuration."
    )

    # Initialize the BoardLoader with the path to the temporary data directory and list the available board configuration files
    loader = BoardLoader(data_path=str(d))
    available_boards = loader.list_available_boards()

    # Validate that the list of available boards contains the expected filenames
    assert len(available_boards) == 3
    assert set(available_boards) == {f"board_{i}.json" for i in range(3)}
    assert "readme.txt" not in available_boards


def test_list_available_boards_directory_not_found(tmp_path):
    """
    Validates that the BoardLoader raises a FileNotFoundError when attempting to list available boards from a non-existent data directory.
    :param tmp_path: A pytest fixture providing a temporary directory for file operations during the test.
    """
    # Initialize the BoardLoader with a path to a non-existent data directory and attempt to list available board configuration files
    loader = BoardLoader(data_path=str(tmp_path / "non_existent_data"))

    # Assert that a FileNotFoundError is raised when trying to list available boards from a data directory that does not exist
    with pytest.raises(FileNotFoundError):
        loader.list_available_boards()
