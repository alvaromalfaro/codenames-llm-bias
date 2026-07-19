import json
import re
from pathlib import Path
from typing import List
from backend.app.models.game_schemas import Board


class BoardLoader:
    def __init__(self, data_path: str = "data/boards"):
        # Initialize the loader with the path to the data directory
        self.data_path = Path(data_path)
        # Construct a dictionary to hold the available boards, keyed by their category
        self.boards = self._load_boards()

    def load_board(self, filename: str) -> Board:
        """
        Loads a board configuration from a JSON file and validates it against the Board schema.

        :param self: The instance of the BoardLoader class.
        :param filename: The name of the JSON file containing the board configuration (e.g., "board1.json").
        :return: The validated Board object.
        """
        file_path = self.data_path / filename

        if not file_path.exists():
            raise FileNotFoundError(
                f"Board file '{filename}' not found in '{self.data_path}'."
            )

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return Board.model_validate(data)

    def list_available_boards(self) -> List[str]:
        """
        Lists all available board configuration files in the data directory.

        :param self: The instance of the BoardLoader class.
        :return: A list of filenames for the available board configurations.
        :rtype: List[str]
        """
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Data directory '{self.data_path}' not found."
            )

        return [f.name for f in self.data_path.glob("*.json")]

    def _load_boards(self) -> dict[str, list[Board]]:
        """
        Internal method to load all board configurations from the data directory and store them in a 
        dictionary.

        :param self: The instance of the BoardLoader class.
        :return: A dictionary mapping board categories to their corresponding Board objects.
        :rtype: dict[str, list[Board]]
        """
        boards = {}
        for file_path in self.data_path.glob("*.json"):
            try:
                if re.search(r"balance_report\.json$", str(file_path)) or re.search(r"measurement_frame\.json$", str(file_path)):
                    continue
                board = self.load_board(file_path.name)
                if board.category not in boards:
                    boards[board.category] = []
                boards[board.category].append(board)
            except Exception as e:
                print(f"Error loading board from '{file_path}': {e}")
        return boards
