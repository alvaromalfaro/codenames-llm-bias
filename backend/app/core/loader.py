import json
from pathlib import Path
from typing import List
from app.models.game_schemas import Board


class BoardLoader:
    def __init__(self, data_path: str = "data"):
        # Initialize the loader with the path to the data directory
        self.data_path = Path(data_path)

    def load_board(self, filename: str) -> Board:
        """
        Loads a board configuration from a JSON file and validates it against the Board schema.

        :param self: The instance of the BoardLoader class.
        :param filename: The name of the JSON file containing the board configuration (e.g., "board1.json").
        :type filename: str
        :return: The validated Board object.
        :rtype: Board
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
        :return: A list of filenames for the available board configurations (e.g., ["board1.json", "board2.json"]).
        :rtype: List[str]
        """
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Data directory '{self.data_path}' not found."
            )

        return [f.name for f in self.data_path.glob("*.json")]
