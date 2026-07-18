"""A.4 end-to-end: the 28 sealed bank boards carry the frame_id and the platform loader accepts them.

Exercises the A.1 change (Board.measurement_frame_id optional) against the actually-sealed artifacts:
seal + load, no board dropped. Reads the committed data/boards/ artifacts; no DB, no model.
"""

import glob
import json
import os

from backend.app.core.loader import BoardLoader
from backend.app.models.game_schemas import Board

_BOARDS_DIR = "data/boards"
_SEAL_KEY = "measurement_frame_id"


def _target_frame_id() -> str:
    with open(os.path.join(_BOARDS_DIR, "measurement_frame.json"), encoding="utf-8") as f:
        return json.load(f)["frame_id"]


def _bank_board_files() -> list[str]:
    """The real bank boards: every data/boards/*.json with an ``arbiters`` block."""
    files = []
    for path in sorted(glob.glob(os.path.join(_BOARDS_DIR, "*.json"))):
        with open(path, encoding="utf-8") as f:
            if "arbiters" in json.load(f):
                files.append(path)
    return files


def test_all_28_bank_boards_sealed_and_load_via_backend():
    target = _target_frame_id()
    files = _bank_board_files()
    assert len(files) == 28, f"expected 28 bank boards, found {len(files)}"

    loader = BoardLoader(data_path=_BOARDS_DIR)
    for path in files:
        raw = json.load(open(path, encoding="utf-8"))
        assert raw[_SEAL_KEY] == target, path
        # backend Board.model_validate accepts the sealed artifact and surfaces the frame id.
        board: Board = loader.load_board(os.path.basename(path))
        assert board.measurement_frame_id == target


def test_balance_report_and_example_board_are_not_sealed():
    for name in ("balance_report.json", "example_board.json"):
        raw = json.load(open(os.path.join(_BOARDS_DIR, name), encoding="utf-8"))
        assert _SEAL_KEY not in raw
