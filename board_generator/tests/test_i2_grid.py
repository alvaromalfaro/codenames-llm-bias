"""Every board has 25 unique words on a 5x5 grid."""

from __future__ import annotations

import dataclasses

from board_generator.board import Board, Grid, validate_board_grid


def test_i2_legal_board_passes(control_board: Board) -> None:
    assert validate_board_grid(control_board) is True


def test_i2_rejects_wrong_word_count(control_board: Board) -> None:
    truncated = dataclasses.replace(
        control_board, words=control_board.words[:24])
    assert validate_board_grid(truncated) is False


def test_i2_rejects_duplicate_words(control_board: Board) -> None:
    words = list(control_board.words)
    words[1] = dataclasses.replace(words[1], text=words[0].text)
    board = dataclasses.replace(control_board, words=words)
    assert validate_board_grid(board) is False


def test_i2_rejects_non_5x5_grid(control_board: Board) -> None:
    # 25 cells, but not 5×5: the grid is 5×5, not merely rows*cols == 25.
    board = dataclasses.replace(control_board, grid=Grid(rows=1, cols=25))
    assert validate_board_grid(board) is False
