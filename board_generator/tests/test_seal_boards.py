"""Tests for the board sealer (board_generator.seal_boards) on a temp board dir, no real data."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from board_generator.emit_frame import FRAME_FILENAME
from board_generator.seal_boards import (
    SEAL_KEY,
    StaleSealError,
    read_frame_id,
    seal_board_dir,
)

_FRAME = "a" * 64
_OTHER_FRAME = "b" * 64


def _write(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(
        obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _board(board_id: str) -> dict:
    """A minimal board-shaped dict: an ``arbiters`` block + nested floats to exercise round-trip."""
    return {
        "board_id": board_id,
        "arbiters": {"primary": "sentence-transformers/all-mpnet-base-v2@e8c3b32"},
        "cards": [{"text": "WEDDING", "covariates": {"subtlex_freq": 3.5993, "length": 7}}],
    }


@pytest.fixture
def boards_dir(tmp_path: Path) -> Path:
    _write(tmp_path / "probe-000.json", _board("probe-000"))
    _write(tmp_path / "control-000.json", _board("control-000"))
    _write(tmp_path / "balance_report.json", {"summary": "no arbiters here"})
    _write(tmp_path / "example_board.json",
           {"board_id": "example", "cards": []})
    _write(tmp_path / FRAME_FILENAME, {"frame_id": _FRAME})
    return tmp_path


def test_read_frame_id(boards_dir: Path) -> None:
    assert read_frame_id(boards_dir / FRAME_FILENAME) == _FRAME


def test_seals_only_arbiters_boards_and_skips_the_rest(boards_dir: Path) -> None:
    result = seal_board_dir(boards_dir, _FRAME)
    assert result.sealed == ["control-000.json", "probe-000.json"]
    assert result.already_sealed == []
    # sidecar + the two non-arbiters files are skipped.
    assert set(result.skipped) == {
        "balance_report.json", "example_board.json", FRAME_FILENAME}

    for name in result.sealed:
        assert json.loads((boards_dir / name).read_text())[SEAL_KEY] == _FRAME
    # skipped files gain no key
    assert SEAL_KEY not in json.loads(
        (boards_dir / "balance_report.json").read_text())
    assert SEAL_KEY not in json.loads(
        (boards_dir / "example_board.json").read_text())
    assert SEAL_KEY not in json.loads(
        (boards_dir / FRAME_FILENAME).read_text())


def test_parsed_equality_only_the_one_key_added(boards_dir: Path) -> None:
    before = copy.deepcopy(json.loads(
        (boards_dir / "probe-000.json").read_text()))
    seal_board_dir(boards_dir, _FRAME)
    after = json.loads((boards_dir / "probe-000.json").read_text())
    assert after == {**before, SEAL_KEY: _FRAME}
    # nothing else changed: floats and structure preserved bit-for-bit at the parsed level.
    del after[SEAL_KEY]
    assert after == before


def test_second_run_is_a_byte_level_no_op(boards_dir: Path) -> None:
    seal_board_dir(boards_dir, _FRAME)
    snapshot = {p.name: p.read_bytes() for p in boards_dir.glob("*.json")}
    result = seal_board_dir(boards_dir, _FRAME)
    assert result.sealed == []
    assert result.already_sealed == ["control-000.json", "probe-000.json"]
    assert {p.name: p.read_bytes()
            for p in boards_dir.glob("*.json")} == snapshot


def test_stale_seal_stops(boards_dir: Path) -> None:
    stale = _board("probe-000")
    stale[SEAL_KEY] = _OTHER_FRAME
    _write(boards_dir / "probe-000.json", stale)
    with pytest.raises(StaleSealError):
        seal_board_dir(boards_dir, _FRAME)


def test_default_frame_id_read_from_sidecar(boards_dir: Path) -> None:
    result = seal_board_dir(boards_dir)  # frame_id defaults to the sidecar's
    assert len(result.sealed) == 2
    assert json.loads(
        (boards_dir / "probe-000.json").read_text())[SEAL_KEY] == _FRAME
