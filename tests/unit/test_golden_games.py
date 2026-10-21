"""Golden games: games real models played against Ollama, played again from the record.

Each fixture of tests/fixtures/golden was recorded by scripts/record_golden_game.py: every call
the local clients made to the daemon, and the game as it was stored. Here the same game is played
again through the real local client, with a stand-in for the daemon that serves the recorded
answers in order (backend/app/core/golden.py). Every request must be the one recorded, and the
game must come out as it did: the same clues, proposals, measurements, reveals, redraws and result.

A change that should not change the game, such as a refactor, must keep them passing. One that
changes what the models are sent fails at the first request that differs; record the golden games
again then (the script's docstring says how).
"""

import pytest

from backend.app.core import golden
from backend.app.core.game_runner import SeatSpec, default_client_factory, run_single_game
from backend.app.models.game_schemas import Board
from backend.app.paths import REPO_ROOT

_FIXTURES = sorted((REPO_ROOT / "tests" / "fixtures" / "golden").glob("*.json"))


def test_there_are_golden_games() -> None:
    assert _FIXTURES


async def _replay(fixture: dict):
    """Play ``fixture`` again; return the result, the recorder and the stand-ins for the daemon."""
    stand_ins: dict[int, golden.ReplayOllama] = {}

    def client_factory(seat: int, spec: SeatSpec):
        client = default_client_factory(seat, spec)
        stand_ins[seat] = golden.ReplayOllama(fixture["calls"][seat], seat)
        client._client = stand_ins[seat]  # type: ignore[attr-defined]
        return client

    captured = {}
    result = await run_single_game(
        board=Board.model_validate(fixture["board"]),
        seat_specs=[SeatSpec(**seat) for seat in fixture["seats"]],
        master_seed=fixture["master_seed"],
        temperature=fixture["temperature"],
        game_index=fixture["game_index"],
        persist=False,
        client_factory=client_factory,
        on_dispatch=lambda engine, recorder: captured.update(recorder=recorder),
    )
    return result, captured["recorder"], stand_ins


@pytest.mark.parametrize("path", _FIXTURES, ids=lambda path: path.stem)
async def test_a_golden_game_plays_again_as_recorded(path) -> None:
    fixture = golden.load(path)
    result, recorder, stand_ins = await _replay(fixture)

    expected = fixture["expected"]
    # A request that differs ends the game as an error whose message names it.
    assert (result.status, result.error_code, result.result) == (
        expected["status"],
        expected["error_code"],
        expected["result"],
    ), result.error
    assert golden.transcript(recorder) == expected["transcript"]
    assert {seat: s.unserved() for seat, s in stand_ins.items()} == {0: 0, 1: 0}


async def test_a_request_that_differs_from_the_record_is_named() -> None:
    """As a changed prompt would: the game ends in error at that request, naming what differs."""
    fixture = golden.load(_FIXTURES[0])
    seat, index = next(
        (seat, i)
        for seat, calls in enumerate(fixture["calls"])
        for i, call in enumerate(calls)
        if call["call"] == "chat"
    )
    fixture["calls"][seat][index]["request"]["messages_sha256"] = "0" * 64

    result, _, _ = await _replay(fixture)

    assert result.status == "error"
    assert "differs from the recorded one in ['messages_sha256']" in result.error
