"""Tests for the headless single-game driver (game_runner.run_single_game).

Seed-derivation, reproducibility, the temperature assertion and the error boundary run without a
database. The dispatch + sudden-death end-to-end tests require a live Postgres and are skipped when 
DATABASE_URL is unset."""
import json
import os
import random
import uuid

import pytest
from unittest.mock import AsyncMock, MagicMock

from backend.app.core import game_runner
from backend.app.core.game_runner import (
    SeatSpec, _derive, _game_identity, _seed_engine, _seed_game,
    _seed_meas, _seed_play, run_single_game,
)
from backend.app.core.llm.client import LLMClient
from backend.app.models.game_schemas import Board, CardRole, WordCard
from backend.app.models.llm_errors import LLMRefusalError, LLMTimeoutError
from backend.app.models.llm_schemas import (
    ClueJSONFormat, ConfidenceRankingJSONFormat, GuessJSONFormat,
)


# shared board (the official Duet layout, mirrored from test_game_conductor)
_A, _C, _S = CardRole.AGENT, CardRole.CIVILIAN, CardRole.ASSASSIN
_CARDS = [
    ("BUCKET", _C, _C),    # 0
    ("BRICK", _A, _A),     # 1  shared agent
    ("ANT", _A, _S),       # 2
    ("LEMONADE", _S, _S),  # 3  assassin for BOTH perspectives
    ("RUSSIA", _C, _A),    # 4  seat-1 agent
    ("CAVE", _A, _C),      # 5  seat-0 agent
    ("FIDDLE", _C, _C),    # 6
    ("VAMPIRE", _C, _C),   # 7
    ("TATTOO", _A, _A),    # 8  shared agent
    ("RANCH", _A, _C),     # 9  seat-0 agent
    ("LOCUST", _S, _C),    # 10
    ("RIFLE", _C, _A),     # 11 seat-1 agent
    ("VIRUS", _C, _A),     # 12 seat-1 agent
    ("IGLOO", _C, _C),     # 13
    ("MAKEUP", _C, _S),    # 14
    ("POTTER", _C, _A),    # 15 seat-1 agent
    ("CAESAR", _A, _C),    # 16 seat-0 agent
    ("NAPOLEON", _A, _A),  # 17 shared agent
    ("GOLF", _C, _C),      # 18
    ("PINE", _S, _A),      # 19 seat-1 agent
    ("DOLL", _A, _C),      # 20 seat-0 agent
    ("LUNCH", _A, _C),     # 21 seat-0 agent
    ("SKATES", _C, _C),    # 22
    ("CRAFT", _C, _C),     # 23
    ("PEW", _C, _A),       # 24 seat-1 agent
]


def _board(board_id: str = "tb") -> Board:
    return Board(
        board_id=board_id, category="neutral",
        cards=[WordCard(id=i, text=t, human_perspective_role=h,
                        llm_perspective_role=l, category="neutral")
               for i, (t, h, l) in enumerate(_CARDS)],
    )


# mock clients
def _mock_response(text: str) -> MagicMock:
    resp = MagicMock()
    resp.text = text
    resp.model_used = "test_model"
    resp.latency_ms = 0
    resp.raw_payload = json.loads(text)
    resp.usage = None
    resp.finish_reason = None
    resp.provider = None
    resp.request_id = None
    resp.resolved_model = None
    resp.system_fingerprint = None
    resp.requested_temperature = None
    resp.requested_seed = None
    return resp


def _guess_json(words, confidences=None, reasoning="r", stop_reason="done") -> str:
    confidences = confidences or [0.9] * len(words)
    items = ", ".join(
        f'{{"word": "{w}", "confidence": {c}}}' for w, c in zip(words, confidences))
    return f'{{"reasoning": "{reasoning}", "stop_reason": "{stop_reason}", "proposals": [{items}]}}'


def _rankings_json(pairs=(("BRICK", 0.9), ("CAVE", 0.1)), reasoning="r") -> str:
    items = ", ".join(
        f'{{"word": "{w}", "confidence": {c}}}' for w, c in pairs)
    return f'{{"reasoning": "{reasoning}", "rankings": [{items}]}}'


def _clue_json(clue="OCEAN", count=1, reasoning="r", targets=None) -> str:
    return json.dumps({"reasoning": reasoning, "clue": clue,
                       "count": count, "targets": targets or []})


def _make_client(name: str, guess_supplier, log=None) -> MagicMock:
    """A mock LLMClient. ``guess_supplier()`` is called per guess request and returns the word list
    (or raises to inject an error); clue/ranking are canned. Each response echoes the request seed
    (as real clients do). ``log``, if given, records (name, format, seed) per call in dispatch order.
    """
    def gen(request, expected_format=None):
        if log is not None:
            log.append((name, expected_format, request.seed))
        if expected_format is ClueJSONFormat:
            text = _clue_json()
        elif expected_format is ConfidenceRankingJSONFormat:
            text = _rankings_json()
        elif expected_format is GuessJSONFormat:
            text = _guess_json(guess_supplier())
        else:
            raise AssertionError(
                f"unexpected expected_format {expected_format!r}")
        resp = _mock_response(text)
        resp.requested_seed = request.seed
        resp.requested_temperature = request.temperature
        return resp

    client = MagicMock(spec=LLMClient)
    client.model_name = name
    client.generate = AsyncMock(side_effect=gen)
    client.close = AsyncMock()
    return client


def _queue_supplier(word_lists):
    """A stateful guess supplier popping the next scripted word list per call."""
    it = iter(word_lists)
    return lambda: next(it)


def _unique_game_index() -> int:
    """A fresh game_index per invocation so the deterministic game_id is unique across test runs on a
    persistent DB (the driver's game_id folds in game_index)."""
    return uuid.uuid4().int % (2**31)


def _seed_with_start_player(target: int, game_index=0) -> int:
    """Find a master seed whose derived engine RNG picks ``target`` as the start player, so the
    scripted dispatch order is deterministic. Uses the driver's own derivation (the seam under test).
    """
    for master_seed in range(100000):
        _, _, seed_engine = _game_identity(master_seed, game_index)
        if random.Random(seed_engine).choice([0, 1]) == target:
            return master_seed
    raise AssertionError("no master seed found for the requested start player")


_SPECS = (SeatSpec("ollama", "m0"), SeatSpec("openrouter", "m1"))


# seed derivation units
def test_derive_is_stable_and_part_sensitive():
    # Same inputs -> same output.
    assert _derive(42, b"a", b"b") == _derive(42, b"a", b"b")
    # Different seed / parts -> different output.
    assert _derive(42, b"a") != _derive(43, b"a")
    assert _derive(42, b"a") != _derive(42, b"b")
    # Always a 64-bit value.
    assert 0 <= _derive(2**70, b"x") < 2**64


def test_seed_scheme_labels_and_turns_distinct():
    sg = _seed_game(12345, "game-abc")
    # play vs measurement for the same (seat, turn) differ (distinct labels).
    assert _seed_play(sg, 0, 1) != _seed_meas(sg, 0, 1)
    # seat-sensitive and turn-sensitive.
    assert _seed_play(sg, 0, 1) != _seed_play(sg, 1, 1)
    assert _seed_play(sg, 0, 1) != _seed_play(sg, 0, 2)
    assert _seed_meas(sg, 0, 1) != _seed_meas(sg, 1, 1)
    # engine seed derives from the game seed and is stable.
    assert _seed_engine(sg) == _seed_engine(sg)
    # game seed folds in the game_id.
    assert _seed_game(12345, "game-abc") != _seed_game(12345, "game-xyz")


# temperature assertion
@pytest.mark.asyncio
async def test_temperature_required_raises_before_any_client(monkeypatch):
    monkeypatch.setattr(game_runner, "_db_enabled", lambda: False)
    factory = MagicMock()  # would blow up if called
    with pytest.raises(ValueError, match="temperature"):
        await run_single_game(board=_board(), seat_specs=_SPECS, master_seed=1,
                              temperature=None, persist=False, client_factory=factory)
    factory.assert_not_called()


# mandatory persistence
@pytest.mark.asyncio
async def test_persist_true_without_database_url_raises_before_any_client(monkeypatch):
    """persist=True (the default) with no DATABASE_URL fails loud before any client is built, so a
    misconfigured run burns zero provider calls."""
    monkeypatch.setattr(game_runner, "_db_enabled", lambda: False)
    factory = MagicMock()  # would blow up if called
    with pytest.raises(RuntimeError, match="DATABASE_URL"):
        await run_single_game(board=_board(), seat_specs=_SPECS, master_seed=1,
                              temperature=0.4, client_factory=factory)
    factory.assert_not_called()


@pytest.mark.asyncio
async def test_persist_false_plays_without_touching_db(monkeypatch):
    """persist=False plays a full game and returns a result while never calling into the writer."""
    persist_game = MagicMock(side_effect=AssertionError(
        "writer must not be called for persist=False"))
    monkeypatch.setattr(game_runner.writer, "persist_game", persist_game)
    clients = {0: _make_client("m0", lambda: ["LEMONADE"]),
               1: _make_client("m1", lambda: ["LEMONADE"])}
    res = await run_single_game(board=_board(), seat_specs=_SPECS, master_seed=777,
                                temperature=0.4, persist=False,
                                client_factory=lambda i, spec: clients[i])
    assert res.status == "completed" and res.result == "loss_assassin"
    persist_game.assert_not_called()


# reproducibility (first-order acceptance)
async def _run_lemonade(master_seed, monkeypatch, log):
    """A 2-dispatch game: the first guessing seat guesses LEMONADE (assassin for BOTH perspectives),
    so the game ends in a loss regardless of which seat starts - fully deterministic."""
    monkeypatch.setattr(game_runner, "_db_enabled", lambda: False)
    clients = {0: _make_client("m0", lambda: ["LEMONADE"], log),
               1: _make_client("m1", lambda: ["LEMONADE"], log)}
    return await run_single_game(
        board=_board(), seat_specs=_SPECS, master_seed=master_seed, temperature=0.4,
        persist=False, client_factory=lambda i, spec: clients[i]), clients


@pytest.mark.asyncio
async def test_reproducibility_identical_across_runs(monkeypatch):
    log1, log2 = [], []
    res1, _ = await _run_lemonade(777, monkeypatch, log1)
    res2, _ = await _run_lemonade(777, monkeypatch, log2)

    assert res1.status == "completed" and res1.result == "loss_assassin"
    assert res1.game_id == res2.game_id
    assert res1.seed_game == res2.seed_game
    # Identical sequence of (seat, format, seed) requests -> identical derived seeds & dispatch order.
    assert log1 == log2
    assert len(log1) >= 3  # at least: one clue, one guess, one measurement


@pytest.mark.asyncio
async def test_request_seeds_match_derivation_scheme(monkeypatch):
    master_seed = 777
    log = []
    res, _ = await _run_lemonade(master_seed, monkeypatch, log)

    game_id, seed_game, seed_engine = _game_identity(master_seed, 0)
    start = random.Random(seed_engine).choice([0, 1])
    clue_giver, guesser = start, 1 - start

    seen = {(name, fmt): seed for name, fmt, seed in log}
    cg_name = "m0" if clue_giver == 0 else "m1"
    g_name = "m0" if guesser == 0 else "m1"
    # Turn 1: clue by the clue-giver, guess+measurement by the guesser, each with its own seed.
    assert seen[(cg_name, ClueJSONFormat)] == _seed_play(
        seed_game, clue_giver, 1)
    assert seen[(g_name, GuessJSONFormat)] == _seed_play(seed_game, guesser, 1)
    assert seen[(g_name, ConfidenceRankingJSONFormat)
                ] == _seed_meas(seed_game, guesser, 1)


# identity determinism (run_id must NOT feed the derivation)
@pytest.mark.asyncio
async def test_identity_ignores_run_id(monkeypatch):
    """Same master_seed + game_index but different run_id (including a random uuid4, simulating a 
    DB-minted run.id) must yield the same game_id and seed_game. run_id is persistence bookkeeping, 
    never identity."""
    monkeypatch.setattr(game_runner, "_db_enabled", lambda: False)

    async def _play(run_id):
        clients = {0: _make_client("m0", lambda: ["LEMONADE"]),
                   1: _make_client("m1", lambda: ["LEMONADE"])}
        return await run_single_game(
            board=_board(), seat_specs=_SPECS, master_seed=777, temperature=0.4,
            run_id=run_id, game_index=3, persist=False,
            client_factory=lambda i, spec: clients[i])

    res_fixed = await _play("fixed-run-id")
    # a DB-minted run.id is a random uuid4
    res_random = await _play(str(uuid.uuid4()))
    assert res_fixed.run_id != res_random.run_id  # the run ids really did differ
    assert res_fixed.game_id == res_random.game_id
    assert res_fixed.seed_game == res_random.seed_game


def test_identity_is_game_index_sensitive():
    id0, sg0, _ = _game_identity(777, 0)
    id1, sg1, _ = _game_identity(777, 1)
    assert id0 != id1
    assert sg0 != sg1


def test_identity_is_master_seed_sensitive():
    id0, sg0, _ = _game_identity(777, 0)
    id1, sg1, _ = _game_identity(778, 0)
    assert id0 != id1
    assert sg0 != sg1


# error boundary
@pytest.mark.asyncio
async def test_retriable_error_bounded_same_seed_retry(monkeypatch):
    monkeypatch.setattr(game_runner, "_db_enabled", lambda: False)
    monkeypatch.setattr(game_runner, "_RETRY_BACKOFF_BASE_S",
                        0.0)  # no real sleeping

    def _raise_timeout():
        raise LLMTimeoutError()

    # Both seats: clue succeeds, the guess always times out (retriable). Whichever seat guesses,
    # the driver retries the same dispatch k times, then the boundary flushes as error.
    clients = {0: _make_client("m0", _raise_timeout),
               1: _make_client("m1", _raise_timeout)}
    res = await run_single_game(board=_board(), seat_specs=_SPECS, master_seed=5,
                                temperature=0.4, persist=False,
                                client_factory=lambda i, spec: clients[i])

    assert res.status == "error"
    assert "timed out" in (res.error or "")
    # Exactly one seat guessed; its guess dispatch was attempted k times, all with the same seed.
    guess_seeds = []
    for c in clients.values():
        for call in c.generate.call_args_list:
            if call.kwargs.get("expected_format") is GuessJSONFormat:
                guess_seeds.append(call.args[0].seed)
    assert len(guess_seeds) == game_runner._RETRY_ATTEMPTS
    assert len(set(guess_seeds)) == 1  # never reseeded


@pytest.mark.asyncio
async def test_deterministic_error_not_retried(monkeypatch):
    monkeypatch.setattr(game_runner, "_db_enabled", lambda: False)

    def _refuse():
        raise LLMRefusalError()

    clients = {0: _make_client("m0", _refuse), 1: _make_client("m1", _refuse)}
    res = await run_single_game(board=_board(), seat_specs=_SPECS, master_seed=5,
                                temperature=0.4, persist=False,
                                client_factory=lambda i, spec: clients[i])

    assert res.status == "error"
    # A non-retriable LLM error is not retried: the guess dispatch happened exactly once.
    guess_calls = sum(
        1 for c in clients.values() for call in c.generate.call_args_list
        if call.kwargs.get("expected_format") is GuessJSONFormat)
    assert guess_calls == 1


# DB-gated end-to-end
_db_required = pytest.mark.skipif(
    not os.environ.get("DATABASE_URL"), reason="requires a live Postgres database")


def _insert_board(session, board_id):
    from backend.app.db.models import BoardModel, WordCardModel
    session.add(BoardModel(board_id=board_id, type="control"))
    for i, (text, human, llm) in enumerate(_CARDS):
        session.add(WordCardModel(
            board_id=board_id, card_id=i, text=text,
            human_perspective_role=human.value, llm_perspective_role=llm.value))
    session.flush()


@_db_required
@pytest.mark.asyncio
async def test_dispatch_full_game_persists_two_seats():
    """A 2-turn game with start_player 0 forced: seat 1 guesses BRICK (agent both -> pass), then seat
    0 guesses LEMONADE (assassin -> loss). Both seats give a clue, guess, and are measured; the game
    persists with two real seat identities and both turns."""
    from sqlalchemy import select

    from backend.app.db.models import GameModel, GameSeatModel, GuessProposalModel, TurnModel
    from backend.app.db.session import session_scope

    board_id = f"runner-{uuid.uuid4()}"
    with session_scope() as session:
        _insert_board(session, board_id)

    game_index = _unique_game_index()
    master_seed = _seed_with_start_player(0, game_index=game_index)
    clients = {0: _make_client("m0", _queue_supplier([["LEMONADE"]])),
               1: _make_client("m1", _queue_supplier([["BRICK"]]))}
    res = await run_single_game(board=_board(board_id), seat_specs=_SPECS,
                                master_seed=master_seed, temperature=0.4,
                                game_index=game_index,
                                client_factory=lambda i, spec: clients[i])

    assert res.status == "completed" and res.result == "loss_assassin"
    with session_scope() as session:
        game = session.get(GameModel, res.game_id)
        assert game is not None
        assert game.game_status == "completed"
        assert game.run_id == res.run_id and game.run_id is not None
        assert game.start_player == 0

        seats = session.execute(select(GameSeatModel).where(
            GameSeatModel.game_id == res.game_id)).scalars().all()
        by_seat = {s.seat_index: s for s in seats}
        assert set(by_seat) == {0, 1}
        assert (by_seat[0].provider, by_seat[0].model_ref) == ("ollama", "m0")
        assert (by_seat[1].provider, by_seat[1].model_ref) == (
            "openrouter", "m1")
        # Runner seat sampling: constant temperature on both seats; requested_seed NULL on both
        # (the runner seeds per (seat, turn), so llm_call.requested_seed is the source of truth).
        assert float(by_seat[0].requested_temperature) == 0.4
        assert float(by_seat[1].requested_temperature) == 0.4
        assert by_seat[0].requested_seed is None
        assert by_seat[1].requested_seed is None

        turns = session.execute(select(TurnModel).where(
            TurnModel.game_id == res.game_id)).scalars().all()
        assert len(turns) == 2
        assert {t.clue_giver_seat for t in turns} == {0, 1}
        assert all(t.phase == "normal" for t in turns)

        # Both seats were measured: a measurement proposal exists for each guesser seat.
        meas = session.execute(select(GuessProposalModel).where(
            GuessProposalModel.kind == "measurement",
            GuessProposalModel.turn_id.in_([t.id for t in turns]))).scalars().all()
        assert {m.guesser_seat for m in meas} == {0, 1}


@_db_required
@pytest.mark.asyncio
async def test_sudden_death_both_seats_persists():
    """A game driven to natural sudden death (start_player 0): nine civilian turns exhaust the timer,
    then seat 0 (SUDDEN_DEATH_LLM) reveals all its agents and hands off to seat 1
    (SUDDEN_DEATH_HUMAN) for the win. The single SD turn persists both seats' play + measurement."""
    from sqlalchemy import select

    from backend.app.db.models import GuessProposalModel, TurnModel
    from backend.app.db.session import session_scope

    board_id = f"runner-sd-{uuid.uuid4()}"
    with session_scope() as session:
        _insert_board(session, board_id)

    game_index = _unique_game_index()
    master_seed = _seed_with_start_player(0, game_index=game_index)
    # Guesser order under start_player 0 is 1,0,1,0,1,0,1,0,1 across the nine civilian turns.
    seat0_agents = ["BRICK", "ANT", "CAVE", "TATTOO", "RANCH",
                    # all seat-0 (human) agents
                    "CAESAR", "NAPOLEON", "DOLL", "LUNCH"]
    seat1_agents = ["RUSSIA", "RIFLE", "VIRUS", "POTTER",
                    "PINE", "PEW"]  # seat-1 leftovers post-SD
    client0 = _make_client("m0", _queue_supplier(
        # 4 civilians, then SD
        [["RUSSIA"], ["RIFLE"], ["VIRUS"], ["MAKEUP"], seat0_agents]))
    client1 = _make_client("m1", _queue_supplier(
        # 5 civ, then SD
        [["BUCKET"], ["FIDDLE"], ["VAMPIRE"], ["IGLOO"], ["GOLF"], seat1_agents]))
    clients = {0: client0, 1: client1}

    res = await run_single_game(board=_board(board_id), seat_specs=_SPECS,
                                master_seed=master_seed, temperature=0.4,
                                game_index=game_index,
                                client_factory=lambda i, spec: clients[i])

    assert res.status == "completed" and res.result == "victory_sd"
    with session_scope() as session:
        sd_turns = session.execute(select(TurnModel).where(
            TurnModel.game_id == res.game_id,
            TurnModel.phase == "sudden_death")).scalars().all()
        assert len(sd_turns) == 1
        proposals = session.execute(select(GuessProposalModel).where(
            GuessProposalModel.turn_id == sd_turns[0].id)).scalars().all()
        assert {(p.kind, p.guesser_seat) for p in proposals} == {
            ("play", 0), ("play", 1), ("measurement", 0), ("measurement", 1)}
