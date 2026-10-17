"""Tests for the batch orchestrator (batch_runner.run_batch).

Two failure levels are exercised separately: Level-1 preconditions (bank shortfall, collision) abort
before any game; Level-2 fault isolation (count-and-continue + per-pairing consecutive-failure abort)
is tested by patching run_single_game so the loop's own logic is what's under test. One hermetic
end-to-end test plays all 192 games with deterministic mock clients (no DB, no daemon).

The threshold reset test is the discriminating one: 4 errors + 1 completion + 4 errors must not
abort, proving the counter is consecutive, not cumulative.
"""
import json
import os
from pathlib import Path

import pytest

from backend.app.core import batch_runner
from backend.app.core.batch_runner import (
    BatchPreconditionError, dry_run_client_factory, run_batch,
)
from backend.app.core.batch_schedule import ScheduleError, build_schedule
from backend.app.core.game_runner import GameRunResult

# Reuse the validated fixture bank + models from the schedule tests (same tests/unit package dir).
from test_batch_schedule import _bank, _models, _MASTER_SEED

_BOARDS_DIR = Path(__file__).resolve().parents[2] / "data" / "boards"


# helpers: a patched run_single_game whose per-game outcome is scripted by game_index
def _fake_runner(outcome):
    """Build a fake run_single_game returning a scripted GameRunResult per game_index.

    ``outcome(game_index) -> (status, error)``. It records nothing and touches no DB - it stands in
    for the runner so the loop's count-and-continue / threshold logic is what the test exercises.
    """
    async def fake(**kwargs):
        gi = kwargs["game_index"]
        status, error = outcome(gi)
        return GameRunResult(game_id=f"g{gi}", run_id=kwargs["run_id"], seed_game=0,
                             status=status, result=None, error=error)
    return fake


def _spy_never_called():
    async def spy(**kwargs):  # pragma: no cover - must never run
        raise AssertionError(
            "run_single_game must not be called after a Level-1 abort")
    return spy


# Level 1 preconditions: abort before any game
async def test_bank_shortfall_aborts_before_any_game(monkeypatch) -> None:
    monkeypatch.setattr(batch_runner, "run_single_game", _spy_never_called())
    with pytest.raises(ScheduleError, match="science"):
        await run_batch(models=_models(), boards=_bank(science=3), master_seed=_MASTER_SEED,
                        temperature=0.4, persist=False, enforce_digests=False)


@pytest.mark.skipif(not os.environ.get("DATABASE_URL"),
                    reason="requires a live Postgres database")
async def test_collision_check_aborts_before_any_game(monkeypatch) -> None:
    from backend.app.db.models import BoardModel, GameModel
    from backend.app.db.session import session_scope

    # A derived game id from this schedule (identity is (master_seed, game_index) only).
    seed = 987654
    schedule = build_schedule(_models(), _bank(), seed)
    colliding_id = schedule[0].game_id
    board_id = "collision-test-board"

    with session_scope() as s:
        s.add(BoardModel(board_id=board_id))
        s.flush()
        s.add(GameModel(id=colliding_id, board_id=board_id))
    try:
        monkeypatch.setattr(batch_runner, "run_single_game",
                            _spy_never_called())
        with pytest.raises(BatchPreconditionError, match="already recorded"):
            await run_batch(models=_models(), boards=_bank(), master_seed=seed,
                            temperature=0.4, persist=True, enforce_digests=False)
    finally:
        with session_scope() as s:
            s.query(GameModel).filter_by(id=colliding_id).delete()
            s.query(BoardModel).filter_by(board_id=board_id).delete()


# Level 1d: the bank goes into the database before the run is minted
async def test_persisted_batch_ingests_the_bank_after_the_collision_check_and_before_minting(
        monkeypatch) -> None:
    """With persistence, the batch stores the frame and the boards it schedules, from the directory
    it was given, before the run row exists; no database is touched here (the steps are stubbed)."""
    steps = []
    monkeypatch.setenv("DATABASE_URL", "postgresql://unused")
    monkeypatch.setattr(batch_runner, "_check_schema_at_head", lambda: steps.append("schema"))
    monkeypatch.setattr(batch_runner, "_check_no_collisions",
                        lambda game_ids, master_seed: steps.append("collisions"))
    monkeypatch.setattr(batch_runner, "_ingest_bank",
                        lambda bank_dir, board_ids: steps.append(("ingest", bank_dir, board_ids)))

    def fake_create_run(**kwargs):
        steps.append("mint")
        return "run-1", None
    monkeypatch.setattr(batch_runner, "create_run", fake_create_run)
    monkeypatch.setattr(batch_runner, "run_single_game",
                        _fake_runner(lambda gi: ("completed", None)))

    report = await run_batch(models=_models(), boards=_bank(), master_seed=_MASTER_SEED,
                             temperature=0.4, persist=True, enforce_digests=False,
                             bank_dir="some/bank")

    assert [s if isinstance(s, str) else s[0] for s in steps] == [
        "schema", "collisions", "ingest", "mint"]
    _, bank_dir, board_ids = steps[2]
    assert bank_dir == Path("some/bank")
    scheduled = {c.board_id for c in build_schedule(_models(), _bank(), _MASTER_SEED)}
    assert sorted(scheduled) == board_ids
    assert report.completed == 192


# Level 1e: the schema the games are written to
@pytest.mark.parametrize("stamped", [set(), {"0005_run_delete_cascade"}])
async def test_a_schema_behind_the_migrations_aborts_before_any_game(monkeypatch, stamped) -> None:
    """A database never migrated, or one migration behind: the games would be played and then fail
    to be stored, so the batch stops first - before the collision check and the ingestion."""
    monkeypatch.setenv("DATABASE_URL", "postgresql://unused")
    monkeypatch.setattr(batch_runner, "_database_revisions", lambda: stamped)
    for step in ("_check_no_collisions", "_ingest_bank", "create_run", "run_single_game"):
        monkeypatch.setattr(batch_runner, step, _spy_never_called())

    with pytest.raises(BatchPreconditionError, match="upgrade head"):
        await run_batch(models=_models(), boards=_bank(), master_seed=_MASTER_SEED,
                        temperature=0.4, persist=True, enforce_digests=False)


def test_the_migrations_have_a_single_head_and_it_is_their_newest_file() -> None:
    versions = Path(__file__).resolve().parents[2] / "backend" / "migrations" / "versions"
    newest = max(p.stem for p in versions.glob("[0-9]*.py"))
    assert batch_runner._migration_heads() == {newest}


@pytest.mark.skipif(not os.environ.get("DATABASE_URL"),
                    reason="requires a live Postgres database")
def test_a_database_at_head_passes_the_schema_check() -> None:
    batch_runner._check_schema_at_head()


async def test_dry_run_never_ingests(monkeypatch) -> None:
    def spy(bank_dir, board_ids):  # pragma: no cover - must never run
        raise AssertionError("a batch that does not persist must not touch the database")
    monkeypatch.setattr(batch_runner, "_ingest_bank", spy)

    report = await run_batch(models=_models(), boards=_bank(), master_seed=_MASTER_SEED,
                             temperature=0.4, persist=False, enforce_digests=False,
                             make_client_factory=dry_run_client_factory)

    assert report.completed == 192


def test_ingest_bank_needs_the_measurement_frame(tmp_path) -> None:
    """The sealed boards point at the frame: without its sidecar the batch stops before storing any
    board, and before opening a database session."""
    with pytest.raises(BatchPreconditionError, match="no measurement frame"):
        batch_runner._ingest_bank(tmp_path, ["control-000"])


_needs_db = pytest.mark.skipif(not os.environ.get("DATABASE_URL"),
                               reason="requires a live Postgres database")


@_needs_db
def test_ingest_bank_stores_the_frame_and_the_boards_once() -> None:
    from sqlalchemy import select

    from backend.app.core.loader import BoardLoader
    from backend.app.db.models import BoardModel, MeasurementFrameModel
    from backend.app.db.session import session_scope

    board_ids = sorted(b.board_id for group in BoardLoader(str(_BOARDS_DIR)).boards.values()
                       for b in group)
    frame_id = json.loads((_BOARDS_DIR / "measurement_frame.json").read_text("utf-8"))["frame_id"]

    batch_runner._ingest_bank(_BOARDS_DIR, board_ids)
    batch_runner._ingest_bank(_BOARDS_DIR, board_ids)  # a second batch adds nothing, and passes

    with session_scope() as session:
        assert session.get(MeasurementFrameModel, frame_id) is not None
        stored = set(session.execute(select(BoardModel.board_id).where(
            BoardModel.board_id.in_(board_ids))).scalars().all())
    assert stored == set(board_ids)


@_needs_db
def test_ingest_bank_stops_when_a_scheduled_board_is_not_stored() -> None:
    with pytest.raises(BatchPreconditionError, match="no-such-board"):
        batch_runner._ingest_bank(_BOARDS_DIR, ["control-000", "no-such-board"])


@_needs_db
def test_ingest_bank_stops_on_a_frame_that_contradicts_the_stored_one(tmp_path) -> None:
    """Same frame_id, different encoder: the frame is immutable, so the batch stops instead of
    overwriting it."""
    batch_runner._ingest_bank(_BOARDS_DIR, [])  # the real frame is stored
    tampered = json.loads((_BOARDS_DIR / "measurement_frame.json").read_text("utf-8"))
    tampered["encoder"] = {**tampered["encoder"], "name": "different/model"}
    (tmp_path / "measurement_frame.json").write_text(json.dumps(tampered), encoding="utf-8")

    with pytest.raises(BatchPreconditionError, match="contradicts"):
        batch_runner._ingest_bank(tmp_path, [])


# Level 2: count-and-continue
async def test_count_and_continue_plays_all_cells(monkeypatch) -> None:
    # Alternate error/complete by game_index: consecutive errors never exceed 1, so no pairing aborts
    # and every one of the 192 cells is attempted.
    def outcome(gi):
        return ("error", "provider timeout") if gi % 2 == 0 else ("completed", None)
    monkeypatch.setattr(batch_runner, "run_single_game", _fake_runner(outcome))

    report = await run_batch(models=_models(), boards=_bank(), master_seed=_MASTER_SEED,
                             temperature=0.4, persist=False, enforce_digests=False)

    assert report.attempted == 192
    assert report.completed == 96
    assert report.errored == 96
    assert report.pairings_aborted == 0
    for pr in report.pairings:
        assert pr.attempted == 32 and pr.completed == 16 and pr.errored == 16
        assert not pr.aborted
        # error strings classified as provider failures (data, not noise).
        assert pr.error_kinds == {"provider": 16}


def _llm_error_classes():
    from backend.app.models import llm_errors
    return [cls for cls in vars(llm_errors).values()
            if isinstance(cls, type) and issubclass(cls, llm_errors.LLMError)
            and cls is not llm_errors.LLMError]


@pytest.mark.parametrize("cls", _llm_error_classes(), ids=lambda cls: cls.__name__)
def test_every_llm_error_code_has_a_kind(cls) -> None:
    """An LLM error is classified by its code, never by its message: the message of every LLMError
    names its provider ("provider=ollama"), which once made a model that loops read as a provider
    failure."""
    exc = cls(provider="ollama")
    assert exc.code in batch_runner._PROVIDER_ERROR_CODES | batch_runner._MODEL_ERROR_CODES
    assert batch_runner._classify_error(str(exc), exc.code) in {"provider", "model"}


@pytest.mark.parametrize("code, kind", [
    ("truncated_response", "model"), ("degenerate_response", "model"), ("empty_response", "model"),
    ("parse_error", "model"), ("refused", "model"),
    ("timeout", "provider"), ("rate_limit", "provider"), ("provider_unavailable", "provider"),
    ("model_not_found", "provider"), ("invalid_request", "provider"),
])
def test_an_llm_error_is_classified_by_its_code(code, kind) -> None:
    assert batch_runner._classify_error(
        f"LLMError(code={code}, message=..., provider=ollama)", code) == kind


async def test_the_report_counts_a_model_that_loops_as_a_model_error(monkeypatch) -> None:
    from backend.app.models.llm_errors import LLMTruncatedResponseError
    exc = LLMTruncatedResponseError(provider="ollama")

    async def fake(**kwargs):
        gi = kwargs["game_index"]
        if gi == 0:
            return GameRunResult(game_id="g0", run_id=kwargs["run_id"], seed_game=0,
                                 status="error", result=None, error=str(exc), error_code=exc.code)
        return GameRunResult(game_id=f"g{gi}", run_id=kwargs["run_id"], seed_game=0,
                             status="completed", result=None)
    monkeypatch.setattr(batch_runner, "run_single_game", fake)

    report = await run_batch(models=_models(), boards=_bank(), master_seed=_MASTER_SEED,
                             temperature=0.4, persist=False, enforce_digests=False)

    assert report.pairings[0].error_kinds == {"model": 1}


# Level 2: per-pairing consecutive-failure threshold
async def test_threshold_aborts_pairing_and_next_pairing_still_runs(monkeypatch) -> None:
    # Pairing 0 is game_index 0..31. Error its first 5 games -> abort after the 5th; the remaining 27
    # are skipped. Everything else completes, so pairing 1+ run normally.
    def outcome(gi):
        return ("error", "boom") if gi < 5 else ("completed", None)
    monkeypatch.setattr(batch_runner, "run_single_game", _fake_runner(outcome))

    report = await run_batch(models=_models(), boards=_bank(), master_seed=_MASTER_SEED,
                             temperature=0.4, persist=False, enforce_digests=False,
                             consecutive_failure_threshold=5)

    p0 = report.pairings[0]
    assert p0.aborted is True
    assert p0.attempted == 5 and p0.errored == 5 and p0.completed == 0
    assert p0.skipped == 27
    assert "consecutive" in (p0.abort_reason or "")
    assert report.pairings_aborted == 1
    # the other five pairings each ran their full 32 games.
    for pr in report.pairings[1:]:
        assert pr.attempted == 32 and pr.completed == 32 and not pr.aborted


async def test_threshold_counter_is_consecutive_not_cumulative(monkeypatch) -> None:
    # Pairing 0: 4 errors, 1 completion (resets), 4 more errors = 8 errors but never 5 in a row.
    # A cumulative counter would abort at the 5th error; a consecutive one must NOT.
    erroring = {0, 1, 2, 3, 5, 6, 7, 8}  # game_index 4 and >=9 complete

    def outcome(gi):
        return ("error", "boom") if gi in erroring else ("completed", None)
    monkeypatch.setattr(batch_runner, "run_single_game", _fake_runner(outcome))

    report = await run_batch(models=_models(), boards=_bank(), master_seed=_MASTER_SEED,
                             temperature=0.4, persist=False, enforce_digests=False,
                             consecutive_failure_threshold=5)

    p0 = report.pairings[0]
    assert p0.aborted is False  # the discriminating assertion: consecutive, not cumulative
    assert p0.attempted == 32
    assert p0.errored == 8 and p0.completed == 24
    assert report.pairings_aborted == 0


# Level 2: a game lost to the provider is played again
def _scripted_runner(script, calls):
    """A fake run_single_game playing each game as ``script`` says: ``script[game_index]`` is the
    list of ``(status, error_code)`` outcomes of its first, second, ... play; a game past its script
    (or without one) completes. Every play's game_index is appended to ``calls``."""
    async def fake(**kwargs):
        gi = kwargs["game_index"]
        calls.append(gi)
        outcomes = script.get(gi, [])
        nth = calls.count(gi) - 1
        status, code = outcomes[nth] if nth < len(outcomes) else ("completed", None)
        return GameRunResult(game_id=f"g{gi}", run_id=kwargs["run_id"], seed_game=0,
                             status=status, result=None,
                             error=None if code is None else f"LLMError(code={code})",
                             error_code=code)
    return fake


def _provider(healthy: bool, probes: list):
    """A fake _provider_healthy that always answers ``healthy``, recording the game_index of each
    probe in ``probes``."""
    async def fake(cell, factory):
        probes.append(cell.game_index)
        return healthy
    return fake


async def test_a_game_lost_to_the_provider_is_played_again(monkeypatch) -> None:
    calls, probes = [], []
    monkeypatch.setattr(batch_runner, "run_single_game",
                        _scripted_runner({3: [("error", "provider_unavailable")]}, calls))
    monkeypatch.setattr(batch_runner, "_provider_healthy", _provider(True, probes))

    report = await run_batch(models=_models(), boards=_bank(), master_seed=_MASTER_SEED,
                             temperature=0.4, persist=False, enforce_digests=False)

    assert calls.count(3) == 2 and probes == [3]
    assert report.attempted == 192 and report.completed == 192 and report.errored == 0
    assert report.aborted_reason is None


async def test_the_error_row_goes_before_the_replay(monkeypatch) -> None:
    """The replay writes the same deterministic game.id, so the error row must go first."""
    calls, forgotten = [], []
    monkeypatch.setenv("DATABASE_URL", "postgresql://unused")
    for step in ("_check_schema_at_head", "_check_no_collisions", "_ingest_bank"):
        monkeypatch.setattr(batch_runner, step, lambda *args, **kwargs: None)
    monkeypatch.setattr(batch_runner, "create_run", lambda **kwargs: ("run-1", None))
    monkeypatch.setattr(batch_runner, "run_single_game",
                        _scripted_runner({3: [("error", "timeout")]}, calls))
    monkeypatch.setattr(batch_runner, "_provider_healthy", _provider(True, []))
    monkeypatch.setattr(batch_runner, "_forget_error_game",
                        lambda game_id, run_id: forgotten.append((game_id, run_id, len(calls))))

    report = await run_batch(models=_models(), boards=_bank(), master_seed=_MASTER_SEED,
                             temperature=0.4, persist=True, enforce_digests=False)

    # deleted once, after the first play of game 3 and before its replay
    assert forgotten == [("g3", "run-1", calls.index(3) + 1)]
    assert report.completed == 192


async def test_an_outage_does_not_abort_the_pairing(monkeypatch) -> None:
    """Six games in a row lost to the provider and then replayed: none of them counts towards the
    threshold of five, which is for games the model loses."""
    calls = []
    monkeypatch.setattr(batch_runner, "run_single_game", _scripted_runner(
        {gi: [("error", "timeout")] for gi in range(6)}, calls))
    monkeypatch.setattr(batch_runner, "_provider_healthy", _provider(True, []))

    report = await run_batch(models=_models(), boards=_bank(), master_seed=_MASTER_SEED,
                             temperature=0.4, persist=False, enforce_digests=False,
                             consecutive_failure_threshold=5)

    assert report.pairings_aborted == 0 and report.completed == 192


async def test_a_provider_that_does_not_come_back_stops_the_batch(monkeypatch) -> None:
    calls, probes = [], []
    monkeypatch.setattr(batch_runner, "run_single_game",
                        _scripted_runner({40: [("error", "provider_unavailable")]}, calls))
    monkeypatch.setattr(batch_runner, "_provider_healthy", _provider(False, probes))

    report = await run_batch(models=_models(), boards=_bank(), master_seed=_MASTER_SEED,
                             temperature=0.4, persist=False, enforce_digests=False,
                             provider_wait_s=0)

    # game 40 (pairing 1) is played once and not again; nothing after it is played
    assert calls == list(range(41)) and probes == [40]
    assert report.attempted == 41 and report.completed == 40 and report.errored == 1
    assert "game_index 40" in report.aborted_reason
    assert report.as_dict()["aborted_reason"] == report.aborted_reason
    p1 = report.pairings[1]
    assert p1.error_kinds == {"provider": 1} and p1.skipped == 23 and not p1.aborted
    assert report.pairings_aborted == 0
    assert all(pr.attempted == 0 for pr in report.pairings[2:])


async def test_a_game_the_provider_keeps_failing_is_an_error_after_its_replays(monkeypatch) -> None:
    """The provider answers its health check but fails the same game every time: after two replays
    the game stays an error, and five of them in a row abort the pairing like any other error."""
    calls = []
    monkeypatch.setattr(batch_runner, "run_single_game", _scripted_runner(
        {gi: [("error", "timeout")] * 3 for gi in range(5)}, calls))
    monkeypatch.setattr(batch_runner, "_provider_healthy", _provider(True, []))

    report = await run_batch(models=_models(), boards=_bank(), master_seed=_MASTER_SEED,
                             temperature=0.4, persist=False, enforce_digests=False,
                             provider_replays=2, consecutive_failure_threshold=5)

    assert [calls.count(gi) for gi in range(5)] == [3] * 5
    p0 = report.pairings[0]
    assert p0.aborted and p0.errored == 5 and p0.error_kinds == {"provider": 5}
    assert report.aborted_reason is None


@pytest.mark.parametrize("code", [
    "model_not_found", "invalid_request", "auth", "truncated_response", "parse_error", None])
async def test_only_a_game_lost_to_the_provider_is_played_again(monkeypatch, code) -> None:
    """A setup error would fail the replay the same way, and a model's error is data."""
    calls, probes = [], []
    monkeypatch.setattr(batch_runner, "run_single_game",
                        _scripted_runner({0: [("error", code)]}, calls))
    monkeypatch.setattr(batch_runner, "_provider_healthy", _provider(True, probes))

    report = await run_batch(models=_models(), boards=_bank(), master_seed=_MASTER_SEED,
                             temperature=0.4, persist=False, enforce_digests=False)

    assert calls.count(0) == 1 and probes == []
    assert report.errored == 1


@pytest.mark.parametrize("cls", _llm_error_classes(), ids=lambda cls: cls.__name__)
def test_the_games_played_again_are_those_lost_to_a_retriable_provider_error(cls) -> None:
    exc = cls(provider="ollama")
    assert (exc.code in batch_runner._TRANSIENT_PROVIDER_CODES) == (
        exc.code in batch_runner._PROVIDER_ERROR_CODES and exc.retriable)


async def _wait(answers, max_wait_s):
    """Run _wait_for_provider against a probe answering ``answers`` in order (then True), with a
    sleep that only records; return (result, probes made, sleeps)."""
    it, probes, sleeps = iter(answers), [], []

    async def probe():
        probes.append(None)
        return next(it, True)

    async def sleep(s):
        sleeps.append(s)

    result = await batch_runner._wait_for_provider(probe, max_wait_s=max_wait_s, sleep=sleep)
    return result, len(probes), sleeps


async def test_the_wait_ends_as_soon_as_the_provider_answers() -> None:
    assert await _wait([True], 600) == (True, 1, [])
    assert await _wait([False, False, True], 600) == (True, 3, [15, 30])


async def test_the_wait_gives_up_after_its_budget() -> None:
    result, probes, sleeps = await _wait([False] * 100, 100)
    assert result is False and sleeps == [15, 30, 55] and probes == 4


async def test_the_wait_probes_at_least_every_five_minutes() -> None:
    result, _, sleeps = await _wait([False] * 100, 3600)
    assert result is False and sum(sleeps) == 3600
    assert sleeps[:6] == [15, 30, 60, 120, 240, 300] and max(sleeps) == 300


async def test_a_wait_without_budget_probes_once() -> None:
    assert await _wait([False], 0) == (False, 1, [])


class _ProbeClient:
    def __init__(self, healthy):
        self.healthy, self.closed = healthy, False

    async def health_check(self):
        return self.healthy

    async def close(self):
        self.closed = True


@pytest.mark.parametrize("seat0, seat1, expected", [
    (True, True, True), (True, False, False), (False, True, False)])
async def test_the_provider_is_healthy_when_both_seats_answer(seat0, seat1, expected) -> None:
    cell = build_schedule(_models(), _bank(), _MASTER_SEED)[0]
    built = []

    def factory(seat_index, spec):
        built.append((seat_index, spec))
        return _ProbeClient(seat0 if seat_index == 0 else seat1)

    assert await batch_runner._provider_healthy(cell, factory) is expected
    assert [(s, spec) for s, spec in built] == [(0, cell.seat0_spec), (1, cell.seat1_spec)]


async def test_the_health_check_clients_are_closed() -> None:
    cell = build_schedule(_models(), _bank(), _MASTER_SEED)[0]
    clients = []

    def factory(seat_index, spec):
        clients.append(_ProbeClient(healthy=False))
        return clients[-1]

    await batch_runner._provider_healthy(cell, factory)
    assert len(clients) == 2 and all(c.closed for c in clients)


# report shape + provenance
async def test_report_shape_and_provenance(monkeypatch) -> None:
    def outcome(gi):
        # make exactly pairing 2 abort (its first 5 games error); the rest complete.
        return ("error", "model produced no legal play") if 64 <= gi < 69 else ("completed", None)
    monkeypatch.setattr(batch_runner, "run_single_game", _fake_runner(outcome))

    report = await run_batch(models=_models(), boards=_bank(), master_seed=_MASTER_SEED,
                             temperature=0.4, persist=False, enforce_digests=False)

    # provenance of the batch is on the report.
    assert report.master_seed == _MASTER_SEED
    assert isinstance(report.run_id, str) and report.run_id
    # per-pairing completed/errored/aborted are all legible.
    assert len(report.pairings) == 6
    aborted = [pr for pr in report.pairings if pr.aborted]
    assert len(aborted) == 1 and aborted[0].pairing_ordinal == 2
    # classified as model-no-legal-play
    assert aborted[0].error_kinds == {"model": 5}
    # a round-trippable dict for the analysis.
    d = report.as_dict()
    assert d["run_id"] == report.run_id and d["master_seed"] == _MASTER_SEED
    assert len(d["pairings"]) == 6 and d["pairings"][2]["aborted"] is True


# hermetic end-to-end: real loop + real run_single_game + deterministic mock clients (no DB/daemon)
async def test_dry_run_plays_all_games_to_completion() -> None:
    # Exercises the whole wiring: build_schedule -> create_run(persist=False) -> the loop ->
    # run_single_game -> the engine -> DryRunClient (guessing each board's shared assassin) via the
    # make_client_factory seam. Every game completes; nothing errors or aborts.
    report = await run_batch(models=_models(), boards=_bank(), master_seed=_MASTER_SEED,
                             temperature=0.4, persist=False, enforce_digests=False,
                             make_client_factory=dry_run_client_factory)

    assert report.attempted == 192
    assert report.completed == 192
    assert report.errored == 0
    assert report.pairings_aborted == 0
    assert report.run_id  # synthesized deterministic run id (persist=False)
