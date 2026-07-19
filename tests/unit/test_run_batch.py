"""Tests for the batch orchestrator (batch_runner.run_batch).

Two failure levels are exercised separately: Level-1 preconditions (bank shortfall, collision) abort
before any game; Level-2 fault isolation (count-and-continue + per-pairing consecutive-failure abort)
is tested by patching run_single_game so the loop's own logic is what's under test. One hermetic
end-to-end test plays all 192 games with deterministic mock clients (no DB, no daemon).

The threshold reset test is the discriminating one: 4 errors + 1 completion + 4 errors must not
abort, proving the counter is consecutive, not cumulative.
"""
import os

import pytest

from backend.app.core import batch_runner
from backend.app.core.batch_runner import (
    BatchPreconditionError, dry_run_client_factory, run_batch,
)
from backend.app.core.batch_schedule import ScheduleError, build_schedule
from backend.app.core.game_runner import GameRunResult

# Reuse the validated fixture bank + models from the schedule tests (same tests/unit package dir).
from test_batch_schedule import _bank, _models, _MASTER_SEED


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
    # a round-trippable dict for the step-7 analysis.
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
