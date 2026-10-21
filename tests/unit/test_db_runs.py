"""Tests for backend.app.db.runs: runs read back with their games counted by status.

The database tests (marked ``db``) run on the test session's Postgres (tests/conftest.py). Each
inserts its own board, runs and games (unique ids) and deletes them afterwards."""
import uuid
from decimal import Decimal

import pytest

from backend.app.db.runs import get_run, list_runs

_needs_db = pytest.mark.db


def test_an_id_that_is_not_a_run_id_finds_no_run(monkeypatch) -> None:
    """Postgres would reject it as a uuid; it is no run, and no database is asked."""
    monkeypatch.delenv("DATABASE_URL", raising=False)
    assert get_run("not-a-run-id") is None


@pytest.fixture
def two_runs():
    """Two runs on one board: the first with 2 completed games and 1 error, the second with none.
    Each run is stored in a transaction of its own: created_at is the transaction's now(), and
    runs created together would tie."""
    from backend.app.db.models import BoardModel, GameModel, RunModel
    from backend.app.db.session import session_scope
    from backend.app.db.writer import delete_run

    board_id = f"test-runs-{uuid.uuid4()}"
    run_ids = [str(uuid.uuid4()), str(uuid.uuid4())]
    snapshot = {"0": {"provider": "ollama", "model_name": "m", "identity_kind": "ollama_digest",
                      "identity": "abc"}}
    with session_scope() as s:
        s.add(BoardModel(board_id=board_id))
        s.add(RunModel(id=run_ids[0], master_seed=Decimal(2026), temperature=0.8,
                       code_version="abc1234", prompt_template_version="tpl",
                       model_registry_snapshot=snapshot, notes="a note"))
        s.flush()
        for status in ("completed", "completed", "error"):
            s.add(GameModel(id=str(uuid.uuid4()), run_id=run_ids[0], board_id=board_id,
                            game_status=status))
    with session_scope() as s:
        s.add(RunModel(id=run_ids[1], temperature=0.4))
    yield run_ids, snapshot
    for run_id in run_ids:
        delete_run(run_id)
    with session_scope() as s:
        s.query(BoardModel).filter_by(board_id=board_id).delete()


@_needs_db
def test_a_run_is_read_back_with_its_games_by_status(two_runs) -> None:
    (first, second), snapshot = two_runs

    run = get_run(first)
    assert (run.run_id, run.master_seed, run.temperature, run.code_version) == (
        first, 2026, 0.8, "abc1234")
    assert run.prompt_template_version == "tpl" and run.model_registry_snapshot == snapshot
    assert run.notes == "a note" and run.created_at.tzinfo is not None
    assert run.games_by_status == {"completed": 2, "error": 1} and run.games == 3

    empty = get_run(second)
    assert empty.master_seed is None and empty.games_by_status == {} and empty.games == 0
    assert get_run(str(uuid.uuid4())) is None


@_needs_db
def test_every_run_is_listed_oldest_first(two_runs) -> None:
    (first, second), _ = two_runs
    listed = [run.run_id for run in list_runs()]
    assert listed.index(first) < listed.index(second)
