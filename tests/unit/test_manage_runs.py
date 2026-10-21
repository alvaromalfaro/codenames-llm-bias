"""Tests for scripts/manage_runs.py: what list and show print, and that delete asks before it
deletes. The database is stubbed; tests/unit/test_db_runs.py reads real runs."""
import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from backend.app.db.runs import RunSummary
from backend.app.db.writer import DeleteRunResult

_PATH = Path(__file__).resolve().parents[2] / "scripts" / "manage_runs.py"
_SPEC = importlib.util.spec_from_file_location("manage_runs", _PATH)
cli = importlib.util.module_from_spec(_SPEC)
sys.modules["manage_runs"] = cli
_SPEC.loader.exec_module(cli)

_RUN_ID = "11111111-1111-4111-8111-111111111111"


def _run(run_id=_RUN_ID, **changes):
    run = dict(
        run_id=run_id, created_at=datetime(2026, 10, 12, 21, 4, 11, tzinfo=timezone.utc),
        master_seed=2026, temperature=0.8, code_version="26a115a",
        prompt_template_version="templates@3f9a",
        model_registry_snapshot={
            "1": {"provider": "ollama", "model_name": "qwen2.5:14b",
                  "identity_kind": "ollama_digest", "identity": "b" * 64},
            "0": {"provider": "ollama", "model_name": "llama3.1:8b",
                  "identity_kind": "ollama_digest", "identity": "a" * 64}},
        stimulus_manifest={f"board-{i:02d}": f"{i:064x}" for i in range(16)},
        environment_snapshot={"python": "3.12.13", "ollama": {"host": "http://localhost:11434",
                                                              "version": "0.40.2"}},
        notes="2026-10-13T08:00:00+00:00 resumed at code_version 26a115a",
        games_by_status={"completed": 189, "error": 3})
    return RunSummary(**{**run, **changes})


@pytest.fixture
def db(monkeypatch):
    """Stub the runs in the database; return the run ids delete_run was called with."""
    runs = {_RUN_ID: _run()}
    deleted = []

    def delete_run(run_id):
        deleted.append(run_id)
        run = runs.pop(run_id, None)
        return DeleteRunResult(found=run is not None, games_deleted=run.games if run else 0)
    monkeypatch.setattr(cli, "list_runs", lambda: list(runs.values()))
    monkeypatch.setattr(cli, "get_run", runs.get)
    monkeypatch.setattr(cli, "delete_run", delete_run)
    return deleted


def _answer(monkeypatch, answer):
    """Answer the delete question with ``answer``; None is no terminal to answer from."""
    asked = []

    def fake_input(question):
        asked.append(question)
        if answer is None:
            raise EOFError
        return answer
    monkeypatch.setattr("builtins.input", fake_input)
    return asked


def test_list_shows_every_run_with_its_games(db, capsys) -> None:
    assert cli.main(["list"]) == 0
    out = capsys.readouterr().out
    assert _RUN_ID in out and "2026-10-12 21:04" in out and "26a115a" in out
    assert "189 completed, 3 error" in out


def test_list_says_when_there_are_no_runs(monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "list_runs", lambda: [])
    assert cli.main(["list"]) == 0
    assert "no runs" in capsys.readouterr().out


def test_show_prints_the_provenance_the_models_and_the_notes(db, capsys) -> None:
    assert cli.main(["show", _RUN_ID]) == 0
    out = capsys.readouterr().out
    for text in ("master_seed       2026", "temperature       0.8", "templates@3f9a",
                 "16 in its stimulus manifest", "python 3.12.13, ollama 0.40.2",
                 "192 (189 completed, 3 error)", "resumed at code_version 26a115a",
                 f"--resume {_RUN_ID} --retry-errors"):
        assert text in out
    # the models in seat order
    assert out.index("llama3.1:8b") < out.index("qwen2.5:14b") and "a" * 64 in out


def test_show_offers_no_retry_when_no_game_ended_in_error(monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "get_run", lambda run_id: _run(games_by_status={"completed": 192}))
    assert cli.main(["show", _RUN_ID]) == 0
    assert "--retry-errors" not in capsys.readouterr().out


def test_show_of_an_unknown_run_fails(db, capsys) -> None:
    assert cli.main(["show", "22222222-2222-4222-8222-222222222222"]) == 1
    assert "no run 22222222" in capsys.readouterr().err


def test_delete_shows_the_run_and_deletes_it_on_a_yes(db, monkeypatch, capsys) -> None:
    asked = _answer(monkeypatch, "y")
    assert cli.main(["delete", _RUN_ID]) == 0
    assert db == [_RUN_ID]
    assert "its 192 game(s)?" in asked[0]
    out = capsys.readouterr().out
    assert "master_seed" in out and f"deleted run {_RUN_ID} and its 192 game(s)" in out


@pytest.mark.parametrize("answer", ["", "n", "no", "maybe", None])
def test_delete_deletes_nothing_without_a_yes(db, monkeypatch, capsys, answer) -> None:
    _answer(monkeypatch, answer)
    assert cli.main(["delete", _RUN_ID]) == 1
    assert db == [] and "not deleted" in capsys.readouterr().out


def test_delete_with_yes_does_not_ask(db, monkeypatch) -> None:
    asked = _answer(monkeypatch, "n")
    assert cli.main(["delete", _RUN_ID, "--yes"]) == 0
    assert db == [_RUN_ID] and asked == []


def test_delete_of_an_unknown_run_asks_nothing(db, monkeypatch, capsys) -> None:
    asked = _answer(monkeypatch, "y")
    assert cli.main(["delete", "22222222-2222-4222-8222-222222222222"]) == 1
    assert db == [] and asked == [] and "no run" in capsys.readouterr().err


def test_without_a_database_it_stops(monkeypatch, capsys) -> None:
    def no_database():
        raise RuntimeError("DATABASE_URL is not set")
    monkeypatch.setattr(cli, "list_runs", no_database)
    assert cli.main(["list"]) == 2
    assert "DATABASE_URL is not set" in capsys.readouterr().err


def test_a_command_is_required() -> None:
    with pytest.raises(SystemExit):
        cli.main([])
