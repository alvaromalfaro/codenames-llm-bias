"""Tests for the resume flags of scripts/run_batch.py: what they need, and what the CLI hands the
batch. The batch itself is a stub here; tests/unit/test_run_batch.py covers the resume."""

import importlib.util
import sys
from pathlib import Path

import pytest

from backend.app.core.batch_runner import BatchPreconditionError, BatchReport, StoredRun

_PATH = Path(__file__).resolve().parents[2] / "scripts" / "run_batch.py"
_SPEC = importlib.util.spec_from_file_location("run_batch_cli", _PATH)
cli = importlib.util.module_from_spec(_SPEC)
sys.modules["run_batch_cli"] = cli
_SPEC.loader.exec_module(cli)

_RUN_ID = "11111111-1111-4111-8111-111111111111"
_RUN = StoredRun(
    run_id=_RUN_ID,
    master_seed=2026,
    temperature=0.7,
    prompt_template_version="tpl",
    code_version="abc1234",
    model_registry_snapshot={},
)


@pytest.fixture
def batch(monkeypatch):
    """Stub the bank and the batch; return the keyword arguments the CLI ran the batch with."""
    calls = []

    async def fake_run_batch(**kwargs):
        calls.append(kwargs)
        return BatchReport(
            run_id=_RUN_ID,
            master_seed=kwargs["master_seed"],
            attempted=0,
            completed=0,
            errored=0,
            pairings_aborted=0,
            pairings=[],
            resumed=kwargs["resume_run_id"] is not None,
        )

    monkeypatch.setattr(cli, "_load_bank", lambda: ["a board"])
    monkeypatch.setattr(cli, "run_batch", fake_run_batch)
    monkeypatch.setattr(cli, "load_run", lambda run_id: _RUN if run_id == _RUN_ID else None)
    return calls


def test_a_resume_takes_the_seed_and_the_temperature_of_its_run(batch) -> None:
    assert cli.main(["--resume", _RUN_ID, "--retry-errors"]) == 0
    [kwargs] = batch
    assert kwargs["master_seed"] == 2026 and kwargs["temperature"] == 0.7
    assert kwargs["resume_run_id"] == _RUN_ID and kwargs["retry_errors"] is True
    assert kwargs["persist"] and kwargs["enforce_digests"]


def test_a_seed_given_with_resume_goes_to_the_batch_to_be_checked(batch) -> None:
    assert cli.main(["--resume", _RUN_ID, "--master-seed", "1", "--temperature", "0.7"]) == 0
    assert batch[0]["master_seed"] == 1 and batch[0]["retry_errors"] is False


def test_a_new_batch_resumes_nothing(batch) -> None:
    assert cli.main(["--master-seed", "1", "--temperature", "0.7"]) == 0
    assert batch[0]["resume_run_id"] is None and batch[0]["retry_errors"] is False


def test_an_unknown_run_is_a_precondition_failure(batch, capsys) -> None:
    assert cli.main(["--resume", "22222222-2222-4222-8222-222222222222"]) == 2
    assert "no run 22222222" in capsys.readouterr().err and batch == []


def test_reading_the_run_without_a_database_is_a_precondition_failure(
    batch, monkeypatch, capsys
) -> None:
    def no_database(run_id):
        raise BatchPreconditionError(f"reading run {run_id} requires DATABASE_URL.")

    monkeypatch.setattr(cli, "load_run", no_database)
    assert cli.main(["--resume", _RUN_ID]) == 2
    assert "DATABASE_URL" in capsys.readouterr().err and batch == []


def test_a_bank_with_a_file_that_is_not_a_board_is_a_precondition_failure(
    batch, monkeypatch, capsys
) -> None:
    from backend.app.core.loader import InvalidBoardFileError

    def invalid_bank():
        raise InvalidBoardFileError("data/boards/bad.json is not a valid board")

    monkeypatch.setattr(cli, "_load_bank", invalid_bank)
    assert cli.main(["--master-seed", "1", "--temperature", "0.7"]) == 2
    assert "bad.json" in capsys.readouterr().err and batch == []


@pytest.mark.parametrize(
    "argv, message",
    [
        (["--resume", _RUN_ID, "--dry-run"], "--dry-run"),
        (["--master-seed", "1", "--temperature", "0.7", "--retry-errors"], "goes with --resume"),
    ],
)
def test_flags_that_do_not_go_together(batch, argv, message) -> None:
    with pytest.raises(SystemExit, match=message):
        cli.main(argv)
    assert batch == []
