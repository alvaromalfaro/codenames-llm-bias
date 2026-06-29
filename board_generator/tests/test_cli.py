"""Offline tests for the CLI subcommand dispatch (board_generator.cli).

These exercise only the argparse wiring: which subcommand runs, with which arguments. The dilemma
flow itself is interactive and needs the primary arbiter φ* / Hugging Face / stdin, so it is not
invoked here - run_dilemma_flow is monkeypatched to capture the kwargs it would receive. The bank
flow's full dispatch is covered by test_bank.py's CLI tests.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from board_generator import cli, dilemma_flow


def _capture_dilemma(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Replace run_dilemma_flow with a recorder; returns the dict that captures its call."""
    captured: dict[str, Any] = {}

    def fake(specification: str, **kwargs: Any) -> Path:
        captured["spec"] = specification
        captured["kwargs"] = kwargs
        return Path("/tmp/dilemma_stub.json")

    monkeypatch.setattr(cli, "run_dilemma_flow", fake)
    return captured


def test_dilemma_dispatch_passes_explicit_args(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured = _capture_dilemma(monkeypatch)

    cli.main(
        [
            "dilemma",
            "--spec", "gender-career",
            "--words-dir", str(tmp_path / "words"),
            "--subtlex-path", str(tmp_path / "subtlex.csv"),
            "--out-dir", str(tmp_path / "dilemmas"),
            "--k", "5",
            "--attempt-cap", "12",
        ]
    )

    assert captured["spec"] == "gender-career"
    assert captured["kwargs"] == {
        "words_dir": tmp_path / "words",
        "subtlex_path": tmp_path / "subtlex.csv",
        "out_dir": tmp_path / "dilemmas",
        "k": 5,
        "attempt_cap": 12,
    }


def test_dilemma_dispatch_uses_documented_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _capture_dilemma(monkeypatch)

    cli.main(["dilemma", "--spec", "gender-science"])

    assert captured["spec"] == "gender-science"
    kwargs = captured["kwargs"]
    assert kwargs["words_dir"] == cli.DEFAULT_WORDS_DIR
    assert kwargs["subtlex_path"] == cli.DEFAULT_SUBTLEX_PATH
    assert kwargs["out_dir"] == dilemma_flow.DEFAULT_DILEMMA_DIR
    assert kwargs["k"] == dilemma_flow.DEFAULT_K
    assert kwargs["attempt_cap"] is None


def test_no_subcommand_exits_non_zero() -> None:
    with pytest.raises(SystemExit):
        cli.main([])


def test_bank_requires_manifest() -> None:
    with pytest.raises(SystemExit):
        cli.main(["bank"])


def test_dilemma_rejects_unknown_spec() -> None:
    with pytest.raises(SystemExit):
        cli.main(["dilemma", "--spec", "gender-occupation"])
