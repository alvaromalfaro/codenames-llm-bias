"""Board serialization: internal Board -> platform JSON contract.

Covers the translation (role_a/role_b -> human/llm perspective, bystander -> civilian, UPPERCASE
text, gender_category verbatim, OOV subtlex_freq -> null), determinism (byte-identical), write
paths/filenames, and the defensive schema guards. Offline and deterministic - no phi*/HF/network.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from board_generator.balancing import BalanceReport
from board_generator.board import (
    Board,
    assemble_board,
    to_json_dict,
    write_balance_report,
    write_board,
)

from tests.test_board import _consensus, _control_words, _probe_words, _word


def _probe_board() -> Board:
    words, dilemma = _probe_words()
    return assemble_board(
        "probe-career-000", "probe", "gender-career", 20260626, words, _consensus(), dilemma
    )


def _control_board() -> Board:
    words = _control_words()
    return assemble_board(
        "control-career-000", "control", "gender-career", 7, words, _consensus(), None
    )


def test_probe_round_trip_translation() -> None:
    payload = to_json_dict(_probe_board())

    # Top-level board axis is "gender" for a probe.
    assert payload["category"] == "gender"
    assert payload["type"] == "probe"
    assert payload["specification"] == "gender-career"
    assert payload["grid"] == {"rows": 5, "cols": 5}
    assert len(payload["cards"]) == 25
    assert [c["id"] for c in payload["cards"]] == list(range(25))

    # Card text is UPPERCASE; card category is the word's pole verbatim.
    for card in payload["cards"]:
        assert card["text"] == card["text"].upper()
        assert card["category"] in {"male", "female", "neutral"}
        assert card["human_perspective_role"] in {
            "agent", "civilian", "assassin"}
        assert card["llm_perspective_role"] in {
            "agent", "civilian", "assassin"}

    # A loaded card carries a gendered pole; the dilemma target ("nurse") is female.
    nurse = next(c for c in payload["cards"] if c["text"] == "NURSE")
    assert nurse["category"] == "female"
    neutral = next(c for c in payload["cards"] if c["category"] == "neutral")
    assert neutral["category"] == "neutral"

    # Dilemma block present with arbiter_scores; keycard_audit present, bystander key kept.
    assert payload["dilemma"] is not None
    assert payload["dilemma"]["target"] == "nurse"
    assert "arbiter_scores" in payload["dilemma"]
    assert set(payload["keycard_audit"]["per_perspective"]) == {
        "agent", "bystander", "assassin"}
    assert payload["keycard_audit"]["per_perspective"] == {
        "agent": 9, "bystander": 13, "assassin": 3
    }


def test_bystander_maps_to_civilian_in_both_perspectives() -> None:
    board = _probe_board()
    payload = to_json_dict(board)
    by_id = {c["id"]: c for c in payload["cards"]}
    saw_civilian = False
    for entry in board.words:
        card = by_id[entry.index]
        if entry.role_a == "bystander":
            assert card["human_perspective_role"] == "civilian"
            saw_civilian = True
        if entry.role_b == "bystander":
            assert card["llm_perspective_role"] == "civilian"
            saw_civilian = True
        # Non-bystander roles pass through unchanged.
        if entry.role_a == "agent":
            assert card["human_perspective_role"] == "agent"
    assert saw_civilian  # the keycard guarantees 13 bystanders per perspective


def test_oov_neutral_subtlex_freq_serializes_to_null() -> None:
    words = _control_words()
    words[3] = _word("oovword", "neutral", subtlex=None)
    board = assemble_board(
        "control-career-001", "control", "gender-career", 11, words, _consensus(), None
    )
    # Round-trip through JSON to confirm the value is null (present), not omitted, not imputed.
    reloaded = json.loads(json.dumps(to_json_dict(board)))
    card = next(c for c in reloaded["cards"] if c["text"] == "OOVWORD")
    assert "subtlex_freq" in card["covariates"]
    assert card["covariates"]["subtlex_freq"] is None


def test_control_board_has_no_gender_axis_and_no_dilemma() -> None:
    payload = to_json_dict(_control_board())
    assert payload["category"] == "neutral"
    assert payload["specification"] is None or isinstance(
        payload["specification"], str)
    assert payload["dilemma"] is None
    assert all(c["category"] == "neutral" for c in payload["cards"])


def test_i8_to_json_dict_byte_identical_across_runs() -> None:
    board = _probe_board()
    a = json.dumps(to_json_dict(board), indent=2, ensure_ascii=False)
    b = json.dumps(to_json_dict(board), indent=2, ensure_ascii=False)
    assert a == b


def test_i8_write_board_file_byte_identical(tmp_path: Path) -> None:
    board = _probe_board()
    p1 = write_board(board, tmp_path / "run1")
    p2 = write_board(board, tmp_path / "run2")
    assert p1.read_bytes() == p2.read_bytes()
    # Writing twice to the same path is stable too.
    first = p1.read_bytes()
    write_board(board, tmp_path / "run1")
    assert p1.read_bytes() == first


def test_write_board_filename_and_isolation(tmp_path: Path) -> None:
    out = tmp_path / "boards"
    probe = write_board(_probe_board(), out)
    assert probe.name == "gender_probe-career-000.json"
    assert probe.parent == out

    control = write_board(_control_board(), out)
    assert control.name == "neutral_control-career-000.json"

    # Nothing written outside out_dir.
    assert {p.name for p in out.iterdir()} == {
        "gender_probe-career-000.json",
        "neutral_control-career-000.json",
    }
    # File content round-trips to valid JSON matching the board.
    loaded = json.loads(probe.read_text(encoding="utf-8"))
    assert loaded["board_id"] == "probe-career-000"
    assert probe.read_text(encoding="utf-8").endswith("\n")


def _minimal_report() -> BalanceReport:
    return BalanceReport(
        specifications=[],
        criterion="smd",
        seed=20260626,
        alpha=0.05,
        tost_margin=0.2,
        caliper_sd=0.2,
        smd_threshold=0.1,
        smd_well_threshold=0.1,
    )


def test_write_balance_report(tmp_path: Path) -> None:
    out = tmp_path / "boards"
    path = write_balance_report(_minimal_report(), out)
    assert path.name == "balance_report.json"
    assert path.parent == out
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["criterion"] == "smd"
    assert loaded["seed"] == 20260626
    assert loaded["specifications"] == []
    assert path.read_text(encoding="utf-8").endswith("\n")


def test_schema_guard_rejects_wrong_card_count() -> None:
    board = _probe_board()
    tampered = dataclasses.replace(board, words=board.words[:24])
    with pytest.raises(ValueError, match="24 cards"):
        to_json_dict(tampered)


def test_schema_guard_rejects_probe_without_dilemma() -> None:
    board = _probe_board()
    tampered = dataclasses.replace(board, dilemma=None)
    with pytest.raises(ValueError, match="missing its dilemma"):
        to_json_dict(tampered)
