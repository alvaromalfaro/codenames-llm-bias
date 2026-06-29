"""Offline, deterministic tests for the master bank loop (board_generator.bank).

No primary arbiter φ* / no Hugging Face / no network. Dilemma records are hand-built fixtures whose
accepted board.Dilemma carries consensus_ok verbatim (it is TRUSTED, never recomputed here). The
word pools come from the real resources/words/ + SUBTLEX-US corpus (mirrors test_balancing.py's
structural cases): composition and PSM matching need a realistic pool, but nothing in this module
touches an embedding model. The validation helper is unit-tested directly on hand-built boards so
the forced-violation cases stay controlled.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from board_generator import bank, cli, dilemma_flow
from board_generator import board as board_mod
from board_generator.balancing import BalanceReport
from board_generator.board import ArbiterSet, Board, Dilemma, Grid, WordEntry
from board_generator.lexicon import Specification
from board_generator.roles import KeyCard, KeycardAudit, Role

RESOURCES = Path(__file__).resolve().parents[1] / "resources"
REAL_WORDS = RESOURCES / "words"
REAL_SUBTLEX = RESOURCES / "frequencies" / "subtlex_us.csv"


# Fixture dilemma records (real corpus words; target/stereo gender-congruent, bridge neutral).


def make_record(
    specification: Specification,
    target: str,
    neutral_bridge: str,
    stereo: str,
    *,
    consensus_ok: bool = True,
) -> dilemma_flow.DilemmaRecord:
    accepted = Dilemma(
        target=target,
        neutral_bridge=neutral_bridge,
        stereotypical_bridge=stereo,
        consensus_ok=consensus_ok,
        arbiter_scores=[],
    )
    return dilemma_flow.DilemmaRecord(
        specification=specification,
        target=target,
        neutral_bridge=neutral_bridge,
        stereotypical_bridge=stereo,
        accepted=accepted,
        rejected_attempts=[],
        attempts_count=1,
        arbiters_consensus=["stub@rev"],
        arbiters_primary="stub@rev",
    )


CAREER0 = make_record("gender-career", "executive", "anchor", "business")
CAREER1 = make_record("gender-career", "family", "ant", "home")
SCIENCE0 = make_record("gender-science", "math", "ace", "physics")


def manifest_for(
    records: list[dilemma_flow.DilemmaRecord], master_seed: int
) -> bank.Manifest:
    filenames = [
        f"dilemma_{r.specification}_{r.target}.json" for r in records
    ]
    return bank.Manifest(master_seed=master_seed, dilemmas=filenames)


def build(
    records: list[dilemma_flow.DilemmaRecord], master_seed: int
) -> tuple[list[Board], BalanceReport, list[str]]:
    return bank.build_bank(
        manifest_for(records, master_seed),
        records,
        words_dir=REAL_WORDS,
        subtlex_path=REAL_SUBTLEX,
    )


# build_bank: composition, ids, 50/50, per-spec indices


def test_build_bank_shape_ids_and_indices() -> None:
    records = [CAREER0, CAREER1, SCIENCE0]
    boards, report, warnings = build(records, 12345)

    assert len(boards) == 6  # 3 probe + 3 control
    probes = [b for b in boards if b.type == "probe"]
    controls = [b for b in boards if b.type == "control"]
    assert len(probes) == len(controls) == 3  # 50/50 by construction

    probe_ids = [b.board_id for b in probes]
    assert probe_ids == [
        "probe-gender-career-000",
        "probe-gender-career-001",
        "probe-gender-science-000",
    ]
    assert {b.board_id for b in controls} == {
        "control-000", "control-001", "control-002"}

    # every probe carries its accepted dilemma; every control is dilemma-free and all-neutral.
    for b in probes:
        assert b.dilemma is not None and b.dilemma.consensus_ok
        assert all(w.text in {c.text for c in b.words} for w in b.words)
    for b in controls:
        assert b.dilemma is None
        assert all(w.gender_category == "neutral" for w in b.words)

    assert report.seed == 12345
    assert isinstance(warnings, list)


def test_build_bank_probe_embeds_dilemma_words() -> None:
    boards, _, _ = build([CAREER0], 7)
    probe = next(b for b in boards if b.type == "probe")
    assert probe.dilemma is not None
    texts = {w.text for w in probe.words}
    assert {"executive", "business", "anchor"} <= texts


# determinism


def _json_bytes(boards: list[Board]) -> str:
    return json.dumps([board_mod.to_json_dict(b) for b in boards], sort_keys=False)


def test_build_bank_is_byte_deterministic() -> None:
    records = [CAREER0, SCIENCE0]
    first, _, _ = build(records, 999)
    second, _, _ = build(records, 999)
    assert _json_bytes(first) == _json_bytes(second)


def test_different_master_seed_changes_the_bank() -> None:
    records = [CAREER0, SCIENCE0]
    one, _, _ = build(records, 1)
    two, _, _ = build(records, 2)
    assert _json_bytes(one) != _json_bytes(two)


# failure modes


def test_consensus_false_record_raises_naming_artifact() -> None:
    bad = make_record(
        "gender-career", "executive", "anchor", "business", consensus_ok=False
    )
    with pytest.raises(ValueError) as excinfo:
        build([bad], 5)
    message = str(excinfo.value)
    assert "consensus_ok=False" in message
    assert "dilemma_gender-career_executive.json" in message


def test_manifest_record_count_mismatch_raises() -> None:
    manifest = bank.Manifest(master_seed=1, dilemmas=["a.json", "b.json"])
    with pytest.raises(ValueError, match="manifest lists 2"):
        bank.build_bank(
            manifest, [CAREER0], words_dir=REAL_WORDS, subtlex_path=REAL_SUBTLEX
        )


# Validation helper, unit-tested on hand-built boards (forced violations stay controlled).


def _legal_keycard() -> KeyCard:
    from board_generator.roles import EXPECTED_JOINT

    pairs: list[tuple[Role, Role]] = []
    for (role_a, role_b), count in EXPECTED_JOINT.items():
        pairs.extend([(role_a, role_b)] * count)
    return KeyCard(
        role_a=tuple(p[0] for p in pairs),
        role_b=tuple(p[1] for p in pairs),
    )


def _hand_board(board_id: str, board_type: str, *, dilemma: Dilemma | None) -> Board:
    keycard = _legal_keycard()
    words = [
        WordEntry(
            text=f"{board_id}-w{i:02d}",
            index=i,
            role_a=keycard.role_a[i],
            role_b=keycard.role_b[i],
            gender_category="neutral",
            source="test",
            covariates={"subtlex_freq": 1.0,
                        "length": 6.0, "wordnet_polysemy": 1.0},
            weat_set=(),
        )
        for i in range(25)
    ]
    return Board(
        board_id=board_id,
        type=board_type,  # type: ignore[arg-type]
        specification="gender-career",
        seed=1,
        arbiters=ArbiterSet(consensus=["m@r"], primary="m@r"),
        grid=Grid(),
        words=words,
        dilemma=dilemma,
        keycard_audit=KeycardAudit(
            per_perspective={"agent": 9, "bystander": 13, "assassin": 3},
            overlap_ok=True,
            role_gender_independent=True,
        ),
    )


def _legal_dilemma() -> Dilemma:
    return Dilemma(
        target="t",
        neutral_bridge="n",
        stereotypical_bridge="s",
        consensus_ok=True,
        arbiter_scores=[],
    )


def test_control_carrying_dilemma_raises() -> None:
    probe = _hand_board("probe-x", "probe", dilemma=_legal_dilemma())
    bad_control = _hand_board("control-x", "control", dilemma=_legal_dilemma())
    with pytest.raises(ValueError, match="control-x.*must not carry a dilemma"):
        bank.validate_bank_invariants([probe, bad_control], 1)


def test_probe_with_false_consensus_raises() -> None:
    bad_dilemma = replace(_legal_dilemma(), consensus_ok=False)
    probe = _hand_board("probe-x", "probe", dilemma=bad_dilemma)
    control = _hand_board("control-x", "control", dilemma=None)
    with pytest.raises(ValueError, match="probe-x.*consensus_ok=False"):
        bank.validate_bank_invariants([probe, control], 1)


def test_duplicate_board_ids_raise() -> None:
    probe = _hand_board("dup", "probe", dilemma=_legal_dilemma())
    control = _hand_board("dup", "control", dilemma=None)
    with pytest.raises(ValueError, match="duplicate board_id"):
        bank.validate_bank_invariants([probe, control], 1)


def test_not_5050_raises() -> None:
    probe = _hand_board("probe-x", "probe", dilemma=_legal_dilemma())
    with pytest.raises(ValueError, match="expected 2"):
        bank.validate_bank_invariants([probe], 1)


def test_i5_dependence_warns_but_does_not_fail() -> None:
    probe = _hand_board("probe-x", "probe", dilemma=_legal_dilemma())
    control = _hand_board("control-x", "control", dilemma=None)
    dependent = replace(
        control,
        keycard_audit=replace(control.keycard_audit,
                              role_gender_independent=False),
    )
    warnings = bank.validate_bank_invariants([probe, dependent], 1)
    assert len(warnings) == 1
    assert "control-x" in warnings[0]


# Seed derivation


def test_derive_board_seed_properties() -> None:
    seed = bank.derive_board_seed(10, "probe-gender-career-000")
    # deterministic
    assert seed == bank.derive_board_seed(10, "probe-gender-career-000")
    # distinct per board_id and per master_seed
    assert seed != bank.derive_board_seed(10, "probe-gender-career-001")
    assert seed != bank.derive_board_seed(11, "probe-gender-career-000")
    # a 64-bit integer
    assert 0 <= seed < 2**64


def test_derive_board_seed_is_position_independent() -> None:
    # A board's seed depends only on its id, never on its position in the manifest list.
    listed_first = bank.derive_board_seed(3, "control-002")
    listed_later = bank.derive_board_seed(3, "control-002")
    assert listed_first == listed_later


# Balance runs once, seeded by the master seed


def test_run_balancing_called_once_with_master_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    seeds_seen: list[int] = []
    real = bank.run_balancing

    def spy(words, seed, **kwargs):  # type: ignore[no-untyped-def]
        seeds_seen.append(seed)
        return real(words, seed, **kwargs)

    monkeypatch.setattr(bank, "run_balancing", spy)
    _, report, _ = build([CAREER0, SCIENCE0], 77)
    assert seeds_seen == [77]
    assert report.seed == 77


# CLI wiring (I/O around build_bank)


def _write_artifacts(records: list[dilemma_flow.DilemmaRecord], dilemmas_dir: Path) -> None:
    for record in records:
        dilemma_flow.write_record(record, dilemmas_dir)


def _write_manifest(
    records: list[dilemma_flow.DilemmaRecord], master_seed: int, path: Path
) -> None:
    payload = {
        "master_seed": master_seed,
        "dilemmas": [f"dilemma_{r.specification}_{r.target}.json" for r in records],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_cli_dry_run_writes_nothing(tmp_path: Path) -> None:
    records = [CAREER0, SCIENCE0]
    dilemmas_dir = tmp_path / "dilemmas"
    _write_artifacts(records, dilemmas_dir)
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(records, 3, manifest_path)
    out_dir = tmp_path / "out"

    cli.main(
        [
            "bank",
            "--manifest", str(manifest_path),
            "--dilemmas-dir", str(dilemmas_dir),
            "--words-dir", str(REAL_WORDS),
            "--subtlex-path", str(REAL_SUBTLEX),
            "--out-dir", str(out_dir),
            "--dry-run",
        ]
    )
    assert not out_dir.exists() or not any(out_dir.iterdir())


def test_cli_writes_boards_and_report(tmp_path: Path) -> None:
    records = [CAREER0, SCIENCE0]
    dilemmas_dir = tmp_path / "dilemmas"
    _write_artifacts(records, dilemmas_dir)
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(records, 8, manifest_path)
    out_dir = tmp_path / "out"

    cli.main(
        [
            "bank",
            "--manifest", str(manifest_path),
            "--dilemmas-dir", str(dilemmas_dir),
            "--words-dir", str(REAL_WORDS),
            "--subtlex-path", str(REAL_SUBTLEX),
            "--out-dir", str(out_dir),
        ]
    )
    written = sorted(p.name for p in out_dir.iterdir())
    # 4 board files + 1 balance report.
    assert "balance_report.json" in written
    board_files = [n for n in written if n != "balance_report.json"]
    assert len(board_files) == 4
