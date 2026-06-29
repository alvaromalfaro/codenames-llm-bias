"""Regression tests for the neutral-candidate writer.

These lock the invariants of scripts/build_neutral_candidates.py and its engine
(board_generator.neutral) so a future regeneration cannot silently reintroduce a fault. They are
OFFLINE and DETERMINISTIC: lexicon + WordNet only, no embeddings / no primary arbiter φ* / no
Hugging Face / no network / no RNG. They complement tests/test_neutral.py (which exercises the
in-memory builder) by also covering the writer-level / load_words round-trip / anti-finalization
properties.

Each test builds a tiny, fully controlled environment under tmp_path; the real 390-line duet.txt is
never touched. The deck deliberately mixes every category we care about: a denotational stoplist
token (mother), two words that genuinely live in the loaded career/science pools (executive, math),
clean neutrals (apple, river), an OOV-in-SUBTLEX neutral (kayak), a multi-token phrase (big bang)
and a proper noun (boston). Requires the WordNet corpus (``uv run python -m nltk.downloader
wordnet``).
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import warnings
from pathlib import Path

import pytest

from board_generator.lexicon import _check_playability, load_words
from board_generator.neutral import (
    EXCLUDED_BY_ALREADY_LOADED,
    EXCLUDED_BY_DENOTATIONAL,
    NeutralCandidatesResult,
    build_neutral_candidates,
    candidates_csv_text,
    loaded_pool_words,
    provenance_json_text,
)

# The writer script lives in scripts/ (no __init__), so load it from its path.
SCRIPT_PATH = Path(__file__).resolve(
).parents[1] / "scripts" / "build_neutral_candidates.py"

CANDIDATES_HEADER = "word,gender_category,word_kind,source,weat_set,specification"

# Deck tokens, mixed-case to also exercise normalization. One token per line.
#   mother    -> denotational stoplist hit (dropped)
#   executive -> already loaded (gender-career, dropped)
#   math      -> already loaded (gender-science, dropped)
#   apple     -> clean neutral (kept)
#   river     -> clean neutral (kept)
#   kayak     -> clean neutral, absent from SUBTLEX -> OOV covariate, still kept
#   big bang  -> multi-token phrase (dropped by playability)
#   boston    -> proper noun, dom_pos=Name in SUBTLEX -> invalid (dropped by playability)
DECK_LINES = [
    "MOTHER",
    "Executive",
    "math",
    "APPLE",
    "river",
    "Kayak",
    "big bang",
    "Boston",
]

EXPECTED_CANDIDATES = ["apple", "kayak", "river"]


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _make_env(tmp_path: Path, deck_lines: list[str] | None = None) -> dict[str, Path]:
    """Write a minimal, self-contained build environment under tmp_path."""
    deck = tmp_path / "duet.txt"
    _write(deck, "\n".join(DECK_LINES if deck_lines is None else deck_lines) + "\n")

    subtlex = tmp_path / "subtlex.csv"
    # kayak is intentionally omitted (OOV); boston is a Name so it is rejected as a common word.
    _write(
        subtlex,
        "word,zipf,dom_pos\n"
        "apple,4.5,Noun\n"
        "river,5.0,Noun\n"
        "mother,6.0,Noun\n"
        "executive,4.8,Noun\n"
        "math,4.2,Noun\n"
        "boston,4.0,Name\n",
    )

    stoplist = tmp_path / "stoplist.csv"
    _write(
        stoplist,
        "token,normalized,reason,reviewer,date\nMOTHER,mother,denotational_gender,,\n",
    )

    words_dir = tmp_path / "words"
    words_dir.mkdir()
    _write(
        words_dir / "gender_career.csv",
        "word,gender_category,word_kind,source,weat_set,specification\n"
        "EXECUTIVE,male,common,weat,weat-6,gender-career\n",
    )
    _write(
        words_dir / "gender_science.csv",
        "word,gender_category,word_kind,source,weat_set,specification\n"
        "MATH,male,common,weat,weat-7,gender-science\n",
    )

    return {"pool": deck, "subtlex": subtlex, "stoplist": stoplist, "words_dir": words_dir}


@pytest.fixture
def env(tmp_path: Path) -> dict[str, Path]:
    return _make_env(tmp_path)


def _build(env: dict[str, Path]) -> NeutralCandidatesResult:
    return build_neutral_candidates(
        env["pool"], env["subtlex"], env["stoplist"], env["words_dir"]
    )


def _candidate_texts(result: NeutralCandidatesResult) -> set[str]:
    return {word.text for word in result.candidates}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _exclusion(result: NeutralCandidatesResult, token: str) -> object:
    matches = [e for e in result.exclusions if e.token == token]
    assert len(matches) == 1, (token, result.exclusions)
    return matches[0]


# Denotational stoplist exclusion


def test_g1_stoplist_token_absent_from_candidates(env: dict[str, Path]) -> None:
    assert "mother" not in _candidate_texts(_build(env))


def test_g1_stoplist_exclusion_reason_and_detail_is_file_hash(env: dict[str, Path]) -> None:
    result = _build(env)
    mother = _exclusion(result, "mother")
    # type: ignore[attr-defined]
    assert mother.excluded_by == EXCLUDED_BY_DENOTATIONAL
    # detail is the sha256 of the stoplist file: recompute independently and compare.
    assert mother.detail == _sha256(
        env["stoplist"])  # type: ignore[attr-defined]


def test_g1_normalized_match_is_case_insensitive(tmp_path: Path) -> None:
    # stoplist normalized form is lowercase "apple"; the deck token is uppercase "APPLE".
    env = _make_env(tmp_path)
    _write(
        env["stoplist"],
        "token,normalized,reason,reviewer,date\nAPPLE,apple,denotational_gender,,\n",
    )
    result = _build(env)
    assert "apple" not in _candidate_texts(result)
    # type: ignore[attr-defined]
    assert _exclusion(result, "apple").excluded_by == EXCLUDED_BY_DENOTATIONAL


# Disjointness with the loaded pools


def test_g2_loaded_tokens_absent_with_spec_detail(env: dict[str, Path]) -> None:
    result = _build(env)
    texts = _candidate_texts(result)
    assert "executive" not in texts
    assert "math" not in texts
    # type: ignore[attr-defined]
    assert _exclusion(
        result, "executive").excluded_by == EXCLUDED_BY_ALREADY_LOADED
    # type: ignore[attr-defined]
    assert _exclusion(result, "executive").detail == "gender-career"
    # type: ignore[attr-defined]
    assert _exclusion(result, "math").excluded_by == EXCLUDED_BY_ALREADY_LOADED
    # type: ignore[attr-defined]
    assert _exclusion(result, "math").detail == "gender-science"


def test_g2_candidates_disjoint_from_loaded_pools(env: dict[str, Path]) -> None:
    result = _build(env)
    # career U science
    loaded = set(loaded_pool_words(env["words_dir"], env["subtlex"]))
    assert _candidate_texts(result).isdisjoint(loaded)


# Playability of everything emitted


def test_g3_every_emitted_token_is_playable(env: dict[str, Path]) -> None:
    result = _build(env)
    for word in result.candidates:
        status, _ambiguous, _reason = _check_playability(
            word.text, "common", word.dom_pos)
        assert status != "invalid", (word.text, status)


def test_g3_emitted_csv_reloads_without_playability_error(
    env: dict[str, Path], tmp_path: Path
) -> None:
    # A single unplayable token would make load_words raise for the WHOLE bank, not just the neutral
    # pool. Loading a dir holding ONLY the emitted candidates must not raise.
    out_dir = tmp_path / "only_candidates"
    out_dir.mkdir()
    _write(out_dir / "neutral_candidates.csv",
           candidates_csv_text(_build(env)))
    with warnings.catch_warnings():  # kayak is OOV -> warns; OOV is not under test here
        warnings.simplefilter("ignore")
        load_words(out_dir, env["subtlex"])  # must not raise


def test_g3_phrase_and_proper_noun_dropped(env: dict[str, Path]) -> None:
    texts = _candidate_texts(_build(env))
    assert "big bang" not in texts
    assert "bang" not in texts  # the phrase is not split into tokens either
    assert "boston" not in texts


# Output schema


def test_g4_header_is_exact(env: dict[str, Path]) -> None:
    assert candidates_csv_text(_build(env)).splitlines()[
        0] == CANDIDATES_HEADER


def test_g4_every_row_is_neutral_common_duet_blank(env: dict[str, Path]) -> None:
    rows = list(csv.DictReader(candidates_csv_text(_build(env)).splitlines()))
    assert [r["word"] for r in rows] == EXPECTED_CANDIDATES
    for row in rows:
        assert row["gender_category"] == "neutral"
        assert row["word_kind"] == "common"
        assert row["source"] == "duet"
        assert row["weat_set"] == ""
        assert row["specification"] == ""


def test_g4_roundtrip_through_load_words(env: dict[str, Path], tmp_path: Path) -> None:
    out_dir = tmp_path / "roundtrip"
    out_dir.mkdir()
    _write(out_dir / "neutral_candidates.csv",
           candidates_csv_text(_build(env)))
    with warnings.catch_warnings():  # kayak OOV
        warnings.simplefilter("ignore")
        loaded = load_words(out_dir, env["subtlex"])
    assert {w.text for w in loaded.words} == set(EXPECTED_CANDIDATES)
    for word in loaded.words:
        assert word.gender_category == "neutral"
        assert word.specification is None
        assert not word.weat_set


# Provenance, OOV semantics


def test_g5_rho_w_never_drives_exclusion(env: dict[str, Path]) -> None:
    payload = json.loads(provenance_json_text(_build(env)))
    assert payload["rho_w_used_for_exclusion"] is False
    # Explicit negative assert: load / ρ_w based exclusion NEVER happens.
    reasons = {e["excluded_by"] for e in payload["exclusions"]}
    assert "rho_w" not in reasons
    assert "load" not in reasons
    assert reasons <= {EXCLUDED_BY_DENOTATIONAL, EXCLUDED_BY_ALREADY_LOADED}


def test_g5_provenance_stoplist_hash_matches_file(env: dict[str, Path]) -> None:
    payload = json.loads(provenance_json_text(_build(env)))
    assert payload["stoplist"]["sha256"] == _sha256(env["stoplist"])


def test_g5_counts_reconcile_arithmetically(env: dict[str, Path]) -> None:
    result = _build(env)
    counts = result.counts
    stoplist_hits = sum(
        1 for e in result.exclusions if e.excluded_by == EXCLUDED_BY_DENOTATIONAL
    )
    already_loaded_hits = sum(
        1 for e in result.exclusions if e.excluded_by == EXCLUDED_BY_ALREADY_LOADED
    )
    emitted_rows = len(candidates_csv_text(
        result).splitlines()) - 1  # minus header

    assert counts["playable"] - stoplist_hits == counts["after_stoplist"]
    assert counts["after_stoplist"] - \
        already_loaded_hits == counts["after_already_loaded"]
    assert counts["after_already_loaded"] == counts["final_candidates"] == emitted_rows
    # The concrete fixture expectations (would have caught the 331-vs-332 drift by hand).
    assert counts == {
        "raw_lines": 8,
        "playable": 6,
        "after_stoplist": 5,
        "after_already_loaded": 3,
        "final_candidates": 3,
    }


def test_g5_oov_is_not_exclusion(env: dict[str, Path]) -> None:
    result = _build(env)
    by_text = {w.text: w for w in result.candidates}
    # kayak is absent from SUBTLEX: it STILL appears, with subtlex_freq resolving to None.
    assert "kayak" in by_text
    assert by_text["kayak"].covariates["subtlex_freq"] is None
    assert all(e.token != "kayak" for e in result.exclusions)


# Determinism


def test_g6_two_builds_are_byte_identical(env: dict[str, Path]) -> None:
    first, second = _build(env), _build(env)
    # Provenance carries no timestamp, so byte-determinism is total (no field excluded).
    assert candidates_csv_text(first) == candidates_csv_text(second)
    assert provenance_json_text(first) == provenance_json_text(second)


def test_g6_output_independent_of_deck_order(tmp_path: Path) -> None:
    env = _make_env(tmp_path)
    first = _build(env)
    # Rewrite the SAME deck path with reversed lines: paths are unchanged, so provenance (which
    # records the path) must also stay byte-identical, not just the CSV.
    _write(env["pool"], "\n".join(reversed(DECK_LINES)) + "\n")
    second = _build(env)
    assert candidates_csv_text(first) == candidates_csv_text(second)
    assert provenance_json_text(first) == provenance_json_text(second)


# Anti-finalization safeguard (do not automate the manual gate)


def _run_writer(env: dict[str, Path], out_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spec = importlib.util.spec_from_file_location(
        "_writer_under_test", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_neutral_candidates.py",
            "--pool", str(env["pool"]),
            "--subtlex", str(env["subtlex"]),
            "--stoplist", str(env["stoplist"]),
            "--words-dir", str(env["words_dir"]),
            "--out-dir", str(out_dir),
        ],
    )
    module.main()


def test_g7_writer_emits_candidates_never_neutral_csv(
    env: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out_dir = tmp_path / "out"
    _run_writer(env, out_dir, monkeypatch)
    assert (out_dir / "neutral_candidates.csv").exists()
    # finalization stays the human's manual gate
    assert not (out_dir / "neutral.csv").exists()


def test_g7_writer_does_not_touch_existing_neutral_csv(
    env: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    sentinel = (
        "word,gender_category,word_kind,source,weat_set,specification\n"
        "sentinel,neutral,common,human,,\n"
    )
    _write(out_dir / "neutral.csv", sentinel)
    _run_writer(env, out_dir, monkeypatch)
    assert (out_dir / "neutral.csv").read_text(encoding="utf-8") == sentinel
