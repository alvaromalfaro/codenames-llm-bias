"""Tests for the denotational-gender candidate detector (board_generator.neutral).

Offline and deterministic: WordNet only, no embeddings / no φ* / no network. A tiny fixture deck
mixes known denotational-gender tokens with clear neutrals and asserts the gendered ones are flagged
(with the expected reason) while the neutrals are not. Requires the WordNet corpus
(``uv run python -m nltk.downloader wordnet``).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from board_generator.neutral import (
    EXCLUDED_BY_ALREADY_LOADED,
    EXCLUDED_BY_DENOTATIONAL,
    NeutralCandidatesResult,
    build_neutral_candidates,
    candidates_csv_text,
    flag_token,
    provenance_json_text,
)

# Clear neutrals: ordinary board nouns with no denotational gender.
NEUTRALS = ["hospital", "river", "engine", "anchor", "bridge", "banana"]


def test_seed_token_flagged_by_seed() -> None:
    flags = flag_token("mother")
    assert flags.flagged
    assert "seed" in flags.reasons


def test_morphology_suffix_flagged() -> None:
    # -ess fires the morphology check regardless of whether it is also a seed word.
    assert "morphology" in flag_token("waitress").reasons


def test_wordnet_person_gloss_flagged() -> None:
    # "king" is not in woman/man hypernym closure (king -> monarch -> ruler -> person), so it relies
    # on the noun.person + gendered-gloss branch ("a male sovereign ...").
    flags = flag_token("king")
    assert "wordnet" in flags.reasons
    assert flags.wordnet_synset is not None


def test_known_gendered_tokens_all_flagged() -> None:
    for token in ("mother", "bride", "waitress", "king"):
        assert flag_token(token).flagged, token


def test_neutrals_not_flagged() -> None:
    for token in NEUTRALS:
        flags = flag_token(token)
        assert not flags.flagged, (token, flags.reasons)


def test_reasons_are_deterministically_ordered() -> None:
    # "waitress" trips seed, morphology and wordnet; order is always seed, morphology, wordnet.
    reasons = flag_token("waitress").reasons
    assert reasons == ("seed", "morphology", "wordnet")


# --- Neutral-pool candidate builder
# Offline + deterministic: lexicon + WordNet only, no embeddings / φ* / network / RNG. A small deck
# mixes a stoplist token (mother), an already-loaded token (executive, in the career CSV) and
# clean neutrals (apple, river); each is handled with the exact reason / row shape.


@pytest.fixture
def build_env(tmp_path: Path) -> dict[str, Path]:
    """Write a minimal, self-contained build environment under tmp_path."""
    deck = tmp_path / "duet.txt"
    deck.write_text("APPLE\nRIVER\nMOTHER\nEXECUTIVE\n", encoding="utf-8")

    subtlex = tmp_path / "subtlex.csv"
    subtlex.write_text(
        "word,zipf,dom_pos\n"
        "apple,4.5,Noun\n"
        "river,5.0,Noun\n"
        "mother,6.0,Noun\n"
        "executive,4.8,Noun\n",
        encoding="utf-8",
    )

    stoplist = tmp_path / "stoplist.csv"
    stoplist.write_text(
        "token,normalized,reason,reviewer,date\nMOTHER,mother,denotational_gender,,\n",
        encoding="utf-8",
    )

    words_dir = tmp_path / "words"
    words_dir.mkdir()
    (words_dir / "gender_career.csv").write_text(
        "word,gender_category,word_kind,source,weat_set,specification\n"
        "EXECUTIVE,male,common,weat,weat-6,gender-career\n",
        encoding="utf-8",
    )

    return {
        "pool": deck,
        "subtlex": subtlex,
        "stoplist": stoplist,
        "words_dir": words_dir,
    }


def _build(env: dict[str, Path]) -> NeutralCandidatesResult:
    return build_neutral_candidates(
        env["pool"], env["subtlex"], env["stoplist"], env["words_dir"]
    )


def test_stoplist_token_excluded_denotational(build_env: dict[str, Path]) -> None:
    result = _build(build_env)
    mother = [e for e in result.exclusions if e.token == "mother"]
    assert len(mother) == 1
    assert mother[0].excluded_by == EXCLUDED_BY_DENOTATIONAL
    assert mother[0].detail == result.stoplist_sha256


def test_loaded_token_excluded_already_loaded(build_env: dict[str, Path]) -> None:
    result = _build(build_env)
    executive = [e for e in result.exclusions if e.token == "executive"]
    assert len(executive) == 1
    assert executive[0].excluded_by == EXCLUDED_BY_ALREADY_LOADED
    assert executive[0].detail == "gender-career"


def test_clean_neutrals_survive_with_exact_row_shape(build_env: dict[str, Path]) -> None:
    result = _build(build_env)
    by_text = {word.text: word for word in result.candidates}
    assert set(by_text) == {"apple", "river"}
    for word in result.candidates:
        assert word.gender_category == "neutral"
        assert word.word_kind == "common"
        assert word.source == "duet"
        assert word.specification is None
        assert word.weat_set == ()  # legal "no WEAT set" encoding (empty tuple, not None)
        assert set(word.covariates) == {
            "subtlex_freq", "length", "wordnet_polysemy"}


def test_emitted_csv_row_shape(build_env: dict[str, Path]) -> None:
    csv_text = candidates_csv_text(_build(build_env))
    lines = csv_text.splitlines()
    assert lines[0] == "word,gender_category,word_kind,source,weat_set,specification"
    assert lines[1:] == ["apple,neutral,common,duet,,",
                         "river,neutral,common,duet,,"]


def test_candidates_disjoint_from_loaded_pools(build_env: dict[str, Path]) -> None:
    result = _build(build_env)
    loaded = {"executive"}  # career U science in this fixture
    candidate_texts = {word.text for word in result.candidates}
    assert candidate_texts.isdisjoint(loaded)


def test_build_is_deterministic_byte_for_byte(build_env: dict[str, Path]) -> None:
    first, second = _build(build_env), _build(build_env)
    assert candidates_csv_text(first) == candidates_csv_text(second)
    assert provenance_json_text(first) == provenance_json_text(second)


def test_provenance_records_hash_and_no_load_reason(build_env: dict[str, Path]) -> None:
    result = _build(build_env)
    payload = json.loads(provenance_json_text(result))
    assert payload["stoplist"]["sha256"] == result.stoplist_sha256
    assert len(result.stoplist_sha256) == 64
    # ρ_w / load NEVER drives exclusion: only the two legal reasons appear.
    assert set(payload["exclusion_reasons"]) <= {
        EXCLUDED_BY_DENOTATIONAL,
        EXCLUDED_BY_ALREADY_LOADED,
    }
    assert payload["rho_w_used_for_exclusion"] is False
    assert all(
        e["excluded_by"] in {EXCLUDED_BY_DENOTATIONAL,
                             EXCLUDED_BY_ALREADY_LOADED}
        for e in payload["exclusions"]
    )


def test_counts_reconcile(build_env: dict[str, Path]) -> None:
    counts = _build(build_env).counts
    assert counts["playable"] == 4
    assert counts["after_stoplist"] == 3  # mother dropped
    assert counts["after_already_loaded"] == 2  # executive dropped
    assert counts["final_candidates"] == 2
