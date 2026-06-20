"""Tests for the word loader, covariate annotation and playability checks.

Positive cases run against the real resources/words/ + SUBTLEX-US table; negative and edge cases use 
small, controlled temp CSVs so behaviour does not depend on the 74k-row corpus. Requires the WordNet 
corpus (``uv run python -m nltk.downloader wordnet``).
"""

from __future__ import annotations

import csv
import warnings
from pathlib import Path

import pytest
from nltk.corpus import wordnet

from board_generator.lexicon import LoadResult, Word, load_words

RESOURCES = Path(__file__).resolve().parents[1] / "resources"
REAL_WORDS = RESOURCES / "words"
REAL_SUBTLEX = RESOURCES / "frequencies" / "subtlex_us.csv"

WORD_HEADER = ["word", "gender_category", "word_kind", "source", "weat_set", "specification"]
FREQ_HEADER = ["word", "zipf", "dom_pos"]

# Arts words present under both WEAT-7 and WEAT-8 in gender_science.csv (collapse on load).
ARTS_DUPLICATED = {"poetry", "art", "dance",
                   "literature", "novel", "symphony", "drama"}


def _write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def _make_dirs(tmp_path: Path, word_rows: list[list[str]], freq_rows: list[list[str]]
               ) -> tuple[Path, Path]:
    """Write a one-file words dir and a frequency table; return (words_dir, subtlex_path)."""
    words_dir = tmp_path / "words"
    words_dir.mkdir()
    _write_csv(words_dir / "words.csv", WORD_HEADER, word_rows)
    subtlex_path = tmp_path / "subtlex.csv"
    _write_csv(subtlex_path, FREQ_HEADER, freq_rows)
    return words_dir, subtlex_path


@pytest.fixture(scope="module")
def real_load() -> LoadResult:
    with warnings.catch_warnings():  # proper nouns are expected OOV; not under test here
        warnings.simplefilter("ignore")
        return load_words(REAL_WORDS, REAL_SUBTLEX)


def _by_text(words: list[Word]) -> dict[str, Word]:
    return {w.text: w for w in words}


def test_loads_real_career_and_science(real_load: LoadResult) -> None:
    # Pool grows as expansions land; derive the count from the per-specification split rather than
    # hardcoding a magic total, and assert only that both axes are populated.
    by_spec: dict[str | None, int] = {}
    for w in real_load.words:
        by_spec[w.specification] = by_spec.get(w.specification, 0) + 1
    assert by_spec["gender-career"] > 0
    assert by_spec["gender-science"] > 0
    assert len(real_load.words) == sum(by_spec.values())
    assert real_load.excluded == []  # no phrases in these two cores
    assert all(w.text == w.text.lower() for w in real_load.words)

    by_text = _by_text(real_load.words)
    # The 7 duplicated arts words collapse to one Word each with merged weat_set {weat-7, weat-8};
    # both rows carry the same gender-science specification, so they merge cleanly.
    for arts in ARTS_DUPLICATED:
        assert by_text[arts].weat_set == ("weat-7", "weat-8"), arts
        assert by_text[arts].specification == "gender-science", arts
    # Words appearing in only one set keep a singleton provenance.
    assert by_text["sculpture"].weat_set == ("weat-7",)
    assert by_text["shakespeare"].weat_set == ("weat-8",)
    # A She Figures expansion word carries the gender-science specification but NO weat_set.
    engineering = by_text["engineering"]
    assert engineering.specification == "gender-science"
    assert engineering.weat_set == ()
    # Career words carry the gender-career specification.
    assert by_text["salary"].specification == "gender-career"


def test_dedup_conflict_on_gender_raises(tmp_path: Path) -> None:
    words_dir, subtlex = _make_dirs(
        tmp_path,
        [
            ["NURSE", "female", "common", "weat", "weat-6"],
            ["NURSE", "male", "common", "other", "weat-6"],
        ],
        [["nurse", "4.6526", "Noun"]],
    )
    with pytest.raises(ValueError, match="conflicting gender_category"):
        load_words(words_dir, subtlex)


def test_unknown_word_kind_raises(tmp_path: Path) -> None:
    words_dir, subtlex = _make_dirs(
        tmp_path,
        [["NURSE", "female", "wibble", "weat", "weat-6", "gender-career"]],
        [["nurse", "4.6526", "Noun"]],
    )
    with pytest.raises(ValueError, match="unknown word_kind"):
        load_words(words_dir, subtlex)


def test_she_figures_row_loads_with_specification_and_no_weat_set(tmp_path: Path) -> None:
    # A non-WEAT expansion row: a specification but an EMPTY weat_set (She Figures style).
    words_dir, subtlex = _make_dirs(
        tmp_path,
        [["ENGINEERING", "male", "common", "she-figures", "", "gender-science"]],
        [["engineering", "4.5", "Noun"]],
    )
    word = _by_text(load_words(words_dir, subtlex).words)["engineering"]
    assert word.specification == "gender-science"
    assert word.weat_set == ()


def test_empty_specification_loads_as_none(tmp_path: Path) -> None:
    # Neutral words carry no intrinsic axis: an empty specification maps to None (no error).
    words_dir, subtlex = _make_dirs(
        tmp_path,
        [["RIVER", "neutral", "common", "duet", "", ""]],
        [["river", "4.8", "Noun"]],
    )
    word = _by_text(load_words(words_dir, subtlex).words)["river"]
    assert word.specification is None


def test_unknown_specification_raises(tmp_path: Path) -> None:
    words_dir, subtlex = _make_dirs(
        tmp_path,
        [["NURSE", "female", "common", "weat", "weat-6", "gender-wibble"]],
        [["nurse", "4.6526", "Noun"]],
    )
    with pytest.raises(ValueError, match="unknown specification"):
        load_words(words_dir, subtlex)


def test_dedup_conflict_on_specification_raises(tmp_path: Path) -> None:
    # Same word, two different specifications -> a hard conflict (only source/weat_set may differ).
    words_dir, subtlex = _make_dirs(
        tmp_path,
        [
            ["NURSE", "female", "common", "weat", "weat-6", "gender-career"],
            ["NURSE", "female", "common", "other", "", "gender-science"],
        ],
        [["nurse", "4.6526", "Noun"]],
    )
    with pytest.raises(ValueError, match="conflicting specification"):
        load_words(words_dir, subtlex)


def test_common_noun_is_playable(tmp_path: Path) -> None:
    words_dir, subtlex = _make_dirs(
        tmp_path,
        [["NURSE", "female", "common", "weat", "weat-6"]],
        [["nurse", "4.6526", "Noun"]],
    )
    result = load_words(words_dir, subtlex)
    nurse = _by_text(result.words)["nurse"]
    assert nurse.ambiguous_pos is False
    assert nurse.dom_pos == "Noun"


def test_noun_with_nonnoun_dompos_is_ambiguous(tmp_path: Path) -> None:
    # "dance" has a WordNet noun sense but SUBTLEX dom_pos "Verb" -> playable AND ambiguous.
    words_dir, subtlex = _make_dirs(
        tmp_path,
        [["DANCE", "female", "common", "weat", "weat-7"]],
        [["dance", "5.1698", "Verb"]],
    )
    dance = _by_text(load_words(words_dir, subtlex).words)["dance"]
    assert dance.ambiguous_pos is True
    assert dance.dom_pos == "Verb"


def test_common_word_with_name_dompos_is_rejected(tmp_path: Path) -> None:
    words_dir, subtlex = _make_dirs(
        tmp_path,
        [["EINSTEIN", "male", "common", "weat", "weat-8"]],  # mislabeled as common
        [["einstein", "3.7232", "Name"]],
    )
    with pytest.raises(ValueError, match="dom_pos=Name"):
        load_words(words_dir, subtlex)


def test_phrase_is_excluded_not_an_error(tmp_path: Path) -> None:
    words_dir, subtlex = _make_dirs(
        tmp_path,
        [["ICE CREAM", "neutral", "phrase", "duet", ""]],
        [["other", "4.0", "Noun"]],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the phrase is OOV; irrelevant here
        result = load_words(words_dir, subtlex)
    assert result.words == []
    assert len(result.excluded) == 1
    assert result.excluded[0].text == "ice cream"


def test_proper_noun_passes_without_wordnet_noun_check(tmp_path: Path) -> None:
    # "einstein" has no common-noun sense; as a proper noun only the minimal token check applies.
    words_dir, subtlex = _make_dirs(
        tmp_path,
        [["EINSTEIN", "male", "proper", "weat", "weat-8"]],
        [["einstein", "3.7232", "Name"]],
    )
    einstein = _by_text(load_words(words_dir, subtlex).words)["einstein"]
    assert einstein.word_kind == "proper"
    # dom_pos not cross-checked for proper nouns
    assert einstein.ambiguous_pos is False


def test_strict_mode_reports_all_invalids_in_one_error(tmp_path: Path) -> None:
    words_dir, subtlex = _make_dirs(
        tmp_path,
        [
            ["EINSTEIN", "male", "common", "weat",
                "weat-8"],  # rejected: dom_pos=Name
            # rejected: no WordNet noun sense
            ["ZZQXWORD", "neutral", "common", "made-up", ""],
        ],
        [["einstein", "3.7232", "Name"]],  # zzqxword is OOV (dom_pos None)
    )
    with pytest.raises(ValueError) as excinfo:
        load_words(words_dir, subtlex)
    message = str(excinfo.value)
    assert "2 unplayable" in message
    assert "einstein" in message
    assert "zzqxword" in message


def test_oov_word_has_none_freq_and_warns(tmp_path: Path) -> None:
    # Controlled table that deliberately omits "nurse" so it is OOV.
    words_dir, subtlex = _make_dirs(
        tmp_path,
        [["NURSE", "female", "common", "weat", "weat-6"]],
        [["other", "4.0", "Noun"]],
    )
    with pytest.warns(UserWarning, match="nurse"):
        result = load_words(words_dir, subtlex)
    assert result.oov == ["nurse"]
    nurse = _by_text(result.words)["nurse"]
    assert nurse.covariates["subtlex_freq"] is None
    assert nurse.dom_pos is None
    # OOV must not be flagged ambiguous merely for dom_pos=None
    assert nurse.ambiguous_pos is False


def test_covariate_length_and_polysemy(real_load: LoadResult) -> None:
    family = _by_text(real_load.words)["family"]
    assert family.covariates["length"] == len("family") == 6
    # Polysemy is the WordNet SENSE count (not synonyms); compare against a live computation.
    assert family.covariates["wordnet_polysemy"] == len(
        wordnet.synsets("family"))
    assert family.covariates["wordnet_polysemy"] > 1  # genuinely polysemous
    # "family" is in SUBTLEX-US
    assert isinstance(family.covariates["subtlex_freq"], float)
