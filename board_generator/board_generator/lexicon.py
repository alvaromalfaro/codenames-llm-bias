"""Word-list loading, covariate annotation and playability validation.

Loads every word CSV under resources/words/ (the WEAT cores plus the gender-disparity expansion 
share one schema), deduplicates rows into unique board Words, and annotates each with the three 
confound-control covariates (subtlex_freq, length, wordnet_polysemy) plus its gender_category, 
source and provenance weat_set.

Playability validation keeps multi-token, mislabeled or non-board entries out of the board-eligible 
pool before anything is embedded or placed on a grid. Validation is strict: every offending word 
across the whole load is collected and reported in a single error.
"""

from __future__ import annotations

import csv
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, get_args

from nltk.corpus import wordnet

# Gender specifications a board can belong to.
Specification = Literal["gender-career", "gender-science"]

# Gender labels. Binary axis is a documented simplification; gender only.
GenderCategory = Literal["male", "female", "neutral"]

# Lexical kind of a source row. phrase is multi-token and never a board card.
WordKind = Literal["common", "proper", "phrase"]

# The exact covariate keys. board.WordEntry.covariates relies on these (no more, no less).
COVARIATE_KEYS = ("subtlex_freq", "length", "wordnet_polysemy")

_VALID_GENDER: frozenset[str] = frozenset(get_args(GenderCategory))
_VALID_KIND: frozenset[str] = frozenset(get_args(WordKind))
_VALID_SPEC: frozenset[str] = frozenset(get_args(Specification))


@dataclass(frozen=True, slots=True)
class Word:
    """A board-eligible word with its gender label, provenance and covariates."""

    text: str  # lowercased
    gender_category: GenderCategory
    word_kind: WordKind
    source: str  # extraction source(s), for traceability
    # provenance WEAT set(s); merged & sorted across duplicate rows
    weat_set: tuple[str, ...]
    dom_pos: (
        str | None
        # dominant POS from SUBTLEX-US, verbatim ("Noun"/"Verb"/"Name"); None if OOV
    )
    ambiguous_pos: bool  # common word playable as a noun but whose corpus dom_pos disagrees

    # Pre-imputation covariates: subtlex_freq is None for OOV words (proper nouns are expected OOV).
    # board.WordEntry.covariates is dict[str, float] because imputation (balancing.py) runs and
    # fills the Nones before board assembly; this float | None vs float difference is deliberate,
    # not a mismatch to "fix".
    covariates: Mapping[str, float | None]

    # Axis-routing field: which gender specification this word belongs to, or None for neutral words
    # (which carry no intrinsic axis). Orthogonal to weat_set provenance: weat-7 and weat-8 both map
    # to gender-science, and non-WEAT sources carry a specification with no weat_set. Has a default
    # so positional Word construction works.
    specification: Specification | None = None


@dataclass(frozen=True, slots=True)
class LoadResult:
    """Outcome of :func:load_words: board-eligible words plus auditable diagnostics."""

    words: list[Word]  # playable, board-eligible words
    excluded: list[Word]  # phrases (multi-token) - excluded by design
    # words absent from the SUBTLEX-US table (subtlex_freq is None)
    oov: list[str]


def load_words(words_dir: Path, subtlex_path: Path) -> LoadResult:
    """Load, dedup, annotate and validate every board word under words_dir.

    Reads every *.csv in words_dir (schema word,gender_category,word_kind,source,weat_set); 
    empty/header-only files such as a pending neutral.csv are tolerated. Words are lowercased and 
    joined to the SUBTLEX-US table at subtlex_path by lowercased word.

    Hard errors (raised immediately): an unknown gender_category/word_kind/specification value, or 
    two rows for the same word disagreeing on gender_category, word_kind, or specification. Strict 
    playability: every invalid word is collected and reported together in a single error. OOV words 
    are not an error - they are collected and surfaced in one warning. Imputation of OOV 
    subtlex_freq is balancing.py's job, not this loader's.
    """
    frequency = _load_frequency_table(subtlex_path)
    merged = _merge_rows(_read_word_rows(words_dir))

    words: list[Word] = []
    excluded: list[Word] = []
    oov: list[str] = []
    invalid: list[tuple[str, str]] = []

    for text in sorted(merged):  # deterministic order
        gender_category, word_kind, source, weat_set, specification = merged[text]
        freq_entry = frequency.get(text)
        if freq_entry is None:
            subtlex_freq, dom_pos = None, None
            oov.append(text)
        else:
            subtlex_freq, dom_pos = freq_entry

        status, ambiguous_pos, reason = _check_playability(
            text, word_kind, dom_pos)
        if status == "invalid":
            assert reason is not None
            invalid.append((text, reason))
            continue

        covariates: dict[str, float | None] = {
            "subtlex_freq": subtlex_freq,
            "length": len(text),
            # WordNet senses (polysemy)
            "wordnet_polysemy": len(wordnet.synsets(text)),
        }
        word = Word(
            text=text,
            gender_category=gender_category,
            word_kind=word_kind,
            source=source,
            weat_set=weat_set,
            dom_pos=dom_pos,
            ambiguous_pos=ambiguous_pos,
            covariates=covariates,
            specification=specification,
        )
        (excluded if status == "excluded" else words).append(word)

    if invalid:
        listing = "\n".join(
            f"  - {text}: {reason}" for text, reason in invalid)
        raise ValueError(
            f"{len(invalid)} unplayable board word(s):\n{listing}")

    if oov:
        warnings.warn(
            f"{len(oov)} word(s) absent from SUBTLEX-US (subtlex_freq=None; expected for proper "
            f"nouns): {', '.join(oov)}",
            stacklevel=2,
        )

    return LoadResult(words=words, excluded=excluded, oov=oov)


def _check_playability(text: str, word_kind: WordKind, dom_pos: str | None
                       ) -> tuple[Literal["playable", "excluded", "invalid"], bool, str | None]:
    """Classify a word for board use. Returns (status, ambiguous_pos, reason).

    reason is set only for "invalid". dom_pos is None for OOV words: the advisory checks (ambiguous_pos / 
    Name reject) apply only when dom_pos is known, so an OOV common word is never rejected, nor 
    flagged ambiguous, merely for being OOV.
    """
    if word_kind == "phrase":
        # Multi-token breaks the single-word embedding unit - excluded by design, not an error.
        return ("excluded", False, None)

    if word_kind == "proper":
        # Irregular WordNet coverage -> skip the noun check; do not cross-check dom_pos ("Name" is
        # expected/consistent for a proper noun). Minimal check only.
        if text.isalpha():
            return ("playable", False, None)
        return ("invalid", False, "proper noun is not a single alphabetic token")

    # word_kind == "common": primary check is a WordNet NOUN sense; dom_pos is advisory.
    if dom_pos == "Name":
        # A "common" word the corpus sees as a proper noun -> mislabeled (hard reject).
        return ("invalid", False, "common word labeled as a proper noun by SUBTLEX-US (dom_pos=Name)")
    if wordnet.synsets(text, pos=wordnet.NOUN):
        # Playable; flag the advisory mismatch only when the corpus reports a non-noun dom_pos.
        ambiguous_pos = dom_pos is not None and dom_pos != "Noun"
        return ("playable", ambiguous_pos, None)
    return ("invalid", False, "common word has no WordNet noun sense")


def _load_frequency_table(subtlex_path: Path) -> dict[str, tuple[float, str]]:
    """Load SUBTLEX-US (word,zipf,dom_pos) into a lookup keyed by lowercased word."""
    table: dict[str, tuple[float, str]] = {}
    with subtlex_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            word = (row["word"] or "").strip().lower()
            if not word:
                continue
            table[word] = (float(row["zipf"]), (row["dom_pos"] or "").strip())
    return table


def _read_word_rows(words_dir: Path) -> list[tuple[str, GenderCategory, WordKind, str, str]]:
    """Read and enum-validate every row across the word CSVs.

    Returns (word, gender_category, word_kind, source, weat_set, specification) tuples. Unknown enum 
    values (gender-category/word-kind/specification) are a hard error - a typo must never silently 
    relax validation. Blank words are skipped, so an empty/header-only neutral.csv is tolerated.
    """
    rows: list[tuple[str, GenderCategory, WordKind, str, str]] = []
    for csv_path in sorted(words_dir.glob("*.csv")):
        with csv_path.open(newline="", encoding="utf-8") as handle:
            for line_no, row in enumerate(
                csv.DictReader(handle), start=2
            ):  # +1 header, +1 to 1-based
                word = (row["word"] or "").strip().lower()
                if not word:
                    continue
                gender = (row["gender_category"] or "").strip()
                kind = (row["word_kind"] or "").strip()
                spec = (row.get("specification") or "").strip()
                if gender not in _VALID_GENDER:
                    raise ValueError(
                        f"{csv_path.name}:{line_no}: unknown gender_category {gender!r} ({word!r})"
                    )
                if kind not in _VALID_KIND:
                    raise ValueError(
                        f"{csv_path.name}:{line_no}: unknown word_kind {kind!r} for {word!r}"
                    )
                if spec and spec not in _VALID_SPEC:
                    raise ValueError(
                        f"{csv_path.name}:{line_no}: unknown specification {spec!r} for {word!r}"
                    )
                # Validated against the Literals above; the casts only narrow the type for mypy.
                # An empty specification (neutral words) maps to None.
                gender_cat: GenderCategory = gender  # type: ignore[assignment]
                word_kind: WordKind = kind  # type: ignore[assignment]
                # type: ignore[assignment]
                specification: Specification | None = spec or None
                rows.append(
                    (
                        word,
                        gender_cat,
                        word_kind,
                        (row["source"] or "").strip(),
                        (row["weat_set"] or "").strip(),
                        specification
                    )
                )
    return rows


def _merge_rows(rows: list[tuple[str, GenderCategory, WordKind, str, str, Specification | None]],
                ) -> dict[str, tuple[GenderCategory, WordKind, str, tuple[str, ...], Specification | None]]:
    """Dedup rows to unique words, merging source/weat_set provenance.

    Raises if two rows for the same word disagree on gender_category, word_kind, or specification 
    (the same word must agree on all three; only source and weat_set provenance may differ). source 
    and weat_set are merged deterministically (distinct values, sorted). The arts words shared by
    weat-7 and weat-8 carry the same gender-science specification, so they merge cleanly to one
    word.
    """
    acc: dict[str, tuple[GenderCategory, WordKind,
                         Specification | None, set[str], set[str]]] = {}
    for word, gender, kind, source, weat, spec in rows:
        if word not in acc:
            acc[word] = (gender, kind, spec, set(), set())
        prev_gender, prev_kind, prev_spec, sources, weat_sets = acc[word]
        if gender != prev_gender:
            raise ValueError(
                f"conflicting gender_category for {word!r}: {prev_gender!r} vs {gender!r}"
            )
        if kind != prev_kind:
            raise ValueError(
                f"conflicting word_kind for {word!r}: {prev_kind!r} vs {kind!r}")
        if spec != prev_spec:
            raise ValueError(
                f"conflicting specification for {word!r}: {prev_spec!r} vs {spec!r}")
        if source:
            sources.add(source)
        if weat:
            weat_sets.add(weat)

    return {
        word: (gender, kind, "; ".join(
            sorted(sources)), tuple(sorted(weat_sets)), spec)
        for word, (gender, kind, spec, sources, weat_sets) in acc.items()
    }
