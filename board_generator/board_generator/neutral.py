"""Recall-oriented enumeration of denotational-gender candidates among neutral-pool tokens.

This is a suggestion aid for the human curator of neutral.csv. It never decides inclusion, never
writes neutral.csv or a stoplist, and uses no embeddings / no primary arbiter φ* / no Hugging Face.
Detection is purely lexical + WordNet, fully offline and deterministic.

Three independent, recall-oriented checks flag a token as a denotational-gender candidate. Each
records which check fired so the reviewer can trust high-precision hits (seed) and scrutinise the
noisier ones (morphology, wordnet):

  - SEED        token is in GENDER_DENOTATIONAL_SEED, a curated gender-word set.
  - MORPHOLOGY  token carries a gendered suffix/affix (-ess, -man, -woman, ...). Expect false
                positives ("human", "dress") - the human prunes them.
  - WORDNET     some synset of the token has woman.n.01 or man.n.01 in its hypernym closure, or is a
                noun.person whose gloss carries a gendered cue. Recall backstop.

The hypernym-closure check alone has poor recall (e.g. "king" -> "monarch" -> "ruler" -> "person",
never "man"), which is exactly why the gloss-cue branch exists.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from nltk.corpus import wordnet

from board_generator.lexicon import (
    COVARIATE_KEYS,
    Word,
    _check_playability,
    _load_frequency_table,
    load_words,
)

# SEED - curated canonical denotational-gender words. Provenance: standard gendered noun pairs used
# in gender-bias word lists plus common kinship/title/role pairs. Curated inline by hand; high
# precision. Listed as female/male pairs in comments for readability; membership test is
# order-insensitive.
GENDER_DENOTATIONAL_SEED: frozenset[str] = frozenset(
    {
        # kinship
        "mother", "father", "mom", "dad", "mommy", "daddy", "mum",
        "sister", "brother", "daughter", "son",
        "wife", "husband", "aunt", "uncle", "niece", "nephew",
        "grandmother", "grandfather", "grandma", "grandpa", "granny",
        "widow", "widower", "bride", "groom", "bridegroom",
        "girlfriend", "boyfriend", "fiancee", "fiance",
        # generic person / pronoun-like
        "woman", "man", "girl", "boy", "lady", "gentleman",
        "female", "male", "she", "he", "her", "him", "hers",
        "madam", "sir", "miss", "mister", "mistress", "master",
        "lass", "lad", "maiden", "spinster", "bachelor",
        # titles / royalty / religious
        "queen", "king", "princess", "prince", "duchess", "duke",
        "empress", "emperor", "countess", "count", "baroness", "baron",
        "nun", "monk", "priestess", "priest", "goddess", "god",
        "witch", "wizard", "heroine", "hero",
        # occupational gender pairs
        "actress", "actor", "waitress", "waiter", "hostess", "host",
        "stewardess", "steward", "seamstress", "tailor",
        "policewoman", "policeman", "saleswoman", "salesman",
        "businesswoman", "businessman",
    }
)

# MORPHOLOGY - gendered suffixes/affixes. Recall aid; a plain endswith check, so false positives are
# expected (e.g. "human"/"woman" end in -man, "dress"/"address" end in -ess). The human prunes.
GENDERED_SUFFIXES: tuple[str, ...] = (
    "ess",
    "woman",
    "man",
    "girl",
    "boy",
    "princess",
    "prince",
    "queen",
    "king",
    "maid",
    "wife",
)

# Gendered cue lemmas for the WordNet gloss branch. Matched whole-word against a noun.person gloss.
# Recall-oriented; the bare pronouns add recall at some precision cost, which is why wordnet-only
# hits are the ones the reviewer is told to scrutinise.
_GENDER_GLOSS_CUES: frozenset[str] = frozenset(
    {
        "male", "female", "man", "woman", "men", "women",
        "boy", "girl", "boys", "girls",
        "he", "she", "his", "her", "him", "hers",
        "son", "sons", "daughter", "daughters",
        "husband", "wife", "brother", "sister",
        "father", "mother", "king", "queen",
        "lady", "ladies", "gentleman", "widow", "bride", "groom",
        "masculine", "feminine", "maternal", "paternal", "nun", "monk",
    }
)

_GLOSS_CUE_RE = re.compile(
    r"\b(?:" + "|".join(sorted(_GENDER_GLOSS_CUES)) + r")\b")

# Sense anchors for the hypernym-closure branch.
_WOMAN = "woman.n.01"
_MAN = "man.n.01"


@dataclass(frozen=True, slots=True)
class DenotationalFlags:
    """Which denotational-gender checks fired for a token, plus the WordNet sense that triggered it.

    reasons is a deterministically-ordered subset of ("seed", "morphology", "wordnet"); empty means
    the token is not flagged. wordnet_synset is the name of the first synset that fired the WordNet
    check (for audit), or None.
    """

    reasons: tuple[str, ...]
    wordnet_synset: str | None

    @property
    def flagged(self) -> bool:
        return bool(self.reasons)


def _seed_hit(normalized: str) -> bool:
    """Membership in the curated canonical gender-word set."""
    return normalized in GENDER_DENOTATIONAL_SEED


def _morphology_hit(normalized: str) -> bool:
    """Token carries a gendered suffix and is longer than it (avoids bare-suffix noise)."""
    return any(
        normalized.endswith(suffix) and len(normalized) > len(suffix)
        for suffix in GENDERED_SUFFIXES
    )


def _gloss_has_gender_cue(gloss: str) -> bool:
    return _GLOSS_CUE_RE.search(gloss.lower()) is not None


def _wordnet_hit(normalized: str) -> str | None:
    """Returns the name of the first synset that fires the WordNet check, else None.

    A synset fires if woman.n.01 or man.n.01 is in its hypernym closure, or it is a noun.person
    whose gloss carries a gendered cue. Iterates every synset (any part of speech) in deterministic
    WordNet order, so the reported sense is stable for a given token.
    """
    woman = wordnet.synset(_WOMAN)
    man = wordnet.synset(_MAN)
    for synset in wordnet.synsets(normalized):
        closure = set(synset.closure(lambda s: s.hypernyms()))
        if woman in closure or man in closure:
            return synset.name()
        if synset.lexname() == "noun.person" and _gloss_has_gender_cue(synset.definition()):
            return synset.name()
    return None


def flag_token(normalized: str) -> DenotationalFlags:
    """Run all three recall-oriented checks on a normalized token and aggregate the result.

    The caller is responsible for normalization + playability (reuse board_generator.lexicon); this
    operates on an already-normalized, already-playable token.
    """
    reasons: list[str] = []
    if _seed_hit(normalized):
        reasons.append("seed")
    if _morphology_hit(normalized):
        reasons.append("morphology")
    wordnet_synset = _wordnet_hit(normalized)
    if wordnet_synset is not None:
        reasons.append("wordnet")
    return DenotationalFlags(reasons=tuple(reasons), wordnet_synset=wordnet_synset)


# ---------------------------------------------------------------------------
# Neutral-pool candidate builder.
#
# Deterministic and offline: no embeddings, no primary arbiter φ*, no Hugging Face, no network. It
# emits CANDIDATES for the human review gate; it never finalizes neutral.csv. Tokens are excluded
# ONLY because they denote gender (the curated denotational stoplist) or are already loaded by a
# gender specification (career U science). The gender load ρ_w plays no role in exclusion.
# ---------------------------------------------------------------------------

# Exact header of the emitted candidates CSV - matches lexicon.load_words' schema verbatim so the
# human-reviewed neutral.csv produced from it loads without change.
CANDIDATES_CSV_HEADER = (
    "word", "gender_category", "word_kind", "source", "weat_set", "specification"
)

# The only legal exclusion reasons. Never ρ_w / load.
EXCLUDED_BY_DENOTATIONAL = "denotational_gender"
EXCLUDED_BY_ALREADY_LOADED = "already_loaded"


@dataclass(frozen=True, slots=True)
class PlayableToken:
    """A pool token that survived lexicon normalization + playability, with its covariate inputs."""

    normalized: str
    dom_pos: str | None
    ambiguous_pos: bool
    subtlex_freq: float | None


@dataclass(frozen=True, slots=True)
class StoplistData:
    """The denotational-gender stoplist: its normalized tokens plus the file's content hash."""

    normalized: frozenset[str]
    sha256: str  # sha256 of the raw file bytes, recorded in provenance
    path: str


@dataclass(frozen=True, slots=True)
class Exclusion:
    """One dropped token: why it was dropped and the supporting detail (stoplist hash or spec)."""

    token: str
    excluded_by: str  # EXCLUDED_BY_DENOTATIONAL | EXCLUDED_BY_ALREADY_LOADED
    # stoplist sha256 (denotational) or the specification it collided with (loaded)
    detail: str


@dataclass(frozen=True, slots=True)
class NeutralCandidatesResult:
    """Outcome of the candidate build: surviving Words, the audit trail and reconcilable counts."""

    candidates: tuple[Word, ...]
    exclusions: tuple[Exclusion, ...]
    counts: Mapping[str, int]
    pool_path: str
    subtlex_path: str
    stoplist_path: str
    stoplist_sha256: str


def enumerate_playable_tokens(pool_path: Path, subtlex_path: Path) -> list[PlayableToken]:
    """Normalize the pool and keep board-playable tokens, reusing the real loader's rules.

    Normalization is strip + lower; a whitespace-bearing token is a multi-token phrase (dropped,
    like a word_kind="phrase" row); single tokens run through lexicon._check_playability as "common"
    words with their SUBTLEX dom_pos, so a "Name" dom_pos or a missing WordNet noun sense drops them
    exactly as load_words would. Deterministic file order; duplicate normalized forms keep the 1st.
    """
    frequency = _load_frequency_table(subtlex_path)
    playable: list[PlayableToken] = []
    seen: set[str] = set()
    for raw in pool_path.read_text(encoding="utf-8").splitlines():
        normalized = raw.strip().lower()
        if not normalized or normalized in seen:
            continue
        if " " in normalized or "\t" in normalized:
            continue  # multi-token phrase: excluded by design
        freq_entry = frequency.get(normalized)
        subtlex_freq, dom_pos = freq_entry if freq_entry is not None else (
            None, None)
        status, ambiguous_pos, _reason = _check_playability(
            normalized, "common", dom_pos)
        if status == "playable":
            seen.add(normalized)
            playable.append(PlayableToken(
                normalized, dom_pos, ambiguous_pos, subtlex_freq))
    return playable


def load_denotational_stoplist(path: Path) -> StoplistData:
    """Load the curated denotational-gender stoplist.

    Matches on the normalized column, re-normalized (strip + lower) to the same surface form the
    lexicon uses. The exclusion reason is always denotational_gender; the file's content hash
    (sha256 of the raw bytes) is recorded so the provenance pins which curation drove exclusions.
    """
    data = path.read_bytes()
    sha256 = hashlib.sha256(data).hexdigest()
    normalized: set[str] = set()
    for row in csv.DictReader(data.decode("utf-8").splitlines()):
        token = (row.get("normalized") or "").strip().lower()
        if token:
            normalized.add(token)
    return StoplistData(normalized=frozenset(normalized), sha256=sha256, path=str(path))


def loaded_pool_words(words_dir: Path, subtlex_path: Path) -> dict[str, str]:
    """Map every already-loaded word (career U science) to its specification, via the real loader.

    Reuses lexicon.load_words over resources/words/ so the "already loaded" set is exactly what the
    platform will load. Returns {normalized_text -> specification}; a word with no specification
    maps to "unspecified".
    """
    result = load_words(words_dir, subtlex_path)
    loaded: dict[str, str] = {}
    for word in result.words:
        loaded.setdefault(word.text, word.specification or "unspecified")
    return loaded


def build_neutral_candidates(
    pool_path: Path,
    subtlex_path: Path,
    stoplist_path: Path,
    words_dir: Path,
) -> NeutralCandidatesResult:
    """Build neutral-pool candidates from the Duet deck.

    Pipeline: normalize + playability -> drop denotational-gender tokens (stoplist) -> drop
    already-loaded tokens (career U science) -> build a neutral Word per survivor. ρ_w is never
    consulted; the only exclusion reasons are denotational_gender and already_loaded.
    """
    playable = enumerate_playable_tokens(pool_path, subtlex_path)
    stoplist = load_denotational_stoplist(stoplist_path)
    loaded = loaded_pool_words(words_dir, subtlex_path)

    raw_lines = sum(
        1 for line in pool_path.read_text(encoding="utf-8").splitlines() if line.strip()
    )

    exclusions: list[Exclusion] = []
    survivors: list[PlayableToken] = []
    after_stoplist = 0
    for token in playable:
        if token.normalized in stoplist.normalized:
            exclusions.append(
                Exclusion(token.normalized,
                          EXCLUDED_BY_DENOTATIONAL, stoplist.sha256)
            )
            continue
        after_stoplist += 1
        if token.normalized in loaded:
            exclusions.append(
                Exclusion(token.normalized, EXCLUDED_BY_ALREADY_LOADED,
                          loaded[token.normalized])
            )
            continue
        survivors.append(token)

    candidates = sorted(
        (
            Word(
                text=token.normalized,
                gender_category="neutral",
                word_kind="common",
                source="duet",
                weat_set=(),  # legal "no WEAT set" encoding (empty tuple, not None)
                dom_pos=token.dom_pos,
                ambiguous_pos=token.ambiguous_pos,
                covariates={
                    "subtlex_freq": token.subtlex_freq,
                    "length": len(token.normalized),
                    "wordnet_polysemy": len(wordnet.synsets(token.normalized)),
                },
                specification=None,
            )
            for token in survivors
        ),
        key=lambda word: word.text,
    )
    exclusions.sort(key=lambda exclusion: (
        exclusion.token, exclusion.excluded_by))

    counts = {
        "raw_lines": raw_lines,
        "playable": len(playable),
        "after_stoplist": after_stoplist,
        "after_already_loaded": len(survivors),
        "final_candidates": len(candidates),
    }
    return NeutralCandidatesResult(
        candidates=tuple(candidates),
        exclusions=tuple(exclusions),
        counts=counts,
        pool_path=str(pool_path),
        subtlex_path=str(subtlex_path),
        stoplist_path=stoplist.path,
        stoplist_sha256=stoplist.sha256,
    )


def candidates_csv_text(result: NeutralCandidatesResult) -> str:
    """Serialize candidates to the load_words CSV schema, sorted by word. Deterministic (LF lines).

    Rows look like `apple,neutral,common,duet,,` — blank weat_set (empty tuple) and blank
    specification (None). This is the automatic artifact for review.
    """
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(CANDIDATES_CSV_HEADER)
    for word in result.candidates:
        writer.writerow(
            (
                word.text,
                word.gender_category,
                word.word_kind,
                word.source,
                ";".join(word.weat_set),  # empty tuple -> ""
                word.specification or "",  # None -> ""
            )
        )
    return buffer.getvalue()


def provenance_json_text(result: NeutralCandidatesResult) -> str:
    """Serialize the audit trail as deterministic JSON (sorted keys; LF-terminated).

    Documents that ρ_w / load played no role in exclusion: the only excluded_by reasons are
    denotational_gender and already_loaded, and the flag is asserted explicitly.
    """
    payload = {
        "deck_source": result.pool_path,
        "subtlex_source": result.subtlex_path,
        "stoplist": {"path": result.stoplist_path, "sha256": result.stoplist_sha256},
        "lexicon_markers": {
            "covariate_keys": list(COVARIATE_KEYS),
            "playability": "board_generator.lexicon._check_playability",
            "normalization": "strip + lower; whitespace -> phrase (dropped)",
            "uses_embeddings_or_phi_star_or_network": False,
        },
        "counts": dict(result.counts),
        "exclusions": [
            {"token": e.token, "excluded_by": e.excluded_by, "detail": e.detail}
            for e in result.exclusions
        ],
        "exclusion_reasons": sorted({e.excluded_by for e in result.exclusions}),
        "rho_w_used_for_exclusion": False,
    }
    return json.dumps(payload, sort_keys=True, indent=2) + "\n"
