#!/usr/bin/env python3
"""Three read-only diagnostics over the Duet neutral pool and the word-loader contract.

DIAGNOSTIC / INSPECTION only. It DECIDES nothing and WRITES nothing to disk: the three reports go to
stdout (a clean, parseable document) and the per-report summaries to stderr. It deletes nothing,
modifies no resource, and uses NO embeddings / NO primary arbiter φ* / NO Hugging Face / NO network.
Detection is purely lexical + WordNet, fully offline and deterministic.

REPORT 1 - Royalty/kinship recall probe (false-negative check on the enumerator).
  Loads the pool and applies the lexicon's normalization ONLY (lowercase/strip) - no playability
  filter - then tests an extended inline denotational probe list (canonical royalty/kinship/title
  terms) for raw presence in the deck. If a probe term is absent, the enumerator missed nothing in
  that category; if present, it is a recall gap to fold into the stoplist review.

REPORT 2 - Playability drop breakdown (pool-size / external-validity check).
  Re-runs the lexicon's normalization + playability over the pool and classifies every dropped token
  by the reason the lexicon used (multi_token_phrase | proper_noun (dom_pos=Name) | no_nominal_sense
  | other), with the full token list per reason. This quantifies what the playability step removes -
  e.g. whether the "nominal sense required" rule prunes game-legal non-nominal words (GREEN, COLD).
  It reuses lexicon._check_playability so the single-token verdicts are the loader's verdicts; the
  whitespace -> phrase rule is applied at SCRIPT level, mirroring the loader path used by
  neutral_denotational_report.py (the loader excludes a token as a phrase only when its row carries
  word_kind="phrase"; for a bare pool we mirror that with the same whitespace test).

REPORT 3 - load_words contract introspection (unblocks neutral.csv emission).
  Reports the exact schema the neutral pool must conform to, derived from the lexicon source and
  type hints (signature, Literals, constants, and the row[...] / row.get(...) dereferences in
  _read_word_rows). It does NOT call load_words on data.

Usage:
    uv run python scripts/neutral_pool_audit_report.py
    uv run python scripts/neutral_pool_audit_report.py --pool resources/neutral_pool/duet.txt
"""

from __future__ import annotations

import argparse
import inspect
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import get_args

# Reuse the real loader's normalization + playability rather than reimplementing them, so the drop
# verdicts match what the pipeline will actually admit. These are module-private helpers of the
# loader; we read them, we do not modify the loader.
from board_generator import lexicon
from board_generator.lexicon import (
    COVARIATE_KEYS,
    GenderCategory,
    Specification,
    WordKind,
    _check_playability,
    _load_frequency_table,
    _read_word_rows,
    load_words,
)

# Extended royalty/kinship/title probe. Inline-only on purpose: this is a RECALL audit of the
# enumerator, NOT a curated seed - it must not be added to GENDER_DENOTATIONAL_SEED.
ROYALTY_KINSHIP_PROBE: tuple[str, ...] = (
    "king", "queen", "prince", "princess", "knight", "nun", "monk", "lady", "gentleman",
    "widow", "widower", "niece", "nephew", "uncle", "aunt", "duke", "duchess", "earl",
    "lord", "dame", "maid", "bachelor", "spinster", "husband", "wife", "son", "daughter",
    "boy", "girl", "man", "woman", "mr", "mrs", "sir", "madam", "witch", "priest",
    "priestess", "hostess", "actor", "actress", "hero", "heroine", "god", "goddess",
    "czar", "emperor", "empress", "queenmother",
)


def _normalized_pool(pool_path: Path) -> list[str]:
    """Every non-blank pool line under the lexicon's normalization ONLY (lower + strip).

    No playability filter: REPORT 1 wants raw presence in the deck, including multi-token entries
    verbatim. Mirrors the loader's normalization step (lowercase, strip) without admission rules.
    """
    out: list[str] = []
    for raw in pool_path.read_text(encoding="utf-8").splitlines():
        normalized = raw.strip().lower()
        if normalized:
            out.append(normalized)
    return out


# --- REPORT 1 — Royalty/kinship recall probe ---
def report_recall_probe(pool_path: Path) -> None:
    deck = set(_normalized_pool(pool_path))
    present = [term for term in ROYALTY_KINSHIP_PROBE if term in deck]
    absent = [term for term in ROYALTY_KINSHIP_PROBE if term not in deck]

    print("=" * 99)
    print("REPORT 1 - Royalty/kinship recall probe (raw presence in the deck, normalization only)")
    print("=" * 99)
    print(f"pool: {pool_path}")
    print("normalization: lowercase + strip ONLY (no playability filter)")
    print(
        f"probe terms: {len(ROYALTY_KINSHIP_PROBE)}   deck tokens: {len(deck)}")
    print()
    print(f"PRESENT in deck (recall gaps to review): {len(present)}")
    if present:
        for term in present:
            print(f"  - {term}")
    else:
        print("  (none - the enumerator missed nothing in this category)")
    print()
    print(f"ABSENT from deck (expected): {len(absent)}")

    n_probe = len(ROYALTY_KINSHIP_PROBE)
    print(
        f"probe terms present in deck: {len(present)}/{n_probe}", file=sys.stderr)


# --- REPORT 2 - Playability drop breakdown ---
# Maps the loader's verbatim "invalid" reason string (from lexicon._check_playability) to a stable
# diagnostic category. An unmapped reason falls through to "other" carrying the precise reason.
_NAME_REASON = "common word labeled as a proper noun by SUBTLEX-US (dom_pos=Name)"
_NO_NOUN_REASON = "common word has no WordNet noun sense"
_REASON_CATEGORY = {
    _NAME_REASON: "proper_noun (dom_pos=Name)",
    _NO_NOUN_REASON: "no_nominal_sense",
}


def report_playability_drops(pool_path: Path, subtlex_path: Path) -> None:
    frequency = _load_frequency_table(subtlex_path)
    dropped: dict[str, list[str]] = defaultdict(list)
    n_total = 0
    n_playable = 0

    for normalized in _normalized_pool(pool_path):
        n_total += 1
        # Whitespace -> phrase, applied at SCRIPT level mirroring the loader (which excludes a row
        # only when word_kind="phrase"). Same test neutral_denotational_report.py uses.
        if " " in normalized or "\t" in normalized:
            dropped["multi_token_phrase"].append(normalized)
            continue
        freq_entry = frequency.get(normalized)
        dom_pos = freq_entry[1] if freq_entry is not None else None
        status, _ambiguous, reason = _check_playability(
            normalized, "common", dom_pos)
        if status == "playable":
            n_playable += 1
            continue
        # status is "invalid" for a single common token (the loader never returns "excluded" here).
        category = _REASON_CATEGORY.get(reason or "", f"other ({reason!r})")
        dropped[category].append(normalized)

    n_dropped = n_total - n_playable

    print("=" * 99)
    print("REPORT 2 - Playability drop breakdown (by reason)")
    print("=" * 99)
    print(f"pool: {pool_path}")
    print(f"subtlex: {subtlex_path}")
    print("MIRRORS the loader: single-token verdicts come from lexicon._check_playability (the")
    print("real function); the whitespace -> phrase rule is applied at SCRIPT level, matching the")
    print("loader path used by neutral_denotational_report.py. Playability logic is unchanged.")
    print()
    print(
        f"normalized tokens: {n_total}   playable: {n_playable}   dropped: {n_dropped}")
    print()
    for category in sorted(dropped):
        tokens = sorted(dropped[category])
        print(f"--- {category}: {len(tokens)} ---")
        for token in tokens:
            print(f"  - {token}")
        print()

    summary = ", ".join(f"{cat}={len(toks)}" for cat,
                        toks in sorted(dropped.items()))
    print(
        f"playable={n_playable} dropped={n_dropped} ({summary})", file=sys.stderr)


# --- REPORT 3 - load_words contract introspection ---
def _csv_columns_dereferenced() -> tuple[list[str], list[str]]:
    """(required, optional) CSV columns, read from the _read_word_rows source (no execution).

    A column read as row["X"] is required (a missing column would KeyError); a column read as
    row.get("X") is optional (tolerated when absent). Derived by scanning the source text so the
    report tracks the loader if it changes.
    """
    src = inspect.getsource(_read_word_rows)
    required = sorted(set(re.findall(r'row\["([^"]+)"\]', src)))
    optional = sorted(set(re.findall(r'row\.get\(\s*"([^"]+)"', src)))
    return required, optional


def report_load_words_contract() -> None:
    required, optional = _csv_columns_dereferenced()

    print("=" * 99)
    print("REPORT 3 - load_words contract introspection (read-only; not executed on data)")
    print("=" * 99)
    print(f"signature: load_words{inspect.signature(load_words)}")
    print(
        f"source module: {lexicon.__name__}  ({inspect.getsourcefile(load_words)})")
    print()

    print("(a) CSV columns the loader dereferences, and canonical header order")
    print("    Canonical header (observed in resources/words/*.csv and docstring):")
    print("      word, gender_category, word_kind, source, weat_set, specification")
    print(
        f"    required (read as row[\"X\"], KeyError if missing): {', '.join(required)}")
    print(
        f"    optional (read as row.get(\"X\"), tolerated if absent): {', '.join(optional)}")
    print()

    print("(b) How an empty `specification` is encoded")
    print("    _read_word_rows: spec=(row.get('specification') or '').strip(); then spec or None.")
    print("    => empty string in the CSV OR the column absent both map to None internally.")
    print("    Neutral words carry NO specification, so leave the cell empty (do not invent one).")
    print()

    print("(c) Is `source` enum-constrained? Would source='duet' be accepted?")
    print(
        "    source is read as (row['source'] or '').strip() - FREE-FORM text, no enum/Literal,")
    print("    merged across duplicate rows. source='duet' is accepted as-is; no registration.")
    print("    NUANCE: it is `specification`, NOT `source`, that is enum-constrained:")
    print(f"      Specification = Literal{get_args(Specification)}")
    print("      Putting 'duet' in the specification column WOULD raise (unknown specification).")
    print(
        f"    gender_category enum: {get_args(GenderCategory)}  (neutral pool uses 'neutral')")
    print(
        f"    word_kind enum:       {get_args(WordKind)}  (single tokens are 'common')")
    print()

    print("(d) Required vs optional columns; covariate dtypes")
    print(f"    required CSV columns: {', '.join(required)}")
    print(f"    optional CSV columns: {', '.join(optional)}")
    print("    blank-word rows are skipped, so an empty/header-only neutral.csv is tolerated.")
    print("    Covariates are NOT CSV columns - load_words COMPUTES them per word:")
    print(f"      covariate keys: {COVARIATE_KEYS}")
    print("      subtlex_freq:     float | None  (None when the word is OOV in SUBTLEX-US)")
    print("      length:           int           (len(text))")
    print("      wordnet_polysemy: int           (len(wordnet.synsets(text)))")
    print(
        "    All three are stored under Word.covariates: Mapping[str, float | None].")
    print()
    print("    => neutral.csv rows should look like:")
    print("       word,gender_category,word_kind,source,weat_set,specification")
    print("       apple,neutral,common,duet,,")

    print(
        f"contract: required={required} optional={optional} covariates={list(COVARIATE_KEYS)}",
        file=sys.stderr,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Three read-only diagnostics over the Duet neutral pool and load_words."
    )
    parser.add_argument(
        "--pool",
        type=Path,
        default=Path("resources/neutral_pool/duet.txt"),
        help="neutral-pool token file (default: resources/neutral_pool/duet.txt)",
    )
    parser.add_argument(
        "--subtlex",
        type=Path,
        default=Path("resources/frequencies/subtlex_us.csv"),
        help="SUBTLEX-US reference CSV (default: resources/frequencies/subtlex_us.csv)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.pool.exists():
        sys.exit(f"neutral pool not found: {args.pool}")
    if not args.subtlex.exists():
        sys.exit(f"SUBTLEX-US reference not found: {args.subtlex}")

    report_recall_probe(args.pool)
    print()
    report_playability_drops(args.pool, args.subtlex)
    print()
    report_load_words_contract()


if __name__ == "__main__":
    main()
