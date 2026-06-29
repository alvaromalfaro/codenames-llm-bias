#!/usr/bin/env python3
"""Enumerate denotational-gender candidates in the Duet neutral pool, for human review.

DIAGNOSTIC / SUGGESTION step, not a pipeline step. It is read-only: it SUGGESTS tokens that may
denote gender, so a human can curate neutral.csv. It does NOT decide inclusion, does NOT write
neutral.csv or any stoplist, deletes nothing, and uses no embeddings / no primary arbiter φ* / no 
Hugging Face / no network. Detection is purely lexical + WordNet, fully offline and deterministic.

Pipeline:
  1. Load resources/neutral_pool/duet.txt. Normalize (lowercase/strip) and apply the real
     playability filter by reusing board_generator.lexicon - multi-token tokens are dropped as
     phrases; proper nouns the corpus reports as Name and common tokens with no WordNet noun sense
     are dropped - so the enumeration reflects the true playable candidate set.
  2. Flag each surviving token via board_generator.neutral.flag_token (seed / morphology / wordnet).
  3. Emit the report as CSV (token, normalized, reasons, wordnet_synset, reviewer_decision) sorted
     alphabetically by normalized, with reviewer_decision left EMPTY for the human.

Like the other report scripts this WRITES NOTHING: the CSV goes to stdout and the summary to stderr,
so stdout is a clean, parseable document and the reviewer redirects it where they want, e.g.:

    uv run python scripts/neutral_denotational_report.py > neutral_denotational_candidates.csv

Usage:
    uv run python scripts/neutral_denotational_report.py
    uv run python scripts/neutral_denotational_report.py --pool resources/neutral_pool/duet.txt
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

# Reuse the real loader's normalization + playability rather than reimplementing them, so the
# candidate set matches what the pipeline will actually admit. These are module-private helpers of
# the loader; we read them, we do not modify the loader.
from board_generator.lexicon import _check_playability, _load_frequency_table
from board_generator.neutral import flag_token

CSV_HEADER = ("token", "normalized", "reasons",
              "wordnet_synset", "reviewer_decision")


@dataclass(frozen=True, slots=True)
class Candidate:
    """One playable, flagged token row of the report."""

    token: str  # original line from the pool, verbatim
    normalized: str
    reasons: tuple[str, ...]
    wordnet_synset: str | None


def _playable_tokens(pool_path: Path, subtlex_path: Path) -> list[tuple[str, str]]:
    """Return (original, normalized) for every playable token in the pool, in file order.

    Normalization and playability come straight from the loader: a whitespace-bearing token is a
    multi-token phrase (dropped); for single tokens we run _check_playability as a "common" word
    with the corpus dom_pos, so a "Name" dom_pos or a missing WordNet noun sense drops it exactly as
    the real load would.
    """
    frequency = _load_frequency_table(subtlex_path)
    playable: list[tuple[str, str]] = []
    for raw in pool_path.read_text(encoding="utf-8").splitlines():
        original = raw.strip()
        if not original:
            continue
        normalized = original.lower()
        if " " in normalized or "\t" in normalized:
            continue  # multi-token phrase: excluded by design, like a word_kind="phrase" row
        freq_entry = frequency.get(normalized)
        dom_pos = freq_entry[1] if freq_entry is not None else None
        status, _ambiguous, _reason = _check_playability(
            normalized, "common", dom_pos)
        if status == "playable":
            playable.append((original, normalized))
    return playable


def build_candidates(pool_path: Path, subtlex_path: Path) -> list[Candidate]:
    """Playable tokens that any check flagged, sorted deterministically by normalized token."""
    candidates = [
        Candidate(
            token=original,
            normalized=normalized,
            reasons=flags.reasons,
            wordnet_synset=flags.wordnet_synset,
        )
        for original, normalized in _playable_tokens(pool_path, subtlex_path)
        if (flags := flag_token(normalized)).flagged
    ]
    return sorted(candidates, key=lambda c: c.normalized)


def write_report(candidates: list[Candidate], handle: object) -> None:
    """Write the candidate rows as CSV to an open text handle."""
    writer = csv.writer(handle)  # type: ignore[arg-type]
    writer.writerow(CSV_HEADER)
    for candidate in candidates:
        writer.writerow(
            (
                candidate.token,
                candidate.normalized,
                ",".join(candidate.reasons),
                candidate.wordnet_synset or "",
                "",  # reviewer_decision: left EMPTY for the human
            )
        )


def _print_summary(n_playable: int, candidates: list[Candidate]) -> None:
    by_check: Counter[str] = Counter()
    for candidate in candidates:
        by_check.update(candidate.reasons)
    print(f"playable tokens:       {n_playable}", file=sys.stderr)
    print(f"flagged candidates:    {len(candidates)}", file=sys.stderr)
    print("by check (overlapping):", file=sys.stderr)
    for check in ("seed", "morphology", "wordnet"):
        print(f"  {check:11} {by_check.get(check, 0)}", file=sys.stderr)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Suggest denotational-gender candidates in the Duet neutral pool (read-only)."
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

    n_playable = len(_playable_tokens(args.pool, args.subtlex))
    candidates = build_candidates(args.pool, args.subtlex)
    write_report(candidates, sys.stdout)
    _print_summary(n_playable, candidates)


if __name__ == "__main__":
    main()
