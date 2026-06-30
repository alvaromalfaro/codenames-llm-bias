#!/usr/bin/env python3
"""Emit neutral-pool candidates from the Duet deck.

Unlike the read-only report scripts (neutral_denotational_report.py, neutral_pool_audit_report.py,
load_filter_report.py, ...), this one writes files by design: it produces

    resources/neutral_pool/neutral_candidates.csv
    resources/neutral_pool/neutral_candidates.provenance.json

It is deterministic and offline - no embeddings, no primary arbiter φ*, no Hugging Face, no network.
The candidates CSV is the input to the manual review gate; it is not neutral.csv. Finalization into
resources/words/neutral.csv stays the human's job, which is also why the outputs land under
resources/neutral_pool/ - outside lexicon.load_words' glob over resources/words/, so they are never
auto-loaded as final boards. Nothing under backend/ or data/boards/ is touched.

Usage:
    uv run python scripts/build_neutral_candidates.py
    uv run python scripts/build_neutral_candidates.py --out-dir resources/neutral_pool
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from board_generator.neutral import (
    build_neutral_candidates,
    candidates_csv_text,
    provenance_json_text,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Emit neutral-pool candidates + provenance from the Duet deck (writes files)."
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
    parser.add_argument(
        "--stoplist",
        type=Path,
        default=Path("resources/curation/gender_denotational_stoplist.csv"),
        help="denotational-gender stoplist (default: resources/neutral_pool/"
        "gender_denotational_stoplist.csv)",
    )
    parser.add_argument(
        "--words-dir",
        type=Path,
        default=Path("resources/words"),
        help="loaded word CSVs, career U science (default: resources/words)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("resources/neutral_pool"),
        help="where to write the candidates + provenance (default: resources/neutral_pool)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    for label, path in (
        ("neutral pool", args.pool),
        ("SUBTLEX-US reference", args.subtlex),
        ("denotational stoplist", args.stoplist),
    ):
        if not path.exists():
            sys.exit(f"{label} not found: {path}")
    if not args.words_dir.is_dir():
        sys.exit(f"words directory not found: {args.words_dir}")

    result = build_neutral_candidates(
        args.pool, args.subtlex, args.stoplist, args.words_dir)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "neutral_candidates.csv"
    provenance_path = args.out_dir / "neutral_candidates.provenance.json"
    csv_path.write_text(candidates_csv_text(result), encoding="utf-8")
    provenance_path.write_text(provenance_json_text(result), encoding="utf-8")

    counts = result.counts
    print(f"wrote {csv_path}", file=sys.stderr)
    print(f"wrote {provenance_path}", file=sys.stderr)
    print(
        "counts: "
        + " ".join(f"{key}={value}" for key, value in counts.items())
        + f"  (stoplist sha256={result.stoplist_sha256[:12]}…)",
        file=sys.stderr,
    )
    print(
        "NOTE: these are CANDIDATES for manual review, not the final neutral.csv.",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
