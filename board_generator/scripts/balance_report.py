#!/usr/bin/env python3
"""Pretty-print the balance report for the real word pool.

One-time diagnostic step - run manually; NOT part of the generator runtime. Loads the annotated
word pool with lexicon.load_words and runs balancing.run_balancing over it, then dumps the resulting 
BalanceReport as indented JSON to stdout. Use it to eyeball balance / equivalence verdicts and 
matching counts without booting the full generator.

This keeps inspection out of the runtime: it only reads the resources and reuses the public balancing 
API; it writes nothing. JSON goes to stdout, any loader warnings (e.g. OOV words) to stderr, so the 
output is a clean, parseable document.

Decisions:
  * the report is emitted with allow_nan=False - the balancing layer sanitizes every non-finite
    statistic to null, so the output is always valid JSON (no NaN/Infinity tokens).
  * the matched Word subsets are intentionally NOT printed (they are not JSON; only the report is).

Usage:
    uv run python scripts/balance_report.py
    uv run python scripts/balance_report.py --seed 1234567 --criterion mann_whitney_cohen
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

from board_generator.balancing import BalanceCriterion, run_balancing
from board_generator.lexicon import load_words


def balance_report(
    words_dir: Path, subtlex_path: Path, seed: int, criterion: BalanceCriterion
) -> str:
    """Load the real pool, run balancing, and return the report as indented JSON."""
    result = load_words(words_dir, subtlex_path)
    balanced = run_balancing(result.words, seed=seed, criterion=criterion)
    payload = dataclasses.asdict(balanced.report)
    return json.dumps(payload, indent=2, allow_nan=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pretty-print the balance report for the real word pool as JSON."
    )
    parser.add_argument(
        "--words",
        type=Path,
        default=Path("resources/words"),
        help="word CSV directory (default: resources/words)",
    )
    parser.add_argument(
        "--subtlex",
        type=Path,
        default=Path("resources/frequencies/subtlex_us.csv"),
        help="SUBTLEX-US reference CSV (default: resources/frequencies/subtlex_us.csv)",
    )
    parser.add_argument(
        "--seed", type=int, default=1234567, help="balancing seed (default: 1234567)"
    )
    parser.add_argument(
        "--criterion",
        choices=("tost", "mann_whitney_cohen"),
        default="tost",
        help="governing equivalence criterion (default: tost)",
    )
    args = parser.parse_args()
    if not args.words.is_dir():
        sys.exit(f"words directory not found: {args.words}")
    if not args.subtlex.exists():
        sys.exit(f"SUBTLEX-US reference not found: {args.subtlex}")

    print(balance_report(args.words, args.subtlex, args.seed, args.criterion))


if __name__ == "__main__":
    main()
