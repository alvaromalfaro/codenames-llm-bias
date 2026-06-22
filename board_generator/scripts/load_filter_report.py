#!/usr/bin/env python3
"""Pretty-print the gender-load filter report for the real word pool.

One-time diagnostic step - run manually; not part of the generator runtime. It loads the real
primary arbiter φ* and embeds words. It loads the annotated word pool (lexicon.load_words) and the
gender-attribute CSV (read_attribute_words), builds the load-filter report and dumps it as indented
JSON to stdout.

ρ_w is a measurement, so it uses the single primary φ* alone (not the consensus average). We still
call load_consensus(DEFAULT_CONSENSUS) - the one HF entry point - and then select the arbiter whose
ref is DEFAULT_CONSENSUS.primary.

This keeps inspection out of the runtime: it only reads the resources and reuses the public
load_filter API; it writes nothing. JSON goes to stdout, any loader warnings (e.g. OOV words) to
stderr, so the output is a clean, parseable document.

Decision: the report is emitted with allow_nan=False - the load-filter layer keeps every statistic
finite (cosines, and the embedded balance report sanitizes non-finite stats to null), so the output
is always valid JSON (no NaN/Infinity tokens).

Usage:
    uv run python scripts/load_filter_report.py
    uv run python scripts/load_filter_report.py --seed 1234567 --quantile 0.10
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

from board_generator.arbiter import DEFAULT_CONSENSUS, Arbiter, load_consensus
from board_generator.lexicon import load_words
from board_generator.load_filter import build_load_filter_report, read_attribute_words


def _primary_arbiter() -> Arbiter:
    """Load the consensus (the one HF entry point) and return the φ* primary arbiter."""
    arbiters = load_consensus(DEFAULT_CONSENSUS)
    for arbiter in arbiters:
        if arbiter.ref == DEFAULT_CONSENSUS.primary:
            return arbiter
    raise RuntimeError(
        "primary φ* not found in the loaded consensus (should be unreachable)")


def load_filter_report(
    words_dir: Path, subtlex_path: Path, attributes_path: Path, seed: int, quantile: float
) -> str:
    """Load the real pool + attributes, build the load-filter report, return it as indented JSON."""
    result = load_words(words_dir, subtlex_path)
    attributes = read_attribute_words(attributes_path)
    phi_star = _primary_arbiter()
    report = build_load_filter_report(
        result.words, attributes, phi_star, seed=seed, quantile=quantile
    )
    return json.dumps(dataclasses.asdict(report), indent=2, allow_nan=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pretty-print the gender-load filter report as JSON."
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
        "--attributes",
        type=Path,
        default=Path("resources/attribute_words/gender_attributes.csv"),
        help="gender-attribute CSV (default: resources/attribute_words/gender_attributes.csv)",
    )
    parser.add_argument(
        "--seed", type=int, default=1234567, help="re-balance seed (default: 1234567)"
    )
    parser.add_argument(
        "--quantile",
        type=float,
        default=0.10,
        help="per-spec core quantile anchoring τ_load (default: 0.10)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.words.is_dir():
        sys.exit(f"words directory not found: {args.words}")
    if not args.subtlex.exists():
        sys.exit(f"SUBTLEX-US reference not found: {args.subtlex}")
    if not args.attributes.exists():
        sys.exit(f"gender-attribute CSV not found: {args.attributes}")

    print(load_filter_report(args.words, args.subtlex,
          args.attributes, args.seed, args.quantile))


if __name__ == "__main__":
    main()
