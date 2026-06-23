#!/usr/bin/env python3
"""Pretty-print the read-only axis diagnostics for the gender-load filter.

One-time diagnostic - run manually; NOT part of the generator runtime. Like the load-filter report,
this is one of the few steps that needs Hugging Face: it loads the real primary arbiter φ* and
embeds words. It loads the annotated word pool (lexicon.load_words) and the gender-attribute CSV
(read_attribute_words), builds the axis diagnostics and dumps them as indented JSON to stdout.

Use this to decide whether the weak/negative τ_load is anisotropy (a large shared offset that
mean-centering recovers from) or a genuine lack of gender signal, before changing the filter. It
only reads resources and reuses the public APIs; it writes nothing.

ρ_w is a measurement, so it uses the single primary φ* alone (not the consensus average). We still
call load_consensus(DEFAULT_CONSENSUS) - the one HF entry point - and then select the arbiter whose
ref is DEFAULT_CONSENSUS.primary.

The report is emitted with allow_nan=False - every statistic is kept finite (cosines, and undefined
effect sizes are None), so the output is always valid JSON (no NaN/Infinity tokens). JSON goes to
stdout, any loader warnings (e.g. OOV words) to stderr.

Usage:
    uv run python scripts/axis_diagnostics.py
    uv run python scripts/axis_diagnostics.py --seed 1234567 --permutations 10000
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

from board_generator.arbiter import DEFAULT_CONSENSUS, Arbiter, load_consensus
from board_generator.axis_diagnostics import build_axis_diagnostics
from board_generator.lexicon import load_words
from board_generator.load_filter import read_attribute_words


def _primary_arbiter() -> Arbiter:
    """Load the consensus (the one HF entry point) and return the φ* primary arbiter."""
    arbiters = load_consensus(DEFAULT_CONSENSUS)
    for arbiter in arbiters:
        if arbiter.ref == DEFAULT_CONSENSUS.primary:
            return arbiter
    raise RuntimeError(
        "primary φ* not found in the loaded consensus (should be unreachable)")


def axis_diagnostics_report(
    words_dir: Path, subtlex_path: Path, attributes_path: Path, seed: int, permutations: int
) -> str:
    """Load the real pool + attributes, build the axis diagnostics, return them as indented JSON."""
    result = load_words(words_dir, subtlex_path)
    attributes = read_attribute_words(attributes_path)
    phi_star = _primary_arbiter()
    diagnostics = build_axis_diagnostics(
        result.words, attributes, phi_star, seed=seed, n_permutations=permutations
    )
    return json.dumps(dataclasses.asdict(diagnostics), indent=2, allow_nan=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pretty-print the read-only axis diagnostics as JSON."
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
        "--seed", type=int, default=1234567, help="permutation RNG seed (default: 1234567)"
    )
    parser.add_argument(
        "--permutations",
        type=int,
        default=10000,
        help="permutation-test label shuffles (default: 10000)",
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

    print(
        axis_diagnostics_report(
            args.words, args.subtlex, args.attributes, args.seed, args.permutations
        )
    )


if __name__ == "__main__":
    main()
