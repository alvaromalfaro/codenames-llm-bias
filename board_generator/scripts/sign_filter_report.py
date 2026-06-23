#!/usr/bin/env python3
"""Pretty-print the CENTERED sign-criterion load filter, comparing δ values side by side.

Diagnostic, not a pipeline step - this only reports ρ_w; it never admits or prunes board words.
Inclusion is by the source criterion; the WEAT core is grandfathered.

One-time diagnostic - run manually; not part of the generator runtime, and not a committed rule
change. Like the load-filter and axis diagnostics, it needs Hugging Face: it loads the real primary
arbiter φ* and embeds words.

The raw core-quantile τ_load collapsed to a negative, non-discriminating threshold because the raw
poles overlap (see scripts/axis_diagnostics.py). Mean-centering fixes the sign/offset (ρ becomes
male > 0 / female < 0). This script previews the alternative admission rule: center the space, then
admit an expansion word iff it lands strictly on the correct side of the centered axis with an
a-priori margin δ (signed_load_centered > δ). It lays out δ ∈ {0.0, 0.01} in one JSON document so
the partitions and re-balance can be compared side by side. No final δ is frozen here.

ρ_w is a measurement, so it uses the single primary φ* alone (not the consensus average). We still
call load_consensus(DEFAULT_CONSENSUS) and then select the arbiter whose ref is
DEFAULT_CONSENSUS.primary. The same μ̄ (build_axis_diagnostics) backs both the reported centered
effect sizes and the sign-filter partitions.

The report is emitted with allow_nan=False - every statistic is kept finite (cosines; undefined
effect sizes are None; the embedded balance report sanitizes non-finite stats to null), so the
output is always valid JSON. JSON goes to stdout, any loader warnings (e.g. OOV words) to stderr.

Usage:
    uv run python scripts/sign_filter_report.py
    uv run python scripts/sign_filter_report.py --seed 1234567 --deltas 0.0 0.01
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
from board_generator.load_filter import build_sign_filter_report, read_attribute_words

# The δ margins previewed side by side by default (a-priori; not a frozen final choice).
DEFAULT_DELTAS: tuple[float, ...] = (0.0, 0.01)


def _primary_arbiter() -> Arbiter:
    """Load the consensus (the one HF entry point) and return the φ* primary arbiter."""
    arbiters = load_consensus(DEFAULT_CONSENSUS)
    for arbiter in arbiters:
        if arbiter.ref == DEFAULT_CONSENSUS.primary:
            return arbiter
    raise RuntimeError(
        "primary φ* not found in the loaded consensus (should be unreachable)")


def sign_filter_report(
    words_dir: Path,
    subtlex_path: Path,
    attributes_path: Path,
    seed: int,
    permutations: int,
    deltas: tuple[float, ...],
) -> str:
    """Build the side-by-side δ comparison document and return it as indented JSON.

    μ̄-norm and the centered effect sizes per specification are reused from build_axis_diagnostics
    (no recomputation); each δ case is a full build_sign_filter_report. The same φ* and the same μ̄
    construction back both, so the effect sizes and the partitions are comparable.
    """
    result = load_words(words_dir, subtlex_path)
    attributes = read_attribute_words(attributes_path)
    phi_star = _primary_arbiter()

    diagnostics = build_axis_diagnostics(
        result.words, attributes, phi_star, seed=seed, n_permutations=permutations
    )
    centered_effect_sizes = [
        {
            "specification": spec.specification,
            "effect_centered": dataclasses.asdict(spec.effect_centered),
        }
        for spec in diagnostics.specifications
    ]
    delta_cases = [
        dataclasses.asdict(
            build_sign_filter_report(
                result.words, attributes, phi_star, seed=seed, delta=delta)
        )
        for delta in deltas
    ]

    document = {
        "arbiter_primary": str(phi_star.ref),
        "seed": seed,
        "mu_bar_norm": diagnostics.mu_bar_norm,
        "n_embedded_items": diagnostics.n_embedded_items,
        "n_permutations": diagnostics.n_permutations,
        "centered_effect_sizes": centered_effect_sizes,
        "deltas": list(deltas),
        "delta_cases": delta_cases,
    }
    return json.dumps(document, indent=2, allow_nan=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pretty-print the centered sign-criterion load filter, comparing δ side by side"
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
        "--seed", type=int, default=1234567, help="re-balance / permutation seed (default: 1234567)"
    )
    parser.add_argument(
        "--permutations",
        type=int,
        default=10000,
        help="permutation-test label shuffles for the centered effect sizes (default: 10000)",
    )
    parser.add_argument(
        "--deltas",
        type=float,
        nargs="+",
        default=list(DEFAULT_DELTAS),
        help="a-priori sign-criterion margins δ to compare side by side (default: 0.0 0.01)",
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
        sign_filter_report(
            args.words,
            args.subtlex,
            args.attributes,
            args.seed,
            args.permutations,
            tuple(args.deltas),
        )
    )


if __name__ == "__main__":
    main()
