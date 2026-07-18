#!/usr/bin/env python3
"""Emit the versioned measurement-frame sidecar to data/boards/measurement_frame.json.

One-time, standalone emitter (run manually); not part of the generator runtime and not wired into
the `board-generator` CLI. Itt extends the generator's output contract (a new sidecar) without 
changing generation logic.

Like the load-filter / sign-filter diagnostics it needs Hugging Face: it loads the real primary
arbiter φ* and embeds words. It also reads φ*'s frozen recipe (pooling + normalize) from the pinned
checkpoint snapshot already in the local HF cache - it never downloads and never hard-codes the
recipe. The gender axis e_gen and centering mean μ̄  are rebuilt with the same generator functions
generation uses (build_gender_axis / build_mu_bar), then serialized losslessly (float64 big-endian
hex) and content-hashed into `frame_id`.

Re-running with the same φ* and reference set writes the same `frame_id` (the wall-clock `created_at` 
and the diagnostic floats sit outside the hashed content subtree).

Usage:
    uv run python scripts/emit_frame.py
    uv run python scripts/emit_frame.py --out-dir ../data/boards
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import board_generator
from board_generator.arbiter import DEFAULT_CONSENSUS, Arbiter, load_consensus
from board_generator.board import DEFAULT_OUTPUT_DIR
from board_generator.emit_frame import build_frame, read_encoder_recipe, write_frame
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


def emit_frame(
    words_dir: Path, subtlex_path: Path, attributes_path: Path, out_dir: Path
) -> Path:
    """Load the real pool + attributes, read φ*'s recipe, build the frame, and write the sidecar."""
    result = load_words(words_dir, subtlex_path)
    attributes = read_attribute_words(attributes_path)
    phi_star = _primary_arbiter()

    ref = DEFAULT_CONSENSUS.primary
    encoder = read_encoder_recipe(ref.model_id, ref.hf_revision)

    frame = build_frame(
        phi_star,
        result.words,
        attributes,
        attribute_source=str(attributes_path),
        encoder=encoder,
        generator_version=board_generator.__version__,
        created_at=datetime.now(UTC).isoformat(),
    )
    return write_frame(frame, out_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Emit the measurement-frame sidecar (data/boards/measurement_frame.json)."
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
        "--out-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"output directory (default: {DEFAULT_OUTPUT_DIR})",
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

    path = emit_frame(args.words, args.subtlex, args.attributes, args.out_dir)
    # Human-readable confirmation to stderr; the artifact is the written file.
    frame_id = json.loads(path.read_text(encoding="utf-8"))["frame_id"]
    print(f"wrote {path}\nframe_id {frame_id}", file=sys.stderr)


if __name__ == "__main__":
    main()
