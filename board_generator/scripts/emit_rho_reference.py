#!/usr/bin/env python3
"""Emit the frozen per-word ρ reference fixture for one reference board.

One-time, standalone emitter (run manually). It reads the already re-emitted measurement-frame 
sidecar (`data/boards/measurement_frame.json`), and for the chosen reference board computes rho_raw /
rho_cent / rho_weat per card through the real primary φ* (raw path only, never _CenteringEncoder /
axis_diagnostics). The fixture is keyed by the sidecar's frame_id and written under
`board_generator/tests/fixtures/`.

The axis and μ̄  rebuilt here from the raw φ* are asserted equal (bit-for-bit) to the vectors 
serialized in the sidecar, so the fixture is computed on the frame's own geometry and its single μ̄.

Usage:
    uv run python scripts/emit_rho_reference.py
    uv run python scripts/emit_rho_reference.py --board <path-to-a-probe-board>.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from board_generator.arbiter import DEFAULT_CONSENSUS, Arbiter, load_consensus
from board_generator.board import DEFAULT_OUTPUT_DIR
from board_generator.emit_frame import FRAME_FILENAME, decode_f64_be_hex
from board_generator.lexicon import load_words
from board_generator.load_filter import build_gender_axis, build_mu_bar, read_attribute_words
from board_generator.rho_reference import (
    build_rho_reference,
    read_board_card_texts,
    write_fixture,
)

_DEFAULT_BOARD = DEFAULT_OUTPUT_DIR / "gender_probe-gender-career-000.json"


def _primary_arbiter() -> Arbiter:
    """Load the consensus (the one HF entry point) and return the φ* primary arbiter."""
    arbiters = load_consensus(DEFAULT_CONSENSUS)
    for arbiter in arbiters:
        if arbiter.ref == DEFAULT_CONSENSUS.primary:
            return arbiter
    raise RuntimeError(
        "primary φ* not found in the loaded consensus (should be unreachable)")


def emit_rho_reference(
    words_dir: Path,
    subtlex_path: Path,
    attributes_path: Path,
    board_path: Path,
    sidecar_path: Path,
) -> Path:
    """Build and write the ρ reference fixture, verifying provenance against the sidecar."""
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    frame_id = sidecar["frame_id"]
    encoder = sidecar["encoder"]
    sidecar_axis = decode_f64_be_hex(
        sidecar["gender_axis"]["vector_f64_be_hex"])
    sidecar_mu_bar = decode_f64_be_hex(sidecar["mu_bar"]["vector_f64_be_hex"])

    words = load_words(words_dir, subtlex_path).words
    attributes = read_attribute_words(attributes_path)
    phi_star = _primary_arbiter()

    # Provenance: the fixture must be computed on the frame's own geometry and its SINGLE μ̄.
    rebuilt_axis = build_gender_axis(attributes, phi_star)
    reference_texts = [a.word for a in attributes] + [
        w.text for w in words if w.specification is not None
    ]
    rebuilt_mu_bar = build_mu_bar(reference_texts, phi_star)
    if phi_star.ref != DEFAULT_CONSENSUS.primary:
        sys.exit("provenance: φ* is not the pinned primary encoder")
    if not np.array_equal(rebuilt_axis, sidecar_axis):
        sys.exit(
            "provenance: rebuilt gender axis does not match the sidecar (wrong φ*/geometry)")
    if not np.array_equal(rebuilt_mu_bar, sidecar_mu_bar):
        sys.exit(
            "provenance: rebuilt μ̄ does not match the sidecar (not the frame's single μ̄)")

    board_id, card_texts = read_board_card_texts(board_path)
    fixture = build_rho_reference(
        board_id=board_id,
        card_texts=card_texts,
        attributes=attributes,
        phi_star=phi_star,
        gender_axis=sidecar_axis,
        mu_bar=sidecar_mu_bar,
        encoder=encoder,
        frame_id=frame_id,
    )
    return write_fixture(fixture)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Emit the frozen per-word ρ reference fixture for one reference board."
    )
    parser.add_argument("--words", type=Path, default=Path("resources/words"))
    parser.add_argument(
        "--subtlex", type=Path, default=Path("resources/frequencies/subtlex_us.csv")
    )
    parser.add_argument(
        "--attributes",
        type=Path,
        default=Path("resources/attribute_words/gender_attributes.csv"),
    )
    parser.add_argument(
        "--board",
        type=Path,
        default=_DEFAULT_BOARD,
        help=f"reference board JSON (default: {_DEFAULT_BOARD})",
    )
    parser.add_argument(
        "--sidecar",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / FRAME_FILENAME,
        help=f"measurement-frame sidecar (default: {DEFAULT_OUTPUT_DIR / FRAME_FILENAME})",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    for label, path in (
        ("words directory", args.words),
        ("SUBTLEX-US reference", args.subtlex),
        ("gender-attribute CSV", args.attributes),
        ("reference board", args.board),
        ("measurement-frame sidecar", args.sidecar),
    ):
        if not path.exists():
            sys.exit(f"{label} not found: {path}")

    path = emit_rho_reference(
        args.words, args.subtlex, args.attributes, args.board, args.sidecar
    )
    fixture = json.loads(path.read_text(encoding="utf-8"))
    print(
        f"wrote {path}\nframe_id {fixture['frame_id']}\nboard {fixture['board_id']} "
        f"({len(fixture['words'])} cards)",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
