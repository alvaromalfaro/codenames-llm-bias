#!/usr/bin/env python3
"""Seal the bank board artifacts with measurement_frame_id.

One-time, standalone maintenance script (run manually); not part of the generator runtime and not
wired into the `board-generator` CLI. It reads the frozen frame_id from the measurement-frame
sidecar (`data/boards/measurement_frame.json`) and writes it as a top-level `measurement_frame_id`
key into every real bank board (files with an `arbiters` block). It skips the sidecar, `balance_report.json`,
and `example_board.json` (no `arbiters` block), and reports exactly what it sealed and skipped.

Each write is guarded by a parsed-equality assertion (re-parsed file == parsed_before + the one key). 
Already-sealed boards are a no-op; a board with a different frame_id aborts the run.

Usage:
    uv run python scripts/seal_boards.py
    uv run python scripts/seal_boards.py --boards-dir ../data/boards
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from board_generator.board import DEFAULT_OUTPUT_DIR
from board_generator.emit_frame import FRAME_FILENAME
from board_generator.seal_boards import StaleSealError, read_frame_id, seal_board_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Seal bank board artifacts with measurement_frame_id."
    )
    parser.add_argument(
        "--boards-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"board artifact directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    sidecar = args.boards_dir / FRAME_FILENAME
    if not args.boards_dir.is_dir():
        sys.exit(f"boards directory not found: {args.boards_dir}")
    if not sidecar.exists():
        sys.exit(f"measurement-frame sidecar not found: {sidecar}")

    frame_id = read_frame_id(sidecar)
    try:
        result = seal_board_dir(args.boards_dir, frame_id)
    except StaleSealError as exc:
        sys.exit(f"STOP — stale seal: {exc}")

    print(f"frame_id: {frame_id}", file=sys.stderr)
    print(f"sealed {len(result.sealed)} board(s)", file=sys.stderr)
    if result.already_sealed:
        print(
            f"already sealed (no-op): {len(result.already_sealed)}", file=sys.stderr
        )
    print(f"skipped {len(result.skipped)}: {result.skipped}", file=sys.stderr)


if __name__ == "__main__":
    main()
