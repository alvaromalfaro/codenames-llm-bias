"""Offline backfill of ``embedding_mpnet`` with extrinsic phi* embeddings.

Populates one row per distinct lowercased board word and clue word for a measurement frame, in the
encoder space that frame declares. This is a post-batch, run-by-hand job: clues are not known a
priori, so it cannot run at startup. It writes vectors only.

The encoder (name, revision, pooling, normalize) is read from the ``measurement_frame`` row, never
hardcoded. The pinned checkpoint must already be in the local HuggingFace cache; nothing here
downloads it.

Requires the optional ``embeddings`` extra (sentence-transformers): ``uv sync --extra embeddings``.

DATABASE_URL must be in the process environment. Run from the REPO ROOT.

Examples:
    set -a; source .env; set +a
    DATABASE_URL="postgresql+psycopg2://${POSTGRES_USER}:${POSTGRES_PASSWORD}@localhost:5432/${POSTGRES_DB}"

    # what would be embedded, without loading the model or writing anything:
    python scripts/backfill_embeddings.py --frame-id 8a404797b3e656dd... --dry-run

    # the real backfill (idempotent; safe to re-run):
    python scripts/backfill_embeddings.py --frame-id 8a404797b3e656dd...
"""
from __future__ import annotations

import argparse
import logging
import sys

from backend.app.db.embed_mpnet import (
    DEFAULT_BATCH_SIZE, EncoderContractError, MissingFrameError, SentenceTransformerEncoder,
    backfill_frame, board_word_texts, clue_texts, load_encoder_spec,
)
from backend.app.db.session import session_scope

logger = logging.getLogger("backfill_embeddings")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="backfill_embeddings.py",
        description="Populate embedding_mpnet with phi* vectors for a measurement frame.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--frame-id",
                   help="measurement_frame.frame_id to embed for (REQUIRED; the encoder spec is "
                        "read from that row).")
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                   help=f"texts encoded and inserted per statement (default {DEFAULT_BATCH_SIZE}).")
    p.add_argument("--dry-run", action="store_true",
                   help="report the candidate counts per phase, load no model and write nothing.")
    return p


def _print_report(frame_id: str, report: dict[str, dict[str, int]]) -> None:
    print("\n" + "=" * 72)
    print(f"EMBEDDING BACKFILL  frame_id={frame_id}")
    print("=" * 72)
    total = 0
    for kind, counts in report.items():
        print(f"  {kind:<11} candidates={counts['candidates']:<5} "
              f"inserted={counts['inserted']:<5} skipped={counts['skipped']}")
        total += counts["inserted"]
    print("-" * 72)
    print(f"  TOTAL rows inserted this run: {total}")
    print("=" * 72)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    args = build_parser().parse_args(argv)

    if not args.frame_id:
        raise SystemExit("missing required argument: --frame-id")
    if args.batch_size < 1:
        raise SystemExit(
            f"--batch-size must be >= 1 (got {args.batch_size}).")

    try:
        with session_scope() as session:
            spec = load_encoder_spec(session, args.frame_id)
            print(f"[backfill] frame={spec.frame_id}")
            print(f"[backfill] encoder={spec.name}@{spec.revision} "
                  f"pooling={spec.pooling} normalize={spec.normalize}")

            if args.dry_run:
                board_words = board_word_texts(session, args.frame_id)
                clues = clue_texts(session, args.frame_id)
                print("[backfill] DRY RUN - no model loaded, nothing written")
                print(f"[backfill] board_word candidates={len(board_words)} "
                      f"clue candidates={len(clues)}")
                return 0

            encoder = SentenceTransformerEncoder(spec)
            report = backfill_frame(
                session, args.frame_id, encoder, batch_size=args.batch_size)
    except (MissingFrameError, EncoderContractError) as e:
        print(f"\n[backfill ABORTED - precondition] {type(e).__name__}: {e}",
              file=sys.stderr)
        return 2
    except RuntimeError as e:  # e.g. DATABASE_URL is not set
        print(f"\n[backfill ABORTED] {type(e).__name__}: {e}", file=sys.stderr)
        return 2

    _print_report(args.frame_id, report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
