"""The batch orchestrator CLI: play the full 192-game model-pairing cross under one run.

This is the real batch - the probe/control calendar, the per-board seat mirror and the C(4,2) 
pairing cross from ``batch_schedule.build_schedule`` - played with count-and-continue fault 
isolation and a per-pairing consecutive-failure abort. It imports from backend.app and modifies 
nothing under backend/.

The orchestration lives in ``backend.app.core.batch_runner``; this file is only the CLI: argument 
parsing, model/board loading, the dry-run wiring and the final report print.

Preconditions (all fail loud before any provider call - see batch_runner.run_batch):
  * the loaded bank has >= 4 career + 4 science + 8 control boards;
  * DATABASE_URL is set (real path) and none of the 192 deterministic game ids already exist;
  * the digest gate passes once at run-mint (enforce_digests=True on the real path).

Environment: export the vars yourself (no dotenv). DATABASE_URL, OLLAMA_HOST and/or
OPENROUTER_API_KEY must be in the process environment. Run from the REPO ROOT.

Examples:
    # real batch (default 4 models from config), enforcing digests:
    set -a; source .env; set +a # loads POSTGRES_* etc.
    OLLAMA_HOST=http://localhost:11434
    DATABASE_URL="postgresql+psycopg2://${POSTGRES_USER}:${POSTGRES_PASSWORD}@localhost:5432/${POSTGRES_DB}"
    python scripts/run_batch.py --master-seed 1234 --temperature 0.7

    # no-DB reproducibility check of the schedule + loop wiring (deterministic mock clients):
    python scripts/run_batch.py --master-seed 1234 --temperature 0.7 --dry-run
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from typing import Any, cast

from backend.app import config
from backend.app.core.batch_runner import (
    BatchPreconditionError, BatchReport, dry_run_client_factory, run_batch,
)
from backend.app.core.batch_schedule import ScheduleError
from backend.app.core.game_runner import ModelDigestMismatchError, SeatSpec
from backend.app.core.loader import BoardLoader

logger = logging.getLogger("run_batch")

_BOARD_DATA_PATH = "data/boards"
_KNOWN_PROVIDERS = ("ollama", "openrouter")


# models
def _models_from_config() -> list[SeatSpec]:
    """The 4 batch models from ``config.llm_models`` (2 Ollama + 2 OpenRouter). Provider keys are
    lowercased to the SeatSpec convention; the local-only ``think`` flag is read per model."""
    specs: list[SeatSpec] = []
    all_models = cast("dict[str, dict[str, Any]]", config.llm_models)
    for provider_key, models in all_models.items():
        provider = provider_key.lower()
        for model_name, cfg in models.items():
            specs.append(SeatSpec(provider=provider, model_name=model_name,
                                  think=bool((cfg or {}).get("think", False))))
    return specs


def _parse_model(arg: str) -> SeatSpec:
    """Parse a ``provider:model`` flag into a SeatSpec (split on the FIRST colon; model names contain
    colons). Mirrors run_pilot._parse_seat's validation."""
    if ":" not in arg:
        raise SystemExit(
            f"--model {arg!r} must be provider:model (e.g. ollama:llama3.1:8b).")
    provider, model_name = arg.split(":", 1)
    provider, model_name = provider.strip().lower(), model_name.strip()
    if not provider or not model_name:
        raise SystemExit(
            f"--model {arg!r} must be provider:model (e.g. ollama:llama3.1:8b).")
    if provider not in _KNOWN_PROVIDERS:
        raise SystemExit(
            f"--model {arg!r} has unknown provider {provider!r}; one of {_KNOWN_PROVIDERS}.")
    return SeatSpec(provider=provider, model_name=model_name)


# CLI
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_batch.py",
        description="Play the full batch (C(4,2) pairing cross) and report.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--master-seed", type=int,
                   help="master seed (REQUIRED; reproducibility is the point).")
    p.add_argument("--temperature", type=float,
                   help="sampling temperature (REQUIRED; no default).")
    p.add_argument("--model", action="append", metavar="PROVIDER:MODEL",
                   help="a batch model as provider:model; repeat exactly 4 times to override the "
                        "config default (2 Ollama + 2 OpenRouter).")
    p.add_argument("--dry-run", action="store_true",
                   help="no-DB reproducibility check of the schedule + loop wiring using "
                        "deterministic mock clients (persist off, digest gate off).")
    return p


def _load_bank() -> list:
    """Load the on-disk board bank once, flattened (BoardLoader keys by category)."""
    loader = BoardLoader(_BOARD_DATA_PATH)
    boards = [b for group in loader.boards.values() for b in group]
    if not boards:
        raise SystemExit(
            f"No boards found under {_BOARD_DATA_PATH!r}. Run from the repo root.")
    return boards


def _print_report(report: BatchReport, *, dry_run: bool) -> None:
    print("\n" + "=" * 72)
    print(f"BATCH REPORT  run_id={report.run_id}  master_seed={report.master_seed}"
          f"{'  (DRY RUN)' if dry_run else ''}")
    print("=" * 72)
    for pr in report.pairings:
        line = (f"  p{pr.pairing_ordinal} {pr.pairing_key[0]} vs {pr.pairing_key[1]}: "
                f"attempted={pr.attempted} completed={pr.completed} errored={pr.errored} "
                f"skipped={pr.skipped}")
        if pr.aborted:
            line += f"  ABORTED ({pr.abort_reason})"
        if pr.error_kinds:
            line += f"  error_kinds={dict(pr.error_kinds)}"
        print(line)
    print("-" * 72)
    print(f"  TOTALS: attempted={report.attempted} completed={report.completed} "
          f"errored={report.errored} pairings_aborted={report.pairings_aborted}")
    print("=" * 72)
    print("\n" + json.dumps(report.as_dict(), indent=2))


# entry point
def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    args = build_parser().parse_args(argv)

    missing = [name for name, val in (("--master-seed", args.master_seed),
                                      ("--temperature", args.temperature)) if val is None]
    if missing:
        raise SystemExit(f"missing required argument(s): {', '.join(missing)}")

    if args.model:
        if len(args.model) != 4:
            raise SystemExit(
                f"--model must be given exactly 4 times (got {len(args.model)}); "
                "omit it to use the 4 config models.")
        models = [_parse_model(m) for m in args.model]
    else:
        models = _models_from_config()

    boards = _load_bank()
    dry = args.dry_run
    print(f"[batch] models={[f'{m.provider}:{m.model_name}' for m in models]}")
    print(f"[batch] master_seed={args.master_seed} temperature={args.temperature} "
          f"dry_run={dry} enforce_digests={not dry}")

    try:
        report = asyncio.run(run_batch(
            models=models, boards=boards, master_seed=args.master_seed,
            temperature=args.temperature,
            persist=not dry, enforce_digests=not dry,
            make_client_factory=(dry_run_client_factory if dry else None),
        ))
    except (ScheduleError, BatchPreconditionError, ModelDigestMismatchError) as e:
        print(
            f"\n[batch ABORTED - precondition] {type(e).__name__}: {e}", file=sys.stderr)
        return 2

    _print_report(report, dry_run=dry)
    return 0


if __name__ == "__main__":
    sys.exit(main())
