"""The batch orchestrator CLI: play the full 192-game model-pairing cross under one run.

This is the real batch - the probe/control calendar, the per-board seat mirror and the C(4,2) 
pairing cross from ``batch_schedule.build_schedule`` - played with count-and-continue fault 
isolation and a per-pairing consecutive-failure abort. It imports from backend.app and modifies 
nothing under backend/.

The orchestration lives in ``backend.app.core.batch_runner``; this file is only the CLI: argument 
parsing, model/board loading, the dry-run wiring and the final report print.

Preconditions (all fail loud before any provider call - see batch_runner.run_batch):
  * the loaded bank has >= 4 career + 4 science + 8 control boards;
  * DATABASE_URL is set (real path), the database schema is at the head of the migrations, and
    none of the 192 deterministic game ids already exist;
  * the measurement frame and the boards under data/boards are stored in the database (the batch
    stores whatever is missing; the database needs only the migrated schema);
  * the digest gate passes once at run-mint (enforce_digests=True on the real path).

A game lost to the provider (Ollama timing out or unavailable) is played again once the provider
answers again. A provider that does not answer within --provider-wait-minutes (default 30) stops the
batch: the report says where, and the exit code is 3.

A batch that stopped (that exit 3, a crash, a Ctrl+C) continues with --resume RUN_ID: the same run,
playing only the games it has not recorded. The run must match this batch - master seed, temperature,
prompt templates, models and the weights served - or nothing plays; another code version only warns.
--master-seed and --temperature may be left out, and are then the run's. The games it recorded as
errors stay so, unless --retry-errors plays them again.

Environment: export the vars yourself (no dotenv). DATABASE_URL, OLLAMA_HOST and/or
OPENROUTER_API_KEY must be in the process environment.

Examples:
    # real batch (default 4 models from config), enforcing digests:
    set -a; source .env; set +a # loads POSTGRES_* etc.
    OLLAMA_HOST=http://localhost:11434
    DATABASE_URL="postgresql+psycopg2://${POSTGRES_USER}:${POSTGRES_PASSWORD}@localhost:5432/${POSTGRES_DB}"
    python scripts/run_batch.py --master-seed 1234 --temperature 0.7

    # no-DB reproducibility check of the schedule + loop wiring (deterministic mock clients):
    python scripts/run_batch.py --master-seed 1234 --temperature 0.7 --dry-run

    # continue a batch that stopped, playing its errors again too:
    python scripts/run_batch.py --resume <RUN_ID> --retry-errors
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
    BatchPreconditionError, BatchReport, dry_run_client_factory, load_run, run_batch,
)
from backend.app.core.batch_schedule import ScheduleError
from backend.app.core.game_runner import ModelDigestMismatchError, SeatSpec
from backend.app.core.loader import BoardLoader
from backend.app.paths import BOARDS_DIR

logger = logging.getLogger("run_batch")

_KNOWN_PROVIDERS = ("ollama", "openrouter")


# models
def _models_from_config() -> list[SeatSpec]:
    """The batch models from ``config.llm_models`` (the schedule needs exactly 4). Provider keys are
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
    colons)."""
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
                   help="master seed (REQUIRED unless --resume; reproducibility is the point).")
    p.add_argument("--temperature", type=float,
                   help="sampling temperature (REQUIRED unless --resume; no default).")
    p.add_argument("--model", action="append", metavar="PROVIDER:MODEL",
                   help="a batch model as provider:model; repeat exactly 4 times to override the "
                        "roster in backend/app/config.py.")
    p.add_argument("--dry-run", action="store_true",
                   help="no-DB reproducibility check of the schedule + loop wiring using "
                        "deterministic mock clients (persist off, digest gate off).")
    p.add_argument("--provider-wait-minutes", type=float, metavar="MIN",
                   help="how long to wait for a provider that stops answering before stopping "
                        "the batch (default 30).")
    p.add_argument("--resume", metavar="RUN_ID",
                   help="continue the batch of run RUN_ID, playing only the games it has not "
                        "recorded; --master-seed and --temperature default to the run's.")
    p.add_argument("--retry-errors", action="store_true",
                   help="with --resume, play again the games the run recorded as errors.")
    return p


def _load_bank() -> list:
    """Load the on-disk board bank once, flattened (BoardLoader keys by category)."""
    loader = BoardLoader(BOARDS_DIR)
    boards = [b for group in loader.boards.values() for b in group]
    if not boards:
        raise SystemExit(
            f"No boards found under {str(BOARDS_DIR)!r}.")
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
        if report.resumed:
            line += (f" previously_completed={pr.previously_completed} "
                     f"previously_errored={pr.previously_errored}")
        if pr.aborted:
            line += f"  ABORTED ({pr.abort_reason})"
        if pr.error_kinds:
            line += f"  error_kinds={dict(pr.error_kinds)}"
        print(line)
    print("-" * 72)
    print(f"  TOTALS: attempted={report.attempted} completed={report.completed} "
          f"errored={report.errored} pairings_aborted={report.pairings_aborted}")
    if report.resumed:
        print(f"  RESUMED: previously_completed={report.previously_completed} "
              f"previously_errored={report.previously_errored}")
    if report.aborted_reason:
        print(f"  STOPPED: {report.aborted_reason}")
    print("=" * 72)
    print("\n" + json.dumps(report.as_dict(), indent=2))


# entry point
def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    args = build_parser().parse_args(argv)

    if args.resume is not None:
        if args.dry_run:
            raise SystemExit("--resume continues a run in the database; it cannot be combined "
                             "with --dry-run.")
        if args.master_seed is None or args.temperature is None:
            try:
                run = load_run(args.resume)
                if run is None:
                    raise BatchPreconditionError(f"there is no run {args.resume} to resume.")
            except BatchPreconditionError as e:
                print(f"\n[batch ABORTED - precondition] {e}", file=sys.stderr)
                return 2
            if args.master_seed is None:
                args.master_seed = run.master_seed
            if args.temperature is None:
                args.temperature = run.temperature
    elif args.retry_errors:
        raise SystemExit("--retry-errors goes with --resume.")

    missing =[name for name, val in (("--master-seed", args.master_seed),
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
    if args.resume is not None:
        print(f"[batch] resuming run {args.resume} retry_errors={args.retry_errors}")

    wait = ({} if args.provider_wait_minutes is None
            else {"provider_wait_s": args.provider_wait_minutes * 60})
    try:
        report = asyncio.run(run_batch(
            models=models, boards=boards, master_seed=args.master_seed,
            temperature=args.temperature,
            persist=not dry, enforce_digests=not dry,
            make_client_factory=(dry_run_client_factory if dry else None),
            bank_dir=BOARDS_DIR, resume_run_id=args.resume,
            retry_errors=args.retry_errors, **wait,
        ))
    except (ScheduleError, BatchPreconditionError, ModelDigestMismatchError) as e:
        print(
            f"\n[batch ABORTED - precondition] {type(e).__name__}: {e}", file=sys.stderr)
        return 2

    _print_report(report, dry_run=dry)
    if report.aborted_reason:
        print(f"\n[batch STOPPED - provider] {report.aborted_reason}. The games played so far are "
              f"stored under run {report.run_id}; once the provider answers, continue it with "
              f"--resume {report.run_id}.", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
