"""List, show and delete the runs in the database.

A run is one batch (``scripts/run_batch.py``): the provenance it was minted with (master seed,
temperature, code version, prompt-template fingerprint, the models and the weights they served),
its notes (each resume appends one) and its games. ``list`` shows every run with its games by
status, ``show`` one run in full, and ``delete`` removes a run with all its games, after asking: a
delete cannot be undone. Deleting a run is how a master seed is played again from scratch; a batch
that stopped is continued with ``run_batch.py --resume`` instead.

DATABASE_URL must be in the process environment.

Examples:
    set -a; source .env; set +a
    DATABASE_URL="postgresql+psycopg2://${POSTGRES_USER}:${POSTGRES_PASSWORD}@localhost:5432/${POSTGRES_DB}"

    python scripts/manage_runs.py list
    python scripts/manage_runs.py show <RUN_ID>
    python scripts/manage_runs.py delete <RUN_ID>          # shows the run and asks first
    python scripts/manage_runs.py delete <RUN_ID> --yes    # does not ask
"""

from __future__ import annotations

import argparse
import sys
from datetime import timezone

from backend.app.db.runs import RunSummary, get_run, list_runs
from backend.app.db.writer import delete_run


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="manage_runs.py",
        description="List, show and delete the runs in the database.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    commands = p.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="every run, oldest first, with its games by status.")
    show = commands.add_parser("show", help="one run's provenance, notes and games.")
    show.add_argument("run_id", metavar="RUN_ID")
    delete = commands.add_parser("delete", help="delete a run and all its games, after asking.")
    delete.add_argument("run_id", metavar="RUN_ID")
    delete.add_argument("--yes", action="store_true", help="delete without asking.")
    return p


def _games(run: RunSummary) -> str:
    """``"189 completed, 3 error"``, or ``"none"``."""
    return ", ".join(f"{n} {status}" for status, n in run.games_by_status.items()) or "none"


def _created(run: RunSummary, fmt: str) -> str:
    return run.created_at.astimezone(timezone.utc).strftime(fmt)


def _dash(value) -> str:
    return "-" if value is None else str(value)


def _environment(snapshot: dict | None) -> str:
    """``"python 3.12.13, ollama 0.40.2"``, or ``"-"`` for a run minted before it was recorded."""
    if not snapshot:
        return "-"
    parts = [f"python {snapshot.get('python')}"]
    if snapshot.get("ollama"):
        parts.append(f"ollama {_dash(snapshot['ollama'].get('version'))}")
    return ", ".join(parts)


def _print_list(runs: list[RunSummary]) -> None:
    if not runs:
        print("no runs in the database")
        return
    print(f"{'RUN_ID':<36}  {'CREATED (UTC)':<16}  {'SEED':>6}  {'TEMP':>5}  {'CODE':<14}  GAMES")
    for run in runs:
        print(
            f"{run.run_id:<36}  {_created(run, '%Y-%m-%d %H:%M'):<16}  "
            f"{_dash(run.master_seed):>6}  {run.temperature:>5g}  "
            f"{_dash(run.code_version):<14}  {_games(run)}"
        )


def _print_run(run: RunSummary) -> None:
    print(f"run {run.run_id}")
    rows = [
        ("created", _created(run, "%Y-%m-%d %H:%M:%S UTC")),
        ("master_seed", _dash(run.master_seed)),
        ("temperature", f"{run.temperature:g}"),
        ("code_version", _dash(run.code_version)),
        ("prompt templates", _dash(run.prompt_template_version)),
        (
            "boards",
            f"{len(run.stimulus_manifest)} in its stimulus manifest"
            if run.stimulus_manifest
            else "-",
        ),
        ("environment", _environment(run.environment_snapshot)),
        ("games", f"{run.games} ({_games(run)})" if run.games else "none"),
    ]
    for label, value in rows:
        print(f"  {label:<17} {value}")
    snapshot = run.model_registry_snapshot or {}
    print("  models" + ("" if snapshot else "            -"))
    for seat in sorted(snapshot, key=int):
        entry = snapshot[seat]
        print(
            f"    {entry.get('provider')}:{entry.get('model_name')}  "
            f"{entry.get('identity_kind')}  {_dash(entry.get('identity'))}"
        )
    if run.notes:
        print("  notes")
        for line in run.notes.splitlines():
            print(f"    {line}")
    errors = run.games_by_status.get("error", 0)
    if errors:
        print(
            f"\n{errors} game(s) ended in error; play them again with:\n"
            f"  python scripts/run_batch.py --resume {run.run_id} --retry-errors"
        )


def _confirm(question: str) -> bool:
    """Ask ``question``; only a yes is a yes, and no answer at all (no terminal) is a no."""
    try:
        return input(question).strip().lower() in ("y", "yes")
    except EOFError:
        print()
        return False


def _delete(run_id: str, *, yes: bool) -> int:
    run = get_run(run_id)
    if run is None:
        print(f"there is no run {run_id}", file=sys.stderr)
        return 1
    _print_run(run)
    print()
    if not yes and not _confirm(
        f"Delete run {run_id} and its {run.games} game(s)? This cannot be undone. [y/N] "
    ):
        print("not deleted")
        return 1
    result = delete_run(run_id)
    if not result.found:  # deleted by someone else since it was read
        print(f"there is no run {run_id}", file=sys.stderr)
        return 1
    print(f"deleted run {run_id} and its {result.games_deleted} game(s)")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "list":
            _print_list(list_runs())
            return 0
        if args.command == "show":
            run = get_run(args.run_id)
            if run is None:
                print(f"there is no run {args.run_id}", file=sys.stderr)
                return 1
            _print_run(run)
            return 0
        return _delete(args.run_id, yes=args.yes)
    except RuntimeError as e:  # e.g. DATABASE_URL is not set
        print(f"\n[manage_runs ABORTED] {type(e).__name__}: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
