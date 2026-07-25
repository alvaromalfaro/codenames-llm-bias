"""CLI for the strategic (skill) metrics on control boards: TV, PA and EP.

Skill is measured on the gender-neutral control boards, so competence is scored where the gender
manipulation is absent. Intervals come from the shared cluster bootstrap over games.

  TV  win rate            per pairing (the honest unit) and per model (marginal over partners)
  PA  guess accuracy      agent share of the cards a model actually played as guesser
  EP  clue efficiency     agents revealed per clue the model gave (a ratio, may exceed 1)

Codenames Duet is cooperative: both seats share one outcome, so a per-model TV necessarily mixes a
model's own skill with its partners'. The report says so where it prints it.

Strictly read-only: one session, compute, print, no writes.

Environment: DATABASE_URL must be in the process environment. Run from the REPO ROOT.

Examples:
    set -a; source .env; set +a
    DATABASE_URL="postgresql+psycopg2://${POSTGRES_USER}:${POSTGRES_PASSWORD}@localhost:5432/${POSTGRES_DB}"
    python scripts/run_skill_metrics.py
    python scripts/run_skill_metrics.py --replicates 500 --json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys

from backend.app.analysis.inference import DEFAULT_REPLICATES, DEFAULT_SEED
from backend.app.analysis.skill_metrics import (
    DEFAULT_MASTER_SEED,
    WIN_RESULTS,
    CellEstimate,
    SkillReport,
    compute_skill_metrics,
)
from backend.app.db.session import session_scope

logger = logging.getLogger("run_skill_metrics")

_RULE = "=" * 92


def _fmt(value: float | None, places: int = 4) -> str:
    return "-" if value is None else f"{value:.{places}f}"


def _interval(cell: CellEstimate | None) -> str:
    if cell is None:
        return "-"
    return f"[{_fmt(cell.ci_low)}, {_fmt(cell.ci_high)}]"


def _print_results(report: SkillReport) -> None:
    print(_RULE)
    print("game.result encoding - inspected, not assumed")
    print(_RULE)
    total = sum(report.result_counts.values())
    for value, count in report.result_counts.items():
        marker = "WIN " if value in WIN_RESULTS else "loss"
        print(f"  {marker}  {value:<22} {count:>4}")
    wins = sum(count for value, count in report.result_counts.items() if value in WIN_RESULTS)
    print(f"\n  win = result IN {sorted(WIN_RESULTS)}  ->  {wins} of {total} games")
    if wins == 0:
        print("  NOTE: no game in this run was won, so TV is identically 0 and every bootstrap")
        print("        resample also yields 0 - the intervals below are degenerate by construction.")


def _print_tv(report: SkillReport) -> None:
    print()
    print(_RULE)
    print("TV - win rate per PAIRING  (the honest unit: the outcome belongs to the pair)")
    print(_RULE)
    print(f"  {'pairing':<40} {'TV':>8} {'95% CI':>21} {'games':>6}")
    for label, cell in report.tv_by_pairing.items():
        print(
            f"  {label:<40} {_fmt(cell.point):>8} {_interval(cell):>21} "
            f"{report.games_per_pairing.get(label, 0):>6}"
        )

    print()
    print(_RULE)
    print("TV - win rate per MODEL  (marginal over partners)")
    print(_RULE)
    print("  CAVEAT: cooperative game. A per-model win rate conflates the model's own skill with")
    print("          its partners' - it is not an attribution to the model.")
    print(f"\n  {'model':<40} {'TV':>8} {'95% CI':>21} {'games':>6}")
    diag = {d.model_ref: d for d in report.diagnostics}
    for model_ref, cell in report.tv_by_model.items():
        print(
            f"  {model_ref:<40} {_fmt(cell.point):>8} {_interval(cell):>21} "
            f"{diag[model_ref].n_games:>6}"
        )


def _print_pa_ep(report: SkillReport) -> None:
    diag = {d.model_ref: d for d in report.diagnostics}

    print()
    print(_RULE)
    print("PA - guess accuracy as guesser  (agent share of played, resolved cards)")
    print(_RULE)
    print(f"  {'model':<26} {'PA':>8} {'95% CI':>21} {'agents':>8} {'played':>8}")
    for model_ref, cell in report.pa_by_model.items():
        d = diag[model_ref]
        print(
            f"  {model_ref:<26} {_fmt(cell.point):>8} {_interval(cell):>21} "
            f"{d.n_agent_cards:>8} {d.n_played_cards:>8}"
        )

    print()
    print(_RULE)
    print("EP - clue efficiency as clue-giver  (agents revealed per clue; a ratio, not a rate)")
    print(_RULE)
    print(f"  {'model':<26} {'EP':>8} {'95% CI':>21} {'agents':>8} {'clues':>8}")
    for model_ref, cell in report.ep_by_model.items():
        d = diag[model_ref]
        print(
            f"  {model_ref:<26} {_fmt(cell.point):>8} {_interval(cell):>21} "
            f"{d.n_agents_revealed_on_own_clues:>8} {d.n_clues:>8}"
        )


def _print_diagnostics(report: SkillReport) -> None:
    print()
    print(_RULE)
    print("Diagnostics - the n behind each estimate")
    print(_RULE)
    print(
        f"  {'model':<26} {'games':>7} {'clues':>7} {'played':>8} {'agent cards':>12} "
        f"{'agents/clues':>13}"
    )
    for d in report.diagnostics:
        print(
            f"  {d.model_ref:<26} {d.n_games:>7} {d.n_clues:>7} {d.n_played_cards:>8} "
            f"{d.n_agent_cards:>12} {d.n_agents_revealed_on_own_clues:>13}"
        )


def print_report(report: SkillReport, *, replicates: int, seed: int) -> None:
    print()
    print(_RULE)
    print("Strategic (skill) metrics - CONTROL BOARDS ONLY")
    print(_RULE)
    print(f"run seed   : {report.master_seed}")
    print(f"coverage   : {report.n_games} completed control games")
    print(f"bootstrap  : B={replicates}, seed={seed}, games as clusters, percentile CI (2.5, 97.5)")
    print("scope      : normal-phase turns (sudden death carries no clues, so EP is undefined there")
    print("             and PA is restricted the same way to keep both metrics in one regime)")
    print()
    _print_results(report)
    _print_tv(report)
    _print_pa_ep(report)
    _print_diagnostics(report)
    print()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute strategic skill metrics (TV, PA, EP) on control boards. Read-only."
    )
    parser.add_argument("--master-seed", type=int, default=DEFAULT_MASTER_SEED)
    parser.add_argument("--replicates", type=int, default=DEFAULT_REPLICATES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    with session_scope() as session:
        report = compute_skill_metrics(
            session,
            master_seed=args.master_seed,
            n_replicates=args.replicates,
            seed=args.seed,
        )

    if report.n_games == 0:
        print(
            f"no completed control games for master_seed={args.master_seed}; nothing to report",
            file=sys.stderr,
        )
        return 1

    if args.json:
        print(json.dumps(dataclasses.asdict(report), indent=2, default=str))
    else:
        print_report(report, replicates=args.replicates, seed=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
