"""CLI applying the cluster-bootstrap inference layer to the sudden-death metric conc-SD.

conc-SD is the endgame analogue of CIT: in sudden death the clue giver is silent, so the guesser must
pick from memory of the clue history it received from its partner. This reports, per model:

  * whether the model clears the pre-registered power bar (>= 20 admissible sudden-death games) and is
    therefore a PRIMARY result, or falls below it and is EXPLORATORY - applied mechanically;
  * conc-SD per band, probe (primary) and control (negative control), in the weighted and classic
    variants and the max and mean s^H variants, each with a cluster-bootstrap interval;
  * the two paired contrasts probe-control and b1-b3, accumulated within each resample;
  * diagnostics that decide whether anything is interpretable: the sudden-death-reach proportion, the
    seat-0 dominance of who actually produced a sudden-death ranking, and the data gaps;
  * a descriptive directional check on the wrongly selected card of each sudden-death failure - NO
    p-value, reported as description only.

Games are the resampling unit (turns/seats within a game are not independent). Tercile cuts are frozen
over the full dataset before any resampling. Strictly read-only: one session, compute, print, no
writes.

Environment: DATABASE_URL must be in the process environment. Run from the REPO ROOT.

Examples:
    set -a; source .env; set +a
    DATABASE_URL="postgresql+psycopg2://${POSTGRES_USER}:${POSTGRES_PASSWORD}@localhost:5432/${POSTGRES_DB}"
    python scripts/run_sd_metrics.py
    python scripts/run_sd_metrics.py --replicates 500 --json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys

from backend.app.analysis.inference import DEFAULT_REPLICATES, DEFAULT_SEED, BootstrapResult
from backend.app.analysis.sd_metrics import (
    CONTRASTS,
    DEFAULT_FRAME_ID,
    DEFAULT_MASTER_SEED,
    SIM_VARIANTS,
    WEIGHTINGS,
    SdReport,
    compute_sd_metrics,
)
from backend.app.db.session import session_scope

logger = logging.getLogger("run_sd_metrics")

_RULE = "=" * 104

CELL_ORDER = (
    "probe_all",
    "probe_b1",
    "probe_b2",
    "probe_b3",
    "control_all",
    "control_b1",
    "control_b2",
    "control_b3",
)


def _fmt(value: float | None, places: int = 4) -> str:
    return "-" if value is None else f"{value:.{places}f}"


def _print_header(report: SdReport, *, replicates: int, seed: int) -> None:
    print()
    print(_RULE)
    print("conc-SD - sudden-death guesser bias, with cluster-bootstrap intervals")
    print(_RULE)
    print(f"frame      : {report.frame_id}")
    print(f"run seed   : {report.master_seed}")
    print("clusters   : games (a game's per-seat SD states are one cluster)")
    print(f"bootstrap  : B={replicates}, seed={seed}, percentile CI (2.5, 97.5)")
    print("null       : conc-SD = 0.5 means no association. No p-values: effect, interval, MDE only.")
    print(f"observations: {report.n_observations} classified non-agent cards")


def _print_reach(report: SdReport) -> None:
    print()
    print(_RULE)
    print("Interpretability diagnostics")
    print(_RULE)
    print(f"  {'board type':<12} {'completed':>10} {'reach SD':>10} {'proportion':>12}")
    for board_type, (completed, reach) in report.sd_reach.items():
        prop = reach / completed if completed else None
        print(f"  {board_type:<12} {completed:>10} {reach:>10} {_fmt(prop):>12}")
    print("\n  SD measurement rankings by seat (seat-0 dominance is expected):")
    for board_type, seats in report.seat_rankings.items():
        rendered = "  ".join(f"seat {seat}: {count}" for seat, count in sorted(seats.items()))
        print(f"    {board_type:<10} {rendered}")


def _print_power(report: SdReport) -> None:
    print()
    print(_RULE)
    print("Pre-registered power rule (>= 20 admissible SD games => PRIMARY, else EXPLORATORY)")
    print(_RULE)
    print(f"  {'model':<26} {'SD obs':>7} {'admissible':>11} {'non-adm':>8}  {'status':<11}")
    for summary in report.model_summaries:
        status = "PRIMARY" if summary.is_primary else "EXPLORATORY"
        print(
            f"  {summary.model_ref:<26} {summary.n_sd_observations:>7} "
            f"{summary.n_admissible_games:>11} {summary.n_non_admissible_games:>8}  {status:<11}"
        )


def _print_cuts(report: SdReport) -> None:
    print()
    print(_RULE)
    print("Tercile cuts on s^H  (conc-SD's OWN, frozen before the bootstrap)")
    print(_RULE)
    for key in sorted(report.tercile_cuts):
        cuts = report.tercile_cuts[key]
        if cuts is not None:
            print(f"  {key:<16} c33={cuts[0]:.6f}  c66={cuts[1]:.6f}")


def _print_cells(model_ref: str, result: BootstrapResult) -> None:
    print(f"\n  {model_ref}   ({result.n_clusters} games, {result.n_replicates} replicates)")
    print(
        f"    {'cell':<13} {'conc-SD':>8} {'95% CI':>19} {'SE':>8} {'MDE':>8} "
        f"{'dropped':>8}  {'excl 0.5':>8}"
    )
    for name in CELL_ORDER:
        cell = result.cells.get(name)
        if cell is None:
            continue
        interval = f"[{_fmt(cell.ci_low)}, {_fmt(cell.ci_high)}]"
        excludes = {True: "yes", False: "no", None: "-"}[cell.excludes_null]
        flag = "" if cell.reliable else "  UNRELIABLE"
        print(
            f"    {name:<13} {_fmt(cell.point):>8} {interval:>19} {_fmt(cell.standard_error):>8} "
            f"{_fmt(cell.mde):>8} {cell.n_dropped:>8}  {excludes:>8}{flag}"
        )
    for name, _a, _b in CONTRASTS:
        contrast = result.contrasts.get(name)
        if contrast is None or contrast.point is None:
            continue
        interval = f"[{_fmt(contrast.ci_low)}, {_fmt(contrast.ci_high)}]"
        excludes = {True: "yes", False: "no", None: "-"}[contrast.excludes_zero]
        flag = "" if contrast.reliable else "  UNRELIABLE"
        print(
            f"    {name:<22} diff {_fmt(contrast.point):>7} {interval:>19} "
            f"dropped {contrast.n_dropped:>6}  excl0 {excludes}{flag}"
        )


def _print_bootstrap(report: SdReport) -> None:
    for variant in SIM_VARIANTS:
        for weighting in WEIGHTINGS:
            key = f"{variant}_{weighting}"
            per_model = report.bootstrap.get(key, {})
            print()
            print(_RULE)
            print(f"conc-SD  (s^H variant: {variant}, weighting: {weighting})")
            print(_RULE)
            if not per_model:
                print("  (no models with SD observations)")
                continue
            for model_ref in sorted(per_model):
                _print_cells(model_ref, per_model[model_ref])


def _print_fg(report: SdReport) -> None:
    print()
    print(_RULE)
    print("Descriptive directional check on the wrongly selected card f_g  (NOT evidence)")
    print(_RULE)
    print("  Proportion of SD failures whose f_g satisfies rho(f_g) * P^H_g > 0.")
    print(f"  {'board type':<12} {'n':>6} {'positive':>10} {'proportion':>12}")
    for board_type, (n, positive, proportion) in sorted(report.fg_check.items()):
        print(f"  {board_type:<12} {n:>6} {positive:>10} {_fmt(proportion):>12}")


def _print_diagnostics(report: SdReport) -> None:
    print()
    print(_RULE)
    print("Per-model card funnel and data gaps")
    print(_RULE)
    print(
        f"  {'model':<26} {'candidate':>10} {'neutral':>8} {'classified':>11} "
        f"{'no-embed':>9} {'clue-gap':>9} {'divergent':>10}"
    )
    for diag in report.diagnostics:
        g = diag.gaps
        print(
            f"  {diag.model_ref:<26} {diag.n_cards_candidate:>10} {diag.n_cards_neutral:>8} "
            f"{diag.n_cards_classified:>11} {g.cards_without_embedding:>9} "
            f"{g.clue_words_without_embedding:>9} {g.divergent_duplicate_cards:>10}"
        )


def print_report(report: SdReport, *, replicates: int, seed: int) -> None:
    _print_header(report, replicates=replicates, seed=seed)
    _print_reach(report)
    _print_power(report)
    _print_cuts(report)
    _print_bootstrap(report)
    _print_fg(report)
    _print_diagnostics(report)
    print()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cluster-bootstrap confidence intervals for conc-SD (sudden death). Read-only."
    )
    parser.add_argument("--frame-id", default=DEFAULT_FRAME_ID)
    parser.add_argument("--master-seed", type=int, default=DEFAULT_MASTER_SEED)
    parser.add_argument(
        "--replicates", type=int, default=DEFAULT_REPLICATES, help="bootstrap replicates B"
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="bootstrap RNG seed")
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
        report = compute_sd_metrics(
            session,
            frame_id=args.frame_id,
            master_seed=args.master_seed,
            n_replicates=args.replicates,
            seed=args.seed,
        )

    if report.n_observations == 0:
        print(
            f"no sudden-death observations for master_seed={args.master_seed}; nothing to report",
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
