"""CLI for the clue-giver-role bias metrics: IAE (eqs 4.6-4.8) and TAC/TAI (eq 4.9).

Reads a completed run out of Postgres and prints the point estimates. Strictly read-only: it opens
one session, computes, prints, and writes nothing back. Confidence intervals are deliberately not
produced here - a later shared cluster-bootstrap step owns them.

The computation lives in ``backend.app.analysis.clue_metrics``; this file is only the CLI: argument
parsing and the report layout.

Environment: export the vars yourself (no dotenv). DATABASE_URL must be in the process environment.
Run from the REPO ROOT.

Examples:
    set -a; source .env; set +a
    DATABASE_URL="postgresql+psycopg2://${POSTGRES_USER}:${POSTGRES_PASSWORD}@localhost:5432/${POSTGRES_DB}"
    python scripts/run_clue_metrics.py

    # machine-readable, for downstream aggregation:
    python scripts/run_clue_metrics.py --json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys

from backend.app.analysis.clue_metrics import (
    DEFAULT_FRAME_ID,
    DEFAULT_MASTER_SEED,
    STRATUM_POOLED,
    ClueMetricsReport,
    compute_clue_metrics,
)
from backend.app.db.session import session_scope

logger = logging.getLogger("run_clue_metrics")

_RULE = "=" * 96


def _fmt(value: float | None, places: int = 4) -> str:
    """Render an optional rate; an undefined rate prints as a dash, never as 0.0."""
    return "-" if value is None else f"{value:.{places}f}"


def _print_iae(report: ClueMetricsReport) -> None:
    print(_RULE)
    print("IAE - implicit association effect (clue-giver role)")
    print(_RULE)
    print(
        f"{'model':<24} {'stratum':<9} {'IAE':>8} {'y=1':>5} {'y=0':>5} "
        f"{'excl':>5} {'none_rate':>10}  by_seat"
    )
    for row in report.iae:
        if row.stratum != STRATUM_POOLED:
            continue
        seats = ", ".join(f"seat{seat}={count}" for seat, count in row.by_seat.items())
        print(
            f"{row.model_ref:<24} {row.stratum:<9} {_fmt(row.iae):>8} {row.n_stereotypical:>5} "
            f"{row.n_neutral:>5} {row.n_excluded:>5} {_fmt(row.none_rate):>10}  {seats}"
        )
    print(
        "\nIAE = #{y=1} / (#{y=0} + #{y=1}); excluded dilemmas are reported, not silently dropped."
    )
    print(
        "by_seat exposes whether dilemmas are single- or dual-perspective: on the shipped boards\n"
        "all three dilemma words are agents only under seat 0, so seat-1 counts are expected to be 0."
    )


def _print_tac_tai(report: ClueMetricsReport) -> None:
    print()
    print(_RULE)
    print("TAC / TAI - thematic association, congruent vs incongruent (clue-giver role)")
    print(_RULE)
    if report.tercile_cuts is None:
        print("no eligible pairs; no tercile cuts to report")
        return
    c33, c66 = report.tercile_cuts
    print(f"global tercile cuts over ALL eligible pairs: c33={c33:.6f}  c66={c66:.6f}")
    print("band 1 = least similar, band 3 = most similar; cuts are global, never per model.\n")
    print(
        f"{'model':<24} {'band':>5} {'TAC':>8} {'TAI':>8} {'gap':>9} "
        f"{'con g/e':>12} {'inc g/e':>12}"
    )
    for row in report.tac_tai:
        if row.stratum != STRATUM_POOLED:
            continue
        band = "all" if row.band is None else str(row.band)
        con = f"{row.n_congruent_grouped}/{row.n_congruent_eligible}"
        inc = f"{row.n_incongruent_grouped}/{row.n_incongruent_eligible}"
        print(
            f"{row.model_ref:<24} {band:>5} {_fmt(row.tac):>8} {_fmt(row.tai):>8} "
            f"{_fmt(row.gap):>9} {con:>12} {inc:>12}"
        )
    print("\ngap = TAC - TAI. The reportable signal is the gap widening as the band decreases.")


def _print_strata(report: ClueMetricsReport) -> None:
    print()
    print(_RULE)
    print("Career / science breakdown - DESCRIPTIVE ONLY (reduced n; pooled rows are primary)")
    print(_RULE)
    print(f"{'model':<24} {'stratum':<9} {'IAE':>8} {'y=1':>5} {'y=0':>5} {'excl':>5}")
    for row in report.iae:
        if row.stratum == STRATUM_POOLED:
            continue
        print(
            f"{row.model_ref:<24} {row.stratum:<9} {_fmt(row.iae):>8} {row.n_stereotypical:>5} "
            f"{row.n_neutral:>5} {row.n_excluded:>5}"
        )
    print()
    print(f"{'model':<24} {'stratum':<9} {'TAC':>8} {'TAI':>8} {'gap':>9}   (all bands)")
    for pair_row in report.tac_tai:
        if pair_row.stratum == STRATUM_POOLED or pair_row.band is not None:
            continue
        print(
            f"{pair_row.model_ref:<24} {pair_row.stratum:<9} {_fmt(pair_row.tac):>8} "
            f"{_fmt(pair_row.tai):>8} {_fmt(pair_row.gap):>9}"
        )


def print_report(report: ClueMetricsReport) -> None:
    print()
    print(_RULE)
    print(f"frame     : {report.frame_id}")
    print(f"run seed  : {report.master_seed}")
    print(
        f"coverage  : {report.n_games} completed probe games, {report.n_turns} clue-turns, "
        f"{report.n_dilemma_observations} dilemma observations, "
        f"{report.n_eligible_pairs} eligible pairs"
    )
    print("estimates : POINT ESTIMATES ONLY - no confidence intervals (later bootstrap step)")
    _print_iae(report)
    _print_tac_tai(report)
    _print_strata(report)
    print()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute clue-giver-role bias metrics (IAE, TAC/TAI). Read-only."
    )
    parser.add_argument(
        "--frame-id", default=DEFAULT_FRAME_ID, help="measurement frame id (default: the live frame)"
    )
    parser.add_argument(
        "--master-seed",
        type=int,
        default=DEFAULT_MASTER_SEED,
        help="run.master_seed selecting the batch (default: %(default)s)",
    )
    parser.add_argument(
        "--json", action="store_true", help="emit the report as JSON instead of text tables"
    )
    parser.add_argument("--verbose", action="store_true", help="log progress to stderr")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    with session_scope() as session:
        report = compute_clue_metrics(
            session, frame_id=args.frame_id, master_seed=args.master_seed
        )

    if report.n_turns == 0:
        print(
            f"no completed probe games for master_seed={args.master_seed}; nothing to report",
            file=sys.stderr,
        )
        return 1

    if args.json:
        print(json.dumps(dataclasses.asdict(report), indent=2, default=str))
    else:
        print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
