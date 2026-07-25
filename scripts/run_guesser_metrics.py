"""CLI for the guesser-role bias metric: CIT (eqs 4.10-4.11).

Reads a completed run out of Postgres and prints the point estimates. Strictly read-only: it opens
one session, computes, prints, and writes nothing back. Confidence intervals are deliberately not
produced here - a later shared cluster-bootstrap step owns them.

CIT is centred on 0.5: above means the guesser ranked gender-congruent cards higher than incongruent
ones, below means the reverse. Control boards are the negative control and should sit near 0.5.

The computation lives in ``backend.app.analysis.guesser_metrics``; this file is only the CLI.

Environment: export the vars yourself (no dotenv). DATABASE_URL must be in the process environment.
Run from the REPO ROOT.

Examples:
    set -a; source .env; set +a
    DATABASE_URL="postgresql+psycopg2://${POSTGRES_USER}:${POSTGRES_PASSWORD}@localhost:5432/${POSTGRES_DB}"
    python scripts/run_guesser_metrics.py

    python scripts/run_guesser_metrics.py --json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys

from backend.app.analysis.guesser_metrics import (
    DEFAULT_FRAME_ID,
    DEFAULT_MASTER_SEED,
    STRATUM_POOLED,
    GuesserMetricsReport,
    compute_guesser_metrics,
)
from backend.app.db.session import session_scope

logger = logging.getLogger("run_guesser_metrics")

_RULE = "=" * 100


def _fmt(value: float | None, places: int = 4) -> str:
    """Render an optional statistic; undefined prints as a dash, never as 0.5."""
    return "-" if value is None else f"{value:.{places}f}"


def _print_cuts(report: GuesserMetricsReport) -> None:
    print(_RULE)
    print("Tercile cuts on thematic similarity - computed ONCE PER BOARD TYPE, never pooled")
    print(_RULE)
    for board_type in sorted(report.tercile_cuts):
        cuts = report.tercile_cuts[board_type]
        if cuts is None:
            print(f"{board_type:<10} no eligible cards")
        else:
            print(f"{board_type:<10} c33={cuts[0]:.6f}  c66={cuts[1]:.6f}")
    print("band 1 = least similar to the clue, band 3 = most similar.")


def _print_cit(report: GuesserMetricsReport) -> None:
    print()
    print(_RULE)
    print("CIT - congruent-incongruent transfer (guesser role).  0.5 = no association")
    print(_RULE)
    print(
        f"{'model':<24} {'band':>5} "
        f"{'probe wtd':>10} {'probe cls':>10} {'probe n':>9}   "
        f"{'ctrl wtd':>10} {'ctrl cls':>10} {'ctrl n':>9}"
    )
    rows = {
        (row.model_ref, row.board_type, row.band): row
        for row in report.cit
        if row.stratum == STRATUM_POOLED
    }
    models = sorted({model for model, _, _ in rows})
    bands: list[int | None] = [None, 1, 2, 3]
    for model in models:
        for band in bands:
            probe = rows.get((model, "probe", band))
            control = rows.get((model, "control", band))
            if probe is None and control is None:
                continue
            label = "all" if band is None else str(band)
            print(
                f"{model:<24} {label:>5} "
                f"{_fmt(probe.cit_weighted) if probe else '-':>10} "
                f"{_fmt(probe.cit_classic) if probe else '-':>10} "
                f"{(probe.n_pairs if probe else 0):>9}   "
                f"{_fmt(control.cit_weighted) if control else '-':>10} "
                f"{_fmt(control.cit_classic) if control else '-':>10} "
                f"{(control.n_pairs if control else 0):>9}"
            )
        print()
    print("wtd = weighted by abs(P_gt)*abs(rho_i - rho_j); cls = classic unweighted Cliff delta.")
    print("n = comparable C+/C- pairs. Control is the NEGATIVE CONTROL and should sit near 0.5.")


def _print_diagnostics(report: GuesserMetricsReport) -> None:
    print()
    print(_RULE)
    print("Diagnostics - these determine CIT's power")
    print(_RULE)
    print(
        f"{'model':<24} {'admis':>7} {'non-adm':>8} {'unmatch':>8} {'no-emb':>7} "
        f"{'div-dup':>8} {'div-rank':>9}"
    )
    for diag in report.diagnostics:
        gaps = diag.gaps
        print(
            f"{diag.model_ref:<24} {diag.n_admissible_turns:>7} {diag.n_non_admissible_turns:>8} "
            f"{gaps.unmatched_ranking_words:>8} {gaps.cards_without_embedding:>7} "
            f"{gaps.divergent_duplicate_cards:>8} {gaps.rankings_with_divergent_duplicates:>9}"
        )
    print(
        "\nadmis/non-adm  : turns kept vs dropped for abs(rho(clue)) <= TAU_P\n"
        "unmatch        : ranking words matching no board card\n"
        "no-emb         : cards absent from the frame's embeddings (excluded, never imputed)\n"
        "div-dup        : cards the model ranked twice with DIFFERING confidence (excluded)\n"
        "div-rank       : rankings containing at least one such divergent duplicate"
    )

    print()
    print("Card funnel - why CIT has the power it has")
    print(
        f"{'model':<24} {'candidate':>10} {'neutral':>9} {'dead-zone':>10} {'classified':>11}"
    )
    for diag in report.diagnostics:
        print(
            f"{diag.model_ref:<24} {diag.n_cards_candidate:>10} {diag.n_cards_neutral:>9} "
            f"{diag.n_cards_dead_zone:>10} {diag.n_cards_classified:>11}"
        )
    print(
        "\ncandidate  : non-target, board-matched, embedded cards on admissible turns\n"
        "neutral    : dropped by the neutral cut,  abs(rho_i) <= TAU_RHO\n"
        "dead-zone  : degenerate rho_i * P == 0; unreachable here, kept as a tripwire\n"
        "classified : grouped C+/C- by sign(rho_i * P) and able to form comparable pairs"
    )


def _print_strata(report: GuesserMetricsReport) -> None:
    print()
    print(_RULE)
    print("Probe CIT by specification - DESCRIPTIVE ONLY (reduced n; pooled rows are primary)")
    print(_RULE)
    print(f"{'model':<24} {'stratum':<9} {'weighted':>10} {'classic':>10} {'pairs':>8}")
    for row in report.cit:
        if row.board_type != "probe" or row.stratum == STRATUM_POOLED or row.band is not None:
            continue
        print(
            f"{row.model_ref:<24} {row.stratum:<9} {_fmt(row.cit_weighted):>10} "
            f"{_fmt(row.cit_classic):>10} {row.n_pairs:>8}"
        )


def print_report(report: GuesserMetricsReport) -> None:
    print()
    print(_RULE)
    print(f"frame     : {report.frame_id}")
    print(f"run seed  : {report.master_seed}")
    print(
        f"coverage  : {report.n_turns} normal-phase guesser turns, "
        f"{report.n_admissible_turns} admissible, "
        f"{report.n_card_observations} grouped card observations"
    )
    print("estimates : POINT ESTIMATES ONLY - no confidence intervals (later bootstrap step)")
    _print_cuts(report)
    _print_cit(report)
    _print_diagnostics(report)
    _print_strata(report)
    print()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute the guesser-role bias metric (CIT). Read-only."
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
        report = compute_guesser_metrics(
            session, frame_id=args.frame_id, master_seed=args.master_seed
        )

    if report.n_turns == 0:
        print(
            f"no completed normal-phase turns for master_seed={args.master_seed}; nothing to report",
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
