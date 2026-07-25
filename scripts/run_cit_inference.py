"""CLI applying the cluster-bootstrap inference layer to CIT.

This adds intervals, and in particular the two paired contrasts that the point estimates cannot settle:

  * ``probe_all - control_all`` - is probe genuinely different from the gender-neutral control, or do
    the two overlap? Increment 2b found control sitting at 0.53-0.69, sometimes above probe;
  * ``probe_b1 - probe_b3`` - the monotone-profile hypothesis, that the effect grows as thematic
    proximity to the clue falls.

Both are accumulated inside each bootstrap replicate from the same resample, so their intervals
reflect how the two cells move together. Games are the resampling unit because turns within a game
share a board, a keycard and an opponent.

The tercile cuts are computed once over the full dataset and frozen into the estimator closure; the
bootstrap has no path to recompute them.

Strictly read-only: one session, compute, print, no writes. No p-values - effect, interval, and a
minimum detectable effect as a sensitivity bound.

Environment: DATABASE_URL must be in the process environment. Run from the REPO ROOT.

Examples:
    set -a; source .env; set +a
    DATABASE_URL="postgresql+psycopg2://${POSTGRES_USER}:${POSTGRES_PASSWORD}@localhost:5432/${POSTGRES_DB}"
    python scripts/run_cit_inference.py
    python scripts/run_cit_inference.py --replicates 500 --variant classic
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from backend.app.analysis.guesser_metrics import (
    DEFAULT_FRAME_ID,
    DEFAULT_MASTER_SEED,
    FrameGeometry,
    assign_bands_by_board_type,
    build_cit_estimator,
    collect_card_observations,
    load_guesser_turns,
)
from backend.app.analysis.inference import (
    DEFAULT_REPLICATES,
    DEFAULT_SEED,
    BootstrapResult,
    cluster_bootstrap,
)
from backend.app.db.session import session_scope

logger = logging.getLogger("run_cit_inference")

_RULE = "=" * 104

CIT_NULL = 0.5
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
CONTRASTS = (
    ("probe_minus_control", "probe_all", "control_all"),
    ("probe_b1_minus_b3", "probe_b1", "probe_b3"),
)


def _fmt(value: float | None, places: int = 4) -> str:
    return "-" if value is None else f"{value:.{places}f}"


def _print_cells(model_ref: str, result: BootstrapResult) -> None:
    print(f"\n{model_ref}   ({result.n_clusters} games as clusters, "
          f"{result.n_replicates} replicates, {result.elapsed_seconds:.1f}s)")
    print(
        f"  {'cell':<13} {'CIT':>8} {'95% CI':>19} {'SE':>8} {'MDE':>8} "
        f"{'dropped':>8}  {'excl 0.5':>8}"
    )
    for name in CELL_ORDER:
        cell = result.cells.get(name)
        if cell is None:
            print(f"  {name:<13} {'-':>8}   (no comparable pairs)")
            continue
        interval = f"[{_fmt(cell.ci_low)}, {_fmt(cell.ci_high)}]"
        excludes = {True: "yes", False: "no", None: "-"}[cell.excludes_null]
        flag = "" if cell.reliable else "  UNRELIABLE"
        print(
            f"  {name:<13} {_fmt(cell.point):>8} {interval:>19} {_fmt(cell.standard_error):>8} "
            f"{_fmt(cell.mde):>8} {cell.n_dropped:>8}  {excludes:>8}{flag}"
        )


def _print_contrasts(result: BootstrapResult) -> None:
    print(f"  {'contrast':<22} {'diff':>9} {'95% CI':>21} {'SE':>8} {'dropped':>8}  {'excl 0':>7}")
    for name, _a, _b in CONTRASTS:
        contrast = result.contrasts.get(name)
        if contrast is None:
            continue
        interval = f"[{_fmt(contrast.ci_low)}, {_fmt(contrast.ci_high)}]"
        excludes = {True: "yes", False: "no",
                    None: "-"}[contrast.excludes_zero]
        flag = "" if contrast.reliable else "  UNRELIABLE"
        print(
            f"  {name:<22} {_fmt(contrast.point):>9} {interval:>21} "
            f"{_fmt(contrast.standard_error):>8} {contrast.n_dropped:>8}  {excludes:>7}{flag}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cluster-bootstrap confidence intervals for CIT. Read-only."
    )
    parser.add_argument("--frame-id", default=DEFAULT_FRAME_ID)
    parser.add_argument("--master-seed", type=int, default=DEFAULT_MASTER_SEED)
    parser.add_argument(
        "--replicates", type=int, default=DEFAULT_REPLICATES, help="bootstrap replicates B"
    )
    parser.add_argument("--seed", type=int,
                        default=DEFAULT_SEED, help="bootstrap RNG seed")
    parser.add_argument(
        "--variant", choices=("weighted", "classic"), default="weighted", help="CIT variant"
    )
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
        geometry = FrameGeometry.load(session, args.frame_id)
        turns = load_guesser_turns(session, master_seed=args.master_seed)
        raw_cards, _diagnostics = collect_card_observations(turns, geometry)

    # Cuts are computed ONCE here, over the full dataset, and are baked into the banded cards the
    # estimator closes over. No replicate can move them.
    cards, cuts = assign_bands_by_board_type(raw_cards)
    if not cards:
        print("no grouped card observations; nothing to bootstrap", file=sys.stderr)
        return 1

    models = sorted({card.model_ref for card in cards})
    results: dict[str, BootstrapResult] = {}
    for model_ref in models:
        estimator, game_ids = build_cit_estimator(
            cards, model_ref=model_ref, variant=args.variant
        )
        results[model_ref] = cluster_bootstrap(
            game_ids,
            estimator,
            n_replicates=args.replicates,
            seed=args.seed,
            contrasts=CONTRASTS,
            null_value=CIT_NULL,
        )

    if args.json:
        print(
            json.dumps(
                {
                    "frame_id": args.frame_id,
                    "master_seed": args.master_seed,
                    "variant": args.variant,
                    "tercile_cuts": {k: list(v) if v else None for k, v in cuts.items()},
                    "models": {
                        model: {
                            "n_clusters": result.n_clusters,
                            "elapsed_seconds": result.elapsed_seconds,
                            "cells": {n: vars(c) for n, c in result.cells.items()},
                            "contrasts": {n: vars(c) for n, c in result.contrasts.items()},
                        }
                        for model, result in results.items()
                    },
                },
                indent=2,
                default=str,
            )
        )
        return 0

    print()
    print(_RULE)
    print(f"CIT with cluster-bootstrap intervals  (variant: {args.variant})")
    print(_RULE)
    print(f"frame      : {args.frame_id}")
    print(f"run seed   : {args.master_seed}")
    print("clusters   : games (turns within a game share a board, keycard and opponent)")
    print(
        f"bootstrap  : B={args.replicates}, seed={args.seed}, percentile CI (2.5, 97.5)")
    for board_type in sorted(cuts):
        pair = cuts[board_type]
        if pair:
            print(
                f"cuts {board_type:<8}: c33={pair[0]:.6f} c66={pair[1]:.6f}  (frozen before bootstrap)")
    print("null       : CIT = 0.5 means no association. MDE at alpha 0.05, power 0.8.")
    print("             No p-values: effect, interval and sensitivity only.")

    print()
    print(_RULE)
    print("Per-model cells")
    print(_RULE)
    for model_ref in models:
        _print_cells(model_ref, results[model_ref])

    print()
    print(_RULE)
    print("Paired contrasts  (both terms from the SAME resample, so the CI carries their correlation)")
    print(_RULE)
    for model_ref in models:
        print(f"\n{model_ref}")
        _print_contrasts(results[model_ref])

    total = sum(result.elapsed_seconds for result in results.values())
    print(f"\ntotal bootstrap wall clock: {total:.1f}s "
          f"({args.replicates} replicates x {len(models)} models)")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
