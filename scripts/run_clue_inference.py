"""CLI applying the cluster-bootstrap inference layer to the clue-giver metrics IAE and TAC/TAI.

This adds intervals, so the clue-giver metrics report on the same footing as CIT, the skill metrics 
and conc-SD. It changes no estimator: every point estimate printed here is produced by calling 
``compute_iae`` / ``compute_tac_tai`` on the full data, exactly as ``run_clue_metrics.py`` does. The
replicates supply dispersion only.

Two things this can settle that the point estimates could not:

  * how much of IAE's distance from indifference survives the small decidable denominators - roughly
    7-13 resolved dilemmas per model, so the interval is expected to be wide and to say so;
  * ``gap(b1) - gap(b3)``, the monotonicity hypothesis of section 4.5.1. Band 1 is the least similar
    tercile, so "the gap widens as thematic proximity falls" predicts a positive contrast;
    found the opposite ordering for every model, and this contrast is what says whether that inversion
    is distinguishable from noise. It is accumulated inside each replicate from the same draw, so its
    interval carries the correlation between the two bands.

Games are the resampling unit, and both metrics share one cluster set per model: the games in which
that model gave clues. The tercile cuts are computed once over the full eligible-pair distribution and
frozen into the estimator closures; the bootstrap has no path to recompute them.

Reference points, neither of which is a hypothesis test. For the TAC/TAI gaps the null is 0 - equal
grouping rates for congruent and incongruent pairs. For IAE the reference is 0.5, standing for
indifference between the stereotypical and the neutral bridge among decidable observations; it is
printed so the MDE has a scale to be read against, NOT as a null being tested. The rate cells (TAC and
TAI themselves) have no meaningful reference point, so no exclusion flag is printed for them.

Strictly read-only: one session, compute, print, no writes.

Environment: DATABASE_URL must be in the process environment. Run from the REPO ROOT.

Examples:
    set -a; source .env; set +a
    DATABASE_URL="postgresql+psycopg2://${POSTGRES_USER}:${POSTGRES_PASSWORD}@localhost:5432/${POSTGRES_DB}"
    python scripts/run_clue_inference.py
    python scripts/run_clue_inference.py --replicates 500 --json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from backend.app.analysis.clue_metrics import (
    DEFAULT_FRAME_ID,
    DEFAULT_MASTER_SEED,
    GAP_MONOTONICITY_CONTRAST,
    GAP_NULL,
    IAE_REFERENCE,
    STRATUM_POOLED,
    FrameGeometry,
    assign_global_bands,
    build_clue_estimators,
    collect_dilemma_observations,
    collect_pair_observations,
    compute_iae,
    load_boards,
    load_turn_states,
)
from backend.app.analysis.inference import (
    DEFAULT_REPLICATES,
    DEFAULT_SEED,
    BootstrapResult,
    cluster_bootstrap,
)
from backend.app.db.session import session_scope

logger = logging.getLogger("run_clue_inference")

_RULE = "=" * 104

BAND_ORDER = ("all", "b1", "b2", "b3")


def _fmt(value: float | None, places: int = 4) -> str:
    return "-" if value is None else f"{value:.{places}f}"


def _interval(cell) -> str:
    return f"[{_fmt(cell.ci_low)}, {_fmt(cell.ci_high)}]"


def _print_iae(models, iae_results, decidable) -> None:
    print()
    print(_RULE)
    print("IAE with cluster-bootstrap intervals")
    print(_RULE)
    print("  Reference 0.5 = indifference between the stereotypical and the neutral bridge.")
    print("  It scales the MDE; it is NOT a null hypothesis being tested.")
    print(
        f"  {'model':<26} {'IAE':>8} {'95% CI':>19} {'SE':>8} {'MDE':>8} "
        f"{'decidable':>10} {'games':>6} {'dropped':>8}  {'excl 0.5':>8}"
    )
    for model_ref in models:
        result = iae_results[model_ref]
        cell = result.cells.get("iae")
        if cell is None:
            print(
                f"  {model_ref:<26} {'-':>8}   (no decidable dilemma in the full data)")
            continue
        excludes = {True: "yes", False: "no", None: "-"}[cell.excludes_null]
        flag = "" if cell.reliable else "  UNRELIABLE"
        print(
            f"  {model_ref:<26} {_fmt(cell.point):>8} {_interval(cell):>19} "
            f"{_fmt(cell.standard_error):>8} {_fmt(cell.mde):>8} {decidable[model_ref]:>10} "
            f"{result.n_clusters:>6} {cell.n_dropped:>8}  {excludes:>8}{flag}"
        )


def _print_tac_tai(model_ref: str, result: BootstrapResult) -> None:
    print(f"\n  {model_ref}   ({result.n_clusters} games as clusters, "
          f"{result.n_replicates} replicates, {result.elapsed_seconds:.1f}s)")
    print(
        f"    {'cell':<9} {'value':>8} {'95% CI':>19} {'SE':>8} {'MDE':>8} "
        f"{'dropped':>8}  {'excl 0':>7}"
    )
    for suffix in BAND_ORDER:
        for name in ("tac", "tai", "gap"):
            cell = result.cells.get(f"{name}_{suffix}")
            if cell is None:
                print(
                    f"    {name + '_' + suffix:<9} {'-':>8}   (no eligible pair in the full data)")
                continue
            # Only the gap has a meaningful reference point; a rate's distance from 0 is not a
            # finding, so the flag is suppressed rather than printed as if it were one.
            excludes = (
                {True: "yes", False: "no",
                    None: "-"}[cell.excludes_null] if name == "gap" else ""
            )
            flag = "" if cell.reliable else "  UNRELIABLE"
            print(
                f"    {name + '_' + suffix:<9} {_fmt(cell.point):>8} {_interval(cell):>19} "
                f"{_fmt(cell.standard_error):>8} {_fmt(cell.mde):>8} {cell.n_dropped:>8}  "
                f"{excludes:>7}{flag}"
            )

    contrast = result.contrasts.get(GAP_MONOTONICITY_CONTRAST[0])
    if contrast is not None:
        excludes = {True: "yes", False: "no",
                    None: "-"}[contrast.excludes_zero]
        flag = "" if contrast.reliable else "  UNRELIABLE"
        print(
            f"    {GAP_MONOTONICITY_CONTRAST[0]:<18} diff {_fmt(contrast.point):>8} "
            f"{_interval(contrast):>19} SE {_fmt(contrast.standard_error):>8} "
            f"dropped {contrast.n_dropped:>5}  excl0 {excludes}{flag}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cluster-bootstrap confidence intervals for IAE and TAC/TAI. Read-only."
    )
    parser.add_argument("--frame-id", default=DEFAULT_FRAME_ID)
    parser.add_argument("--master-seed", type=int, default=DEFAULT_MASTER_SEED)
    parser.add_argument(
        "--replicates", type=int, default=DEFAULT_REPLICATES, help="bootstrap replicates B"
    )
    parser.add_argument("--seed", type=int,
                        default=DEFAULT_SEED, help="bootstrap RNG seed")
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
        boards = load_boards(session)
        turn_states = load_turn_states(
            session, master_seed=args.master_seed, boards=boards)
        observations = collect_dilemma_observations(turn_states, boards)
        raw_pairs = collect_pair_observations(turn_states, boards, geometry)

    # Cuts are computed ONCE here, over the full eligible-pair distribution, and are baked into the
    # banded pairs the estimators close over. No replicate can move them.
    pairs, cuts = assign_global_bands(raw_pairs)
    if not observations and not pairs:
        print("no clue-giver observations; nothing to bootstrap", file=sys.stderr)
        return 1

    models = sorted(
        {obs.model_ref for obs in observations} | {
            pair.model_ref for pair in pairs}
    )

    # The decidable n printed next to each IAE interval, taken from the full data.
    decidable = {
        row.model_ref: row.n_stereotypical + row.n_neutral
        for row in compute_iae(observations)
        if row.stratum == STRATUM_POOLED
    }

    iae_results: dict[str, BootstrapResult] = {}
    tac_tai_results: dict[str, BootstrapResult] = {}
    for model_ref in models:
        iae_estimator, tac_tai_estimator, game_ids = build_clue_estimators(
            observations, pairs, model_ref=model_ref
        )
        if not game_ids:
            continue
        iae_results[model_ref] = cluster_bootstrap(
            game_ids,
            iae_estimator,
            n_replicates=args.replicates,
            seed=args.seed,
            null_value=IAE_REFERENCE,
        )
        tac_tai_results[model_ref] = cluster_bootstrap(
            game_ids,
            tac_tai_estimator,
            n_replicates=args.replicates,
            seed=args.seed,
            contrasts=(GAP_MONOTONICITY_CONTRAST,),
            null_value=GAP_NULL,
        )

    models = [model_ref for model_ref in models if model_ref in iae_results]

    if args.json:
        print(
            json.dumps(
                {
                    "frame_id": args.frame_id,
                    "master_seed": args.master_seed,
                    "replicates": args.replicates,
                    "seed": args.seed,
                    "tercile_cuts": list(cuts) if cuts else None,
                    "models": {
                        model_ref: {
                            "n_clusters": iae_results[model_ref].n_clusters,
                            "n_decidable": decidable.get(model_ref, 0),
                            "iae": {
                                n: vars(c) for n, c in iae_results[model_ref].cells.items()
                            },
                            "tac_tai": {
                                n: vars(c) for n, c in tac_tai_results[model_ref].cells.items()
                            },
                            "contrasts": {
                                n: vars(c)
                                for n, c in tac_tai_results[model_ref].contrasts.items()
                            },
                        }
                        for model_ref in models
                    },
                },
                indent=2,
                default=str,
            )
        )
        return 0

    print()
    print(_RULE)
    print("Clue-giver metrics with cluster-bootstrap intervals  (IAE, TAC/TAI)")
    print(_RULE)
    print(f"frame       : {args.frame_id}")
    print(f"run seed    : {args.master_seed}")
    print("boards      : probe only, completed games")
    print("clusters    : games in which the model gave clues (shared by both metrics)")
    print(
        f"bootstrap   : B={args.replicates}, seed={args.seed}, percentile CI (2.5, 97.5)")
    if cuts:
        print(
            f"tercile cuts: c33={cuts[0]:.6f}  c66={cuts[1]:.6f}  (frozen before the bootstrap)")
    print("MDE         : 2.8016 x bootstrap SE, alpha 0.05, power 0.8. No p-values.")

    _print_iae(models, iae_results, decidable)

    print()
    print(_RULE)
    print("TAC / TAI by proximity band  (b1 = least similar tercile, b3 = most; null for gap = 0)")
    print(_RULE)
    for model_ref in models:
        _print_tac_tai(model_ref, tac_tai_results[model_ref])

    total = sum(
        result.elapsed_seconds
        for results in (iae_results, tac_tai_results)
        for result in results.values()
    )
    print(f"\ntotal bootstrap wall clock: {total:.1f}s")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
