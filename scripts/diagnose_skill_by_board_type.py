"""DIAGNOSTIC: the strategic metrics (TV, PA, EP) recomputed BY BOARD TYPE.

Chapter 4 defines skill on the **control** boards, and that definition is not touched here:
``analysis/skill_metrics.py`` and ``scripts/run_skill_metrics.py`` remain the reference
implementation and the reference numbers. This script is an *auxiliary* analysis that answers one
question the reference report cannot: **are the probe boards intrinsically harder than the control
ones?** If they are, the skill homogeneity established on control does not transfer for free to the
rest of the chapter, and that caveat belongs in the write-up.

Nothing is redefined. The estimators are imported from ``analysis.skill_metrics`` (``win_rate``,
``guess_accuracy``, ``clue_efficiency``) and the resampling from ``analysis.inference``; the only
new code is the stratification and the paired contrast wiring. The regime is identical to the
reference script:

  * completed games (``game_status = 'completed'``) of one run;
  * normal-phase turns only (sudden death carries no clues, so EP is undefined there and PA is
    restricted the same way to keep both metrics in one regime);
  * the same numerators and denominators - PA over play-kind items that resolved into a reveal,
    EP over clue-turns with the agents revealed on that turn;
  * cluster bootstrap with games as clusters, B=2000, seed 2026, percentile CI (2.5, 97.5).

The headline number is ``probe_minus_control``: a per-model paired contrast accumulated **within**
each bootstrap replicate, so both arms come from the same draw of clusters and the difference keeps
their correlation. Differencing two independently-computed intervals would throw that away.

Nothing is imputed. A stratum with no observations is reported empty (null point, null interval,
zero counts) rather than filled in, and thin cells carry ``reliable: false`` from the bootstrap's own
usable-replicate rule. No threshold or inference parameter is changed.

Strictly read-only: one session, compute, print, no writes.

Environment: DATABASE_URL must be in the process environment. Run from the REPO ROOT.

Examples:
    set -a; source .env; set +a
    DATABASE_URL="postgresql+psycopg2://${POSTGRES_USER}:${POSTGRES_PASSWORD}@localhost:5432/${POSTGRES_DB}"
    python scripts/diagnose_skill_by_board_type.py
    python scripts/diagnose_skill_by_board_type.py --json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.analysis.inference import (
    DEFAULT_REPLICATES,
    DEFAULT_SEED,
    CellEstimate,
    ContrastEstimate,
    cluster_bootstrap,
    minimum_detectable_effect,
)

# The estimators and the bootstrap adapter are imported, never reimplemented: this diagnostic must
# produce the same arithmetic as the reference script, stratified differently. ``_bootstrap_cell``
# and ``_by_game`` are module-private helpers of skill_metrics, reused here deliberately so that a
# change to the reference adapter propagates to this diagnostic instead of silently diverging.
from backend.app.analysis.skill_metrics import (  # noqa: PLC2701
    AGENT_ROLE,
    COMPLETED,
    CONTROL_BOARD_TYPE,
    DEFAULT_MASTER_SEED,
    NORMAL_PHASE,
    PLAY_KIND,
    ClueGiven,
    GameRecord,
    PlayedCard,
    _bootstrap_cell,
    _by_game,
    clue_efficiency,
    guess_accuracy,
    win_rate,
)
from backend.app.db.models import (
    BoardModel,
    ClueModel,
    GameModel,
    GameSeatModel,
    GuessProposalItemModel,
    GuessProposalModel,
    RevealEventModel,
    RunModel,
    TurnModel,
)
from backend.app.db.session import session_scope

logger = logging.getLogger("diagnose_skill_by_board_type")

PROBE_BOARD_TYPE = "probe"
BOARD_TYPES: tuple[str, ...] = (CONTROL_BOARD_TYPE, PROBE_BOARD_TYPE)

_RULE = "=" * 100


# ------------------------------------------------------------------------------------------------
# Loading - the reference regime, one filter wider (both board types instead of control only)
# ------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class BoardStratum:
    """Which board a game was played on. ``specification`` is null for control boards."""

    board_type: str
    specification: str | None


@dataclass(frozen=True)
class StratifiedSkillData:
    games: list[GameRecord]
    played: list[PlayedCard]
    clues: list[ClueGiven]
    strata: Mapping[str, BoardStratum]


def load_stratified_skill_data(session: Session, *, master_seed: int) -> StratifiedSkillData:
    """Load the completed games of one run across BOTH board types, tagged by board stratum.

    Mirrors ``skill_metrics.load_skill_data`` exactly, minus its ``board.type = 'control'``
    restriction and plus the per-game board stratum needed to split the results afterwards.
    """
    game_rows = session.execute(
        select(GameModel.id, GameModel.result,
               BoardModel.type, BoardModel.specification)
        .join(RunModel, RunModel.id == GameModel.run_id)
        .join(BoardModel, BoardModel.board_id == GameModel.board_id)
        .where(
            RunModel.master_seed == master_seed,
            GameModel.game_status == COMPLETED,
            BoardModel.type.in_(BOARD_TYPES),
        )
    ).all()
    if not game_rows:
        return StratifiedSkillData(games=[], played=[], clues=[], strata={})

    game_ids = [row[0] for row in game_rows]
    strata = {
        game_id: BoardStratum(board_type=board_type,
                              specification=specification)
        for game_id, _result, board_type, specification in game_rows
    }

    seats_by_game: dict[str, dict[int, str]] = defaultdict(dict)
    for game_id, seat_index, model_ref in session.execute(
        select(GameSeatModel.game_id, GameSeatModel.seat_index, GameSeatModel.model_ref).where(
            GameSeatModel.game_id.in_(game_ids)
        )
    ).all():
        seats_by_game[game_id][seat_index] = model_ref

    games = [
        GameRecord(game_id=game_id, result=result,
                   seat_models=dict(seats_by_game.get(game_id, {})))
        for game_id, result, _board_type, _specification in game_rows
    ]

    # PA: play-kind items that resolved into a reveal, on normal turns.
    played = [
        PlayedCard(game_id=game_id, model_ref=model_ref,
                   result_role=result_role)
        for game_id, model_ref, result_role in session.execute(
            select(TurnModel.game_id, GameSeatModel.model_ref,
                   RevealEventModel.result_role)
            .join(GuessProposalModel, GuessProposalModel.turn_id == TurnModel.id)
            .join(
                GuessProposalItemModel,
                GuessProposalItemModel.guess_proposal_id == GuessProposalModel.id,
            )
            .join(
                RevealEventModel,
                RevealEventModel.id == GuessProposalItemModel.reveal_event_id,
            )
            .join(
                GameSeatModel,
                (GameSeatModel.game_id == TurnModel.game_id)
                & (GameSeatModel.seat_index == GuessProposalModel.guesser_seat),
            )
            .where(
                TurnModel.game_id.in_(game_ids),
                TurnModel.phase == NORMAL_PHASE,
                GuessProposalModel.kind == PLAY_KIND,
            )
        ).all()
    ]

    # EP: one row per clue, carrying the agents revealed on that clue's turn.
    agents_by_turn: dict[int, int] = defaultdict(int)
    for turn_id, _reveal_id in session.execute(
        select(RevealEventModel.turn_id, RevealEventModel.id)
        .join(TurnModel, TurnModel.id == RevealEventModel.turn_id)
        .where(
            TurnModel.game_id.in_(game_ids),
            TurnModel.phase == NORMAL_PHASE,
            RevealEventModel.result_role == AGENT_ROLE,
        )
    ).all():
        agents_by_turn[turn_id] += 1

    clues = [
        ClueGiven(
            game_id=game_id,
            model_ref=model_ref,
            agents_revealed=agents_by_turn.get(turn_id, 0),
        )
        for turn_id, game_id, model_ref in session.execute(
            select(TurnModel.id, TurnModel.game_id, GameSeatModel.model_ref)
            .join(ClueModel, ClueModel.turn_id == TurnModel.id)
            .join(
                GameSeatModel,
                (GameSeatModel.game_id == TurnModel.game_id)
                & (GameSeatModel.seat_index == TurnModel.clue_giver_seat),
            )
            .where(TurnModel.game_id.in_(game_ids), TurnModel.phase == NORMAL_PHASE)
        ).all()
    ]

    return StratifiedSkillData(games=games, played=played, clues=clues, strata=strata)


# Output shapes - a cell always carries the raw counts behind it, so an empty cell reads as empty
@dataclass(frozen=True)
class TvCell:
    point: float | None
    ci_low: float | None
    ci_high: float | None
    standard_error: float | None
    mde: float | None
    reliable: bool
    n_games: int


@dataclass(frozen=True)
class PaCell:
    point: float | None
    ci_low: float | None
    ci_high: float | None
    standard_error: float | None
    mde: float | None
    reliable: bool
    n_played_cards: int
    n_agent_cards: int
    n_games: int


@dataclass(frozen=True)
class EpCell:
    point: float | None
    ci_low: float | None
    ci_high: float | None
    standard_error: float | None
    mde: float | None
    reliable: bool
    n_clues: int
    n_agents_revealed_on_own_clues: int
    n_games: int


@dataclass(frozen=True)
class ContrastCell:
    """probe minus control for one model, paired inside each replicate."""

    point: float | None
    ci_low: float | None
    ci_high: float | None
    standard_error: float | None
    mde: float | None
    reliable: bool
    n_denominator_probe: int
    n_denominator_control: int
    n_games_probe: int
    n_games_control: int


@dataclass(frozen=True)
class BoardTypeReport:
    master_seed: int
    n_replicates: int
    seed: int
    note: str
    models: list[str]
    board_types: list[str]
    specifications: list[str]
    n_games_by_board_type: Mapping[str, int]
    tv_by_board_type: Mapping[str, Mapping[str, Mapping[str, TvCell]]]
    pa_by_board_type: Mapping[str, Mapping[str, PaCell]]
    ep_by_board_type: Mapping[str, Mapping[str, EpCell]]
    probe_minus_control: Mapping[str, Mapping[str, ContrastCell]]
    pa_by_specification: Mapping[str, Mapping[str, PaCell]]
    ep_by_specification: Mapping[str, Mapping[str, EpCell]]


DIAGNOSTIC_NOTE = (
    "Auxiliary diagnostic. The chapter-4 skill metrics (TV, PA, EP) are DEFINED on the control "
    "boards and are produced by scripts/run_skill_metrics.py; those definitions and numbers are "
    "unchanged. The probe-board figures here exist only to test whether the probe boards are "
    "intrinsically harder, i.e. whether skill homogeneity established on control transfers to the "
    "rest of the chapter."
)


def _tv_cell(cell: CellEstimate | None, *, n_games: int) -> TvCell:
    if cell is None:
        return TvCell(None, None, None, None, None, False, n_games)
    return TvCell(
        point=cell.point,
        ci_low=cell.ci_low,
        ci_high=cell.ci_high,
        standard_error=cell.standard_error,
        mde=cell.mde,
        reliable=cell.reliable,
        n_games=n_games,
    )


def _pa_cell(cell: CellEstimate | None, cards: Sequence[PlayedCard]) -> PaCell:
    counts = {
        "n_played_cards": len(cards),
        "n_agent_cards": sum(1 for card in cards if card.is_agent),
        "n_games": len({card.game_id for card in cards}),
    }
    if cell is None:
        return PaCell(None, None, None, None, None, False, **counts)
    return PaCell(
        point=cell.point,
        ci_low=cell.ci_low,
        ci_high=cell.ci_high,
        standard_error=cell.standard_error,
        mde=cell.mde,
        reliable=cell.reliable,
        **counts,
    )


def _ep_cell(cell: CellEstimate | None, clues: Sequence[ClueGiven]) -> EpCell:
    counts = {
        "n_clues": len(clues),
        "n_agents_revealed_on_own_clues": sum(clue.agents_revealed for clue in clues),
        "n_games": len({clue.game_id for clue in clues}),
    }
    if cell is None:
        return EpCell(None, None, None, None, None, False, **counts)
    return EpCell(
        point=cell.point,
        ci_low=cell.ci_low,
        ci_high=cell.ci_high,
        standard_error=cell.standard_error,
        mde=cell.mde,
        reliable=cell.reliable,
        **counts,
    )


# Estimation
def _stratum_cell(
    records: Sequence[object],
    statistic: Callable[[Sequence[object]], float | None],
    *,
    cell: str,
    n_replicates: int,
    seed: int,
) -> CellEstimate | None:
    """One metric on one stratum: its own games are its cluster universe."""
    game_ids = sorted({record.game_id for record in records}
                      )  # type: ignore[attr-defined]
    return _bootstrap_cell(
        records, statistic, game_ids, cell=cell, n_replicates=n_replicates, seed=seed
    )


def _paired_contrast(
    records: Sequence[object],
    strata: Mapping[str, BoardStratum],
    statistic: Callable[[Sequence[object]], float | None],
    *,
    prefix: str,
    n_replicates: int,
    seed: int,
) -> ContrastEstimate | None:
    """probe - control for one model, both arms recomputed from the SAME draw of clusters.

    The cluster universe is the union of the model's games across both board types; a game belongs
    to exactly one stratum, so each draw splits cleanly into the two arms. Accumulating the
    difference inside the replicate is what keeps the correlation between the arms - differencing
    two separately-bootstrapped intervals would inflate the width.
    """
    cluster_ids = sorted({record.game_id for record in records}
                         )  # type: ignore[attr-defined]
    if not cluster_ids:
        return None

    grouped = _by_game(records)
    cell_probe = f"{prefix}_probe"
    cell_control = f"{prefix}_control"

    def estimator(subset: Sequence[str]) -> dict[str, float]:
        drawn: dict[str, list[object]] = {
            CONTROL_BOARD_TYPE: [], PROBE_BOARD_TYPE: []}
        for game_id in subset:
            stratum = strata.get(game_id)
            if stratum is None:
                continue
            bucket = drawn.get(stratum.board_type)
            if bucket is None:
                continue
            bucket.extend(grouped.get(game_id, ()))
        out: dict[str, float] = {}
        for board_type, name in ((PROBE_BOARD_TYPE, cell_probe), (CONTROL_BOARD_TYPE, cell_control)):
            value = statistic(drawn[board_type])
            if value is not None:
                out[name] = value
        return out

    result = cluster_bootstrap(
        cluster_ids,
        estimator,
        n_replicates=n_replicates,
        seed=seed,
        contrasts=[("probe_minus_control", cell_probe, cell_control)],
    )
    return result.contrasts.get("probe_minus_control")


def _contrast_cell(
    contrast: ContrastEstimate | None,
    probe_records: Sequence[object],
    control_records: Sequence[object],
) -> ContrastCell:
    counts = {
        "n_denominator_probe": len(probe_records),
        "n_denominator_control": len(control_records),
        # type: ignore[attr-defined]
        "n_games_probe": len({r.game_id for r in probe_records}),
        # type: ignore[attr-defined]
        "n_games_control": len({r.game_id for r in control_records}),
    }
    if contrast is None:
        return ContrastCell(None, None, None, None, None, False, **counts)
    return ContrastCell(
        point=contrast.point,
        ci_low=contrast.ci_low,
        ci_high=contrast.ci_high,
        standard_error=contrast.standard_error,
        mde=(
            minimum_detectable_effect(contrast.standard_error)
            if contrast.standard_error is not None
            else None
        ),
        reliable=contrast.reliable,
        **counts,
    )


def _split_by(
    records: Sequence[object],
    strata: Mapping[str, BoardStratum],
    key: Callable[[BoardStratum], str | None],
) -> dict[str, list[object]]:
    grouped: dict[str, list[object]] = defaultdict(list)
    for record in records:
        stratum = strata.get(record.game_id)  # type: ignore[attr-defined]
        if stratum is None:
            continue
        bucket = key(stratum)
        if bucket is None:
            continue
        grouped[bucket].append(record)
    return grouped


def compute_board_type_diagnostic(
    session: Session,
    *,
    master_seed: int = DEFAULT_MASTER_SEED,
    n_replicates: int = DEFAULT_REPLICATES,
    seed: int = DEFAULT_SEED,
) -> BoardTypeReport:
    """TV, PA and EP by board type, plus the paired probe-minus-control contrast. Read-only."""
    data = load_stratified_skill_data(session, master_seed=master_seed)
    games, played, clues, strata = data.games, data.played, data.clues, data.strata

    models = sorted(
        {model for game in games for model in game.seat_models.values()})
    specifications = sorted(
        {
            stratum.specification
            for stratum in strata.values()
            if stratum.board_type == PROBE_BOARD_TYPE and stratum.specification is not None
        }
    )

    games_by_type = _split_by(games, strata, lambda s: s.board_type)
    played_by_type = _split_by(played, strata, lambda s: s.board_type)
    clues_by_type = _split_by(clues, strata, lambda s: s.board_type)

    def _probe_spec(stratum: BoardStratum) -> str | None:
        if stratum.board_type != PROBE_BOARD_TYPE:
            return None
        return stratum.specification

    played_by_spec = _split_by(played, strata, _probe_spec)
    clues_by_spec = _split_by(clues, strata, _probe_spec)

    # --- TV, per board type: by pairing (the honest unit) and by model (marginal over partners) ---
    tv_by_board_type: dict[str, dict[str, dict[str, TvCell]]] = {}
    for board_type in BOARD_TYPES:
        type_games: list[GameRecord] = list(
            games_by_type.get(board_type, []))  # type: ignore[arg-type]

        by_pairing: dict[str, TvCell] = {}
        pairings: dict[str, list[GameRecord]] = defaultdict(list)
        for game in type_games:
            pairings[game.pairing_label].append(game)
        for label, pairing_games in sorted(pairings.items()):
            cell = _stratum_cell(
                pairing_games,  # type: ignore[arg-type]
                win_rate,  # type: ignore[arg-type]
                cell="tv",
                n_replicates=n_replicates,
                seed=seed,
            )
            by_pairing[label] = _tv_cell(cell, n_games=len(pairing_games))

        by_model: dict[str, TvCell] = {}
        for model_ref in models:
            model_games = [
                g for g in type_games if model_ref in g.seat_models.values()]
            cell = _stratum_cell(
                model_games,  # type: ignore[arg-type]
                win_rate,  # type: ignore[arg-type]
                cell="tv",
                n_replicates=n_replicates,
                seed=seed,
            )
            by_model[model_ref] = _tv_cell(cell, n_games=len(model_games))

        tv_by_board_type[board_type] = {
            "by_pairing": by_pairing, "by_model": by_model}

    # --- PA and EP, per board type x model ---
    pa_by_board_type: dict[str, dict[str, PaCell]] = {}
    ep_by_board_type: dict[str, dict[str, EpCell]] = {}
    for board_type in BOARD_TYPES:
        type_played: list[PlayedCard] = list(
            played_by_type.get(board_type, []))  # type: ignore[arg-type]
        type_clues: list[ClueGiven] = list(clues_by_type.get(
            board_type, []))  # type: ignore[arg-type]

        pa_cells: dict[str, PaCell] = {}
        ep_cells: dict[str, EpCell] = {}
        for model_ref in models:
            cards = [card for card in type_played if card.model_ref == model_ref]
            pa_cells[model_ref] = _pa_cell(
                _stratum_cell(
                    cards,  # type: ignore[arg-type]
                    guess_accuracy,  # type: ignore[arg-type]
                    cell="pa",
                    n_replicates=n_replicates,
                    seed=seed,
                ),
                cards,
            )
            model_clues = [
                clue for clue in type_clues if clue.model_ref == model_ref]
            ep_cells[model_ref] = _ep_cell(
                _stratum_cell(
                    model_clues,  # type: ignore[arg-type]
                    clue_efficiency,  # type: ignore[arg-type]
                    cell="ep",
                    n_replicates=n_replicates,
                    seed=seed,
                ),
                model_clues,
            )
        pa_by_board_type[board_type] = pa_cells
        ep_by_board_type[board_type] = ep_cells

    # --- the headline: probe - control, paired within each replicate ---
    pa_contrasts: dict[str, ContrastCell] = {}
    ep_contrasts: dict[str, ContrastCell] = {}
    for model_ref in models:
        model_cards = [card for card in played if card.model_ref == model_ref]
        pa_contrasts[model_ref] = _contrast_cell(
            _paired_contrast(
                model_cards,  # type: ignore[arg-type]
                strata,
                guess_accuracy,  # type: ignore[arg-type]
                prefix="pa",
                n_replicates=n_replicates,
                seed=seed,
            ),
            [c for c in model_cards if strata[c.game_id].board_type == PROBE_BOARD_TYPE],
            [c for c in model_cards if strata[c.game_id].board_type == CONTROL_BOARD_TYPE],
        )

        model_clues = [clue for clue in clues if clue.model_ref == model_ref]
        ep_contrasts[model_ref] = _contrast_cell(
            _paired_contrast(
                model_clues,  # type: ignore[arg-type]
                strata,
                clue_efficiency,  # type: ignore[arg-type]
                prefix="ep",
                n_replicates=n_replicates,
                seed=seed,
            ),
            [c for c in model_clues if strata[c.game_id].board_type == PROBE_BOARD_TYPE],
            [c for c in model_clues if strata[c.game_id].board_type == CONTROL_BOARD_TYPE],
        )

    # --- probe broken down by specification ---
    pa_by_specification: dict[str, dict[str, PaCell]] = {}
    ep_by_specification: dict[str, dict[str, EpCell]] = {}
    for specification in specifications:
        spec_played: list[PlayedCard] = list(played_by_spec.get(
            specification, []))  # type: ignore[arg-type]
        spec_clues: list[ClueGiven] = list(clues_by_spec.get(
            specification, []))  # type: ignore[arg-type]

        pa_cells = {}
        ep_cells = {}
        for model_ref in models:
            cards = [card for card in spec_played if card.model_ref == model_ref]
            pa_cells[model_ref] = _pa_cell(
                _stratum_cell(
                    cards,  # type: ignore[arg-type]
                    guess_accuracy,  # type: ignore[arg-type]
                    cell="pa",
                    n_replicates=n_replicates,
                    seed=seed,
                ),
                cards,
            )
            model_clues = [
                clue for clue in spec_clues if clue.model_ref == model_ref]
            ep_cells[model_ref] = _ep_cell(
                _stratum_cell(
                    model_clues,  # type: ignore[arg-type]
                    clue_efficiency,  # type: ignore[arg-type]
                    cell="ep",
                    n_replicates=n_replicates,
                    seed=seed,
                ),
                model_clues,
            )
        pa_by_specification[specification] = pa_cells
        ep_by_specification[specification] = ep_cells

    logger.info(
        "board-type diagnostic: %d games (%s), %d played cards, %d clues",
        len(games),
        ", ".join(
            f"{bt}={len(games_by_type.get(bt, []))}" for bt in BOARD_TYPES),
        len(played),
        len(clues),
    )

    return BoardTypeReport(
        master_seed=master_seed,
        n_replicates=n_replicates,
        seed=seed,
        note=DIAGNOSTIC_NOTE,
        models=models,
        board_types=list(BOARD_TYPES),
        specifications=specifications,
        n_games_by_board_type={
            bt: len(games_by_type.get(bt, [])) for bt in BOARD_TYPES},
        tv_by_board_type=tv_by_board_type,
        pa_by_board_type=pa_by_board_type,
        ep_by_board_type=ep_by_board_type,
        probe_minus_control={"pa": pa_contrasts, "ep": ep_contrasts},
        pa_by_specification=pa_by_specification,
        ep_by_specification=ep_by_specification,
    )


# Plain-text report
def _fmt(value: float | None, places: int = 4) -> str:
    return "-" if value is None else f"{value:.{places}f}"


def _interval(cell: TvCell | PaCell | EpCell | ContrastCell) -> str:
    if cell.ci_low is None or cell.ci_high is None:
        return "-"
    return f"[{_fmt(cell.ci_low)}, {_fmt(cell.ci_high)}]"


def _flag(cell: TvCell | PaCell | EpCell | ContrastCell) -> str:
    return "" if cell.reliable else "  (unreliable)"


def _print_tv(report: BoardTypeReport) -> None:
    for board_type in report.board_types:
        block = report.tv_by_board_type.get(board_type, {})
        print()
        print(_RULE)
        print(f"TV - win rate per PAIRING  [{board_type.upper()} boards]")
        print(_RULE)
        print(f"  {'pairing':<44} {'TV':>8} {'95% CI':>21} {'games':>6}")
        for label, cell in block.get("by_pairing", {}).items():
            print(
                f"  {label:<44} {_fmt(cell.point):>8} {_interval(cell):>21} "
                f"{cell.n_games:>6}{_flag(cell)}"
            )

        print()
        print(_RULE)
        print(
            f"TV - win rate per MODEL  (marginal over partners)  [{board_type.upper()} boards]")
        print(_RULE)
        print("  CAVEAT: cooperative game. A per-model win rate conflates the model's own skill")
        print("          with its partners' - it is not an attribution to the model.")
        print(f"\n  {'model':<44} {'TV':>8} {'95% CI':>21} {'games':>6}")
        for model_ref, cell in block.get("by_model", {}).items():
            print(
                f"  {model_ref:<44} {_fmt(cell.point):>8} {_interval(cell):>21} "
                f"{cell.n_games:>6}{_flag(cell)}"
            )


def _print_pa_table(title: str, cells: Mapping[str, PaCell]) -> None:
    print()
    print(_RULE)
    print(title)
    print(_RULE)
    print(f"  {'model':<30} {'PA':>8} {'95% CI':>21} {'SE':>8} {'MDE':>8} {'agents':>8} {'played':>8}")
    for model_ref, cell in cells.items():
        print(
            f"  {model_ref:<30} {_fmt(cell.point):>8} {_interval(cell):>21} "
            f"{_fmt(cell.standard_error):>8} {_fmt(cell.mde):>8} "
            f"{cell.n_agent_cards:>8} {cell.n_played_cards:>8}{_flag(cell)}"
        )


def _print_ep_table(title: str, cells: Mapping[str, EpCell]) -> None:
    print()
    print(_RULE)
    print(title)
    print(_RULE)
    print(f"  {'model':<30} {'EP':>8} {'95% CI':>21} {'SE':>8} {'MDE':>8} {'agents':>8} {'clues':>8}")
    for model_ref, cell in cells.items():
        print(
            f"  {model_ref:<30} {_fmt(cell.point):>8} {_interval(cell):>21} "
            f"{_fmt(cell.standard_error):>8} {_fmt(cell.mde):>8} "
            f"{cell.n_agents_revealed_on_own_clues:>8} {cell.n_clues:>8}{_flag(cell)}"
        )


def _print_contrasts(report: BoardTypeReport) -> None:
    labels = {
        "pa": "PA  probe - control  (guess accuracy; negative = probe boards harder)",
        "ep": "EP  probe - control  (clue efficiency; negative = probe boards harder)",
    }
    for metric, title in labels.items():
        print()
        print(_RULE)
        print(f"PAIRED CONTRAST - {title}")
        print(_RULE)
        print("  Both arms recomputed from the same bootstrap draw, so the difference keeps their")
        print("  correlation. An interval that straddles 0 means no detected difference; read it")
        print("  against the MDE, which says what this design could have found.")
        print(
            f"\n  {'model':<30} {'diff':>9} {'95% CI':>21} {'SE':>8} {'MDE':>8} "
            f"{'n probe':>9} {'n control':>10}"
        )
        for model_ref, cell in report.probe_minus_control.get(metric, {}).items():
            print(
                f"  {model_ref:<30} {_fmt(cell.point):>9} {_interval(cell):>21} "
                f"{_fmt(cell.standard_error):>8} {_fmt(cell.mde):>8} "
                f"{cell.n_denominator_probe:>9} {cell.n_denominator_control:>10}{_flag(cell)}"
            )


def print_report(report: BoardTypeReport) -> None:
    print()
    print(_RULE)
    print("DIAGNOSTIC - strategic metrics BY BOARD TYPE  (auxiliary, not a redefinition)")
    print(_RULE)
    print("Chapter 4 defines TV, PA and EP on the CONTROL boards; those definitions and the numbers")
    print("in run_skill_metrics.py are unchanged. This report recomputes the same metrics on the")
    print("probe boards as well, to test whether the probe boards are intrinsically harder - if")
    print("they are, skill homogeneity established on control does not transfer automatically.")
    print()
    print(f"run seed   : {report.master_seed}")
    for board_type in report.board_types:
        print(
            f"coverage   : {report.n_games_by_board_type.get(board_type, 0):>4} "
            f"completed {board_type} games"
        )
    print(
        f"bootstrap  : B={report.n_replicates}, seed={report.seed}, games as clusters, "
        "percentile CI (2.5, 97.5)"
    )
    print("scope      : normal-phase turns (sudden death carries no clues, so EP is undefined there")
    print("             and PA is restricted the same way to keep both metrics in one regime)")
    print("empty cells: reported empty, never imputed; thin cells are flagged (unreliable)")

    _print_tv(report)
    for board_type in report.board_types:
        _print_pa_table(
            f"PA - guess accuracy as guesser  [{board_type.upper()} boards]",
            report.pa_by_board_type.get(board_type, {}),
        )
    for board_type in report.board_types:
        _print_ep_table(
            f"EP - clue efficiency as clue-giver  [{board_type.upper()} boards]",
            report.ep_by_board_type.get(board_type, {}),
        )
    _print_contrasts(report)

    for specification in report.specifications:
        _print_pa_table(
            f"PA - probe boards, specification = {specification}",
            report.pa_by_specification.get(specification, {}),
        )
    for specification in report.specifications:
        _print_ep_table(
            f"EP - probe boards, specification = {specification}",
            report.ep_by_specification.get(specification, {}),
        )
    print()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "DIAGNOSTIC: TV, PA and EP by board type, with a paired probe-minus-control contrast. "
            "Auxiliary to run_skill_metrics.py, which remains the chapter-4 definition. Read-only."
        )
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
        report = compute_board_type_diagnostic(
            session,
            master_seed=args.master_seed,
            n_replicates=args.replicates,
            seed=args.seed,
        )

    if not report.models:
        print(
            f"no completed games for master_seed={args.master_seed}; nothing to report",
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
