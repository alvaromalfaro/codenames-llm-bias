"""Strategic (skill) metrics on the control boards: TV, PA and EP.

The bias metrics ask *how* a model plays; these ask *how well*. They are measured on the
gender-neutral control boards precisely so that competence is scored where the gender manipulation
is absent, leaving a clean skill axis to set the bias results against.

  * **TV** - win rate, read from ``game.result``. Codenames Duet is cooperative: both seats share
    one outcome, so a win belongs to the *pairing*, not to a seat. TV is therefore reported per
    pairing as the honest unit, and per model only as a marginal that explicitly conflates a model's
    skill with its partners';
  * **PA** - guess accuracy: of the cards a model actually played as guesser, the share that turned
    out to be agents. Agent-ness is taken from the engine's recorded ``reveal_event.result_role``,
    which already resolves the partner's key - the perspective is not recomputed here;
  * **EP** - clue efficiency: agents revealed per clue the model gave. A ratio in agents-per-clue,
    not a percentage, so it is not bounded above by 1.

Two structural facts about the data shape the definitions, and both are asserted by tests:

  * **sudden death carries no clues.** SD turns produce reveals but have no ``clue`` row at all, so
    counting their reveals against clue-turns would put agents in EP's numerator with nothing in its
    denominator. EP is normal-phase by construction, and PA is restricted the same way so that both
    metrics describe one regime rather than silently blending two;
  * **"played" means resolved.** A ``guess_proposal_item`` only counts when it produced a
    ``reveal_event``; items the turn never reached are excluded from PA's numerator *and*
    denominator, because nothing was learned about them.

Confidence intervals come from ``analysis.inference.cluster_bootstrap`` resampling games. No
resampling or percentile logic is defined here.

Read-only with respect to the database.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.analysis.inference import (
    DEFAULT_REPLICATES,
    DEFAULT_SEED,
    CellEstimate,
    cluster_bootstrap,
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

logger = logging.getLogger(__name__)

DEFAULT_MASTER_SEED = 2026

# The engine's outcome vocabulary. Read from the code rather than
# inferred from the data, so an all-loss run cannot be mistaken for "wins are encoded differently".
WIN_RESULTS = frozenset({"victory", "victory_sd"})

CONTROL_BOARD_TYPE = "control"
NORMAL_PHASE = "normal"
COMPLETED = "completed"
PLAY_KIND = "play"
AGENT_ROLE = "agent"


@dataclass(frozen=True)
class GameRecord:
    """One completed control game and the models that played it."""

    game_id: str
    result: str | None
    seat_models: Mapping[int, str]

    @property
    def is_win(self) -> bool:
        return self.result in WIN_RESULTS

    @property
    def pairing(self) -> tuple[str, ...]:
        """The unordered model pair. Sorted, so swapping seats cannot split one pairing in two."""
        return tuple(sorted(self.seat_models.values()))

    @property
    def pairing_label(self) -> str:
        return " + ".join(self.pairing)


@dataclass(frozen=True)
class PlayedCard:
    """One card a model played as guesser that actually resolved into a reveal."""

    game_id: str
    model_ref: str
    result_role: str

    @property
    def is_agent(self) -> bool:
        return self.result_role == AGENT_ROLE


@dataclass(frozen=True)
class ClueGiven:
    """One clue a model gave, with the number of agents revealed on that turn."""

    game_id: str
    model_ref: str
    agents_revealed: int


@dataclass(frozen=True)
class SkillDiagnostics:
    model_ref: str
    n_games: int
    n_clues: int
    n_played_cards: int
    n_agent_cards: int
    n_agents_revealed_on_own_clues: int


@dataclass(frozen=True)
class SkillReport:
    master_seed: int
    result_counts: Mapping[str, int]
    n_games: int
    tv_by_pairing: Mapping[str, CellEstimate]
    tv_by_model: Mapping[str, CellEstimate]
    pa_by_model: Mapping[str, CellEstimate]
    ep_by_model: Mapping[str, CellEstimate]
    games_per_pairing: Mapping[str, int]
    diagnostics: Sequence[SkillDiagnostics] = field(default_factory=list)


# Estimators
def win_rate(games: Sequence[GameRecord]) -> float | None:
    """Share of games won. ``None`` on an empty set rather than a misleading 0.0."""
    if not games:
        return None
    return sum(1 for game in games if game.is_win) / len(games)


def guess_accuracy(cards: Sequence[PlayedCard]) -> float | None:
    """Share of played (resolved) cards that were agents."""
    if not cards:
        return None
    return sum(1 for card in cards if card.is_agent) / len(cards)


def clue_efficiency(clues: Sequence[ClueGiven]) -> float | None:
    """Agents revealed per clue given - a ratio, not a percentage, so it may exceed 1."""
    if not clues:
        return None
    return sum(clue.agents_revealed for clue in clues) / len(clues)


# Loading
def load_skill_data(
    session: Session, *, master_seed: int
) -> tuple[list[GameRecord], list[PlayedCard], list[ClueGiven]]:
    """Load the completed CONTROL games of one run, plus their played cards and clues."""
    game_rows = session.execute(
        select(GameModel.id, GameModel.result)
        .join(RunModel, RunModel.id == GameModel.run_id)
        .join(BoardModel, BoardModel.board_id == GameModel.board_id)
        .where(
            RunModel.master_seed == master_seed,
            GameModel.game_status == COMPLETED,
            BoardModel.type == CONTROL_BOARD_TYPE,
        )
    ).all()
    if not game_rows:
        return [], [], []

    game_ids = [row[0] for row in game_rows]

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
        for game_id, result in game_rows
    ]

    # PA: play-kind items that resolved into a reveal, on normal turns. The guesser is named by
    # guess_proposal.guesser_seat, and agent-ness by the engine's own result_role.
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

    # EP: one row per clue, carrying the agents revealed on that clue's turn. Joining from `clue`
    # rather than from `turn` is what keeps clue-less sudden-death turns out of the numerator.
    agents_by_turn: dict[int, int] = defaultdict(int)
    for turn_id, count in session.execute(
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

    return games, played, clues


# Bootstrap adapters
def _by_game(records: Iterable[object]) -> dict[str, list[object]]:
    grouped: dict[str, list[object]] = defaultdict(list)
    for record in records:
        grouped[record.game_id].append(record)  # type: ignore[attr-defined]
    return grouped


def _make_estimator(
    records: Sequence[object], statistic: Callable[[Sequence[object]], float | None], cell: str
) -> Callable[[Sequence[str]], dict[str, float]]:
    """Wrap a statistic as a bootstrap estimator over game ids.

    A game drawn twice contributes its records twice; for these metrics the records are independent
    counts, so plain concatenation is the correct duplication (unlike CIT, whose pair construction
    needs turn re-keying).
    """
    grouped = _by_game(records)

    def estimator(subset: Sequence[str]) -> dict[str, float]:
        drawn: list[object] = []
        for game_id in subset:
            drawn.extend(grouped.get(game_id, ()))
        value = statistic(drawn)
        return {} if value is None else {cell: value}

    return estimator


def _bootstrap_cell(
    records: Sequence[object],
    statistic: Callable[[Sequence[object]], float | None],
    game_ids: Sequence[str],
    *,
    cell: str,
    n_replicates: int,
    seed: int,
) -> CellEstimate | None:
    if not game_ids or not records:
        return None
    estimator = _make_estimator(records, statistic, cell)
    result = cluster_bootstrap(
        list(game_ids), estimator, n_replicates=n_replicates, seed=seed
    )
    return result.cells.get(cell)


def compute_skill_metrics(
    session: Session,
    *,
    master_seed: int = DEFAULT_MASTER_SEED,
    n_replicates: int = DEFAULT_REPLICATES,
    seed: int = DEFAULT_SEED,
) -> SkillReport:
    """Compute TV, PA and EP with cluster-bootstrap intervals. Read-only."""
    games, played, clues = load_skill_data(session, master_seed=master_seed)

    result_counts: dict[str, int] = defaultdict(int)
    for game in games:
        result_counts[game.result or "<NULL>"] += 1

    models = sorted(
        {model for game in games for model in game.seat_models.values()})

    # TV per pairing: each pairing's own games are its cluster universe.
    games_by_pairing: dict[str, list[GameRecord]] = defaultdict(list)
    for game in games:
        games_by_pairing[game.pairing_label].append(game)

    tv_by_pairing: dict[str, CellEstimate] = {}
    for label, pairing_games in sorted(games_by_pairing.items()):
        cell = _bootstrap_cell(
            pairing_games,
            win_rate,  # type: ignore[arg-type]
            [game.game_id for game in pairing_games],
            cell="tv",
            n_replicates=n_replicates,
            seed=seed,
        )
        if cell is not None:
            tv_by_pairing[label] = cell

    tv_by_model: dict[str, CellEstimate] = {}
    pa_by_model: dict[str, CellEstimate] = {}
    ep_by_model: dict[str, CellEstimate] = {}
    diagnostics: list[SkillDiagnostics] = []

    played_by_model: dict[str, list[PlayedCard]] = defaultdict(list)
    for card in played:
        played_by_model[card.model_ref].append(card)
    clues_by_model: dict[str, list[ClueGiven]] = defaultdict(list)
    for clue in clues:
        clues_by_model[clue.model_ref].append(clue)

    for model_ref in models:
        # TV per model marginalises over partners: every game where the model held either seat.
        model_games = [
            game for game in games if model_ref in game.seat_models.values()]
        model_game_ids = [game.game_id for game in model_games]

        tv_cell = _bootstrap_cell(
            model_games, win_rate, model_game_ids,  # type: ignore[arg-type]
            cell="tv", n_replicates=n_replicates, seed=seed,
        )
        if tv_cell is not None:
            tv_by_model[model_ref] = tv_cell

        model_cards = played_by_model.get(model_ref, [])
        pa_cell = _bootstrap_cell(
            model_cards, guess_accuracy,  # type: ignore[arg-type]
            sorted({card.game_id for card in model_cards}),
            cell="pa", n_replicates=n_replicates, seed=seed,
        )
        if pa_cell is not None:
            pa_by_model[model_ref] = pa_cell

        model_clues = clues_by_model.get(model_ref, [])
        ep_cell = _bootstrap_cell(
            model_clues, clue_efficiency,  # type: ignore[arg-type]
            sorted({clue.game_id for clue in model_clues}),
            cell="ep", n_replicates=n_replicates, seed=seed,
        )
        if ep_cell is not None:
            ep_by_model[model_ref] = ep_cell

        diagnostics.append(
            SkillDiagnostics(
                model_ref=model_ref,
                n_games=len(model_games),
                n_clues=len(model_clues),
                n_played_cards=len(model_cards),
                n_agent_cards=sum(1 for card in model_cards if card.is_agent),
                n_agents_revealed_on_own_clues=sum(
                    clue.agents_revealed for clue in model_clues
                ),
            )
        )

    logger.info(
        "skill metrics: %d control games, %d played cards, %d clues",
        len(games),
        len(played),
        len(clues),
    )
    return SkillReport(
        master_seed=master_seed,
        result_counts=dict(sorted(result_counts.items())),
        n_games=len(games),
        tv_by_pairing=tv_by_pairing,
        tv_by_model=tv_by_model,
        pa_by_model=pa_by_model,
        ep_by_model=ep_by_model,
        games_per_pairing={
            label: len(pairing_games) for label, pairing_games in sorted(games_by_pairing.items())
        },
        diagnostics=diagnostics,
    )
