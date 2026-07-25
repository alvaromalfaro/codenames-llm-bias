"""Sudden-death guesser bias metric: conc-SD

In sudden death the clue giver is silent: the guesser must pick agents from memory of the clue
history it *received* from its partner. conc-SD is the endgame analogue of CIT - a weighted
Cliff-style sign delta mapped onto [0, 1], where **0.5 means no association** - but three things
change from the normal-phase metric:

  * the unit of analysis is the **game**, not the turn. Each game contributes at most one sudden-death
    state per seat, so the comparison set is the per-seat sudden-death observation and the sum runs
    over games (eq 4.12);
  * the clue polarity is a **history mean** ``P^H_g = mean_k rho(clue_k)`` over the clues the guesser
    received from its partner (eq 4.13), not a single clue. Admissibility is ``is_admissible(P^H_g)``;
  * thematic proximity is measured against the **whole clue history**: ``s^H_i = max_k
    thematic_sim(w_i, clue_k)`` by default (eq 4.14), with the mean as a reserved robustness column.

The card set is the **non-agents from the guesser's perspective**. The sudden-death guesser at seat
``s`` hunts its partner's agents, so a card is a non-agent exactly when its role under
``perspective_column(1 - s)`` is not ``'agent'``. Using column ``s`` would score the wrong player's
key - that inversion is the metric's central trap and lives behind ``perspective`` accordingly.

The confidence signal is the **sudden-death** measurement ranking
(``guess_proposal.kind='measurement'`` on a ``turn.phase='sudden_death'`` turn), never a normal-phase
one and never the play proposal.

Grouping, banding, weighting and the [0, 1] mapping are deliberately identical to CIT so the two
endgame/normal metrics cannot drift; the tercile cuts, however, are conc-SD's **own**, computed over
the ``s^H`` distribution (card-vs-history), which is a different variable from CIT's ``s``
(card-vs-single-clue). The generic cluster bootstrap in ``analysis.inference`` supplies the intervals
unchanged - this module adds an adapter, never resampling logic.

A pre-registered power rule caps interpretation: conc-SD is a PRIMARY result for a model only with at
least ``PRIMARY_MIN_ADMISSIBLE_GAMES`` admissible sudden-death games, and EXPLORATORY below that.
Applied mechanically from the counts.

Read-only with respect to the database, and no encoder: geometry comes from vectors already stored in
``embedding_mpnet`` via ``FrameGeometry``.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Literal

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.analysis.geometry import (
    FrameGeometry,
    MissingEmbeddingError,
    assign_tercile,
    classify_congruence,
    compute_tercile_cuts,
    is_admissible,
)
from backend.app.analysis.guesser_metrics import (
    DEFAULT_FRAME_ID,
    DEFAULT_MASTER_SEED,
    resolve_ranking_confidences,
)
from backend.app.analysis.inference import (
    DEFAULT_REPLICATES,
    DEFAULT_SEED,
    BootstrapResult,
    cluster_bootstrap,
)
from backend.app.analysis.perspective import perspective_column
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
    WordCardModel,
)

logger = logging.getLogger(__name__)

BOARD_TYPES: tuple[str, ...] = ("probe", "control")
MEASUREMENT_KIND = "measurement"
NORMAL_PHASE = "normal"
SD_PHASE = "sudden_death"
COMPLETED = "completed"
AGENT_ROLE = "agent"

SD_FAILURE_RESULTS: frozenset[str] = frozenset(
    {"loss_civilian_sd", "loss_assassin_sd"})

# Pre-registered before seeing results: below this many admissible SD games a model's conc-SD is
# reported as EXPLORATORY, not PRIMARY. Applied mechanically from the counts, never adjusted.
PRIMARY_MIN_ADMISSIBLE_GAMES = 20

CONGRUENT = "C+"
INCONGRUENT = "C-"
STRATUM_POOLED = "pooled"

SimVariant = Literal["max", "mean"]
Weighting = Literal["weighted", "classic"]

SIM_VARIANTS: tuple[SimVariant, ...] = ("max", "mean")
WEIGHTINGS: tuple[Weighting, ...] = ("weighted", "classic")

CONCSD_NULL = 0.5
CONTRASTS = (
    ("probe_minus_control", "probe_all", "control_all"),
    ("probe_b1_minus_b3", "probe_b1", "probe_b3"),
)


def _sign(value: float) -> int:
    """Sign of a difference; exact ties give 0 and contribute nothing to the numerator."""
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


# Records and observations
@dataclass(frozen=True)
class DataGaps:
    """Everything excluded for want of a usable measurement, tallied rather than imputed."""

    unmatched_ranking_words: int = 0
    cards_without_embedding: int = 0
    clue_words_without_embedding: int = 0
    divergent_duplicate_cards: int = 0
    observations_with_divergent_duplicates: int = 0

    def merged_with(self, other: DataGaps) -> DataGaps:
        return DataGaps(
            unmatched_ranking_words=self.unmatched_ranking_words +
            other.unmatched_ranking_words,
            cards_without_embedding=self.cards_without_embedding +
            other.cards_without_embedding,
            clue_words_without_embedding=(
                self.clue_words_without_embedding + other.clue_words_without_embedding
            ),
            divergent_duplicate_cards=(
                self.divergent_duplicate_cards + other.divergent_duplicate_cards
            ),
            observations_with_divergent_duplicates=(
                self.observations_with_divergent_duplicates
                + other.observations_with_divergent_duplicates
            ),
        )


@dataclass(frozen=True)
class SdSeatRecord:
    """One sudden-death seat that produced a measurement ranking, before geometry is applied.

    The comparison set of conc-SD: a single ``(game_id, guesser_seat)``. Carries the partner's clue
    history and both perspective columns of every board card, so the geometry step can decide the
    polarity, the non-agent set and the congruence grouping.
    """

    game_id: str
    guesser_seat: int
    model_ref: str
    board_id: str
    board_type: str
    # clue words received from seat (1-s), normal phase, lowercased
    partner_clues: tuple[str, ...]
    board_roles: Mapping[str, Mapping[str, str]]  # word -> {column_name: role}
    # SD measurement ranking (lowercased word, confidence)
    ranking: Sequence[tuple[str, float]]


@dataclass(frozen=True)
class SdObservation:
    """One non-agent card on one admissible sudden-death seat observation."""

    model_ref: str
    game_id: str
    obs_id: int  # groups the C+/C- comparison set; one per SD-seat observation
    board_type: str
    word: str
    rho: float
    sim_max: float  # s^H, MAX over the clue history (the primary variant)
    # s^H, MEAN over the clue history (reserved robustness variant)
    sim_mean: float
    confidence: float
    group: str  # 'C+' | 'C-'
    abs_ph: float  # |P^H_g|, constant within an observation, varies across them
    band_max: int | None = None
    band_mean: int | None = None


@dataclass(frozen=True)
class ConcSdResult:
    model_ref: str
    board_type: str
    variant: SimVariant
    band: int | None  # None = pooled over bands
    concsd_weighted: float | None
    concsd_classic: float | None
    n_pairs: int
    n_cplus: int
    n_cminus: int
    weight_total: float


@dataclass(frozen=True)
class SdModelSummary:
    """Per-model admissible-game count and the mechanical primary/exploratory flag."""

    model_ref: str
    n_sd_observations: int
    n_admissible_games: int
    n_non_admissible_games: int
    is_primary: bool


@dataclass(frozen=True)
class SdDiagnostics:
    model_ref: str
    n_admissible_games: int
    n_non_admissible_games: int
    n_cards_candidate: int
    n_cards_neutral: int
    n_cards_dead_zone: int
    n_cards_classified: int
    gaps: DataGaps


@dataclass(frozen=True)
class SdReport:
    frame_id: str
    master_seed: int
    # keys are "{board_type}_{variant}"; value is the (c33, c66) cut pair or None.
    tercile_cuts: Mapping[str, tuple[float, float] | None]
    model_summaries: Sequence[SdModelSummary]
    # keys are "{variant}_{weighting}"; inner keys are model_ref.
    bootstrap: Mapping[str, Mapping[str, BootstrapResult]]
    # board_type -> (completed, reaching SD)
    sd_reach: Mapping[str, tuple[int, int]]
    # board_type -> {seat -> n SD measurements}
    seat_rankings: Mapping[str, Mapping[int, int]]
    # board_type -> (n, n_positive, prop)
    fg_check: Mapping[str, tuple[int, int, float | None]]
    diagnostics: Sequence[SdDiagnostics]
    n_observations: int


# Perspective (the trap)
def partner_clue_history(
    clues: Sequence[tuple[int, str]], guesser_seat: int
) -> tuple[str, ...]:
    """The clue history H_g the guesser at seat ``s`` RECEIVED: clues given by the partner (1 - s).

    The seat's OWN clues are never part of its received history. ``clues`` are ``(giver_seat, word)``
    pairs in turn order; the returned order is preserved so ``P^H`` and ``s^H`` see the same history.
    """
    return tuple(word for seat, word in clues if seat == 1 - guesser_seat)


def non_agent_words(
    board_roles: Mapping[str, Mapping[str, str]], guesser_seat: int
) -> set[str]:
    """The non-agent card set A^c_g from the sudden-death guesser's perspective.

    The guesser at seat ``s`` hunts the *partner's* agents, so the relevant key column is
    ``perspective_column(1 - s)``. A card is a non-agent exactly when its role there is not
    ``'agent'``. Reading column ``s`` would score the wrong seat's key and invert every result.
    """
    column = perspective_column(1 - guesser_seat)
    return {word for word, roles in board_roles.items() if roles[column] != AGENT_ROLE}


# Loading
def load_sd_seat_records(session: Session, *, master_seed: int) -> list[SdSeatRecord]:
    """Load every sudden-death seat of the run that carries a measurement ranking. Read-only."""
    seat_rows = session.execute(
        select(
            TurnModel.id,
            TurnModel.game_id,
            GuessProposalModel.id,
            GuessProposalModel.guesser_seat,
            GameSeatModel.model_ref,
            GameModel.board_id,
            BoardModel.type,
        )
        .join(GameModel, GameModel.id == TurnModel.game_id)
        .join(RunModel, RunModel.id == GameModel.run_id)
        .join(BoardModel, BoardModel.board_id == GameModel.board_id)
        .join(
            GuessProposalModel,
            (GuessProposalModel.turn_id == TurnModel.id)
            & (GuessProposalModel.kind == MEASUREMENT_KIND),
        )
        .join(
            GameSeatModel,
            (GameSeatModel.game_id == TurnModel.game_id)
            & (GameSeatModel.seat_index == GuessProposalModel.guesser_seat),
        )
        .where(
            RunModel.master_seed == master_seed,
            GameModel.game_status == COMPLETED,
            TurnModel.phase == SD_PHASE,
            BoardModel.type.in_(BOARD_TYPES),
        )
    ).all()
    if not seat_rows:
        return []

    proposal_ids = [row[2] for row in seat_rows]
    board_ids = {row[5] for row in seat_rows}
    game_ids = {row[1] for row in seat_rows}

    # Perspective roles for every card of every board involved, keyed by board and lowercased word.
    roles_by_board: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for board_id, text, llm_role, human_role in session.execute(
        select(
            WordCardModel.board_id,
            WordCardModel.text,
            WordCardModel.llm_perspective_role,
            WordCardModel.human_perspective_role,
        ).where(WordCardModel.board_id.in_(board_ids))
    ).all():
        roles_by_board[board_id][text.lower()] = {
            "llm_perspective_role": llm_role,
            "human_perspective_role": human_role,
        }

    # The clue history of each game: every normal-phase clue with the seat that gave it.
    clues_by_game: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for game_id, giver_seat, clue_word in session.execute(
        select(TurnModel.game_id, TurnModel.clue_giver_seat, ClueModel.clue_word)
        .join(ClueModel, ClueModel.turn_id == TurnModel.id)
        .where(
            TurnModel.game_id.in_(game_ids),
            TurnModel.phase == NORMAL_PHASE,
        )
        .order_by(TurnModel.game_id, TurnModel.turn_number)
    ).all():
        clues_by_game[game_id].append((giver_seat, clue_word.lower()))

    ranking_by_proposal: dict[int, list[tuple[str, float]]] = defaultdict(list)
    for proposal_id, word, confidence in session.execute(
        select(
            GuessProposalItemModel.guess_proposal_id,
            GuessProposalItemModel.word,
            GuessProposalItemModel.confidence,
        )
        .where(GuessProposalItemModel.guess_proposal_id.in_(proposal_ids))
        .order_by(GuessProposalItemModel.guess_proposal_id, GuessProposalItemModel.position)
    ).all():
        if confidence is None:
            continue
        ranking_by_proposal[proposal_id].append(
            (word.lower(), float(confidence)))

    records: list[SdSeatRecord] = []
    for _turn_id, game_id, proposal_id, guesser_seat, model_ref, board_id, board_type in seat_rows:
        partner_clues = partner_clue_history(
            clues_by_game.get(game_id, ()), guesser_seat)
        records.append(
            SdSeatRecord(
                game_id=game_id,
                guesser_seat=guesser_seat,
                model_ref=model_ref,
                board_id=board_id,
                board_type=board_type,
                partner_clues=partner_clues,
                board_roles=roles_by_board.get(board_id, {}),
                ranking=tuple(ranking_by_proposal.get(proposal_id, ())),
            )
        )
    return records


# Observations
def collect_sd_observations(
    records: Iterable[SdSeatRecord], geometry: FrameGeometry
) -> tuple[list[SdObservation], dict[str, SdDiagnostics]]:
    """Turn admissible SD-seat records into grouped per-card observations, tallying exclusions."""
    observations: list[SdObservation] = []
    admissible: dict[str, int] = defaultdict(int)
    non_admissible: dict[str, int] = defaultdict(int)
    gaps: dict[str, DataGaps] = defaultdict(DataGaps)
    candidate: dict[str, int] = defaultdict(int)
    neutral: dict[str, int] = defaultdict(int)
    dead_zone: dict[str, int] = defaultdict(int)

    for obs_id, record in enumerate(records):
        # The embedded clue history: a clue with no vector is a counted gap and drops from BOTH the
        # polarity mean P^H and the s^H similarities, so the two are always over the same history.
        embedded_clues: list[str] = []
        clue_rhos: list[float] = []
        clue_gap = 0
        for clue in record.partner_clues:
            try:
                clue_rhos.append(geometry.rho(clue))
            except MissingEmbeddingError:
                clue_gap += 1
                continue
            embedded_clues.append(clue)

        if not clue_rhos:
            # No usable history: the observation defines no polarity. Counted as non-admissible.
            non_admissible[record.model_ref] += 1
            gaps[record.model_ref] = gaps[record.model_ref].merged_with(
                DataGaps(clue_words_without_embedding=clue_gap)
            )
            continue

        polarity = sum(clue_rhos) / len(clue_rhos)  # P^H_g, eq 4.13
        if not is_admissible(polarity):
            non_admissible[record.model_ref] += 1
            gaps[record.model_ref] = gaps[record.model_ref].merged_with(
                DataGaps(clue_words_without_embedding=clue_gap)
            )
            continue
        admissible[record.model_ref] += 1

        allowed = non_agent_words(record.board_roles, record.guesser_seat)
        usable, divergent = resolve_ranking_confidences(record.ranking)

        unmatched = 0
        without_embedding = 0
        for word, confidence in usable.items():
            if word not in record.board_roles:
                unmatched += 1
                continue
            if word not in allowed:
                continue  # an agent from the guesser's perspective is not part of A^c_g
            try:
                rho_i = geometry.rho(word)
            except MissingEmbeddingError:
                without_embedding += 1
                continue
            candidate[record.model_ref] += 1
            group = classify_congruence(rho_i, polarity)
            if group == "neutral":
                neutral[record.model_ref] += 1
                continue
            if group != CONGRUENT and group != INCONGRUENT:
                dead_zone[record.model_ref] += 1
                continue
            sims = [geometry.thematic_sim(word, clue)
                    for clue in embedded_clues]
            observations.append(
                SdObservation(
                    model_ref=record.model_ref,
                    game_id=record.game_id,
                    obs_id=obs_id,
                    board_type=record.board_type,
                    word=word,
                    rho=rho_i,
                    sim_max=max(sims),
                    sim_mean=sum(sims) / len(sims),
                    confidence=confidence,
                    group=group,
                    abs_ph=abs(polarity),
                )
            )
        gaps[record.model_ref] = gaps[record.model_ref].merged_with(
            DataGaps(
                unmatched_ranking_words=unmatched,
                cards_without_embedding=without_embedding,
                clue_words_without_embedding=clue_gap,
                divergent_duplicate_cards=len(divergent),
                observations_with_divergent_duplicates=1 if divergent else 0,
            )
        )

    classified: dict[str, int] = defaultdict(int)
    for obs in observations:
        classified[obs.model_ref] += 1

    diagnostics = {
        model_ref: SdDiagnostics(
            model_ref=model_ref,
            n_admissible_games=admissible.get(model_ref, 0),
            n_non_admissible_games=non_admissible.get(model_ref, 0),
            n_cards_candidate=candidate.get(model_ref, 0),
            n_cards_neutral=neutral.get(model_ref, 0),
            n_cards_dead_zone=dead_zone.get(model_ref, 0),
            n_cards_classified=classified.get(model_ref, 0),
            gaps=gaps.get(model_ref, DataGaps()),
        )
        for model_ref in sorted({*admissible, *non_admissible, *gaps})
    }
    return observations, diagnostics


def _similarity(obs: SdObservation, variant: SimVariant) -> float:
    return obs.sim_max if variant == "max" else obs.sim_mean


def _band(obs: SdObservation, variant: SimVariant) -> int | None:
    return obs.band_max if variant == "max" else obs.band_mean


def assign_bands_by_board_type(
    cards: Sequence[SdObservation],
) -> tuple[list[SdObservation], dict[str, tuple[float, float] | None]]:
    """Band SD observations on s^H, with cuts computed ONCE per (board type, variant).

    Both variants are banded in one pass so all four cut pairs are reported. These cuts are conc-SD's
    OWN: s^H is card-vs-history, a different distribution from CIT's card-vs-single-clue s, so CIT's
    cuts are never reused. Per board type because probe and control are different populations.
    """
    cuts_by_key: dict[str, tuple[float, float] | None] = {}
    by_type: dict[str, list[SdObservation]] = defaultdict(list)
    for card in cards:
        by_type[card.board_type].append(card)

    banded_by_id: dict[int, dict[str, int]] = defaultdict(dict)
    for board_type, group in sorted(by_type.items()):
        for variant in SIM_VARIANTS:
            cuts = compute_tercile_cuts(
                [_similarity(card, variant) for card in group])
            cuts_by_key[f"{board_type}_{variant}"] = cuts
            c33, c66 = cuts
            for index, card in enumerate(group):
                banded_by_id[id(card)][variant] = assign_tercile(
                    _similarity(card, variant), c33, c66
                )

    banded = [
        replace(
            card,
            band_max=banded_by_id[id(card)]["max"],
            band_mean=banded_by_id[id(card)]["mean"],
        )
        for card in cards
    ]
    return banded, cuts_by_key


# conc-SD
def _concsd_from_pairs(cards: Sequence[SdObservation]) -> ConcSdResult | None:
    """Accumulate the sign delta over C+/C- pairs, pairing only WITHIN each SD-seat observation.

    Cards from different observations answered different clue histories, so a cross-observation
    comparison of raw confidences would not be like with like. Identical form to CIT's ``_cit_from_
    pairs`` with the history weight ``abs_ph * abs(rho_i - rho_j)``.
    """
    by_obs: dict[int, list[SdObservation]] = defaultdict(list)
    for card in cards:
        by_obs[card.obs_id].append(card)

    weighted_numerator = 0.0
    weighted_denominator = 0.0
    classic_numerator = 0
    n_pairs = 0
    for obs_cards in by_obs.values():
        plus = [card for card in obs_cards if card.group == CONGRUENT]
        minus = [card for card in obs_cards if card.group == INCONGRUENT]
        for card_i in plus:
            for card_j in minus:
                weight = card_i.abs_ph * abs(card_i.rho - card_j.rho)
                direction = _sign(card_i.confidence - card_j.confidence)
                weighted_numerator += weight * direction
                weighted_denominator += weight
                classic_numerator += direction
                n_pairs += 1

    if n_pairs == 0:
        return None
    weighted = (
        0.5 * (1.0 + weighted_numerator / weighted_denominator)
        if weighted_denominator > 0
        else None
    )
    classic = 0.5 * (1.0 + classic_numerator / n_pairs)
    return ConcSdResult(
        model_ref="",
        board_type="",
        variant="max",
        band=None,
        concsd_weighted=weighted,
        concsd_classic=classic,
        n_pairs=n_pairs,
        n_cplus=sum(1 for card in cards if card.group == CONGRUENT),
        n_cminus=sum(1 for card in cards if card.group == INCONGRUENT),
        weight_total=weighted_denominator,
    )


def compute_concsd(
    cards: Sequence[SdObservation], *, variant: SimVariant = "max"
) -> list[ConcSdResult]:
    """Aggregate banded observations into per-(model, board type, band) conc-SD rows for a variant."""
    bands: list[int | None] = [
        None,
        *sorted({b for card in cards if (b := _band(card, variant)) is not None}),
    ]
    results: list[ConcSdResult] = []
    for model_ref in sorted({card.model_ref for card in cards}):
        for board_type in sorted({card.board_type for card in cards}):
            for band in bands:
                subset = [
                    card
                    for card in cards
                    if card.model_ref == model_ref
                    and card.board_type == board_type
                    and (band is None or _band(card, variant) == band)
                ]
                if not subset:
                    continue
                accumulated = _concsd_from_pairs(subset)
                if accumulated is None:
                    continue
                results.append(
                    replace(
                        accumulated,
                        model_ref=model_ref,
                        board_type=board_type,
                        variant=variant,
                        band=band,
                    )
                )
    return results


# Bootstrap adapter
def concsd_cells(
    cards: Sequence[SdObservation],
    *,
    model_ref: str,
    variant: SimVariant = "max",
    weighting: Weighting = "weighted",
) -> dict[str, float]:
    """Flatten ``compute_concsd`` for ONE model into named scalar cells for the bootstrap.

    Cell names are ``{board_type}_all`` and ``{board_type}_b{band}``. A cell with no comparable
    C+/C- pairs is simply absent, which the bootstrap records as a dropped replicate.
    """
    cells: dict[str, float] = {}
    for row in compute_concsd(
        [card for card in cards if card.model_ref == model_ref], variant=variant
    ):
        value = row.concsd_weighted if weighting == "weighted" else row.concsd_classic
        if value is None:
            continue
        suffix = "all" if row.band is None else f"b{row.band}"
        cells[f"{row.board_type}_{suffix}"] = value
    return cells


def build_concsd_estimator(
    cards: Sequence[SdObservation],
    *,
    model_ref: str,
    variant: SimVariant = "max",
    weighting: Weighting = "weighted",
) -> tuple[Callable[[Sequence[str]], dict[str, float]], list[str]]:
    """Return ``(estimator, game_ids)`` for bootstrapping one model's conc-SD over games.

    The closure captures cards ALREADY banded over the full dataset and only filters them - there is
    no path back to ``compute_tercile_cuts``, so the cuts are frozen structurally.

    A game drawn twice must contribute twice, but ``compute_concsd`` groups cards into comparison
    sets by ``obs_id``; concatenating a game's cards twice would merge the copies. Each copy
    therefore gets freshly minted ``obs_id`` values (and a copy-tagged ``game_id``), so it forms its
    own comparison set. Both seats of a two-seat game travel together under one ``game_id`` cluster.
    """
    own = [card for card in cards if card.model_ref == model_ref]
    by_game: dict[str, list[SdObservation]] = defaultdict(list)
    for card in own:
        by_game[card.game_id].append(card)
    game_ids = sorted(by_game)

    def estimator(subset: Sequence[str]) -> dict[str, float]:
        resampled: list[SdObservation] = []
        next_obs_id = 0
        for copy_index, game_id in enumerate(subset):
            obs_remap: dict[int, int] = {}
            for card in by_game.get(game_id, ()):
                if card.obs_id not in obs_remap:
                    obs_remap[card.obs_id] = next_obs_id
                    next_obs_id += 1
                resampled.append(
                    replace(
                        card,
                        obs_id=obs_remap[card.obs_id],
                        game_id=f"{game_id}#{copy_index}",
                    )
                )
        return concsd_cells(resampled, model_ref=model_ref, variant=variant, weighting=weighting)

    return estimator, game_ids


# Diagnostics: SD reach, seat dominance and the descriptive f_g check
def load_sd_reach(session: Session, *, master_seed: int) -> dict[str, tuple[int, int]]:
    """Per board type: (completed games, games reaching sudden death). Read-only."""
    reach_rows = session.execute(
        select(BoardModel.type, GameModel.id, TurnModel.game_id)
        .select_from(GameModel)
        .join(RunModel, RunModel.id == GameModel.run_id)
        .join(BoardModel, BoardModel.board_id == GameModel.board_id)
        .outerjoin(
            TurnModel,
            (TurnModel.game_id == GameModel.id) & (
                TurnModel.phase == SD_PHASE),
        )
        .where(
            RunModel.master_seed == master_seed,
            GameModel.game_status == COMPLETED,
            BoardModel.type.in_(BOARD_TYPES),
        )
    ).all()
    completed: dict[str, set[str]] = defaultdict(set)
    reached: dict[str, set[str]] = defaultdict(set)
    for board_type, game_id, sd_game_id in reach_rows:
        # Set union (|=) rather than .add: the package-wide read-only guard forbids the ``add``
        # token, and the whole package deliberately expresses set accumulation this way.
        completed[board_type] |= {game_id}
        if sd_game_id is not None:
            reached[board_type] |= {game_id}
    return {
        board_type: (len(completed[board_type]), len(reached[board_type]))
        for board_type in sorted(completed)
    }


def seat_ranking_counts(records: Sequence[SdSeatRecord]) -> dict[str, dict[int, int]]:
    """SD measurement rankings per (board type, seat) - the seat-0 dominance diagnostic."""
    counts: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for record in records:
        counts[record.board_type][record.guesser_seat] += 1
    return {board_type: dict(seats) for board_type, seats in sorted(counts.items())}


def fg_directional_proportion(
    items: Sequence[tuple[float, float]],
) -> tuple[int, int, float | None]:
    """Descriptive: over (rho_fg, P^H) pairs, count those with ``rho_fg * P^H > 0``.

    Returns ``(n, n_positive, proportion)``. NOT a test - no p-value, reported as description only.
    """
    n = len(items)
    positive = sum(1 for rho_fg, p_h in items if rho_fg * p_h > 0)
    return n, positive, (positive / n if n else None)


def directional_fg_check(
    session: Session, geometry: FrameGeometry, *, master_seed: int
) -> dict[str, tuple[int, int, float | None]]:
    """The exploratory directional check, per board type. Descriptive only, read-only.

    ``f_g`` is the wrongly selected card of a sudden-death failure: the SD-phase reveal that ended the
    game with a non-agent role. ``P^H_g`` uses the acting seat's partner clue history.
    """
    fail_rows = session.execute(
        select(
            BoardModel.type,
            GameModel.id,
            RevealEventModel.acting_seat,
            WordCardModel.text,
        )
        .select_from(GameModel)
        .join(RunModel, RunModel.id == GameModel.run_id)
        .join(BoardModel, BoardModel.board_id == GameModel.board_id)
        .join(TurnModel, (TurnModel.game_id == GameModel.id) & (TurnModel.phase == SD_PHASE))
        .join(
            RevealEventModel,
            (RevealEventModel.turn_id == TurnModel.id)
            & (RevealEventModel.ended_game.is_(True))
            & (RevealEventModel.result_role != AGENT_ROLE),
        )
        .join(
            WordCardModel,
            (WordCardModel.board_id == GameModel.board_id)
            & (WordCardModel.card_id == RevealEventModel.card_id),
        )
        .where(
            RunModel.master_seed == master_seed,
            GameModel.game_status == COMPLETED,
            GameModel.result.in_(tuple(SD_FAILURE_RESULTS)),
            BoardModel.type.in_(BOARD_TYPES),
        )
    ).all()
    if not fail_rows:
        return {}

    game_ids = {row[1] for row in fail_rows}
    clues_by_game: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for game_id, giver_seat, clue_word in session.execute(
        select(TurnModel.game_id, TurnModel.clue_giver_seat, ClueModel.clue_word)
        .join(ClueModel, ClueModel.turn_id == TurnModel.id)
        .where(TurnModel.game_id.in_(game_ids), TurnModel.phase == NORMAL_PHASE)
    ).all():
        clues_by_game[game_id].append((giver_seat, clue_word.lower()))

    items_by_type: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for board_type, game_id, acting_seat, text in fail_rows:
        partner_rhos: list[float] = []
        for seat, clue_word in clues_by_game.get(game_id, ()):
            if seat != 1 - acting_seat:
                continue
            try:
                partner_rhos.append(geometry.rho(clue_word))
            except MissingEmbeddingError:
                continue
        if not partner_rhos:
            continue
        try:
            rho_fg = geometry.rho(text.lower())
        except MissingEmbeddingError:
            continue
        p_h = sum(partner_rhos) / len(partner_rhos)
        items_by_type[board_type].append((rho_fg, p_h))

    return {
        board_type: fg_directional_proportion(items)
        for board_type, items in sorted(items_by_type.items())
    }


# Entry point
def _summarise_models(
    diagnostics: Mapping[str, SdDiagnostics],
) -> list[SdModelSummary]:
    summaries: list[SdModelSummary] = []
    for model_ref in sorted(diagnostics):
        diag = diagnostics[model_ref]
        summaries.append(
            SdModelSummary(
                model_ref=model_ref,
                n_sd_observations=diag.n_admissible_games + diag.n_non_admissible_games,
                n_admissible_games=diag.n_admissible_games,
                n_non_admissible_games=diag.n_non_admissible_games,
                is_primary=diag.n_admissible_games >= PRIMARY_MIN_ADMISSIBLE_GAMES,
            )
        )
    return summaries


def compute_sd_metrics(
    session: Session,
    *,
    frame_id: str = DEFAULT_FRAME_ID,
    master_seed: int = DEFAULT_MASTER_SEED,
    n_replicates: int = DEFAULT_REPLICATES,
    seed: int = DEFAULT_SEED,
) -> SdReport:
    """Compute conc-SD over the sudden-death states of one run, with bootstrap intervals. Read-only.

    All tercile cuts are frozen over the full dataset before any resampling. Every (variant,
    weighting) combination is bootstrapped per model; small n means many degenerate replicates, which
    the per-cell dropped counts and ``reliable`` flags surface rather than hide.
    """
    geometry = FrameGeometry.load(session, frame_id)
    records = load_sd_seat_records(session, master_seed=master_seed)
    raw_obs, diagnostics = collect_sd_observations(records, geometry)
    cards, cuts = assign_bands_by_board_type(raw_obs)

    models = sorted({card.model_ref for card in cards})
    bootstrap: dict[str, dict[str, BootstrapResult]] = {}
    for variant in SIM_VARIANTS:
        for weighting in WEIGHTINGS:
            key = f"{variant}_{weighting}"
            per_model: dict[str, BootstrapResult] = {}
            for model_ref in models:
                estimator, game_ids = build_concsd_estimator(
                    cards, model_ref=model_ref, variant=variant, weighting=weighting
                )
                if not game_ids:
                    continue
                per_model[model_ref] = cluster_bootstrap(
                    game_ids,
                    estimator,
                    n_replicates=n_replicates,
                    seed=seed,
                    contrasts=CONTRASTS,
                    null_value=CONCSD_NULL,
                )
            bootstrap[key] = per_model

    logger.info(
        "sd metrics: %d SD-seat records, %d observations, %d models",
        len(records),
        len(cards),
        len(models),
    )
    return SdReport(
        frame_id=frame_id,
        master_seed=master_seed,
        tercile_cuts=cuts,
        model_summaries=_summarise_models(diagnostics),
        bootstrap=bootstrap,
        sd_reach=load_sd_reach(session, master_seed=master_seed),
        seat_rankings=seat_ranking_counts(records),
        fg_check=directional_fg_check(
            session, geometry, master_seed=master_seed),
        diagnostics=[diagnostics[key] for key in sorted(diagnostics)],
        n_observations=len(cards),
    )
