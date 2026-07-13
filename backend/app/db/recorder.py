"""In-memory per-game accumulator for the persistence write-path.

The recorder is a pure in-memory structure with no database imports. One instance is created at
game start (in ``GET /play``) and lives across HTTP requests inside the routes' ``_games`` map. Its
thin ``record_*`` methods are called at the existing capture points; nothing here talks to a
provider, touches game rules, or performs I/O. At game end the writer reads the accumulated records
and flushes the whole game to Postgres in one transaction.

Deliberately not stored (reconstructable / transient): ``sd_measurement_pending`` and per-card time
markers (``time_marker_by``) - the latter is derivable downstream from ``reveal_event`` rows with
``result_role='civilian'``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from backend.app.models.game_schemas import ClueEntry, ConfidenceRanking, ResolvedTarget
from backend.app.models.llm_schemas import GuessProposal, LLMCallRecord


def result_role_of(result_str: str) -> str:
    """Map a ``resolve_guess`` return string to a ``reveal_event.result_role`` card role.

    The engine returns an outcome string (``'agent'``/``'victory'``/``'loss_assassin_sd'`` ...);
    the reveal row records the underlying card role in ``('agent','assassin','civilian')``. A
    victory is always an agent reveal; a sudden-death loss names the losing role.
    """
    if result_str in ("agent", "victory", "victory_sd"):
        return "agent"
    if result_str in ("assassin", "loss_assassin_sd"):
        return "assassin"
    if result_str in ("civilian", "loss_civilian_sd"):
        return "civilian"
    raise ValueError(
        f"Unmappable resolve_guess result {result_str!r}: cannot derive a reveal_event.result_role")


@dataclass
class SeatRecord:
    """One seat's persisted identity + the sampling actually requested of it."""
    seat_index: int
    provider: Optional[str]
    model_ref: Optional[str]
    requested_temperature: Optional[float] = None
    requested_seed: Optional[int] = None


@dataclass
class ClueRecord:
    """A turn's clue, plus every model attempt that produced it (empty for a human clue)."""
    clue_word: str
    count: int
    reasoning: Optional[str]
    targets_raw: list  # verbatim intended target set S, unresolved
    targets_resolved: list[ResolvedTarget]
    llm_calls: list[LLMCallRecord]


@dataclass
class RevealRecord:
    """One card resolution, derived from a ``resolve_guess`` return + reads of ``engine.state``."""
    card_id: int
    result_role: str
    timer_tokens_after: Optional[int]
    ended_turn: bool
    ended_game: bool
    # Index into the play proposal's items this reveal came from (index-aligned backfill); None for
    # a human reveal, which has no proposal.
    proposal_index: Optional[int]
    acting_seat: int


@dataclass
class TurnRecord:
    """One turn: a clue and its guessing phase (normal), or the single sudden-death turn."""
    turn_number: int
    phase: str  # 'normal' | 'sudden_death'
    clue_giver_seat: int
    clue: Optional[ClueRecord] = None
    play_proposal: Optional[GuessProposal] = None       # kind='play'
    measurement: Optional[ConfidenceRanking] = None     # kind='measurement'
    reveals: list[RevealRecord] = field(default_factory=list)
    # Guesser seats that produced SD proposals/measurements on this (single) sudden-death turn. In
    # sudden death BOTH seats guess, so the guesser is NOT derivable from clue_giver_seat and must be
    # carried explicitly. The single-seat case (all that is reachable pre-runner) yields one seat,
    # which the writer uses for guesser_seat; two distinct seats cannot share one SD turn under
    # UNIQUE(turn_id, kind) and are the writer's named-raise seam. Empty on normal turns.
    sd_guesser_seats: set[int] = field(default_factory=set)


class GameRecorder:
    """Accumulates the per-game signals of one Codenames Duet play for a single terminal flush."""

    def __init__(self, *, game_id: str, board_id: str, start_player: Optional[int], llm_client):
        self.game_id = game_id
        self.board_id = board_id
        # Runner concepts, not written by the interactive path.
        self.run_id: Optional[str] = None
        self.derived_seed: Optional[int] = None
        self.start_player = start_player
        # Game outcome, set at flush time from engine.state.
        self.result: Optional[str] = None
        self.timer_tokens_final: Optional[int] = None
        # One-use latch: a second flush of the same game is a no-op.
        self.flushed = False

        # Seat 0 is the LLM (from the client); seat 1 is the human.
        client_name = type(llm_client).__name__
        provider0 = "openrouter" if "OpenRouter" in client_name else "ollama"
        self.seats: list[SeatRecord] = [
            SeatRecord(0, provider0, getattr(llm_client, "model_name", None)),
            SeatRecord(1, "human", None),
        ]

        self.turns: list[TurnRecord] = []
        self._sd_turn: Optional[TurnRecord] = None

    # internal helpers
    def _observe_seat0_sampling(self, records) -> None:
        """Record seat 0's requested temperature/seed on first observation of one of its calls."""
        seat0 = self.seats[0]
        for rec in records:
            if rec is None:
                continue
            if seat0.requested_temperature is None and rec.requested_temperature is not None:
                seat0.requested_temperature = rec.requested_temperature
            if seat0.requested_seed is None and rec.requested_seed is not None:
                seat0.requested_seed = rec.requested_seed

    def _current_turn(self) -> TurnRecord:
        if not self.turns:
            raise RuntimeError(
                "No turn is open; record_clue must be called before guesses/reveals.")
        return self.turns[-1]

    # normal play
    def record_clue(self, clue_entry: ClueEntry, proposal=None) -> None:
        """Open a normal turn and record its clue. ``proposal`` is the LLM ClueProposal (carrying all
        attempts + reasoning) or None for a human clue (no llm_calls, empty targets)."""
        llm_calls = list(proposal.llm_calls) if proposal is not None else []
        reasoning = proposal.reasoning if proposal is not None else None
        clue = ClueRecord(
            clue_word=clue_entry.clue,
            count=clue_entry.count,
            reasoning=reasoning,
            targets_raw=list(clue_entry.targets),
            targets_resolved=list(clue_entry.targets_resolved),
            llm_calls=llm_calls,
        )
        self.turns.append(
            TurnRecord(
                turn_number=len(self.turns),
                phase="normal",
                clue_giver_seat=clue_entry.clue_giver,
                clue=clue,
            )
        )
        self._observe_seat0_sampling(llm_calls)

    def record_play_proposal(self, guess_proposal: GuessProposal) -> None:
        self._current_turn().play_proposal = guess_proposal
        self._observe_seat0_sampling([guess_proposal.llm_call])

    def record_measurement(self, ranking: Optional[ConfidenceRanking]) -> None:
        """Attach the out-of-band confidence ranking to the current turn. No-op if measurement failed
        (ranking is None)."""
        if ranking is None:
            return
        self._current_turn().measurement = ranking
        self._observe_seat0_sampling([ranking.llm_call])

    def record_reveal(
        self,
        *,
        card_id: int,
        result_str: str,
        timer_tokens_after: Optional[int],
        ended_game: bool,
        proposal_index: Optional[int],
        acting_seat: int,
    ) -> None:
        self._current_turn().reveals.append(
            RevealRecord(
                card_id=card_id,
                result_role=result_role_of(result_str),
                timer_tokens_after=timer_tokens_after,
                ended_turn=result_str != "agent",
                ended_game=ended_game,
                proposal_index=proposal_index,
                acting_seat=acting_seat,
            )
        )

    # sudden death (single per-game turn, no clue)
    def ensure_sudden_death_turn(self, clue_giver_seat: int) -> TurnRecord:
        """Open the single sudden-death turn once; idempotent thereafter."""
        if self._sd_turn is None:
            self._sd_turn = TurnRecord(
                turn_number=len(self.turns),
                phase="sudden_death",
                clue_giver_seat=clue_giver_seat,
            )
            self.turns.append(self._sd_turn)
        return self._sd_turn

    def record_sd_measurement(self, ranking: Optional[ConfidenceRanking], clue_giver_seat: int,
                              guesser_seat: int = 0) -> None:
        if ranking is None:
            return
        turn = self.ensure_sudden_death_turn(clue_giver_seat)
        turn.measurement = ranking
        turn.sd_guesser_seats.add(guesser_seat)
        self._observe_seat0_sampling([ranking.llm_call])

    def record_sd_play_proposal(self, guess_proposal: GuessProposal, clue_giver_seat: int,
                                guesser_seat: int = 0) -> None:
        turn = self.ensure_sudden_death_turn(clue_giver_seat)
        turn.play_proposal = guess_proposal
        turn.sd_guesser_seats.add(guesser_seat)
        self._observe_seat0_sampling([guess_proposal.llm_call])

    def record_sd_reveal(self, *, clue_giver_seat: int, card_id: int, result_str: str,
                         timer_tokens_after: Optional[int], ended_game: bool, proposal_index: Optional[int],
                         acting_seat: int,
                         ) -> None:
        self.ensure_sudden_death_turn(clue_giver_seat).reveals.append(
            RevealRecord(
                card_id=card_id,
                result_role=result_role_of(result_str),
                timer_tokens_after=timer_tokens_after,
                ended_turn=result_str != "agent",
                ended_game=ended_game,
                proposal_index=proposal_index,
                acting_seat=acting_seat,
            )
        )

    # outcome
    def set_outcome(self, result: Optional[str], timer_tokens_final: Optional[int]) -> None:
        """Capture the game-level result and final timer bank (read from engine.state at flush)."""
        self.result = result
        self.timer_tokens_final = timer_tokens_final
