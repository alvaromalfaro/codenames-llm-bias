"""The rules a game's state must keep between two dispatches.

``check_invariants`` reads an engine's state and returns one message per rule it breaks, none when
the state is sound. The game runner checks the state before the first dispatch, after every
dispatch, and once the game is over, before it is stored (``game_runner.run_single_game``): the
batch ends a game that breaks one in error (``GameInvariantError``), and the test games log it and
play on. Section numbers (§) are those of the Duet rules the engine cites.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from backend.app.models.game_schemas import CardRole, GamePhase, WordCard

if TYPE_CHECKING:
    from backend.app.core.engine import CodenamesDuetEngine

logger = logging.getLogger(__name__)

# Every result a finished game can have.
GAME_RESULTS = frozenset(
    {
        "victory",
        "loss_assassin",
        "victory_sd",
        "loss_civilian_sd",
        "loss_assassin_sd",
        "loss_stopped_sd",
    }
)


class GameInvariantError(RuntimeError):
    """The state of a game broke one of its rules: the engine, or what drives it, has a bug, and the
    game is no valid observation. ``code`` is what ``game.error_code`` records for it."""

    code = "invariant_broken"

    def __init__(self, problems: list[str]):
        super().__init__("; ".join(problems))
        self.problems = list(problems)


def _side(card: WordCard, seat: int) -> CardRole:
    """What ``card`` is on ``seat``'s side of the key card."""
    return card.llm_perspective_role if seat == 0 else card.human_perspective_role


def _pending_on_board(engine: CodenamesDuetEngine) -> list[int]:
    """pending_words as the board says it (§2): green on the OTHER seat's side, not covered."""
    cards = engine.state.board.cards
    return [
        sum(1 for c in cards if _side(c, 1 - seat) == CardRole.AGENT and not c.revealed)
        for seat in (0, 1)
    ]


def check_invariants(engine: CodenamesDuetEngine) -> list[str]:
    """Every rule of the game state that must hold between two turns; one message per breach."""
    s = engine.state
    cards = s.board.cards
    problems = []

    spent = s.bystander_tokens + s.check_tokens + s.penalty_tokens
    if s.timer_tokens + spent != 9:
        problems.append(
            f"tokens do not add up to 9 (§12): reserve={s.timer_tokens} "
            f"bystander={s.bystander_tokens} check={s.check_tokens} "
            f"penalty={s.penalty_tokens}"
        )
    if not 0 <= s.timer_tokens <= 9:
        problems.append(f"reserve out of range: {s.timer_tokens}")
    on_words = sum(len(c.time_marker_by) for c in cards)
    if on_words != s.bystander_tokens:
        problems.append(
            f"{on_words} time token(s) on words but bystander_tokens={s.bystander_tokens}"
        )
    if s.pending_words != _pending_on_board(engine):
        problems.append(
            f"pending_words={s.pending_words} but the board says {_pending_on_board(engine)} (§2)"
        )

    for c in cards:
        if len(c.time_marker_by) > 2 or len(set(c.time_marker_by)) != len(c.time_marker_by):
            problems.append(f"{c.text}: time tokens {c.time_marker_by} (at most one per direction)")
        for guesser in c.time_marker_by:
            if _side(c, 1 - guesser) != CardRole.CIVILIAN:
                problems.append(
                    f"{c.text}: time token from seat {guesser}, but the word is "
                    f"{_side(c, 1 - guesser).value} for the clue giver (§6.4)"
                )
        if c.revealed and CardRole.AGENT not in (_side(c, 0), _side(c, 1)):
            problems.append(f"{c.text}: covered by an agent card but green on neither side")

    visible = {c.text.lower() for c in cards if not c.is_covered}
    validator = set(engine.clue_validator.visible_words)
    if visible != validator:
        problems.append(
            f"visible words for clue validation differ from the board (§8.2): "
            f"validator has {sorted(validator - visible)} extra, misses "
            f"{sorted(visible - validator)}"
        )

    if s.is_game_over:
        if s.result not in GAME_RESULTS:
            problems.append(f"unknown result {s.result!r}")
        won = s.result in ("victory", "victory_sd")
        if won != (s.pending_words == [0, 0]):
            problems.append(f"result {s.result} with pending_words={s.pending_words}")
    else:
        phase = s.current_phase
        if s.timer_tokens == 0 and phase not in (
            GamePhase.SUDDEN_DEATH_LLM,
            GamePhase.SUDDEN_DEATH_HUMAN,
        ):
            problems.append(f"the reserve is empty but the phase is {phase.value} (§7.3)")
        if phase in (GamePhase.GIVING_CLUE, GamePhase.GUESSING):
            if s.clue_giver == s.guesser:
                problems.append(f"seat {s.clue_giver} is both clue giver and guesser")
            if s.pending_words[s.guesser] == 0:
                problems.append(f"seat {s.guesser} guesses with nothing pending (§6.8)")
        if phase == GamePhase.SUDDEN_DEATH_LLM and s.pending_words[0] == 0:
            problems.append("SUDDEN_DEATH_LLM with nothing pending for seat 0")
        if phase == GamePhase.SUDDEN_DEATH_HUMAN and (
            s.pending_words[1] == 0 or s.pending_words[0] != 0
        ):
            problems.append(f"SUDDEN_DEATH_HUMAN with pending_words={s.pending_words}")
    return problems


class InvariantGuard:
    """Checks one game's state as it is played: the rules of ``check_invariants``, and that the turn
    number never goes back. Each breach is logged as an ERROR and kept in ``violations``; with
    ``strict`` the first state that breaks one raises ``GameInvariantError`` instead of letting the
    game go on."""

    def __init__(self, game_id: str, *, strict: bool):
        self.game_id = game_id
        self.strict = strict
        self.violations: list[str] = []
        self._last_turn = 0
        self._final_checked = False

    def check(self, engine: CodenamesDuetEngine) -> None:
        """Check the state ``engine`` is in now."""
        problems = check_invariants(engine)
        turn = engine.state.turn_number
        if turn < self._last_turn:
            problems.append(f"turn number went back from {self._last_turn} to {turn}")
        self._last_turn = turn
        for problem in problems:
            logger.error("invariant broken: game_id=%s: %s", self.game_id, problem)
        self.violations.extend(problems)
        if problems and self.strict:
            raise GameInvariantError(problems)

    def check_final(self, engine: CodenamesDuetEngine) -> None:
        """Check the state of the finished game, once however often it is called: before the game
        is stored, so a strict guard stores a broken game as an error, never as completed."""
        if not self._final_checked:
            self._final_checked = True
            self.check(engine)
