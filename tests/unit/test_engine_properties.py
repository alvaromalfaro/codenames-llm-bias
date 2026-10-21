"""Property tests of the engine: any sequence of actions on any board of the bank keeps the rules.

Hypothesis plays the engine directly. At every step it picks an action from the state: most often a
legal one (a valid or an invalid clue, a guess at a word the guesser may still touch, a stop, a
concession in sudden death), sometimes one the engine must refuse (the wrong seat, a covered word,
a stop before any guess, a malformed clue). After each accepted action the state must keep every
invariant of backend/app/core/invariants.py; a refused action must leave the state as it was; and
every game must end.
"""

import random

from hypothesis import HealthCheck, event, given, settings
from hypothesis import strategies as st

from backend.app.core.engine import CodenamesDuetEngine
from backend.app.core.invariants import InvariantGuard
from backend.app.core.loader import BoardLoader
from backend.app.models.game_schemas import CardRole, GamePhase

_BOARDS = [board for boards in BoardLoader().boards.values() for board in boards]
# A clue in neither MorphoLex nor WordNet, so it is valid on any board (batch_runner's dry run
# plays it too); a visible board word is the invalid one, played with a penalty token (§8.4).
_VALID_CLUE = "ZZXQJ"
# Far above the longest game: 9 turns of at most 9 guesses each, and a sudden death.
_MAX_ACTIONS = 200
# How a game picks its guesses: at random, steering clear of the words that would end it on the
# assassin, the agents first, or clear of the assassin until sudden death and at random in it.
# Random play mostly ends on an early assassin; the others reach the long games, sudden death and
# the wins, and the last one its losses on an assassin.
_POLICIES = ("any", "no assassin", "agents first", "no assassin until sudden death")
_SUDDEN_DEATH = (GamePhase.SUDDEN_DEATH_LLM, GamePhase.SUDDEN_DEATH_HUMAN)


def _resolves_to(card, seat: int) -> CardRole:
    """What touching ``card`` reveals for ``seat``: the partner's side of the key card."""
    return card.human_perspective_role if seat == 0 else card.llm_perspective_role


def _legal_actions(engine: CodenamesDuetEngine, policy: str) -> list:
    """The actions the engine must accept in its current state, as (name, callable)."""
    s = engine.state
    phase = s.current_phase
    if phase == GamePhase.GIVING_CLUE:
        visible = [card.text for card in s.board.cards if not card.is_covered]
        return [
            (
                f"clue {clue} {count}",
                lambda c=clue, n=count: engine.receive_clue(c, n, s.clue_giver),
            )
            for clue in (_VALID_CLUE, *visible[:3])
            for count in (1, 2, 4)
        ]
    seat = (
        s.guesser
        if phase == GamePhase.GUESSING
        else (0 if phase == GamePhase.SUDDEN_DEATH_LLM else 1)
    )
    cards = [card for card in s.board.cards if card.is_guessable_by(seat)]
    if policy == "no assassin until sudden death" and phase in _SUDDEN_DEATH:
        policy = "any"
    if policy != "any":
        safe = [card for card in cards if _resolves_to(card, seat) != CardRole.ASSASSIN]
        agents = [card for card in safe if _resolves_to(card, seat) == CardRole.AGENT]
        cards = (agents if policy == "agents first" and agents else safe) or cards
    actions = [
        (f"guess {card.text}", lambda c=card.id: engine.resolve_guess(c, seat)) for card in cards
    ]
    if phase == GamePhase.GUESSING and s.guesses_made_this_turn >= 1:
        actions.append(("stop", lambda: engine.pass_turn(seat)))
    if phase in _SUDDEN_DEATH:
        actions.append(("concede", lambda: engine.concede_sudden_death(seat)))
    return actions


def _refused_actions(engine: CodenamesDuetEngine) -> list:
    """Actions the engine must refuse in its current state, as (name, callable)."""
    s = engine.state
    phase = s.current_phase
    actions = [
        ("clue by the guesser", lambda: engine.receive_clue(_VALID_CLUE, 1, 1 - s.clue_giver)),
        ("stop out of turn", lambda: engine.pass_turn(1 - s.guesser)),
    ]
    if phase == GamePhase.GIVING_CLUE:
        actions.append(("clue for 0", lambda: engine.receive_clue(_VALID_CLUE, 0, s.clue_giver)))
        actions.append(("guess with no clue", lambda: engine.resolve_guess(0, s.guesser)))
    if phase == GamePhase.GUESSING:
        actions.append(("guess by the giver", lambda: engine.resolve_guess(0, s.clue_giver)))
        if s.guesses_made_this_turn == 0:
            actions.append(("stop before guessing", lambda: engine.pass_turn(s.guesser)))
        covered = [card.id for card in s.board.cards if card.is_covered]
        if covered:
            actions.append(
                ("guess a covered word", lambda: engine.resolve_guess(covered[0], s.guesser))
            )
    if phase in _SUDDEN_DEATH:
        other = 1 if phase == GamePhase.SUDDEN_DEATH_LLM else 0
        actions.append(("guess by the other seat", lambda: engine.resolve_guess(0, other)))
        actions.append(("concede by the other seat", lambda: engine.concede_sudden_death(other)))
    return actions


@settings(max_examples=300, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(data=st.data(), board=st.sampled_from(_BOARDS), seed=st.integers(0, 2**32 - 1))
def test_any_game_keeps_the_rules_and_ends(data, board, seed):
    engine = CodenamesDuetEngine(board, rng=random.Random(seed))
    policy = data.draw(st.sampled_from(_POLICIES))
    guard = InvariantGuard("property", strict=True)
    guard.check(engine)
    phases = set()
    for _ in range(_MAX_ACTIONS):
        if engine.state.is_game_over:
            break
        phases.add(engine.state.current_phase)
        # One draw per step picks the legal action and, one step in four, an action to try first
        # that the engine must refuse, and that must change nothing.
        choice = data.draw(st.integers(0, 2**16 - 1))
        legal = _legal_actions(engine, policy)
        if choice % 4 == 0:
            refusals = _refused_actions(engine)
            name, refused = refusals[(choice // 4) % len(refusals)]
            before = engine.state.model_dump()
            try:
                refused()
            except (ValueError, PermissionError):
                pass
            else:
                raise AssertionError(f"the engine accepted {name!r}")
            assert engine.state.model_dump() == before, (
                f"{name!r} was refused but changed the state"
            )
        _, action = legal[(choice // 64) % len(legal)]
        action()
        guard.check(engine)  # strict: a broken rule raises GameInvariantError
    event(f"result: {engine.state.result}")
    if phases & set(_SUDDEN_DEATH):
        event("reached sudden death")
    if engine.state.penalty_tokens:
        event("played an invalid clue")
    assert engine.state.is_game_over, f"the game did not end in {_MAX_ACTIONS} actions"
