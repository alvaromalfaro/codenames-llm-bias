"""Tests for the invariant checker of scripts/play_games.py: a game played by the rules stays clean
after every play, and each kind of broken state is reported."""
import importlib.util
import sys
from pathlib import Path

from backend.app.core.engine import CodenamesDuetEngine
from backend.app.models.game_schemas import Board, GamePhase

_PATH = Path(__file__).resolve().parents[2] / "scripts" / "play_games.py"
_SPEC = importlib.util.spec_from_file_location("play_games", _PATH)
play_games = importlib.util.module_from_spec(_SPEC)
sys.modules["play_games"] = play_games  # its dataclasses look their module up while being built
_SPEC.loader.exec_module(play_games)
check_invariants = play_games.check_invariants

LLM, HUMAN = 0, 1


def _engine(valid_board_data: dict) -> CodenamesDuetEngine:
    engine = CodenamesDuetEngine(board=Board(**valid_board_data))
    engine.state.clue_giver, engine.state.guesser = LLM, HUMAN
    return engine


def test_a_game_played_by_the_rules_breaks_no_invariant(valid_board_data: dict):
    """A shared agent, a miss that leaves a time token, a word found under the partner's token, a
    stop, an invalid clue's penalty and the stop forced by the last pending word: clean after each."""
    engine = _engine(valid_board_data)
    plays = [
        lambda: engine.receive_clue("battle", 2, LLM),
        lambda: engine.resolve_guess(card_id=1, player_id=HUMAN),   # BRICK, shared agent
        lambda: engine.resolve_guess(card_id=5, player_id=HUMAN),   # CAVE, beige for the LLM
        lambda: engine.receive_clue("insect", 1, HUMAN),
        lambda: engine.resolve_guess(card_id=5, player_id=LLM),     # CAVE, under the human's token
        lambda: engine.pass_turn(LLM),
        lambda: engine.receive_clue("bucket", 7, LLM),              # invalid: a visible word
        *[lambda card_id=card_id: engine.resolve_guess(card_id=card_id, player_id=HUMAN)
          for card_id in (4, 8, 11, 12, 15, 17, 19, 24)],          # the human's last 8 words
    ]
    assert check_invariants(engine) == []
    for play in plays:
        play()
        assert check_invariants(engine) == []
    assert engine.state.penalty_tokens == 1
    assert engine.state.pending_words[HUMAN] == 0


def test_broken_states_are_reported(valid_board_data: dict):
    """Each corruption of the state is caught, with a message naming it."""
    def problems(corrupt) -> str:
        engine = _engine(valid_board_data)
        corrupt(engine)
        return " | ".join(check_invariants(engine))

    def lose_a_token(engine):
        engine.state.timer_tokens -= 1

    def miscount_pending(engine):
        engine.state.pending_words[LLM] -= 1

    def token_on_a_giver_green(engine):
        engine.state.board.cards[1].time_marker_by = [HUMAN]  # BRICK is green for the LLM
        engine.state.bystander_tokens += 1
        engine.state.timer_tokens -= 1

    def hide_a_visible_word(engine):
        engine.clue_validator.remove_word("BUCKET")

    def guess_with_nothing_pending(engine):
        engine.state.current_phase = GamePhase.GUESSING
        engine.state.pending_words[HUMAN] = 0
        for card in engine.state.board.cards:  # cover the human's words so the counter is right
            if card.llm_perspective_role.value == "agent":
                card.revealed = True
                engine.clue_validator.remove_word(card.text)
        engine.state.pending_words[LLM] = 6  # 3 of its 9 were shared, and are now covered

    assert "do not add up to 9" in problems(lose_a_token)
    assert "pending_words" in problems(miscount_pending)
    assert "agent for the clue giver" in problems(token_on_a_giver_green)
    assert "visible words for clue validation" in problems(hide_a_visible_word)
    assert "guesses with nothing pending" in problems(guess_with_nothing_pending)
