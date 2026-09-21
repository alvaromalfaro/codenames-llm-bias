"""Regression tests for the board-card partial.

The rules-critical property of `_card.html` is that it must never offer the human a card the
engine would refuse. When the two disagree the human clicks a word that is already out of play,
the click is rejected (or, before the coverage fix, silently resolved against the other face),
and the UI stops being a faithful view of the game.

These tests drive the real Jinja environment the app uses, so a template edit that breaks the
agreement fails here rather than in a live game.
"""
import pytest
from jinja2 import Environment, FileSystemLoader

from backend.app.core.engine import CodenamesDuetEngine
from backend.app.models.game_schemas import Board, GamePhase

_TEMPLATES_DIR = "backend/app/templates"

LLM, HUMAN = 0, 1

# Fixture-board cards used below, by id.
BUCKET, BRICK, ANT, LEMONADE, RUSSIA = 0, 1, 2, 3, 4


@pytest.fixture
def card_template():
    return Environment(loader=FileSystemLoader(_TEMPLATES_DIR)).get_template(
        "partials/_card.html")


@pytest.fixture
def engine_with_marked_board(valid_board_data: dict) -> CodenamesDuetEngine:
    """An engine whose board carries every card state the guesser view has to tell apart.

    - BUCKET: the human's own time token -> out of reach for the human, still open to the LLM.
    - BRICK: the LLM's time token -> out of reach for the LLM, still open to the human.
    - ANT: covered by an agent card the LLM placed -> out of reach for BOTH seats, even though
      the human never touched it and is absent from revealed_by.
    - LEMONADE: two time tokens -> covered, out of reach for both.
    - RUSSIA: untouched.
    """
    engine = CodenamesDuetEngine(board=Board(**valid_board_data))
    cards = engine.state.board.cards
    cards[BUCKET].time_marker_by = [HUMAN]
    cards[BRICK].time_marker_by = [LLM]
    cards[ANT].revealed = True
    cards[ANT].revealed_by = [LLM]
    cards[LEMONADE].time_marker_by = [LLM, HUMAN]
    return engine


def _render(template, engine: CodenamesDuetEngine, card_id: int) -> str:
    return template.render(card=engine.state.board.cards[card_id],
                           state=engine.state, game_id="g", oob=False)


@pytest.mark.parametrize("card_id", [BUCKET, BRICK, ANT, LEMONADE, RUSSIA])
def test_guesser_view_is_clickable_exactly_when_the_engine_allows_the_guess(
        card_template, engine_with_marked_board, card_id: int):
    """A card is clickable in the human guesser's view if and only if resolve_guess would accept
    it from the human. Coverage is seat-blind, so the card the LLM covered is inert too."""
    engine = engine_with_marked_board
    engine.state.clue_giver, engine.state.guesser = LLM, HUMAN
    engine.state.current_phase = GamePhase.GUESSING

    html = _render(card_template, engine, card_id)
    card = engine.state.board.cards[card_id]

    assert ("hx-post" in html) is card.is_guessable_by(HUMAN)
    # A word still within reach must not be struck through.
    assert ("line-through" in html) is not card.is_guessable_by(HUMAN)


@pytest.mark.parametrize("card_id", [BUCKET, BRICK, ANT, LEMONADE, RUSSIA])
def test_clue_giver_view_strikes_exactly_the_words_the_guesser_cannot_take(
        card_template, engine_with_marked_board, card_id: int):
    """The human clue giver's view strikes a word out if and only if the LLM can no longer touch
    it - including a word covered by an agent card that the human alone found."""
    engine = engine_with_marked_board
    engine.state.clue_giver, engine.state.guesser = HUMAN, LLM
    engine.state.current_phase = GamePhase.GIVING_CLUE

    html = _render(card_template, engine, card_id)
    card = engine.state.board.cards[card_id]

    assert ("line-through" in html) is not card.is_guessable_by(LLM)


def test_guesser_view_hides_a_card_covered_by_the_other_seat(
        card_template, engine_with_marked_board):
    """The specific regression: ANT is an agent on the human's face and an ASSASSIN on the LLM's.
    Once the LLM covers it, the human must not be able to click it - doing so used to resolve
    against the LLM's face and lose the game."""
    engine = engine_with_marked_board
    engine.state.clue_giver, engine.state.guesser = LLM, HUMAN
    engine.state.current_phase = GamePhase.GUESSING

    html = _render(card_template, engine, ANT)

    assert "hx-post" not in html
    assert "AGENT" in html


def test_sudden_death_view_also_hides_covered_cards(
        card_template, engine_with_marked_board):
    """The human's sudden-death view shares the guesser branch, so coverage must hold there too -
    in sudden death any wrong touch ends the game outright."""
    engine = engine_with_marked_board
    engine.state.current_phase = GamePhase.SUDDEN_DEATH_HUMAN

    assert "hx-post" not in _render(card_template, engine, ANT)
    assert "hx-post" not in _render(card_template, engine, LEMONADE)
    assert "hx-post" in _render(card_template, engine, RUSSIA)
