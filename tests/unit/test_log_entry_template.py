"""Regression tests for the log-entry partial.

These tests drive the real Jinja environment the app uses, so a template edit that drops what the
log must show fails here rather than in a live game.
"""
from jinja2 import Environment, FileSystemLoader

from backend.app.core.engine import CodenamesDuetEngine
from backend.app.models.game_schemas import Board

_TEMPLATES_DIR = "backend/app/templates"


def _render_clue(engine: CodenamesDuetEngine, clue) -> str:
    template = Environment(loader=FileSystemLoader(_TEMPLATES_DIR)).get_template(
        "partials/_log_entry.html")
    return template.render(card=None, result="clue", state=engine.state, clue=clue, player="LLM")


def test_invalid_clue_shows_the_reason_and_the_penalty(valid_board_data: dict):
    """An invalid clue is played with a penalty token (§8.4): the log says why it was invalid and
    that a token was discarded. When the penalty takes the last token the turn is already over, so
    the entry shows the clue's own turn, not the state's."""
    engine = CodenamesDuetEngine(board=Board(**valid_board_data))
    engine.state.clue_giver, engine.state.guesser = 0, 1
    engine.state.timer_tokens = 1

    entry = engine.receive_clue("bucket", 1, 0)
    assert engine.state.turn_number == 2

    html = _render_clue(engine, entry)
    assert "Turn 1" in html
    assert "Invalid clue:" in html
    assert "is a visible word on the board." in html
    assert "timer token discarded" in html


def test_valid_clue_shows_no_penalty(valid_board_data: dict):
    engine = CodenamesDuetEngine(board=Board(**valid_board_data))
    engine.state.clue_giver, engine.state.guesser = 0, 1

    entry = engine.receive_clue("battle", 1, 0)

    html = _render_clue(engine, entry)
    assert '"battle"' in html
    assert "Invalid clue" not in html
    assert "timer token discarded" not in html
