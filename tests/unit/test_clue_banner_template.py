"""Regression tests for the clue-banner partial.

These tests drive the real Jinja environment the app uses, so a template edit that leaks hidden
information to the human fails here rather than in a live game.
"""
from jinja2 import Environment, FileSystemLoader

from backend.app.core.engine import CodenamesDuetEngine
from backend.app.models.game_schemas import Board, GamePhase

_TEMPLATES_DIR = "backend/app/templates"


def test_sudden_death_banner_does_not_reveal_how_many_agents_are_left(valid_board_data: dict):
    """Nobody is told how many words they still have to guess (§9). The human's number depends on
    the LLM's key side, so the human's sudden-death banner must not change with it."""
    template = Environment(loader=FileSystemLoader(_TEMPLATES_DIR)).get_template(
        "partials/_clue_banner.html")
    engine = CodenamesDuetEngine(board=Board(**valid_board_data))
    engine.state.current_phase = GamePhase.SUDDEN_DEATH_HUMAN

    def render(agents_left: int) -> str:
        engine.state.pending_words = [0, agents_left]
        return template.render(state=engine.state, game_id="g", oob=False)

    one_left = render(1)
    assert one_left == render(4)
    assert "Guess your remaining agents directly" in one_left
