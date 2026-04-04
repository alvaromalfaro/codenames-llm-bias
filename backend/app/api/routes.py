import random
from pathlib import Path
from fastapi import APIRouter, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

from backend.app.core.loader import BoardLoader
from backend.app.core.engine import CodenamesDuetEngine
from backend.app.config.llm_models import llm_models


_TEMPLATES_DIR = Path(__file__).resolve().parent.parent.parent / "templates"
_DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data"
_TEMPLATES_DIR = Path(__file__).resolve().parent.parent.parent / "templates"

templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))
router = APIRouter()

_board_loader = BoardLoader(data_path=str(_DATA_DIR / "boards"))
_games: dict[str, CodenamesDuetEngine] = {}


@router.get("/")
async def home(request: Request):
    return templates.TemplateResponse(request, "index.html")


@router.get("/config")
async def game_configuration(request: Request):
    return templates.TemplateResponse(
        request, "game_config.html", {
            "providers": ["Auto"] + list(llm_models.keys()),
            "bias_categories": _board_loader.boards.keys()
        }
    )


@router.get("/config/models")
async def get_models(request: Request, model_provider: str | None = None):
    """
    Retrieves available models based on the selected provider and returns them as HTML options.
    """
    if not model_provider or model_provider == "Auto":
        return HTMLResponse("")

    models = [model["name"] for model in llm_models.get(model_provider, [])]

    html = "".join(
        f'<option value="{model}">{model}</option>' for model in models)
    return HTMLResponse(content=html)


@router.get("/play")
async def play(request: Request, model_provider: str, bias_category: str, model_name: str | None = None):
    """
    Start the game engine, load the board configuration, and render the game page.
    """
    board = _board_loader.load_board("example_board.json")
    engine = CodenamesDuetEngine(board)
    _games[engine.state.game_id] = engine

    if model_provider == "Auto":
        # Auto-select the provider and model randomly
        model_provider = random.choice(list(llm_models.keys()))
        model_name = random.choice(llm_models[model_provider])["name"]

    return templates.TemplateResponse(request, "game.html", {
        "model_provider": model_provider,
        "bias_category": bias_category,
        "model_name": model_name,
        "state": engine.state,
        "game_id": engine.state.game_id,
    })


@router.post("/play/{game_id}/clue")
async def give_clue(game_id: str, clue: str = Form(...), count: int = Form(...), player_id: int = Form(...)):
    """

    """
    engine = _games.get(game_id)
    if not engine:
        return HTMLResponse("Game not found.", status_code=404)

    try:
        engine.receive_clue(clue, count, player_id)
    except (ValueError, PermissionError) as e:
        return HTMLResponse(f"<div class='text-red-500 text-sm p-2'>{str(e)}</div>", status_code=400)

    log_html = templates.get_template("partials/_log_entry.html").render({
        "card": None,
        "result": "clue",
        "state": engine.state,
        "clue": engine.state.current_clue
    })
    clue_html = templates.get_template("partials/_clue_banner.html").render({
        "state": engine.state,
        "game_id": game_id,
        "oob": True
    })

    return HTMLResponse(content=log_html + clue_html)


@router.post("/play/{game_id}/guess")
async def make_guess(game_id: str, card_id: str = Form(...), player_id: int = Form(...)):
    """
    Handle a guess made by a player, update the game state, and return the updated log and clue 
    banner.
    """
    engine = _games.get(game_id)
    if not engine:
        return HTMLResponse("Game not found.", status_code=404)

    try:
        result = engine.resolve_guess(card_id, player_id)
    except (ValueError, PermissionError) as e:
        return HTMLResponse(f"<div class='text-red-500 text-sm p-2'>{str(e)}</div>", status_code=400)

    card = engine.state.board.cards.get(card_id)

    log_html = templates.get_template("partials/_log_entry.html").render({
        "card": card,
        "result": result,
        "state": engine.state
    })
    card_html = templates.get_template("partials/_card.html").render({
        "card": card,
        "game_id": game_id,
        "state": engine.state,
        "oob": True
    })
    clue_html = templates.get_template("partials/_clue_banner.html").render({
        "state": engine.state,
        "game_id": game_id,
        "oob": True
    })
    stats_html = templates.get_template("partials/_stats.html").render({
        "state": engine.state,
        "oob": True
    })

    return HTMLResponse(content=log_html + card_html + clue_html + stats_html)


@router.post("/play/{game_id}/pass")
async def pass_turn(game_id: str, player_id: int = Form(...)):
    """
    Handle a pass action by a player, update the game state, and return the updated clue banner.
    """
    engine = _games.get(game_id)
    if not engine:
        return HTMLResponse("Game not found.", status_code=404)

    try:
        engine.pass_turn(player_id)
    except (ValueError, PermissionError) as e:
        return HTMLResponse(f"<div class='text-red-500 text-sm p-2'>{str(e)}</div>", status_code=400)

    log_html = templates.get_template("partials/_log_entry.html").render({
        "card": None,
        "result": "pass",
        "state": engine.state
    })
    clue_html = templates.get_template("partials/_clue_banner.html").render({
        "state": engine.state,
        "game_id": game_id,
        "oob": True
    })
    stats_html = templates.get_template("partials/_game_stats.html").render({
        "state": engine.state,
        "oob": True
    })

    return HTMLResponse(content=log_html + clue_html + stats_html)
