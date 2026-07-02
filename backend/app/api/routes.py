import random
from pathlib import Path
from fastapi import APIRouter, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

from backend.app.core.loader import BoardLoader
from backend.app.core.engine import CodenamesDuetEngine
from backend.app.core.llm_service import LLMService
from backend.app.core.llm.client import LLMClient
from backend.app.core.llm.client_local import LLMClientLocal
from backend.app.core.llm.client_openrouter import LLMClientOpenRouter
from backend.app.config import llm_models
from backend.app.models.game_schemas import GamePhase


_TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"
_DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data"

templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))
router = APIRouter()

_board_loader = BoardLoader(data_path=str(_DATA_DIR / "boards"))
_model_providers = list(llm_models.keys())
_model_names = {provider: list(models.keys())
                for provider, models in llm_models.items()}
_llm_service = LLMService()
_games: dict[str, tuple[CodenamesDuetEngine, LLMClient]] = {}


@router.get("/")
async def home(request: Request):
    return templates.TemplateResponse(request, "index.html")


@router.get("/about")
async def about(request: Request):
    return templates.TemplateResponse(request, "about.html")


@router.get("/config")
async def game_configuration(request: Request):
    return templates.TemplateResponse(
        request, "game_config.html", {
            "providers": ["Auto"] + _model_providers,
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

    html = "".join(
        f'<option value="{model}">{model}</option>' for model in _model_names.get(model_provider, []))
    return HTMLResponse(content=html)


@router.get("/play")
async def play(request: Request, model_provider: str, bias_category: str, model_name: str | None = None):
    """
    Start the game engine, load the board configuration, and render the game page.
    """
    bias_boards = _board_loader.boards[bias_category]
    board = bias_boards[random.randint(0, len(bias_boards) - 1)]
    engine = CodenamesDuetEngine(board)

    if model_provider == "Auto":
        model_provider = random.choice(_model_providers)
        model_name = random.choice(_model_names.get(model_provider, []))

    model_config = llm_models.get(model_provider, {}).get(model_name, {})
    if model_provider == "OpenRouter":
        llm_client = LLMClientOpenRouter(model_name=model_name)
    else:
        llm_client = LLMClientLocal(
            model_name=model_name, think=model_config.get("think", False))

    _games[engine.state.game_id] = (engine, llm_client)

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
    game = _games.get(game_id)
    if not game:
        return HTMLResponse("Game not found.", status_code=404)

    engine, _ = game
    try:
        engine.receive_clue(clue, count, player_id)
    except (ValueError, PermissionError) as e:
        return HTMLResponse(f"<div class='text-red-500 text-sm p-2'>{str(e)}</div>", status_code=400)

    log_html = templates.get_template("partials/_log_entry.html").render({
        "card": None,
        "result": "clue",
        "state": engine.state,
        "clue": engine.state.current_clue,
        "player": "Human"
    })
    clue_html = templates.get_template("partials/_clue_banner.html").render({
        "state": engine.state,
        "game_id": game_id,
        "oob": True
    })

    return HTMLResponse(content=log_html + clue_html)


@router.post("/play/{game_id}/guess")
async def make_guess(game_id: str, card_id: int = Form(...), player_id: int = Form(...)):
    """
    Handle a guess made by a player, update the game state, and return the updated log and clue 
    banner.
    """
    game = _games.get(game_id)
    if not game:
        return HTMLResponse("Game not found.", status_code=404)

    engine, _ = game
    try:
        result = engine.resolve_guess(card_id, player_id)
    except (ValueError, PermissionError) as e:
        return HTMLResponse(f"<div class='text-red-500 text-sm p-2'>{str(e)}</div>", status_code=400)

    card = engine.state.board.cards[card_id]

    log_html = templates.get_template("partials/_log_entry.html").render({
        "card": card,
        "result": result,
        "state": engine.state,
        "player": "Human"
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
    stats_html = templates.get_template("partials/_game_stats.html").render({
        "state": engine.state,
        "oob": True
    })

    cards_html = ""
    if result in ("civilian", "assassin", "victory", "victory_sd"):
        cards_html = _render_cards_oob(engine, game_id)
        return HTMLResponse(content=log_html + cards_html + clue_html + stats_html)

    return HTMLResponse(content=log_html + card_html + clue_html + stats_html)


@router.post("/play/{game_id}/pass")
async def pass_turn(game_id: str, player_id: int = Form(...)):
    """
    Handle a pass action by a player, update the game state, and return the updated clue banner.
    """
    game = _games.get(game_id)
    if not game:
        return HTMLResponse("Game not found.", status_code=404)

    engine, _ = game
    try:
        engine.pass_turn(player_id)
    except (ValueError, PermissionError) as e:
        return HTMLResponse(f"<div class='text-red-500 text-sm p-2'>{str(e)}</div>", status_code=400)

    log_html = templates.get_template("partials/_log_entry.html").render({
        "card": None,
        "result": "pass",
        "state": engine.state,
        "player": "Human"
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
    cards_html = _render_cards_oob(engine, game_id)

    return HTMLResponse(content=log_html + clue_html + stats_html + cards_html)


@router.post("/play/{game_id}/llm-clue")
async def llm_give_clue(game_id: str):
    game = _games.get(game_id)
    if not game:
        return HTMLResponse("Game not found.", status_code=404)

    engine, llm_client = game

    proposal = await _llm_service.propose_clue(llm_client, engine.state, engine.clue_validator)

    engine.receive_clue(proposal.clue, proposal.count,
                        player_id=0, raw_payload=proposal.raw_payload)

    print(
        f"LLM proposed clue: {proposal.clue} ({proposal.count}) with reasoning: {proposal.reasoning}")

    cards_html = _render_cards_oob(engine, game_id)
    log_html = templates.get_template("partials/_log_entry.html").render({
        "card": None,
        "result": "clue",
        "state": engine.state,
        "clue": engine.state.current_clue,
        "player": "LLM"
    })
    clue_html = templates.get_template("partials/_clue_banner.html").render({
        "state": engine.state,
        "game_id": game_id,
        "oob": True
    })

    return HTMLResponse(content=cards_html + log_html + clue_html)


@router.post("/play/{game_id}/llm-guess")
async def llm_make_guess(game_id: str):
    game = _games.get(game_id)
    if not game:
        return HTMLResponse("Game not found.", status_code=404)

    engine, llm_client = game

    try:
        proposal = await _llm_service.propose_guess(llm_client, engine.state, player_id=0)
    except (ValueError, PermissionError) as e:
        print(f"Error during LLM guess proposal: {str(e)}")

    html = ""
    for word in proposal.proposals:
        card_id = engine.state.board.get_card_id_by_word(word)
        if card_id is None:
            # LLM hallucinated a word not on the board
            continue

        try:
            result = engine.resolve_guess(card_id, player_id=0)
        except (ValueError, PermissionError):
            break

        print(
            f"LLM proposed guess: {word}")

        card = engine.state.board.cards[card_id]

        html += templates.get_template("partials/_log_entry.html").render({
            "card": card, "result": result, "state": engine.state, "player": "LLM"
        })
        html += templates.get_template("partials/_card.html").render({
            "card": card, "game_id": game_id, "state": engine.state, "oob": True
        })
        html += templates.get_template("partials/_game_stats.html").render({
            "state": engine.state, "oob": True
        })

        if result != "agent":
            # civilian, assassin, or victory - turn or game ended
            if result in ("civilian", "assassin", "victory", "victory_sd"):
                html += _render_cards_oob(engine, game_id)
            break
    else:
        # All proposals were correct agents - LLM has no more guesses, pass the turn
        try:
            engine.pass_turn(0)
        except (ValueError, PermissionError):
            pass

    html += templates.get_template("partials/_clue_banner.html").render({
        "state": engine.state, "game_id": game_id, "oob": True
    })

    return HTMLResponse(content=html)


@router.post("/play/{game_id}/llm-sd-guess")
async def llm_make_guess_sd(game_id: str):
    """
    Handle the LLM's sudden death guessing phase. The LLM proposes guesses without a clue,
    relying on the full clue history to identify its remaining agents.
    """
    game = _games.get(game_id)
    if not game:
        return HTMLResponse("Game not found.", status_code=404)

    engine, llm_client = game

    if engine.state.current_phase != GamePhase.SUDDEN_DEATH_LLM:
        return HTMLResponse("Not in LLM sudden death phase.", status_code=400)

    try:
        proposal = await _llm_service.propose_guess_sd(llm_client, engine.state, player_id=0)
    except (ValueError, PermissionError) as e:
        print(f"Error during LLM sudden death guess proposal: {str(e)}")
        return HTMLResponse(f"<div class='text-red-500 text-sm p-2'>{str(e)}</div>", status_code=400)

    html = ""
    for word in proposal.proposals:
        card_id = engine.state.board.get_card_id_by_word(word)
        if card_id is None:
            continue

        try:
            result = engine.resolve_guess(card_id, player_id=0)
        except (ValueError, PermissionError):
            break

        print(f"LLM sudden death guess: {word}")

        card = engine.state.board.cards[card_id]
        html += templates.get_template("partials/_log_entry.html").render({
            "card": card, "result": result, "state": engine.state, "player": "LLM"
        })
        html += templates.get_template("partials/_card.html").render({
            "card": card, "game_id": game_id, "state": engine.state, "oob": True
        })
        html += templates.get_template("partials/_game_stats.html").render({
            "state": engine.state, "oob": True
        })

        if result != "agent":
            if result in ("victory_sd", "victory"):
                html += _render_cards_oob(engine, game_id)
            break

    html += templates.get_template("partials/_clue_banner.html").render({
        "state": engine.state, "game_id": game_id, "oob": True
    })
    return HTMLResponse(content=html)


def _render_cards_oob(engine: CodenamesDuetEngine, game_id: str) -> str:
    return "".join(
        templates.get_template("partials/_card.html").render({
            "card": card,
            "game_id": game_id,
            "state": engine.state,
            "oob": True
        })
        for card in engine.state.board.cards
    )
