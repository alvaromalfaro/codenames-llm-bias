from pathlib import Path
from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse


_TEMPLATES_DIR = Path(__file__).resolve().parent.parent.parent / "templates"
templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))
router = APIRouter()


@router.get("/")
async def home(request: Request):
    return templates.TemplateResponse(request, "index.html")


@router.get("/config")
async def game_configuration(request: Request):
    # TODO: Implement provider and bias category retrieval logic
    return templates.TemplateResponse(
        request, "game_config.html", {
            "providers": ["Auto", "Ollama"],
            "bias_categories": ["example"]
        }
    )


@router.get("/config/models")
async def get_models(request: Request, model_provider: str | None = None):
    """
    Retrieves available models based on the selected provider and returns them as HTML options.
    """
    # TODO: Implement model retrieval logic based on provider
    if not model_provider or model_provider == "Auto":
        return HTMLResponse("")

    models = ["llama3.2:latest"]

    html = "".join(
        f'<option value="{model}">{model}</option>' for model in models)
    return HTMLResponse(content=html)
