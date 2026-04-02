from pathlib import Path
from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates


_TEMPLATES_DIR = Path(__file__).resolve().parent.parent.parent / "templates"
templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))
router = APIRouter()


@router.get("/")
async def home(request: Request):
    return templates.TemplateResponse(request, "index.html")
