from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from backend.app.api.routes import router

_BASE_DIR = Path(__file__).resolve().parent.parent
_DATA_DIR = _BASE_DIR.parent / "data"
if not _DATA_DIR.exists():
    _DATA_DIR = _BASE_DIR / "data"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup tasks
    print("Starting up the application...")
    yield
    # Shutdown tasks
    print("Shutting down the application...")

app = FastAPI(title="Codenames Duet LLM Bias", lifespan=lifespan)
app.include_router(router)
app.mount("/static", StaticFiles(directory="backend/app/static"), name="static")
