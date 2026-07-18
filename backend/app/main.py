import logging
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from backend.app.api.routes import router
from backend.app.db.ingest_boards import ingest_boards_if_absent
from backend.app.db.ingest_frame import ingest_frame_if_absent
from backend.app.db.session import session_scope

logger = logging.getLogger(__name__)

_BASE_DIR = Path(__file__).resolve().parent.parent
_DATA_DIR = _BASE_DIR.parent / "data"
if not _DATA_DIR.exists():
    _DATA_DIR = _BASE_DIR / "data"


def _ingest_frame_on_startup() -> None:
    """Ingest the measurement frame before boards, so the board FK target exists.

    The board.measurement_frame_id FK is enforced (immediate): a sealed board cannot be inserted
    until its frame row exists. Same defensive contract as board ingestion, a missing DATABASE_URL
    or unreachable database is caught and logged, never fatal.
    """
    try:
        with session_scope() as session:
            inserted = ingest_frame_if_absent(
                session, _DATA_DIR / "boards" / "measurement_frame.json"
            )
        logger.info(
            "Measurement-frame ingestion complete (inserted=%s).", inserted)
    except Exception as exc:
        logger.warning(
            "Skipping measurement-frame ingestion (database unavailable): %s", exc)


def _ingest_boards_on_startup() -> None:
    """Ingest board artifacts defensively.

    A missing DATABASE_URL or an unreachable database must not break app startup, so any failure is 
    caught and logged as a warning.
    """
    try:
        with session_scope() as session:
            inserted = ingest_boards_if_absent(session, _DATA_DIR / "boards")
        logger.info("Board ingestion complete (%d new boards).", inserted)
    except Exception as exc:
        logger.warning(
            "Skipping board ingestion (database unavailable): %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup tasks: frame before boards (FK ordering).
    print("Starting up the application...")
    _ingest_frame_on_startup()
    _ingest_boards_on_startup()
    yield
    # Shutdown tasks
    print("Shutting down the application...")

app = FastAPI(title="Codenames Duet LLM Bias", lifespan=lifespan)
app.include_router(router)
app.mount("/static", StaticFiles(directory="backend/app/static"), name="static")
