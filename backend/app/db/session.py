"""Sync SQLAlchemy engine and session wiring.

Mirrors the ``os.environ.get`` idiom used in the backend rather than introducing a pydantic-settings
``BaseSettings`` class. ``DATABASE_URL`` is read on first use, not at import time, so this module 
imports cleanly even when ``DATABASE_URL`` is unset - failure is deferred to the first attempt to 
build an engine.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

_engine: Engine | None = None
_SessionLocal: sessionmaker[Session] | None = None


def _database_url() -> str:
    """Return the configured DATABASE_URL, normalised to the sync psycopg2 driver.

    Raises ``RuntimeError`` if ``DATABASE_URL`` is unset so callers can handle a missing 
    configuration explicitly.
    """
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL is not set")
    if url.startswith("postgresql://"):
        url = "postgresql+psycopg2://" + url[len("postgresql://"):]
    return url


def get_engine() -> Engine:
    """Return a lazily-constructed, process-wide sync engine."""
    global _engine
    if _engine is None:
        _engine = create_engine(
            _database_url(), future=True, pool_pre_ping=True)
    return _engine


def get_sessionmaker() -> sessionmaker[Session]:
    """Return a sessionmaker bound to the engine."""
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(
            bind=get_engine(), autoflush=False, expire_on_commit=False, future=True
        )
    return _SessionLocal


@contextmanager
def session_scope() -> Iterator[Session]:
    """Provide a transactional scope: commit on success, rollback on error, always close."""
    session = get_sessionmaker()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db() -> Iterator[Session]:
    """FastAPI dependency yielding a session (no implicit commit)."""
    session = get_sessionmaker()()
    try:
        yield session
    finally:
        session.close()
