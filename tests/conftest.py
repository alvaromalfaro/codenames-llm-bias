"""Test-session setup: a Postgres of its own for the database tests.

The tests marked ``db`` need a Postgres with pgvector, migrated to head. A session that collects any
of them gets one, and only then:

* ``TEST_DATABASE_URL`` set: that database, migrated to head first. Point it only at a database the
  tests may fill: they insert rows and never delete them.
* otherwise, a throwaway container of the image docker-compose.yml pins for ``db``, on a free
  loopback port and with its data in memory, migrated to head and removed when the session ends.
* neither (no ``TEST_DATABASE_URL`` and no Docker): the database tests are skipped, saying why.

``DATABASE_URL`` from the environment is never used: a shell or a container set up for the
development database (``.env`` sets it) cannot point the tests at it. ``pytest -m "not db"`` runs
the rest without starting anything.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import time
import uuid

import psycopg2
import pytest

from backend.app.paths import ALEMBIC_INI, REPO_ROOT

_COMPOSE_FILE = REPO_ROOT / "docker-compose.yml"
_CONTAINER_PREFIX = "codenames-test-db-"
_CREDENTIALS = "test"
# A first run pulls the image; after that the server is up in seconds.
_PULL_TIMEOUT_S = 600
_READY_TIMEOUT_S = 60


class _NoDatabase(Exception):
    """No database the tests may use: the reason the database tests are skipped."""


def pytest_configure(config: pytest.Config) -> None:
    os.environ.pop("DATABASE_URL", None)


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """After collection and deselection: get the database tests a database, or skip them."""
    db_items = [item for item in items if item.get_closest_marker("db")]
    if not db_items:
        return
    try:
        url = os.environ.get("TEST_DATABASE_URL") or _start_container(config)
        _migrate(url)
    except _NoDatabase as e:
        skip = pytest.mark.skip(reason=f"needs a Postgres: {e}")
        for item in db_items:
            item.add_marker(skip)
        return
    os.environ["DATABASE_URL"] = url


def _docker(*args: str, timeout: float = 60) -> str:
    """Run a docker command and return its output; ``_NoDatabase`` if it cannot be run or fails."""
    try:
        done = subprocess.run(["docker", *args], capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise _NoDatabase(f"no TEST_DATABASE_URL, and `docker {args[0]}` failed: {e}") from e
    if done.returncode != 0:
        raise _NoDatabase(f"no TEST_DATABASE_URL, and `docker {args[0]}` failed: "
                          f"{done.stderr.strip() or done.stdout.strip()}")
    return done.stdout.strip()


def _compose_db_image() -> str:
    """The pgvector image docker-compose.yml pins for ``db``, so the tests run on the batch's."""
    match = re.search(r"^\s*image:\s*(pgvector/pgvector:\S+)\s*$",
                      _COMPOSE_FILE.read_text(encoding="utf-8"), re.MULTILINE)
    if match is None:
        raise RuntimeError(f"no pgvector/pgvector image in {_COMPOSE_FILE}")
    return match.group(1)


def _start_container(config: pytest.Config) -> str:
    """Start a throwaway Postgres, stopped (and so removed) when the session ends; its URL."""
    _docker("info", timeout=30)
    name = _CONTAINER_PREFIX + uuid.uuid4().hex[:8]
    _docker("run", "--detach", "--rm", "--name", name,
            "--env", f"POSTGRES_USER={_CREDENTIALS}", "--env", f"POSTGRES_PASSWORD={_CREDENTIALS}",
            "--env", f"POSTGRES_DB={_CREDENTIALS}", "--publish", "127.0.0.1::5432",
            "--tmpfs", "/var/lib/postgresql/data", _compose_db_image(), timeout=_PULL_TIMEOUT_S)
    config.add_cleanup(lambda: subprocess.run(["docker", "stop", name], capture_output=True))
    port = _docker("port", name, "5432/tcp").splitlines()[0].rsplit(":", 1)[1]
    _wait_until_ready(port)
    return (f"postgresql+psycopg2://{_CREDENTIALS}:{_CREDENTIALS}@127.0.0.1:{port}/"
            f"{_CREDENTIALS}")


def _wait_until_ready(port: str) -> None:
    """Wait for the server to take a connection over TCP. While the image initialises the database,
    its temporary server listens on a socket only, so a TCP connection means the final one is up."""
    deadline = time.monotonic() + _READY_TIMEOUT_S
    while True:
        try:
            psycopg2.connect(host="127.0.0.1", port=port, user=_CREDENTIALS,
                             password=_CREDENTIALS, dbname=_CREDENTIALS, connect_timeout=2).close()
            return
        except psycopg2.OperationalError as e:
            if time.monotonic() > deadline:
                raise _NoDatabase(f"the test Postgres took no connection in "
                                  f"{_READY_TIMEOUT_S} s: {e}") from e
            time.sleep(0.5)


def _migrate(url: str) -> None:
    """``alembic upgrade head`` on ``url``, in a process of its own: migrations/env.py configures
    logging from alembic.ini, which here would silence the loggers the tests read."""
    done = subprocess.run(
        [sys.executable, "-m", "alembic", "-c", str(ALEMBIC_INI), "upgrade", "head"],
        env={**os.environ, "DATABASE_URL": url}, capture_output=True, text=True, cwd=REPO_ROOT)
    if done.returncode != 0:
        raise RuntimeError(f"alembic upgrade head failed on the test database:\n{done.stderr}")
