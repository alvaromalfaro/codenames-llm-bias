"""Tests for the Alembic migrations as shipped.

Most read the scripts only. The last, marked ``db``, compares the models with the test session's
database, which tests/conftest.py migrates to head."""

from pathlib import Path

import pytest
from alembic.config import Config
from alembic.script import ScriptDirectory

from backend.app.paths import ALEMBIC_INI

# Alembic records the applied revision in alembic_version.version_num, a VARCHAR(32): a longer id
# makes the upgrade fail at its very end, when it stamps the revision, and roll back whole.
_VERSION_NUM_LENGTH = 32

_SCRIPTS = ScriptDirectory.from_config(Config(str(ALEMBIC_INI)))
_REVISIONS = list(_SCRIPTS.walk_revisions())


@pytest.mark.parametrize("script", _REVISIONS, ids=lambda script: script.revision)
def test_every_revision_id_fits_the_version_table(script) -> None:
    assert len(script.revision) <= _VERSION_NUM_LENGTH


@pytest.mark.parametrize("script", _REVISIONS, ids=lambda script: script.revision)
def test_every_revision_id_is_its_file_name(script) -> None:
    assert script.revision == Path(script.path).stem


def test_the_migrations_are_one_line() -> None:
    """One head, one base, and each revision revises the one before it."""
    assert len(_SCRIPTS.get_heads()) == 1 and len(_SCRIPTS.get_bases()) == 1
    numbers = [int(script.revision.split("_", 1)[0]) for script in reversed(_REVISIONS)]
    assert numbers == list(range(1, len(numbers) + 1))


@pytest.mark.db
def test_the_models_and_the_migrated_schema_agree() -> None:
    """What `alembic check` does: autogenerate finds nothing to migrate between db/models.py and a
    database at head, so no change to the models is missing its migration."""
    from alembic.autogenerate import compare_metadata
    from alembic.migration import MigrationContext

    from backend.app.db.models import Base
    from backend.app.db.session import get_engine

    with get_engine().connect() as connection:
        differences = compare_metadata(MigrationContext.configure(connection), Base.metadata)
    assert differences == []


@pytest.mark.db
def test_0008_fills_the_durations_of_the_calls_recorded_before_it() -> None:
    """Its backfill, run on a temporary llm_call that shadows the table: an ollama call's durations
    come from its raw_payload, in milliseconds; a key ollama left out, and an API call, stay
    NULL."""
    import json

    from sqlalchemy import text

    from backend.app.db.session import get_engine

    backfill = _SCRIPTS.get_revision("0008_llm_call_durations").module.BACKFILL
    ollama = {
        "total_duration": 5_312_078_605,
        "load_duration": 201_789_487,
        "prompt_eval_duration": 2_754_571_000,
        "eval_duration": 1_695_363_000,
    }
    payloads = [ollama, {**ollama, "load_duration": None}, {"id": "gen-1", "provider": "xAI"}]

    with get_engine().connect() as connection:
        connection.execute(
            text(
                "CREATE TEMP TABLE llm_call "
                "(id int, raw_payload jsonb, load_ms int, prompt_eval_ms int, eval_ms int)"
            )
        )
        for i, payload in enumerate(payloads):
            connection.execute(
                text(
                    "INSERT INTO llm_call (id, raw_payload) VALUES (:id, CAST(:payload AS jsonb))"
                ),
                {"id": i, "payload": json.dumps(payload)},
            )
        connection.execute(text(backfill))
        filled = connection.execute(
            text("SELECT load_ms, prompt_eval_ms, eval_ms FROM llm_call ORDER BY id")
        ).all()
        connection.rollback()

    assert [tuple(row) for row in filled] == [
        (202, 2755, 1695),
        (None, 2755, 1695),
        (None, None, None),
    ]
