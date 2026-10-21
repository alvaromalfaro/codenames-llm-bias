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
