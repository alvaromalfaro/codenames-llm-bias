"""Metadata-only introspection tests for the ORM layer."""

from decimal import Decimal

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

from backend.app.db.models import Base

EXPECTED_TABLES = {
    "run",
    "measurement_frame",
    "extraction_recipe",
    "board",
    "word_card",
    "game",
    "game_seat",
    "turn",
    "llm_call",
    "clue",
    "clue_target",
    "reveal_event",
    "guess_proposal",
    "guess_proposal_item",
    "embedding_mpnet",
    "embedding_llama",
    "embedding_mistral",
    "word_load",
}


def _unique_column_sets(table) -> set[frozenset[str]]:
    """Return the set of column-name groups covered by UniqueConstraints on a table."""
    sets = set()
    for constraint in table.constraints:
        if isinstance(constraint, sa.UniqueConstraint):
            sets.add(frozenset(c.name for c in constraint.columns))
    return sets


def _check_texts(table) -> list[str]:
    return [
        str(c.sqltext)
        for c in table.constraints
        if isinstance(c, sa.CheckConstraint)
    ]


def test_all_expected_tables_present():
    assert set(Base.metadata.tables) == EXPECTED_TABLES


def test_vector_dimensions():
    tables = Base.metadata.tables
    assert isinstance(tables["measurement_frame"].c.gender_axis.type, Vector)
    assert tables["measurement_frame"].c.gender_axis.type.dim == 768
    assert tables["embedding_mpnet"].c.embedding.type.dim == 768
    assert tables["embedding_llama"].c.embedding.type.dim == 4096
    assert tables["embedding_mistral"].c.embedding.type.dim == 5120


def test_uuid_primary_keys_are_string_uuid():
    tables = Base.metadata.tables
    for tname in ("run", "game"):
        pk_cols = list(tables[tname].primary_key.columns)
        assert len(pk_cols) == 1
        col = pk_cols[0]
        assert isinstance(col.type, postgresql.UUID)
        # Stored as native UUID but with string ergonomics (as_uuid=False).
        assert col.type.as_uuid is False


def test_numeric_seed_columns_are_20_0():
    tables = Base.metadata.tables
    for tname, cname in (
        ("run", "master_seed"),
        ("board", "seed"),
        ("game", "derived_seed"),
        ("game_seat", "requested_seed"),
        ("llm_call", "requested_seed"),
    ):
        col = tables[tname].c[cname]
        assert isinstance(col.type, sa.Numeric)
        assert col.type.precision == 20
        assert col.type.scale == 0
    # NUMERIC reads back as Decimal.
    assert Decimal("42") == Decimal(42)


def test_bigint_identity_primary_keys():
    tables = Base.metadata.tables
    identity_pk_tables = EXPECTED_TABLES - {
        "run",
        "game",
        "board",
        "measurement_frame",
        "extraction_recipe",
    }
    for tname in identity_pk_tables:
        pk_cols = list(tables[tname].primary_key.columns)
        assert len(pk_cols) == 1, tname
        col = pk_cols[0]
        assert isinstance(col.type, sa.BigInteger), tname
        assert col.identity is not None and col.identity.always is True, tname


def test_word_card_columns_and_types():
    wc = Base.metadata.tables["word_card"]
    assert isinstance(wc.c.weat_set.type, postgresql.ARRAY)
    assert isinstance(wc.c.weat_set.type.item_type, sa.Text)
    assert isinstance(wc.c.subtlex_freq.type, (sa.Double, sa.Float))
    assert isinstance(wc.c.length.type, sa.Integer)
    assert isinstance(wc.c.wordnet_polysemy.type, sa.Integer)
    assert isinstance(wc.c.card_id.type, sa.SmallInteger)
    # board_id FK cascades on delete.
    fk = next(iter(wc.c.board_id.foreign_keys))
    assert fk.column.table.name == "board"
    assert fk.ondelete == "CASCADE"


def test_jsonb_columns():
    tables = Base.metadata.tables
    for tname, cname in (
        ("board", "arbiters"),
        ("board", "dilemma"),
        ("board", "keycard_audit"),
        ("run", "model_registry_snapshot"),
        ("llm_call", "raw_payload"),
        ("llm_call", "rendered_prompt"),
        ("clue", "targets_raw"),
    ):
        assert isinstance(tables[tname].c[cname].type,
                          postgresql.JSONB), (tname, cname)


def test_prompt_capture_columns():
    """0002 prompt-capture columns: both nullable, expected types, on their tables."""
    tables = Base.metadata.tables
    rendered_prompt = tables["llm_call"].c.rendered_prompt
    assert isinstance(rendered_prompt.type, postgresql.JSONB)
    assert rendered_prompt.nullable is True
    template_version = tables["run"].c.prompt_template_version
    assert isinstance(template_version.type, sa.Text)
    assert template_version.nullable is True


def test_unique_constraints():
    tables = Base.metadata.tables
    assert frozenset({"board_id", "card_id"}) in _unique_column_sets(
        tables["word_card"])
    assert frozenset({"board_id", "text"}) in _unique_column_sets(
        tables["word_card"])
    assert frozenset({"frame_id", "text"}) in _unique_column_sets(
        tables["embedding_mpnet"]
    )
    assert frozenset({"frame_id", "text"}) in _unique_column_sets(
        tables["word_load"])
    assert frozenset({"recipe_id", "text"}) in _unique_column_sets(
        tables["embedding_llama"]
    )
    assert frozenset({"recipe_id", "text"}) in _unique_column_sets(
        tables["embedding_mistral"]
    )
    assert frozenset({"game_id", "seat_index"}) in _unique_column_sets(
        tables["game_seat"]
    )
    assert frozenset({"game_id", "turn_number"}
                     ) in _unique_column_sets(tables["turn"])
    assert frozenset({"turn_id", "position_in_turn"}) in _unique_column_sets(
        tables["reveal_event"]
    )
    assert frozenset({"clue_id", "position"}) in _unique_column_sets(
        tables["clue_target"]
    )
    assert frozenset({"guess_proposal_id", "position"}) in _unique_column_sets(
        tables["guess_proposal_item"]
    )
    # 1:1 turn relationships are enforced by a single-column UNIQUE on turn_id.
    assert frozenset({"turn_id"}) in _unique_column_sets(tables["clue"])
    assert frozenset({"turn_id"}) in _unique_column_sets(
        tables["guess_proposal"])


def test_extra_indexes_present():
    def index_cols(table):
        return {frozenset(c.name for c in ix.columns) for ix in table.indexes}

    tables = Base.metadata.tables
    assert frozenset({"game_id"}) in index_cols(tables["llm_call"])
    assert frozenset({"turn_id"}) in index_cols(tables["llm_call"])
    assert frozenset({"turn_id"}) in index_cols(tables["reveal_event"])


def test_check_constraint_vocabularies():
    tables = Base.metadata.tables
    board_checks = " ".join(_check_texts(tables["board"]))
    assert "probe" in board_checks and "control" in board_checks
    turn_checks = " ".join(_check_texts(tables["turn"]))
    assert "sudden_death" in turn_checks
    game_checks = " ".join(_check_texts(tables["game"]))
    assert "in_progress" in game_checks and "aborted" in game_checks


def test_deferred_board_frame_check_absent():
    """The probe/control <-> frame CHECK is deferred and must not be present yet."""
    board = Base.metadata.tables["board"]
    names = {c.name for c in board.constraints}
    assert "ck_board_type_requires_frame" not in names
    for text in _check_texts(board):
        assert "measurement_frame_id" not in text
