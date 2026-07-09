"""initial persistence schema

Revision ID: 0001_initial
Revises:
Create Date: 2026-07-09

Creates the initial persistence schema. Tables are created in FK-dependency order; the ``vector`` 
extension is created before any table with a vector column.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0001_initial"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Must precede any table with a vector column.
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "run",
        sa.Column("id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("master_seed", sa.Numeric(20, 0), nullable=True),
        sa.Column("temperature", sa.Double(), nullable=False),
        sa.Column("regime_label", sa.Text(), nullable=True),
        sa.Column("code_version", sa.Text(), nullable=True),
        sa.Column("model_registry_snapshot",
                  postgresql.JSONB(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "measurement_frame",
        sa.Column("frame_id", sa.Text(), nullable=False),
        sa.Column("encoder_name", sa.Text(), nullable=False),
        sa.Column("encoder_revision", sa.Text(), nullable=False),
        sa.Column("encoder_pooling", sa.Text(), nullable=False),
        sa.Column("encoder_normalize", sa.Boolean(), nullable=False),
        sa.Column("gender_axis", Vector(768), nullable=False),
        sa.Column("axis_construction", postgresql.JSONB(), nullable=False),
        sa.Column("generator_version", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("frame_id"),
    )

    op.create_table(
        "extraction_recipe",
        sa.Column("recipe_id", sa.Text(), nullable=False),
        sa.Column("model_name", sa.Text(), nullable=False),
        sa.Column("model_revision", sa.Text(), nullable=False),
        sa.Column("precision", sa.Text(), nullable=False),
        sa.Column("layer", sa.Text(), nullable=False),
        sa.Column("pooling", sa.Text(), nullable=False),
        sa.Column("template", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("recipe_id"),
    )

    op.create_table(
        "board",
        sa.Column("board_id", sa.Text(), nullable=False),
        sa.Column("measurement_frame_id", sa.Text(), nullable=True),
        sa.Column("type", sa.Text(), nullable=True),
        sa.Column("category", sa.Text(), nullable=True),
        sa.Column("specification", sa.Text(), nullable=True),
        sa.Column("seed", sa.Numeric(20, 0), nullable=True),
        sa.Column("grid_rows", sa.SmallInteger(), nullable=True),
        sa.Column("grid_cols", sa.SmallInteger(), nullable=True),
        sa.Column("arbiters", postgresql.JSONB(), nullable=True),
        sa.Column("dilemma", postgresql.JSONB(), nullable=True),
        sa.Column("keycard_audit", postgresql.JSONB(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint("type IN ('probe','control')",
                           name="ck_board_type"),
        # DEFERRED - couples board probe/control to a measurement frame.
        # sa.CheckConstraint(
        #     "type IS NULL OR measurement_frame_id IS NOT NULL",
        #     name="ck_board_type_requires_frame",
        # ),
        sa.ForeignKeyConstraint(
            ["measurement_frame_id"], ["measurement_frame.frame_id"]
        ),
        sa.PrimaryKeyConstraint("board_id"),
    )

    op.create_table(
        "word_card",
        sa.Column("id", sa.BigInteger(), sa.Identity(
            always=True), nullable=False),
        sa.Column("board_id", sa.Text(), nullable=False),
        sa.Column("card_id", sa.SmallInteger(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("llm_perspective_role", sa.Text(), nullable=False),
        sa.Column("human_perspective_role", sa.Text(), nullable=False),
        sa.Column("category", sa.Text(), nullable=True),
        sa.Column("source", sa.Text(), nullable=True),
        sa.Column("weat_set", postgresql.ARRAY(sa.Text()), nullable=True),
        sa.Column("subtlex_freq", sa.Double(), nullable=True),
        sa.Column("length", sa.Integer(), nullable=True),
        sa.Column("wordnet_polysemy", sa.Integer(), nullable=True),
        sa.CheckConstraint(
            "card_id BETWEEN 0 AND 24", name="ck_word_card_card_id_range"
        ),
        sa.CheckConstraint(
            "llm_perspective_role IN ('agent','assassin','civilian')",
            name="ck_word_card_llm_role",
        ),
        sa.CheckConstraint(
            "human_perspective_role IN ('agent','assassin','civilian')",
            name="ck_word_card_human_role",
        ),
        sa.CheckConstraint(
            "category IN ('male','female','neutral')", name="ck_word_card_category"
        ),
        sa.ForeignKeyConstraint(
            ["board_id"], ["board.board_id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("board_id", "card_id",
                            name="uq_word_card_board_card"),
        sa.UniqueConstraint("board_id", "text",
                            name="uq_word_card_board_text"),
    )

    op.create_table(
        "game",
        sa.Column("id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("run_id", postgresql.UUID(as_uuid=False), nullable=True),
        sa.Column("board_id", sa.Text(), nullable=False),
        sa.Column("derived_seed", sa.Numeric(20, 0), nullable=True),
        sa.Column("start_player", sa.SmallInteger(), nullable=True),
        sa.Column(
            "game_status",
            sa.Text(),
            server_default=sa.text("'in_progress'"),
            nullable=False,
        ),
        sa.Column("result", sa.Text(), nullable=True),
        sa.Column("timer_tokens_final", sa.SmallInteger(), nullable=True),
        sa.Column(
            "started_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint("start_player IN (0,1)",
                           name="ck_game_start_player"),
        sa.CheckConstraint(
            "game_status IN ('in_progress','completed','aborted','error')",
            name="ck_game_status",
        ),
        sa.ForeignKeyConstraint(["board_id"], ["board.board_id"]),
        sa.ForeignKeyConstraint(["run_id"], ["run.id"]),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "game_seat",
        sa.Column("id", sa.BigInteger(), sa.Identity(
            always=True), nullable=False),
        sa.Column("game_id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("seat_index", sa.SmallInteger(), nullable=False),
        sa.Column("model_ref", sa.Text(), nullable=True),
        sa.Column("provider", sa.Text(), nullable=True),
        sa.Column("precision", sa.Text(), nullable=True),
        sa.Column("requested_temperature", sa.Double(), nullable=True),
        sa.Column("requested_seed", sa.Numeric(20, 0), nullable=True),
        sa.CheckConstraint("seat_index IN (0,1)",
                           name="ck_game_seat_seat_index"),
        sa.CheckConstraint(
            "provider IN ('ollama','openrouter','human')", name="ck_game_seat_provider"
        ),
        sa.ForeignKeyConstraint(["game_id"], ["game.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("game_id", "seat_index",
                            name="uq_game_seat_game_seat"),
    )

    op.create_table(
        "turn",
        sa.Column("id", sa.BigInteger(), sa.Identity(
            always=True), nullable=False),
        sa.Column("game_id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("turn_number", sa.Integer(), nullable=False),
        sa.Column("clue_giver_seat", sa.SmallInteger(), nullable=False),
        sa.Column(
            "phase", sa.Text(), server_default=sa.text("'normal'"), nullable=False
        ),
        sa.CheckConstraint(
            "clue_giver_seat IN (0,1)", name="ck_turn_clue_giver_seat"
        ),
        sa.CheckConstraint(
            "phase IN ('normal','sudden_death')", name="ck_turn_phase"
        ),
        sa.ForeignKeyConstraint(["game_id"], ["game.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("game_id", "turn_number",
                            name="uq_turn_game_turn_number"),
    )

    op.create_table(
        "llm_call",
        sa.Column("id", sa.BigInteger(), sa.Identity(
            always=True), nullable=False),
        sa.Column("game_id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("turn_id", sa.BigInteger(), nullable=True),
        sa.Column("seat_index", sa.SmallInteger(), nullable=False),
        sa.Column("role", sa.Text(), nullable=False),
        sa.Column(
            "retry_index", sa.Integer(), server_default=sa.text("0"), nullable=False
        ),
        sa.Column("provider", sa.Text(), nullable=True),
        sa.Column("model_used", sa.Text(), nullable=True),
        sa.Column("resolved_model", sa.Text(), nullable=True),
        sa.Column("system_fingerprint", sa.Text(), nullable=True),
        sa.Column("requested_temperature", sa.Double(), nullable=True),
        sa.Column("requested_seed", sa.Numeric(20, 0), nullable=True),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column("prompt_tokens", sa.Integer(), nullable=True),
        sa.Column("completion_tokens", sa.Integer(), nullable=True),
        sa.Column("finish_reason", sa.Text(), nullable=True),
        sa.Column("execution_mode", sa.Text(), nullable=True),
        sa.Column("request_id", sa.Text(), nullable=True),
        sa.Column("raw_payload", postgresql.JSONB(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint("seat_index IN (0,1)",
                           name="ck_llm_call_seat_index"),
        sa.CheckConstraint(
            "role IN ('clue_giver','guesser','guesser_sd')", name="ck_llm_call_role"
        ),
        sa.ForeignKeyConstraint(["game_id"], ["game.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["turn_id"], ["turn.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_llm_call_game_id", "llm_call", ["game_id"])
    op.create_index("ix_llm_call_turn_id", "llm_call", ["turn_id"])

    op.create_table(
        "clue",
        sa.Column("id", sa.BigInteger(), sa.Identity(
            always=True), nullable=False),
        sa.Column("turn_id", sa.BigInteger(), nullable=False),
        sa.Column("llm_call_id", sa.BigInteger(), nullable=True),
        sa.Column("clue_word", sa.Text(), nullable=False),
        sa.Column("count", sa.Integer(), nullable=False),
        sa.Column("reasoning", sa.Text(), nullable=True),
        sa.Column(
            "targets_raw",
            postgresql.JSONB(),
            server_default=sa.text("'[]'"),
            nullable=False,
        ),
        sa.CheckConstraint("count >= 1", name="ck_clue_count_positive"),
        sa.ForeignKeyConstraint(["llm_call_id"], ["llm_call.id"]),
        sa.ForeignKeyConstraint(["turn_id"], ["turn.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("turn_id"),
    )

    op.create_table(
        "clue_target",
        sa.Column("id", sa.BigInteger(), sa.Identity(
            always=True), nullable=False),
        sa.Column("clue_id", sa.BigInteger(), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("word", sa.Text(), nullable=False),
        sa.Column("card_id", sa.SmallInteger(), nullable=True),
        sa.Column("giver_role", sa.Text(), nullable=True),
        sa.Column("revealed_at_clue", sa.Boolean(), nullable=True),
        sa.CheckConstraint(
            "giver_role IN ('agent','assassin','civilian')",
            name="ck_clue_target_giver_role",
        ),
        sa.ForeignKeyConstraint(["clue_id"], ["clue.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("clue_id", "position",
                            name="uq_clue_target_clue_position"),
    )

    op.create_table(
        "reveal_event",
        sa.Column("id", sa.BigInteger(), sa.Identity(
            always=True), nullable=False),
        sa.Column("turn_id", sa.BigInteger(), nullable=False),
        sa.Column("position_in_turn", sa.Integer(), nullable=False),
        sa.Column("card_id", sa.SmallInteger(), nullable=False),
        sa.Column("acting_seat", sa.SmallInteger(), nullable=False),
        sa.Column("result_role", sa.Text(), nullable=False),
        sa.Column(
            "ended_turn", sa.Boolean(), server_default=sa.text("false"), nullable=False
        ),
        sa.Column(
            "ended_game", sa.Boolean(), server_default=sa.text("false"), nullable=False
        ),
        sa.Column(
            "time_marker_placed",
            sa.Boolean(),
            server_default=sa.text("false"),
            nullable=False,
        ),
        sa.Column("timer_tokens_after", sa.SmallInteger(), nullable=True),
        sa.CheckConstraint("acting_seat IN (0,1)",
                           name="ck_reveal_event_acting_seat"),
        sa.CheckConstraint(
            "result_role IN ('agent','assassin','civilian')",
            name="ck_reveal_event_result_role",
        ),
        sa.ForeignKeyConstraint(["turn_id"], ["turn.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "turn_id", "position_in_turn", name="uq_reveal_event_turn_position"
        ),
    )
    op.create_index("ix_reveal_event_turn_id", "reveal_event", ["turn_id"])

    op.create_table(
        "guess_proposal",
        sa.Column("id", sa.BigInteger(), sa.Identity(
            always=True), nullable=False),
        sa.Column("turn_id", sa.BigInteger(), nullable=False),
        sa.Column("llm_call_id", sa.BigInteger(), nullable=True),
        sa.Column("guesser_seat", sa.SmallInteger(), nullable=False),
        sa.Column("reasoning", sa.Text(), nullable=True),
        sa.Column("stop_reason", sa.Text(), nullable=True),
        sa.CheckConstraint(
            "guesser_seat IN (0,1)", name="ck_guess_proposal_guesser_seat"
        ),
        sa.ForeignKeyConstraint(["llm_call_id"], ["llm_call.id"]),
        sa.ForeignKeyConstraint(["turn_id"], ["turn.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("turn_id"),
    )

    op.create_table(
        "guess_proposal_item",
        sa.Column("id", sa.BigInteger(), sa.Identity(
            always=True), nullable=False),
        sa.Column("guess_proposal_id", sa.BigInteger(), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("word", sa.Text(), nullable=False),
        sa.Column("confidence", sa.Double(), nullable=True),
        sa.Column("resolved_card_id", sa.SmallInteger(), nullable=True),
        sa.Column("reveal_event_id", sa.BigInteger(), nullable=True),
        sa.CheckConstraint(
            "confidence >= 0.0 AND confidence <= 1.0",
            name="ck_guess_proposal_item_confidence",
        ),
        sa.ForeignKeyConstraint(
            ["guess_proposal_id"], ["guess_proposal.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(["reveal_event_id"], ["reveal_event.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "guess_proposal_id", "position", name="uq_guess_proposal_item_position"
        ),
    )

    op.create_table(
        "embedding_mpnet",
        sa.Column("id", sa.BigInteger(), sa.Identity(
            always=True), nullable=False),
        sa.Column("frame_id", sa.Text(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("kind", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(768), nullable=False),
        sa.CheckConstraint(
            "kind IN ('board_word','clue')", name="ck_embedding_mpnet_kind"
        ),
        sa.ForeignKeyConstraint(["frame_id"], ["measurement_frame.frame_id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("frame_id", "text",
                            name="uq_embedding_mpnet_frame_text"),
    )

    op.create_table(
        "embedding_llama",
        sa.Column("id", sa.BigInteger(), sa.Identity(
            always=True), nullable=False),
        sa.Column("recipe_id", sa.Text(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(4096), nullable=False),
        sa.ForeignKeyConstraint(
            ["recipe_id"], ["extraction_recipe.recipe_id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("recipe_id", "text",
                            name="uq_embedding_llama_recipe_text"),
    )

    op.create_table(
        "embedding_mistral",
        sa.Column("id", sa.BigInteger(), sa.Identity(
            always=True), nullable=False),
        sa.Column("recipe_id", sa.Text(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(5120), nullable=False),
        sa.ForeignKeyConstraint(
            ["recipe_id"], ["extraction_recipe.recipe_id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "recipe_id", "text", name="uq_embedding_mistral_recipe_text"
        ),
    )

    op.create_table(
        "word_load",
        sa.Column("id", sa.BigInteger(), sa.Identity(
            always=True), nullable=False),
        sa.Column("frame_id", sa.Text(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("rho", sa.Double(), nullable=False),
        sa.Column("rho_weat", sa.Double(), nullable=True),
        sa.Column(
            "computed_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["frame_id"], ["measurement_frame.frame_id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("frame_id", "text",
                            name="uq_word_load_frame_text"),
    )


def downgrade() -> None:
    # Reverse dependency order. The vector extension is intentionally NOT dropped.
    op.drop_table("word_load")
    op.drop_table("embedding_mistral")
    op.drop_table("embedding_llama")
    op.drop_table("embedding_mpnet")
    op.drop_table("guess_proposal_item")
    op.drop_table("guess_proposal")
    op.drop_index("ix_reveal_event_turn_id", table_name="reveal_event")
    op.drop_table("reveal_event")
    op.drop_table("clue_target")
    op.drop_table("clue")
    op.drop_index("ix_llm_call_turn_id", table_name="llm_call")
    op.drop_index("ix_llm_call_game_id", table_name="llm_call")
    op.drop_table("llm_call")
    op.drop_table("turn")
    op.drop_table("game_seat")
    op.drop_table("game")
    op.drop_table("word_card")
    op.drop_table("board")
    op.drop_table("extraction_recipe")
    op.drop_table("measurement_frame")
    op.drop_table("run")
