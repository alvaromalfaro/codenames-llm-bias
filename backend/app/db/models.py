from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Double,
    ForeignKey,
    Numeric,
    SmallInteger,
    Text,
    UniqueConstraint,
    BigInteger,
    Identity,
    Integer,
    func,
)
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import text as sa_text


class Base(DeclarativeBase):
    """Declarative base for all persistence models."""


def _uuid_str() -> str:
    """Generate a uuid4 as a string so ids keep string ergonomics."""
    return str(uuid.uuid4())


class RunModel(Base):
    """One experimental batch (e.g. the 6x30 model-pairing cross).

    Anchors the master seed and the temperature regime that govern the batch, plus the code version
    and model registry snapshot needed to reproduce it. 

    Games played interactively or in the ecological human-vs-LLM modality do not belong to a batch, 
    so game.run_id is nullable.
    """
    __tablename__ = "run"

    id: Mapped[str] = mapped_column(
        postgresql.UUID(as_uuid=False), primary_key=True, default=_uuid_str
    )
    master_seed: Mapped[Decimal | None] = mapped_column(
        Numeric(20, 0), nullable=True)
    temperature: Mapped[float] = mapped_column(Double, nullable=False)
    regime_label: Mapped[str | None] = mapped_column(Text, nullable=True)
    code_version: Mapped[str | None] = mapped_column(Text, nullable=True)
    model_registry_snapshot: Mapped[dict | None] = mapped_column(
        postgresql.JSONB, nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)


class MeasurementFrameModel(Base):
    """Frozen extrinsic measurement frame: the arbiter encoder (phi*) and the gender axis (e_gen) 
    that both populated the boards and define per-word gender load rho.

    Content-addressed by frame_id (a hash of encoder + axis + construction), hence immutable: 
    changing the yardstick means inserting a new row, which is the correct semantics since changing 
    the measure is changing the experiment. Emitted by the board generator as a sidecar and consumed
    here; the platform never originates it.
    """
    __tablename__ = "measurement_frame"

    frame_id: Mapped[str] = mapped_column(Text, primary_key=True)
    encoder_name: Mapped[str] = mapped_column(Text, nullable=False)
    encoder_revision: Mapped[str] = mapped_column(Text, nullable=False)
    encoder_pooling: Mapped[str] = mapped_column(Text, nullable=False)
    encoder_normalize: Mapped[bool] = mapped_column(Boolean, nullable=False)
    gender_axis: Mapped[list[float]] = mapped_column(
        Vector(768), nullable=False)
    axis_construction: Mapped[dict] = mapped_column(
        postgresql.JSONB, nullable=False)
    generator_version: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class ExtractionRecipeModel(Base):
    """Versioned recipe for extracting an intrinsic embedding from an evaluated (decoder-only) 
    model: (model@revision, precision, layer, pooling, template).

    Content-addressed by recipe_id. The recipe is part of the identity of every intrinsic embedding: 
    two different recipes yield incomparable SEAT/ML-EAT results, so the recipe must gate any 
    intrinsic-bias computation.
    """
    __tablename__ = "extraction_recipe"

    recipe_id: Mapped[str] = mapped_column(Text, primary_key=True)
    model_name: Mapped[str] = mapped_column(Text, nullable=False)
    model_revision: Mapped[str] = mapped_column(Text, nullable=False)
    precision: Mapped[str] = mapped_column(Text, nullable=False)
    layer: Mapped[str] = mapped_column(Text, nullable=False)
    pooling: Mapped[str] = mapped_column(Text, nullable=False)
    template: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class BoardModel(Base):
    """A persisted board artifact (probe or control). 

    The probe/control distinction lives here (via  `type`) rather than being denormalized onto 
    games. Persisting boards makes the database self-contained: the metrics pipeline joins 
    board -> cards -> covariates without reading from disk.

    measurement_frame_id is nullable for now.
    """
    __tablename__ = "board"
    __table_args__ = (
        CheckConstraint(
            "type IN ('probe','control')", name="ck_board_type"
        ),
        # DEFERRED: couples board probe/control to a frame.
        # CheckConstraint(
        #     "type IS NULL OR measurement_frame_id IS NOT NULL",
        #     name="ck_board_type_requires_frame",
        # ),
    )

    board_id: Mapped[str] = mapped_column(Text, primary_key=True)
    measurement_frame_id: Mapped[str | None] = mapped_column(
        Text, ForeignKey("measurement_frame.frame_id"), nullable=True
    )
    type: Mapped[str | None] = mapped_column(Text, nullable=True)
    category: Mapped[str | None] = mapped_column(Text, nullable=True)
    specification: Mapped[str | None] = mapped_column(Text, nullable=True)
    seed: Mapped[Decimal | None] = mapped_column(Numeric(20, 0), nullable=True)
    grid_rows: Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    grid_cols: Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    arbiters: Mapped[dict | None] = mapped_column(
        postgresql.JSONB, nullable=True)
    dilemma: Mapped[dict | None] = mapped_column(
        postgresql.JSONB, nullable=True)
    keycard_audit: Mapped[dict | None] = mapped_column(
        postgresql.JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class WordCardModel(Base):
    """One of the 25 cards of a board. 

    Carries both perspective roles, since the Duet double-sided key card is encoded per card rather 
    than as a separate object.

    Lexical covariates (SUBTLEX frequency, length, WordNet polysemy) are flattened into columns 
    instead of nested JSON, because they are the difficulty confounders the analysis must control 
    for: flattened, that adjustment is a plain join.
    """
    __tablename__ = "word_card"
    __table_args__ = (
        CheckConstraint(
            "card_id BETWEEN 0 AND 24", name="ck_word_card_card_id_range"
        ),
        CheckConstraint(
            "llm_perspective_role IN ('agent','assassin','civilian')",
            name="ck_word_card_llm_role",
        ),
        CheckConstraint(
            "human_perspective_role IN ('agent','assassin','civilian')",
            name="ck_word_card_human_role",
        ),
        CheckConstraint(
            "category IN ('male','female','neutral')", name="ck_word_card_category"
        ),
        UniqueConstraint("board_id", "card_id",
                         name="uq_word_card_board_card"),
        UniqueConstraint("board_id", "text", name="uq_word_card_board_text"),
    )

    id: Mapped[int] = mapped_column(
        BigInteger, Identity(always=True), primary_key=True
    )
    board_id: Mapped[str] = mapped_column(
        Text, ForeignKey("board.board_id", ondelete="CASCADE"), nullable=False
    )
    card_id: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    llm_perspective_role: Mapped[str] = mapped_column(Text, nullable=False)
    human_perspective_role: Mapped[str] = mapped_column(Text, nullable=False)
    category: Mapped[str | None] = mapped_column(Text, nullable=True)
    source: Mapped[str | None] = mapped_column(Text, nullable=True)
    weat_set: Mapped[list[str] | None] = mapped_column(
        postgresql.ARRAY(Text), nullable=True
    )
    subtlex_freq: Mapped[float | None] = mapped_column(Double, nullable=True)
    length: Mapped[int | None] = mapped_column(Integer, nullable=True)
    wordnet_polysemy: Mapped[int | None] = mapped_column(
        Integer, nullable=True)


class GameModel(Base):
    """One complete Duet play.

    game_status gates the analysis: a game is committed atomically at finish and only 'completed' 
    games are valid observations, so a run truncated by an error is marked 'error' and never
    silently pollutes the estimators.

    Links to its board (which carries the probe/control type and the measurement frame) and records
    the per-game derived seed and start player.
    """
    __tablename__ = "game"
    __table_args__ = (
        CheckConstraint("start_player IN (0,1)", name="ck_game_start_player"),
        CheckConstraint(
            "game_status IN ('in_progress','completed','aborted','error')",
            name="ck_game_status",
        ),
    )

    id: Mapped[str] = mapped_column(
        postgresql.UUID(as_uuid=False), primary_key=True, default=_uuid_str
    )
    run_id: Mapped[str | None] = mapped_column(
        postgresql.UUID(as_uuid=False), ForeignKey("run.id"), nullable=True
    )
    board_id: Mapped[str] = mapped_column(
        Text, ForeignKey("board.board_id"), nullable=False
    )
    derived_seed: Mapped[Decimal | None] = mapped_column(
        Numeric(20, 0), nullable=True)
    start_player: Mapped[int | None] = mapped_column(
        SmallInteger, nullable=True)
    game_status: Mapped[str] = mapped_column(
        Text, nullable=False, server_default=sa_text("'in_progress'")
    )
    result: Mapped[str | None] = mapped_column(Text, nullable=True)
    timer_tokens_final: Mapped[int | None] = mapped_column(
        SmallInteger, nullable=True)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


class GameSeatModel(Base):
    """One seat of a game (two rows for an LLM-vs-LLM game).

    Models both modalities with a single table: in the ecological modality the human seat has 
    provider 'human' and a null model_ref. 

    requested_seed is stored per seat because it records the value actually sent to each provider, 
    which can differ from the canonical derived seed (for instance if a local backend narrows a 
    64-bit seed).
    """
    __tablename__ = "game_seat"
    __table_args__ = (
        CheckConstraint("seat_index IN (0,1)", name="ck_game_seat_seat_index"),
        CheckConstraint(
            "provider IN ('ollama','openrouter','human')", name="ck_game_seat_provider"
        ),
        UniqueConstraint("game_id", "seat_index",
                         name="uq_game_seat_game_seat"),
    )

    id: Mapped[int] = mapped_column(
        BigInteger, Identity(always=True), primary_key=True
    )
    game_id: Mapped[str] = mapped_column(
        postgresql.UUID(as_uuid=False),
        ForeignKey("game.id", ondelete="CASCADE"),
        nullable=False,
    )
    seat_index: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    model_ref: Mapped[str | None] = mapped_column(Text, nullable=True)
    provider: Mapped[str | None] = mapped_column(Text, nullable=True)
    precision: Mapped[str | None] = mapped_column(Text, nullable=True)
    requested_temperature: Mapped[float |
                                  None] = mapped_column(Double, nullable=True)
    requested_seed: Mapped[Decimal | None] = mapped_column(
        Numeric(20, 0), nullable=True)
