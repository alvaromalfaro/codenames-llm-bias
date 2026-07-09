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
    func,
)
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


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
