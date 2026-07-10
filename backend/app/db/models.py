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
    Index,
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
    prompt_template_version: Mapped[str | None] = mapped_column(Text, nullable=True)
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


class TurnModel(Base):
    """One turn: a clue plus its guessing phase. 

    `phase` discriminates normal play from sudden death, which the sudden-death bias metric needs to
    isolate.
    """
    __tablename__ = "turn"
    __table_args__ = (
        CheckConstraint(
            "clue_giver_seat IN (0,1)", name="ck_turn_clue_giver_seat"
        ),
        CheckConstraint(
            "phase IN ('normal','sudden_death')", name="ck_turn_phase"
        ),
        UniqueConstraint("game_id", "turn_number",
                         name="uq_turn_game_turn_number"),
    )

    id: Mapped[int] = mapped_column(
        BigInteger, Identity(always=True), primary_key=True
    )
    game_id: Mapped[str] = mapped_column(
        postgresql.UUID(as_uuid=False),
        ForeignKey("game.id", ondelete="CASCADE"),
        nullable=False,
    )
    turn_number: Mapped[int] = mapped_column(Integer, nullable=False)
    clue_giver_seat: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    phase: Mapped[str] = mapped_column(
        Text, nullable=False, server_default=sa_text("'normal'")
    )


class LlmCallModel(Base):
    """Per-invocation sampling telemetry for one model call. 

    retry_index > 0 marks a call regenerated after a clue-legality rejection; these rejected calls 
    are persisted (not only the accepted one) so the effect of eliciting targets on gameplay can be 
    audited. 

    resolved_model and system_fingerprint let us detect after the fact whether an API provider 
    silently swapped the model mid-experiment.
    """
    __tablename__ = "llm_call"
    __table_args__ = (
        CheckConstraint("seat_index IN (0,1)", name="ck_llm_call_seat_index"),
        CheckConstraint(
            "role IN ('clue_giver','guesser','guesser_sd')", name="ck_llm_call_role"
        ),
        Index("ix_llm_call_game_id", "game_id"),
        Index("ix_llm_call_turn_id", "turn_id"),
    )

    id: Mapped[int] = mapped_column(
        BigInteger, Identity(always=True), primary_key=True
    )
    game_id: Mapped[str] = mapped_column(
        postgresql.UUID(as_uuid=False),
        ForeignKey("game.id", ondelete="CASCADE"),
        nullable=False,
    )
    turn_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("turn.id", ondelete="CASCADE"), nullable=True
    )
    seat_index: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    role: Mapped[str] = mapped_column(Text, nullable=False)
    retry_index: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=sa_text("0")
    )
    provider: Mapped[str | None] = mapped_column(Text, nullable=True)
    model_used: Mapped[str | None] = mapped_column(Text, nullable=True)
    resolved_model: Mapped[str | None] = mapped_column(Text, nullable=True)
    system_fingerprint: Mapped[str | None] = mapped_column(Text, nullable=True)
    requested_temperature: Mapped[float |
                                  None] = mapped_column(Double, nullable=True)
    requested_seed: Mapped[Decimal | None] = mapped_column(
        Numeric(20, 0), nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    prompt_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    completion_tokens: Mapped[int | None] = mapped_column(
        Integer, nullable=True)
    finish_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    execution_mode: Mapped[str | None] = mapped_column(Text, nullable=True)
    request_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    raw_payload: Mapped[dict | None] = mapped_column(
        postgresql.JSONB, nullable=True)
    rendered_prompt: Mapped[list | None] = mapped_column(
        postgresql.JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class ClueModel(Base):
    """The clue-giver's action for a turn (one per turn). 

    llm_call_id points to the accepted call; retries live in llm_call with retry_index > 0.
    targets_raw stores the model's raw intended-target list verbatim; the resolved form lives in 
    clue_target.
    """
    __tablename__ = "clue"
    __table_args__ = (
        CheckConstraint("count >= 1", name="ck_clue_count_positive"),
    )

    id: Mapped[int] = mapped_column(
        BigInteger, Identity(always=True), primary_key=True
    )
    turn_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("turn.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )
    llm_call_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("llm_call.id"), nullable=True
    )
    clue_word: Mapped[str] = mapped_column(Text, nullable=False)
    count: Mapped[int] = mapped_column(Integer, nullable=False)
    reasoning: Mapped[str | None] = mapped_column(Text, nullable=True)
    targets_raw: Mapped[list] = mapped_column(
        postgresql.JSONB, nullable=False, server_default=sa_text("'[]'")
    )


class ClueTargetModel(Base):
    """One resolved element of the clue-giver's intended set S.

    Deliberately has no constraint tying the number of targets to the clue's count: the cardinality
    of S is a diagnostic, not a gate, and enforcing it would discard classifiable observations. 

    card_id is null when the target word cannot be mapped to a board card (derivable malformation).
    """
    __tablename__ = "clue_target"
    __table_args__ = (
        CheckConstraint(
            "giver_role IN ('agent','assassin','civilian')",
            name="ck_clue_target_giver_role",
        ),
        UniqueConstraint("clue_id", "position",
                         name="uq_clue_target_clue_position"),
    )

    id: Mapped[int] = mapped_column(
        BigInteger, Identity(always=True), primary_key=True
    )
    clue_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("clue.id", ondelete="CASCADE"), nullable=False
    )
    position: Mapped[int] = mapped_column(Integer, nullable=False)
    word: Mapped[str] = mapped_column(Text, nullable=False)
    card_id: Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    giver_role: Mapped[str | None] = mapped_column(Text, nullable=True)
    revealed_at_clue: Mapped[bool | None] = mapped_column(
        Boolean, nullable=True)


class RevealEventModel(Base):
    """The engine's ground truth for a single card resolution during a turn. 

    This is the authoritative source from which per-card reveal and time-marker state are 
    reconstructed, so that state is not duplicated. Records whether the reveal ended the turn or the
    game, and the time-token bank level afterwards.
    """
    __tablename__ = "reveal_event"
    __table_args__ = (
        CheckConstraint("acting_seat IN (0,1)",
                        name="ck_reveal_event_acting_seat"),
        CheckConstraint(
            "result_role IN ('agent','assassin','civilian')",
            name="ck_reveal_event_result_role",
        ),
        UniqueConstraint(
            "turn_id", "position_in_turn", name="uq_reveal_event_turn_position"
        ),
        Index("ix_reveal_event_turn_id", "turn_id"),
    )

    id: Mapped[int] = mapped_column(
        BigInteger, Identity(always=True), primary_key=True
    )
    turn_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("turn.id", ondelete="CASCADE"), nullable=False
    )
    position_in_turn: Mapped[int] = mapped_column(Integer, nullable=False)
    card_id: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    acting_seat: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    result_role: Mapped[str] = mapped_column(Text, nullable=False)
    ended_turn: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=sa_text("false")
    )
    ended_game: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=sa_text("false")
    )
    time_marker_placed: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=sa_text("false")
    )
    timer_tokens_after: Mapped[int | None] = mapped_column(
        SmallInteger, nullable=True)


class GuessProposalModel(Base):
    """The guesser's single model output for a turn: the ordered list of proposed cards with 
    confidences, produced in one forward pass.

    The individual items live in guess_proposal_item.
    """
    __tablename__ = "guess_proposal"
    __table_args__ = (
        CheckConstraint(
            "guesser_seat IN (0,1)", name="ck_guess_proposal_guesser_seat"
        ),
    )

    id: Mapped[int] = mapped_column(
        BigInteger, Identity(always=True), primary_key=True
    )
    turn_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("turn.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )
    llm_call_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("llm_call.id"), nullable=True
    )
    guesser_seat: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    reasoning: Mapped[str | None] = mapped_column(Text, nullable=True)
    stop_reason: Mapped[str | None] = mapped_column(Text, nullable=True)


class GuessProposalItemModel(Base):
    """One ordered item of a guess proposal, with the model's self-reported confidence.

    Crucially this includes items the engine never reached because an earlier guess ended the turn: 
    reveal_event_id is null for those. That unreached tail (a confident intent toward a card that 
    was never played) is exactly where a bias signal can hide, so it is preserved rather than 
    discarded.
    """
    __tablename__ = "guess_proposal_item"
    __table_args__ = (
        CheckConstraint(
            "confidence >= 0.0 AND confidence <= 1.0",
            name="ck_guess_proposal_item_confidence",
        ),
        UniqueConstraint(
            "guess_proposal_id", "position", name="uq_guess_proposal_item_position"
        ),
    )

    id: Mapped[int] = mapped_column(
        BigInteger, Identity(always=True), primary_key=True
    )
    guess_proposal_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("guess_proposal.id", ondelete="CASCADE"),
        nullable=False,
    )
    position: Mapped[int] = mapped_column(Integer, nullable=False)
    word: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[float | None] = mapped_column(Double, nullable=True)
    resolved_card_id: Mapped[int | None] = mapped_column(
        SmallInteger, nullable=True)
    reveal_event_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("reveal_event.id"), nullable=True
    )


class EmbeddingMpnetModel(Base):
    """Extrinsic embeddings in the arbiter phi* (mpnet) space, keyed by (frame, text). 

    Populated in two phases: board words at platform startup, and clue words after the fact, since
    clues are not known a priori.

    Storage only, with no approximate-nearest-neighbour index, because all downstream scalars are
    computed offline.
    """
    __tablename__ = "embedding_mpnet"
    __table_args__ = (
        CheckConstraint(
            "kind IN ('board_word','clue')", name="ck_embedding_mpnet_kind"
        ),
        UniqueConstraint("frame_id", "text",
                         name="uq_embedding_mpnet_frame_text"),
    )

    id: Mapped[int] = mapped_column(
        BigInteger, Identity(always=True), primary_key=True
    )
    frame_id: Mapped[str] = mapped_column(
        Text, ForeignKey("measurement_frame.frame_id"), nullable=False
    )
    text: Mapped[str] = mapped_column(Text, nullable=False)
    kind: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(768), nullable=False)


class EmbeddingLlamaModel(Base):
    """Intrinsic embeddings in the evaluated Llama model's space, keyed by (recipe, text). 

    Uniqueness on (recipe, text) rather than text alone is what prevents a stale-recipe cache hit: 
    if the extraction recipe changes, the old vector must not be silently reused.
    """
    __tablename__ = "embedding_llama"
    __table_args__ = (
        UniqueConstraint("recipe_id", "text",
                         name="uq_embedding_llama_recipe_text"),
    )

    id: Mapped[int] = mapped_column(
        BigInteger, Identity(always=True), primary_key=True
    )
    recipe_id: Mapped[str] = mapped_column(
        Text, ForeignKey("extraction_recipe.recipe_id"), nullable=False
    )
    text: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(
        Vector(4096), nullable=False)


class EmbeddingMistralModel(Base):
    """Intrinsic embeddings in the evaluated Mistral model's space (5120-dim), keyed by (recipe, 
    text).

    Same recipe-scoped caching rule as the Llama table.
    """
    __tablename__ = "embedding_mistral"
    __table_args__ = (
        UniqueConstraint(
            "recipe_id", "text", name="uq_embedding_mistral_recipe_text"
        ),
    )

    id: Mapped[int] = mapped_column(
        BigInteger, Identity(always=True), primary_key=True
    )
    recipe_id: Mapped[str] = mapped_column(
        Text, ForeignKey("extraction_recipe.recipe_id"), nullable=False
    )
    text: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(
        Vector(5120), nullable=False)


class WordLoadModel(Base):
    """Derived per-word gender load: rho = cos(phi*(w), e_gen), plus the WEAT-by-sets robustness 
    variant.

    Keyed by frame because a load value is meaningless without the frame that produced it. 

    These are post-hoc batch products of the metrics pipeline, not written during play.
    """
    __tablename__ = "word_load"
    __table_args__ = (
        UniqueConstraint("frame_id", "text", name="uq_word_load_frame_text"),
    )

    id: Mapped[int] = mapped_column(
        BigInteger, Identity(always=True), primary_key=True
    )
    frame_id: Mapped[str] = mapped_column(
        Text, ForeignKey("measurement_frame.frame_id"), nullable=False
    )
    text: Mapped[str] = mapped_column(Text, nullable=False)
    rho: Mapped[float] = mapped_column(Double, nullable=False)
    rho_weat: Mapped[float | None] = mapped_column(Double, nullable=True)
    computed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
