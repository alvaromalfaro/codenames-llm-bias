"""Tests for the phi* embedding backfill.

Two gates, for two different costs:
  * the insert-path tests need Postgres (pgvector) and are skipped when DATABASE_URL is unset. They 
    drive a deterministic StubEncoder so they never load torch.
  * the encoder-contract tests need the real ~420MB checkpoint and are skipped when it is not in
    the local HuggingFace cache. Nothing downloads.

Each DB test writes under its own throwaway frame_id, so runs are independent and need no cleanup.
"""
import hashlib
import os
import uuid

import numpy as np
import pytest

_REAL_FRAME_ID = "8a404797b3e656dd00683910aa829bbbc584c6b23c37e9f0de1173d11a9d0cc3"
_REAL_ENCODER = "sentence-transformers/all-mpnet-base-v2"
_REAL_REVISION = "e8c3b32edf5434bc2275fc9bab85f82640a19130"

requires_db = pytest.mark.skipif(
    not os.environ.get("DATABASE_URL"), reason="requires a live Postgres database"
)


class StubEncoder:
    """Deterministic unit vectors keyed by text, so insert-path tests need no model.

    Lowercases like the real encoder, so a vector is a fingerprint of the *stored* text.
    """

    def encode(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            seed = int.from_bytes(
                hashlib.sha256(t.lower().encode()).digest()[:8], "big")
            vec = np.random.default_rng(seed).standard_normal(768)
            out.append((vec / np.linalg.norm(vec)).tolist())
        return out


def _hf_cache_has_pinned_model() -> bool:
    """True when the pinned snapshot is already on disk. Never triggers a download."""
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    return isinstance(
        try_to_load_from_cache(_REAL_ENCODER, "modules.json",
                               revision=_REAL_REVISION),
        str,
    )


requires_real_encoder = pytest.mark.skipif(
    not _hf_cache_has_pinned_model(),
    reason=f"phi* snapshot {_REAL_ENCODER}@{_REAL_REVISION} not in local HF cache",
)


def _make_frame(session) -> str:
    """Insert a throwaway measurement_frame carrying the real encoder contract; return its id."""
    from backend.app.db.models import MeasurementFrameModel

    frame_id = f"test-embed-{uuid.uuid4()}"
    session.add(MeasurementFrameModel(
        frame_id=frame_id,
        encoder_name=_REAL_ENCODER,
        encoder_revision=_REAL_REVISION,
        encoder_pooling="mean",
        encoder_normalize=True,
        gender_axis=[0.0] * 768,
        axis_construction={"method": "test"},
    ))
    session.flush()
    return frame_id


def _rows_for(session, frame_id):
    from sqlalchemy import select

    from backend.app.db.models import EmbeddingMpnetModel

    return session.execute(
        select(EmbeddingMpnetModel).where(
            EmbeddingMpnetModel.frame_id == frame_id)
    ).scalars().all()


# the insert path
@requires_db
def test_overlapping_text_yields_one_board_word_row():
    """A text that is both a board word and a clue is stored once, as kind='board_word'.

    This is the ON CONFLICT path. Removing the conflict clause raises IntegrityError; reversing the
    two phases flips the surviving row's kind to 'clue'.
    """
    from backend.app.db.embed_mpnet import KIND_BOARD_WORD, KIND_CLUE, insert_embeddings
    from backend.app.db.session import session_scope

    with session_scope() as session:
        frame_id = _make_frame(session)
        enc = StubEncoder()

        b_ins, b_skip = insert_embeddings(
            session, frame_id, ["apple"], KIND_BOARD_WORD, enc)
        c_ins, c_skip = insert_embeddings(
            session, frame_id, ["apple", "banana"], KIND_CLUE, enc)

        assert (b_ins, b_skip) == (1, 0)
        assert (c_ins, c_skip) == (1, 1)

        rows = {r.text: r.kind for r in _rows_for(session, frame_id)}
        assert rows == {"apple": KIND_BOARD_WORD, "banana": KIND_CLUE}


@requires_db
def test_stored_text_is_lowercased():
    """The stored text is the lowercased form, so downstream joins on lowercased text hold."""
    from backend.app.db.embed_mpnet import KIND_BOARD_WORD, insert_embeddings
    from backend.app.db.session import session_scope

    with session_scope() as session:
        frame_id = _make_frame(session)
        inserted, skipped = insert_embeddings(
            session, frame_id, ["BUCKET", "bucket"], KIND_BOARD_WORD, StubEncoder())

        # The two spellings collapse to one candidate before any DB round-trip.
        assert (inserted, skipped) == (1, 0)
        assert [r.text for r in _rows_for(session, frame_id)] == ["bucket"]


@requires_db
def test_every_inserted_vector_has_768_dims():
    from backend.app.db.embed_mpnet import KIND_BOARD_WORD, insert_embeddings
    from backend.app.db.session import session_scope

    with session_scope() as session:
        frame_id = _make_frame(session)
        insert_embeddings(session, frame_id, ["alpha", "beta", "gamma"],
                          KIND_BOARD_WORD, StubEncoder())

        rows = _rows_for(session, frame_id)
        assert len(rows) == 3
        assert all(len(r.embedding) == 768 for r in rows)


@requires_db
def test_rerun_is_idempotent():
    """Re-running inserts nothing, raises nothing and leaves the row count unchanged."""
    from backend.app.db.embed_mpnet import KIND_BOARD_WORD, insert_embeddings
    from backend.app.db.session import session_scope

    with session_scope() as session:
        frame_id = _make_frame(session)
        texts = ["alpha", "beta"]
        enc = StubEncoder()

        assert insert_embeddings(
            session, frame_id, texts, KIND_BOARD_WORD, enc) == (2, 0)
        assert insert_embeddings(
            session, frame_id, texts, KIND_BOARD_WORD, enc) == (0, 2)
        assert len(_rows_for(session, frame_id)) == 2


@requires_db
def test_batching_does_not_change_the_result():
    """A batch_size smaller than the input still inserts every row exactly once."""
    from backend.app.db.embed_mpnet import KIND_BOARD_WORD, insert_embeddings
    from backend.app.db.session import session_scope

    with session_scope() as session:
        frame_id = _make_frame(session)
        texts = [f"word{i}" for i in range(10)]
        inserted, skipped = insert_embeddings(
            session, frame_id, texts, KIND_BOARD_WORD, StubEncoder(), batch_size=3)

        assert (inserted, skipped) == (10, 0)
        assert len(_rows_for(session, frame_id)) == 10


# the frame is the source of encoder identity
@requires_db
def test_load_encoder_spec_reads_the_frame():
    from backend.app.db.embed_mpnet import load_encoder_spec
    from backend.app.db.session import session_scope

    with session_scope() as session:
        frame_id = _make_frame(session)
        spec = load_encoder_spec(session, frame_id)

        assert (spec.name, spec.revision) == (_REAL_ENCODER, _REAL_REVISION)
        assert (spec.pooling, spec.normalize) == ("mean", True)


@requires_db
def test_load_encoder_spec_rejects_unknown_frame():
    from backend.app.db.embed_mpnet import MissingFrameError, load_encoder_spec
    from backend.app.db.session import session_scope

    with session_scope() as session:
        with pytest.raises(MissingFrameError):
            load_encoder_spec(session, "no-such-frame")


# the phase queries, against the real batch
@requires_db
def test_board_word_texts_are_lowercased_and_distinct():
    """Fails if lower() is dropped from the query: the bank stores its words uppercased."""
    from backend.app.db.embed_mpnet import board_word_texts
    from backend.app.db.session import session_scope

    with session_scope() as session:
        texts = board_word_texts(session, _REAL_FRAME_ID)
        if not texts:
            pytest.skip("no boards for the real frame in this database")

        assert all(t == t.lower() for t in texts)
        assert len(texts) == len(set(texts))


@requires_db
def test_clue_texts_are_lowercased_and_distinct():
    from backend.app.db.embed_mpnet import clue_texts
    from backend.app.db.session import session_scope

    with session_scope() as session:
        texts = clue_texts(session, _REAL_FRAME_ID)
        if not texts:
            pytest.skip("no clues for the real frame in this database")

        assert all(t == t.lower() for t in texts)
        assert len(texts) == len(set(texts))


# the real phi* contract
@requires_real_encoder
def test_real_encoder_matches_reference():
    """The vector we store equals the reference encode of the lowercased text, L2-normalized.

    Fails if normalize is dropped. Note it does not discriminate the .lower(): this checkpoint's
    tokenizer is uncased, so lowercasing changes the stored key, not the geometry.
    """
    from backend.app.db.embed_mpnet import EncoderSpec, SentenceTransformerEncoder

    pytest.importorskip("sentence_transformers")
    from sentence_transformers import SentenceTransformer

    spec = EncoderSpec(frame_id=_REAL_FRAME_ID, name=_REAL_ENCODER,
                       revision=_REAL_REVISION, pooling="mean", normalize=True)
    ours = np.asarray(SentenceTransformerEncoder(spec).encode(["BUCKET"])[0])

    model = SentenceTransformer(
        _REAL_ENCODER, revision=_REAL_REVISION, device="cpu")
    reference = np.asarray(model.encode("bucket", normalize_embeddings=True))

    assert ours.shape == (768,)
    assert np.linalg.norm(ours) == pytest.approx(1.0, abs=1e-5)
    assert np.max(np.abs(ours - reference)) < 1e-6


@requires_real_encoder
def test_lowercasing_is_geometrically_a_noop_for_this_checkpoint():
    """Pin the assumption that makes the board_word/clue collision safe.

    all-mpnet-base-v2 tokenizes with do_lower_case=True, so 'BUCKET' and 'bucket' yield the same
    vector. That is why reusing one stored row for a text appearing as both a board word and a clue
    loses nothing. If the frame ever moves to a cased encoder this test fails, which is the point:
    the collision-reuse argument would need revisiting rather than silently degrading.
    """
    pytest.importorskip("sentence_transformers")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        _REAL_ENCODER, revision=_REAL_REVISION, device="cpu")
    assert model.tokenizer("BUCKET")["input_ids"] == model.tokenizer(
        "bucket")["input_ids"]

    lower = np.asarray(model.encode("bucket", normalize_embeddings=True))
    upper = np.asarray(model.encode("BUCKET", normalize_embeddings=True))
    assert np.max(np.abs(lower - upper)) < 1e-6


@requires_real_encoder
def test_build_encoder_rejects_pooling_mismatch():
    """A frame claiming a pooling the checkpoint does not use is a hard stop, not a warning."""
    from backend.app.db.embed_mpnet import EncoderContractError, EncoderSpec, SentenceTransformerEncoder

    pytest.importorskip("sentence_transformers")

    spec = EncoderSpec(frame_id=_REAL_FRAME_ID, name=_REAL_ENCODER,
                       revision=_REAL_REVISION, pooling="cls", normalize=True)
    with pytest.raises(EncoderContractError, match="pools with 'mean'"):
        SentenceTransformerEncoder(spec)


@requires_real_encoder
def test_build_encoder_rejects_normalize_mismatch():
    """The checkpoint ends in Normalize; a frame declaring encoder_normalize=False contradicts it."""
    from backend.app.db.embed_mpnet import EncoderContractError, EncoderSpec, SentenceTransformerEncoder

    pytest.importorskip("sentence_transformers")

    spec = EncoderSpec(frame_id=_REAL_FRAME_ID, name=_REAL_ENCODER,
                       revision=_REAL_REVISION, pooling="mean", normalize=False)
    with pytest.raises(EncoderContractError, match="Normalize"):
        SentenceTransformerEncoder(spec)
