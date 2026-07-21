"""Offline population of ``embedding_mpnet`` with extrinsic phi* embeddings.

The encoder identity is read from the ``measurement_frame`` row at runtime and is never hardcoded:
the frame is the source of truth for ``encoder_name/revision/pooling/normalize``. The board generator
owns the same contract on its side, but this module deliberately does not import from it (the two
sides stay dependency-isolated); it re-implements the contract and the frame keeps them honest.

The phi* contract, matching the generator's ``Arbiter.embed``:
  * ``SentenceTransformer(name, revision=revision, device="cpu")``, then ``.eval()``;
  * the loaded module stack must actually provide the frame's pooling mode and 768 dimensions -
    asserted, not assumed, so a swapped checkpoint fails loud instead of writing wrong geometry;
  * every text is ``.lower()``-ed before encoding. This is the single normalization point, so the
    stored ``text`` is the lowercased form and downstream joins are consistent;
  * ``normalize_embeddings`` follows the frame's ``encoder_normalize``.

Two phases, in this order: board words (``kind='board_word'``) then clues (``kind='clue'``). The
order matters. Because texts are lowercased, some clue words collide with board words; the phase-2
insert hits ``uq_embedding_mpnet_frame_text`` and DO NOTHING keeps the existing ``board_word`` row.
That is correct rather than merely tolerable - same text under the same encoder yields the same
vector, so the rows would be identical apart from ``kind``. It also makes the whole backfill
idempotent: re-running inserts nothing and raises nothing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

from sqlalchemy import text as sql_text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from backend.app.db.models import EmbeddingMpnetModel

logger = logging.getLogger(__name__)

EXPECTED_DIM = 768
DEFAULT_BATCH_SIZE = 256

KIND_BOARD_WORD = "board_word"
KIND_CLUE = "clue"


class MissingFrameError(RuntimeError):
    """No ``measurement_frame`` row for the requested frame_id - there is no encoder spec to obey."""


class EncoderContractError(RuntimeError):
    """The loaded checkpoint disagrees with the frame's declared encoder contract."""


@dataclass(frozen=True)
class EncoderSpec:
    """The encoder identity as declared by a ``measurement_frame`` row."""

    frame_id: str
    name: str
    revision: str
    pooling: str
    normalize: bool


class Encoder(Protocol):
    """Anything that maps texts to phi* vectors. Lets tests inject a stub instead of loading torch."""

    def encode(self, texts: list[str]) -> list[list[float]]: ...


def load_encoder_spec(session: Session, frame_id: str) -> EncoderSpec:
    """Read the encoder contract for ``frame_id`` from ``measurement_frame``.

    This is the only source of encoder identity in this module.
    """
    row = session.execute(
        sql_text(
            "SELECT encoder_name, encoder_revision, encoder_pooling, encoder_normalize "
            "FROM measurement_frame WHERE frame_id = :frame_id"
        ),
        {"frame_id": frame_id},
    ).one_or_none()
    if row is None:
        raise MissingFrameError(
            f"no measurement_frame row for frame_id {frame_id!r}")
    return EncoderSpec(
        frame_id=frame_id,
        name=row.encoder_name,
        revision=row.encoder_revision,
        pooling=row.encoder_pooling,
        normalize=row.encoder_normalize,
    )


class SentenceTransformerEncoder:
    """A phi* encoder pinned to a frame's checkpoint, lowercasing every text before encoding."""

    def __init__(self, spec: EncoderSpec) -> None:
        # Imported lazily
        from sentence_transformers import SentenceTransformer

        self.spec = spec
        self._model = SentenceTransformer(
            spec.name, revision=spec.revision, device="cpu")
        self._model.eval()
        _assert_contract(self._model, spec)

    def encode(self, texts: list[str]) -> list[list[float]]:
        """Encode ``texts`` lowercased, honouring the frame's normalize flag. Order is preserved."""
        vectors = self._model.encode(
            [t.lower() for t in texts],
            convert_to_numpy=True,
            normalize_embeddings=self.spec.normalize,
        )
        return [[float(x) for x in row] for row in vectors]


def _assert_contract(model: object, spec: EncoderSpec) -> None:
    """Verify the loaded stack really provides the frame's pooling, normalize and dimensionality.

    Reading these off the loaded model rather than trusting the checkpoint name is what makes a
    swapped or re-tagged revision fail loudly instead of silently writing vectors from the wrong
    geometry into a frame that claims otherwise.
    """
    pooling_mode: str | None = None
    has_normalize = False
    for _, module in model.named_children():  # type: ignore[attr-defined]
        kind = type(module).__name__
        if kind == "Pooling":
            pooling_mode = getattr(module, "pooling_mode", None)
        elif kind == "Normalize":
            has_normalize = True

    if pooling_mode is None:
        raise EncoderContractError(
            f"{spec.name}@{spec.revision} exposes no Pooling module with a readable pooling_mode; "
            f"frame {spec.frame_id} declares pooling={spec.pooling!r}"
        )
    if pooling_mode != spec.pooling:
        raise EncoderContractError(
            f"{spec.name}@{spec.revision} pools with {pooling_mode!r} but frame {spec.frame_id} "
            f"declares pooling={spec.pooling!r}"
        )

    # Renamed in sentence-transformers 5.x; accept either so the pin can move without a silent skip.
    get_dim = getattr(model, "get_embedding_dimension", None) or getattr(
        model, "get_sentence_embedding_dimension")
    dim = get_dim()
    if dim != EXPECTED_DIM:
        raise EncoderContractError(
            f"{spec.name}@{spec.revision} has dimension {dim}, expected {EXPECTED_DIM}"
        )

    # The frame may declare normalize=True either because the module stack ends in Normalize or
    # because we pass normalize_embeddings=True; both hold here. Only the contradiction is an error:
    # a stack that normalizes while the frame says it must not.
    if has_normalize and not spec.normalize:
        raise EncoderContractError(
            f"{spec.name}@{spec.revision} ends in a Normalize module but frame {spec.frame_id} "
            f"declares encoder_normalize=False"
        )


def board_word_texts(session: Session, frame_id: str) -> list[str]:
    """Distinct lowercased ``word_card.text`` for every board sealed against ``frame_id``.

    ``word_card.text`` is unique only per board, and the bank stores words uppercased, so both the
    DISTINCT and the lower() are load-bearing.
    """
    rows = session.execute(
        sql_text(
            "SELECT DISTINCT lower(wc.text) AS text "
            "FROM word_card wc "
            "JOIN board b ON b.board_id = wc.board_id "
            "WHERE b.measurement_frame_id = :frame_id "
            "ORDER BY 1"
        ),
        {"frame_id": frame_id},
    ).all()
    return [row[0] for row in rows]


def clue_texts(session: Session, frame_id: str) -> list[str]:
    """Distinct lowercased ``clue.clue_word`` for every clue played on a board sealed to ``frame_id``.

    ``clue`` carries no frame reference, so the frame is reached the long way round:
    clue -> turn -> game -> board -> measurement_frame_id.
    """
    rows = session.execute(
        sql_text(
            "SELECT DISTINCT lower(c.clue_word) AS text "
            "FROM clue c "
            "JOIN turn tu ON tu.id = c.turn_id "
            "JOIN game g ON g.id = tu.game_id "
            "JOIN board b ON b.board_id = g.board_id "
            "WHERE b.measurement_frame_id = :frame_id "
            "ORDER BY 1"
        ),
        {"frame_id": frame_id},
    ).all()
    return [row[0] for row in rows]


def _dedupe(texts: list[str]) -> list[str]:
    """Order-preserving dedupe. ON CONFLICT cannot resolve duplicates within one statement, so a
    repeated text in a single chunk would abort it - dedupe first rather than rely on the DB."""
    seen: set[str] = set()
    out: list[str] = []
    for t in texts:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def insert_embeddings(
    session: Session,
    frame_id: str,
    texts: list[str],
    kind: str,
    encoder: Encoder,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> tuple[int, int]:
    """Encode ``texts`` and insert them as ``kind`` rows, skipping any (frame_id, text) already stored.

    Returns ``(inserted, skipped)``. ``ON CONFLICT DO NOTHING ... RETURNING`` yields exactly the rows
    actually written, so the counts need no follow-up SELECT.
    """
    candidates = _dedupe([t.lower() for t in texts])
    inserted = 0
    for start in range(0, len(candidates), batch_size):
        chunk = candidates[start:start + batch_size]
        vectors = encoder.encode(chunk)
        stmt = (
            pg_insert(EmbeddingMpnetModel)
            .values([
                {"frame_id": frame_id, "text": t, "kind": kind, "embedding": v}
                for t, v in zip(chunk, vectors, strict=True)
            ])
            .on_conflict_do_nothing(constraint="uq_embedding_mpnet_frame_text")
            .returning(EmbeddingMpnetModel.id)
        )
        inserted += len(session.execute(stmt).fetchall())
    return inserted, len(candidates) - inserted


def backfill_frame(
    session: Session,
    frame_id: str,
    encoder: Encoder,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, dict[str, int]]:
    """Run both phases for ``frame_id``: board words first, then clues. Returns per-phase counts.

    The phase order is the reason a text that is both a board word and a clue ends up stored once,
    as ``kind='board_word'``.
    """
    report: dict[str, dict[str, int]] = {}
    for kind, texts in (
        (KIND_BOARD_WORD, board_word_texts(session, frame_id)),
        (KIND_CLUE, clue_texts(session, frame_id)),
    ):
        inserted, skipped = insert_embeddings(
            session, frame_id, texts, kind, encoder, batch_size=batch_size)
        report[kind] = {"candidates": len(
            texts), "inserted": inserted, "skipped": skipped}
        logger.info("phase %s: candidates=%d inserted=%d skipped=%d",
                    kind, len(texts), inserted, skipped)
    return report
