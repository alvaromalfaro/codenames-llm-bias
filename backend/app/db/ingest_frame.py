"""Idempotent ingestion of the measurement-frame sidecar into ``measurement_frame``.

Reads ``data/boards/measurement_frame.json`` (emitted + content-hashed by the board generator) and
inserts exactly one row. The frame is content-addressed by ``frame_id`` and immutable, so ingestion
is idempotent by that PK: an existing row is a no-op, and an existing row whose encoder identity
disagrees with the sidecar raises :class:`StaleFrameError` (the hash and content would be
contradicting each other — a hard STOP, never a silent overwrite).

Column vs JSONB split:
  * columns: ``frame_id`` (verbatim PK), ``encoder_name/revision/pooling/normalize``, ``gender_axis``
    (the sidecar's ``gender_axis.vector_f64_be_hex`` decoded to a 768-float pgvector),
    ``generator_version``, ``created_at``.
  * ``axis_construction`` JSONB holds the sidecar's ``gender_axis.construction`` verbatim, with the
    FULL ``mu_bar`` block (dim, vector_f64_be_hex, norm, cos_with_axis, centering_reference) nested
    under a ``"mu_bar"`` key. This is a provenance projection, the authoritative identity is the
    frame_id hash of the canonical sidecar bytes, so a partial JSONB representation is fine.
"""

from __future__ import annotations

import json
import logging
import struct
from datetime import datetime
from pathlib import Path

from sqlalchemy.orm import Session

from backend.app.db.models import MeasurementFrameModel

logger = logging.getLogger(__name__)

DEFAULT_FRAME_PATH = "data/boards/measurement_frame.json"


class StaleFrameError(RuntimeError):
    """An existing measurement_frame row disagrees with the sidecar — reconcile, never overwrite."""


def _decode_f64_be_hex(hex_str: str) -> list[float]:
    """Decode a big-endian IEEE-754 float64 hex string to a list of floats (inverse of the emitter)."""
    raw = bytes.fromhex(hex_str)
    return list(struct.unpack(f">{len(raw) // 8}d", raw))


def frame_artifact_to_orm(data: dict) -> MeasurementFrameModel:
    """Map a parsed measurement-frame sidecar into a (non-persisted) ``MeasurementFrameModel``."""
    encoder = data["encoder"]
    axis = data["gender_axis"]
    # axis_construction JSONB = the sidecar's construction, with the full mu_bar block nested in.
    construction = dict(axis["construction"])
    construction["mu_bar"] = data["mu_bar"]
    return MeasurementFrameModel(
        frame_id=data["frame_id"],
        encoder_name=encoder["name"],
        encoder_revision=encoder["revision"],
        encoder_pooling=encoder["pooling"],
        encoder_normalize=encoder["normalize"],
        gender_axis=_decode_f64_be_hex(axis["vector_f64_be_hex"]),
        axis_construction=construction,
        generator_version=data.get("generator_version"),
        created_at=datetime.fromisoformat(data["created_at"]),
    )


def _encoder_identity(data: dict) -> dict:
    """The human-meaningful identity fields used to detect an impossible hash/content disagreement."""
    encoder = data["encoder"]
    return {
        "encoder_name": encoder["name"],
        "encoder_revision": encoder["revision"],
        "encoder_pooling": encoder["pooling"],
        "encoder_normalize": encoder["normalize"],
        "generator_version": data.get("generator_version"),
    }


def ingest_frame_if_absent(
    session: Session, frame_path: str | Path = DEFAULT_FRAME_PATH
) -> bool:
    """Insert the measurement frame if its ``frame_id`` is not already stored. Returns True if inserted.

    A missing sidecar is a logged no-op (returns False). An existing row with the same frame_id is a
    no-op; an existing row with a DIFFERENT encoder identity raises :class:`StaleFrameError`.
    """
    path = Path(frame_path)
    if not path.exists():
        logger.warning(
            "Measurement-frame sidecar %s does not exist; skipping", path)
        return False

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    frame_id = data["frame_id"]

    existing = session.get(MeasurementFrameModel, frame_id)
    if existing is not None:
        want = _encoder_identity(data)
        have = {
            "encoder_name": existing.encoder_name,
            "encoder_revision": existing.encoder_revision,
            "encoder_pooling": existing.encoder_pooling,
            "encoder_normalize": existing.encoder_normalize,
            "generator_version": existing.generator_version,
        }
        if have != want:
            raise StaleFrameError(
                f"measurement_frame {frame_id} already stored with different content: "
                f"have={have} want={want}"
            )
        return False

    session.add(frame_artifact_to_orm(data))
    session.commit()
    return True
