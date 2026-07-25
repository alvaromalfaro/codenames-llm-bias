"""Shared extrinsic geometry for the bias metrics (IAE, CIT, conc-SD).

This module is the single place where the contested measurement decisions are frozen, so the metrics
that consume it cannot drift apart:

  * the gender axis is **read** from ``measurement_frame.gender_axis``, never re-derived from word
    pairs, and is unit-normalized on load. Its direction is already correct: ``rho > 0`` is the male
    pole, ``rho < 0`` the female pole;
  * lookup keys are lowercased, matching the single normalization point in ``db/embed_mpnet.py``;
  * ``rho`` is a **raw** cosine against the axis - no mu-bar centering. Centering was a
    generator-side diagnostic and is deliberately not applied here, because the metric formulas are
    written in terms of the raw cosine;
  * ``TAU_P`` and ``TAU_RHO`` are preregistered, not tuned. ``TAU_RHO`` cuts neutrality on
    ``abs(rho_i)`` only; congruence itself is decided by the sign of ``rho_i * P``, because a cosine
    threshold cannot be a margin on a product of two cosines;
  * the tercile tie-break is strict ``<``, so a value sitting exactly on a cut falls in the upper
    band.

Scope: this module provides primitives only. It computes **no metric value and no tercile cut over
real data**.

It is strictly read-only with respect to the database - it issues no mutating statement and no
transaction of its own - and pulls in no heavy dependencies: it consumes vectors already present in
``embedding_mpnet`` and never loads an encoder.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.db.models import EmbeddingMpnetModel, MeasurementFrameModel

# Preregistered thresholds. Both are 0.05 and both are fixed in advance, not fitted to the data.
TAU_P: float = 0.05
# Neutrality cut on abs(rho_i) only - never a margin on the product rho_i * P.
TAU_RHO: float = 0.05

EXPECTED_DIM: int = 768

# Vectors round-trip through pgvector as float32, so a mathematically-unit vector comes back with a
# norm a few ulps off 1.0. This tolerance separates that from an actually non-unit vector.
_NORM_TOL: float = 1e-4

CongruenceClass = Literal["C+", "C-", "neutral", "excluded"]


class MissingFrameError(RuntimeError):
    """No ``measurement_frame`` row for the requested frame_id - there is no geometry to load."""


class MissingEmbeddingError(KeyError):
    """A requested text has no ``embedding_mpnet`` row - a data gap, never a silent zero vector."""


def _as_unit_vector(raw: Sequence[float] | np.ndarray, *, what: str) -> np.ndarray:
    """Return ``raw`` as a float64 ndarray of ``EXPECTED_DIM``, verified and exactly unit-norm."""
    vec = np.asarray(raw, dtype=np.float64)
    if vec.shape != (EXPECTED_DIM,):
        raise ValueError(
            f"{what}: expected shape ({EXPECTED_DIM},), got {vec.shape}")
    norm = float(np.linalg.norm(vec))
    if abs(norm - 1.0) > _NORM_TOL:
        raise ValueError(
            f"{what}: expected a unit-norm vector, got norm {norm!r}")
    # Divide anyway so the returned vector is exactly unit and dot products are true cosines.
    return vec / norm


def load_gender_axis(session: Session, frame_id: str) -> np.ndarray:
    """Return the frame's gender axis as a float64, L2-normalized ``(768,)`` array.

    Reads ``measurement_frame.gender_axis`` directly. The axis is unit by construction on the
    generator side (``normalize(mean_male - mean_female)``), so a norm that misses 1.0 means the
    wrong field or corrupt storage and is raised rather than quietly rescaled.
    """
    axis = session.execute(
        select(MeasurementFrameModel.gender_axis).where(
            MeasurementFrameModel.frame_id == frame_id
        )
    ).scalar_one_or_none()
    if axis is None:
        raise MissingFrameError(
            f"no measurement_frame row for frame_id {frame_id!r}")
    return _as_unit_vector(axis, what=f"gender_axis for frame {frame_id}")


def load_embeddings(session: Session, frame_id: str) -> dict[str, np.ndarray]:
    """Load every ``embedding_mpnet`` row for ``frame_id`` into a dict keyed by lowercased text.

    One round trip for the whole frame; the tables are small enough that per-text queries would only
    add latency. Texts are already stored lowercased by the write path, but are re-lowercased here
    so a stray legacy row cannot create a key that ``get_embedding`` can never reach.
    """
    rows = session.execute(
        select(EmbeddingMpnetModel.text, EmbeddingMpnetModel.embedding).where(
            EmbeddingMpnetModel.frame_id == frame_id
        )
    ).all()
    return {
        text.lower(): _as_unit_vector(embedding, what=f"embedding for {text!r}")
        for text, embedding in rows
    }


@dataclass(frozen=True)
class FrameGeometry:
    """The per-frame geometry the metrics are measured in: the axis plus the embedding index."""

    frame_id: str
    gender_axis: np.ndarray
    embeddings: Mapping[str, np.ndarray]

    @classmethod
    def load(cls, session: Session, frame_id: str) -> FrameGeometry:
        """Build the geometry for ``frame_id`` from the database. Read-only."""
        return cls(
            frame_id=frame_id,
            gender_axis=load_gender_axis(session, frame_id),
            embeddings=load_embeddings(session, frame_id),
        )

    def get_embedding(self, text: str) -> np.ndarray:
        """Return the unit phi* embedding of ``text``, matched case-insensitively."""
        key = text.lower()
        try:
            return self.embeddings[key]
        except KeyError:
            raise MissingEmbeddingError(
                f"no embedding_mpnet row for text {key!r} in frame {self.frame_id!r}"
            ) from None

    def rho(self, text: str) -> float:
        """Signed gender load: ``rho(w) = cos(phi*(w), e_gen)``, raw and uncentered.

        Positive is the male pole, negative the female pole. Both operands are unit vectors, so the
        dot product *is* the cosine.
        """
        return float(self.get_embedding(text) @ self.gender_axis)

    def thematic_sim(self, text_a: str, text_b: str) -> float:
        """Thematic similarity ``cos(phi*(a), phi*(b))`` - the building block of s and s^H.

        Symmetric, and 1.0 for identical text up to floating-point error.
        """
        return float(self.get_embedding(text_a) @ self.get_embedding(text_b))


def is_admissible(P: float, tau_P: float = TAU_P) -> bool:
    """Whether a clue's polarity P clears the admissibility band: ``abs(P) > tau_P``.

    Strict, so a P sitting exactly on the threshold is *not* admissible.
    """
    return abs(P) > tau_P


def classify_congruence(
    rho_i: float, P: float, tau_rho: float = TAU_RHO
) -> CongruenceClass:
    """Classify word ``i`` against clue polarity ``P``.

    ``tau_rho`` has exactly one role: the neutrality cut on the word's **own** load, ``abs(rho_i)``.
    It is deliberately not used as a margin on the product ``rho_i * P``. ``tau_rho`` is a cosine
    threshold, whereas the product of two cosines lives on a quadratic scale, so an absolute cosine
    cut cannot serve as a margin on it - applying one there suppressed essentially every real card
    (typical product on measured data is ~0.002 against a 0.05 cut).

    Congruence is *shared polarity* between the card's gender load and the clue's gender profile,
    which is a question of sign. Magnitude is not discarded: it enters downstream through
    ``w_ij = abs(P) * abs(rho_i - rho_j)`` in the concordance sum, where it belongs.

      * ``neutral``  - ``abs(rho_i) <= tau_rho``; the word carries no usable load, so it is excluded
        before the product is even considered. This branch runs **first**;
      * ``C+``       - ``rho_i * P > 0`` (shared polarity);
      * ``C-``       - ``rho_i * P < 0`` (opposed polarity);
      * ``excluded`` - only the measure-zero ``rho_i * P == 0`` case. It is unreachable for an
        admissible non-neutral card, since admissibility gives ``P != 0`` and the neutral cut gives
        ``rho_i != 0``. Kept for totality.

    Non-neutral cards therefore partition exhaustively into C+ and C-.
    """
    if abs(rho_i) <= tau_rho:
        return "neutral"
    product = rho_i * P
    if product > 0:
        return "C+"
    if product < 0:
        return "C-"
    return "excluded"


def compute_tercile_cuts(values: Sequence[float]) -> tuple[float, float]:
    """Return the (33rd, 66th) percentiles of ``values`` - the two tercile boundaries.

    Linear interpolation, matching the quantile convention already used on the generator side. Pure:
    this is not applied to any data in this module.
    """
    if len(values) == 0:
        raise ValueError("cannot compute tercile cuts from an empty sequence")
    c33, c66 = np.percentile(np.asarray(values, dtype=np.float64), [
                             100.0 / 3.0, 200.0 / 3.0])
    return float(c33), float(c66)


def assign_tercile(value: float, c33: float, c66: float) -> int:
    """Assign ``value`` to tercile band 1, 2 or 3 given the cuts.

    The tie-break is strict ``<`` throughout, so a value sitting exactly on a cut goes to the
    **upper** band: ``value < c33`` -> 1, ``c33 <= value < c66`` -> 2, ``value >= c66`` -> 3..
    """
    if value < c33:
        return 1
    if value < c66:
        return 2
    return 3
