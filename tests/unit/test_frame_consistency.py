"""A.6 — measurement-frame TRANSPORT consistency (no embeddings).

Closes the §1.5 transport loop: generator → sidecar → sealed boards → DB row → platform read. It
proves the platform holds the SAME frame that produced the A.3 reference fixture, without recomputing
any embedding.

PROVES:
  * The platform holds the correct frame — its identity (frame_id), gender axis (at float32 storage
    precision), and encoder recipe match the fixture's / sidecar's frame; and the 28 sealed boards
    resolve to exactly this frame row.

DEFERS to 6′:
  * That the platform COMPUTES the same rho with that frame — that needs a live φ* embedding runtime
    in the backend, which does not exist yet. rho_raw/rho_cent/rho_weat stay frozen in the fixture.

CONSEQUENCE: after A.6, any rho mismatch found in 6′ is isolable to the embedding path, not to the
frame — that is the diagnostic value of this test.

This module MUST NOT import sentence-transformers, load φ*, embed anything, or recompute rho. Checks
that would need embeddings are marked ``# 6′:`` and deferred, never implemented here.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from backend.app.db.ingest_boards import ingest_boards_if_absent
from backend.app.db.ingest_frame import ingest_frame_if_absent
from backend.app.db.models import BoardModel, MeasurementFrameModel

pytestmark = pytest.mark.skipif(
    not os.environ.get("DATABASE_URL"), reason="requires a live Postgres database"
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BOARDS_DIR = _REPO_ROOT / "data" / "boards"
_SIDECAR = _BOARDS_DIR / "measurement_frame.json"
_FIXTURE = _REPO_ROOT / "board_generator" / "tests" / "fixtures" / "rho_reference_8a404797.json"
_TARGET = "8a404797b3e656dd00683910aa829bbbc584c6b23c37e9f0de1173d11a9d0cc3"


def _load(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def ingested() -> None:
    """Ingest the real frame then boards (idempotent), mirroring the production startup order."""
    from backend.app.db.session import session_scope

    with session_scope() as session:
        ingest_frame_if_absent(session, _SIDECAR)
    with session_scope() as session:
        ingest_boards_if_absent(session, _BOARDS_DIR)


def _frame_row() -> MeasurementFrameModel:
    from backend.app.db.session import session_scope

    with session_scope() as session:
        row = session.get(MeasurementFrameModel, _TARGET)
        assert row is not None, "measurement_frame row not found — ingestion did not run"
        session.expunge(row)
        return row


# ------------------------------------------------------------------------------------------------
# 1. Identity match — three-way, exact string
# ------------------------------------------------------------------------------------------------
def test_three_way_frame_id_identity(ingested) -> None:
    fixture_id = _load(_FIXTURE)["frame_id"]
    sidecar_id = _load(_SIDECAR)["frame_id"]
    db_id = _frame_row().frame_id
    assert fixture_id == sidecar_id == db_id == _TARGET


# ------------------------------------------------------------------------------------------------
# 2. Encoder recipe match — DB row == sidecar == fixture, exact
# ------------------------------------------------------------------------------------------------
def test_encoder_recipe_matches_across_sidecar_fixture_and_db(ingested) -> None:
    sidecar_enc = _load(_SIDECAR)["encoder"]
    fixture_enc = _load(_FIXTURE)["encoder"]
    row = _frame_row()
    db_enc = {
        "name": row.encoder_name,
        "revision": row.encoder_revision,
        "pooling": row.encoder_pooling,
        "normalize": row.encoder_normalize,
    }
    assert db_enc == sidecar_enc == fixture_enc
    # The recipe the platform will embed with in 6′: mpnet@e8c3b32, mean pooling, normalized.
    assert db_enc == {
        "name": "sentence-transformers/all-mpnet-base-v2",
        "revision": "e8c3b32edf5434bc2275fc9bab85f82640a19130",
        "pooling": "mean",
        "normalize": True,
    }


# ------------------------------------------------------------------------------------------------
# 3. Axis match — the delicate one: the DB axis IS the sidecar axis at float32 precision
# ------------------------------------------------------------------------------------------------
def test_db_axis_is_sidecar_axis_at_float32_precision(ingested) -> None:
    sidecar_hex = _load(_SIDECAR)["gender_axis"]["vector_f64_be_hex"]
    a64 = np.frombuffer(bytes.fromhex(sidecar_hex), dtype=">f8").astype(np.float64)
    assert a64.shape == (768,)

    # The exact value the platform reads back: the sidecar axis round-tripped through float32.
    a32_ref = a64.astype(np.float32).astype(np.float64)

    a_db = np.asarray(_frame_row().gender_axis, dtype=np.float64)
    assert a_db.shape == (768,)

    max_abs_dev = float(np.max(np.abs(a_db - a32_ref)))
    # Primary guard: elementwise near-exact — the DB axis IS the float32 image of the sidecar axis.
    assert max_abs_dev <= 1e-6, f"axis transport bug: max abs deviation {max_abs_dev} > 1e-6"
    assert np.allclose(a_db, a32_ref, atol=1e-7, rtol=0), f"max abs deviation {max_abs_dev}"

    # Secondary sanity: direction preserved vs the full-precision axis.
    cos = float(np.dot(a_db, a64) / (np.linalg.norm(a_db) * np.linalg.norm(a64)))
    assert cos > 1 - 1e-6

    print(f"\n[A.6] axis max abs deviation (DB float32 vs sidecar float32 image) = {max_abs_dev:.3e}")


# ------------------------------------------------------------------------------------------------
# 4. mu_bar + centering_reference survived transport (provenance projection intact)
# ------------------------------------------------------------------------------------------------
def test_mu_bar_and_centering_reference_survived_transport(ingested) -> None:
    sidecar_mu = _load(_SIDECAR)["mu_bar"]
    construction = _frame_row().axis_construction
    db_mu = construction["mu_bar"]

    assert db_mu["cos_with_axis"] == sidecar_mu["cos_with_axis"]
    assert db_mu["norm"] == sidecar_mu["norm"]

    db_cref = db_mu["centering_reference"]
    sidecar_cref = sidecar_mu["centering_reference"]
    assert db_cref["n_reference_items"] == sidecar_cref["n_reference_items"] == 124
    assert db_cref["attribute_counts_by_pole"]["male"] == 19
    assert db_cref["attribute_counts_by_pole"]["female"] == 19
    assert db_cref["loaded_counts_by_pole"]["male"] == 46
    assert db_cref["loaded_counts_by_pole"]["female"] == 40


# ------------------------------------------------------------------------------------------------
# 5. Sealed boards resolve to this frame (closing the A.4 ↔ A.5 transport link)
# ------------------------------------------------------------------------------------------------
def test_all_28_sealed_boards_resolve_to_this_frame(ingested) -> None:
    from sqlalchemy import func, select

    from backend.app.db.session import session_scope

    with session_scope() as session:
        joined = session.execute(
            select(func.count())
            .select_from(BoardModel)
            .join(
                MeasurementFrameModel,
                BoardModel.measurement_frame_id == MeasurementFrameModel.frame_id,
            )
            .where(MeasurementFrameModel.frame_id == _TARGET)
        ).scalar_one()
        assert joined == 28

        distinct = session.execute(
            select(BoardModel.measurement_frame_id)
            .where(BoardModel.measurement_frame_id.isnot(None))
            .distinct()
        ).scalars().all()
        assert distinct == [_TARGET]


# 6′: platform recomputes rho_raw/rho_cent/rho_weat with this frame and matches the fixture
#     within ~1e-6 — DEFERRED: needs a live φ* embedding runtime in the backend (not present).
