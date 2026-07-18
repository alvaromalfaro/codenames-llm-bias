"""DB-gated tests for measurement-frame ingestion and the frame→boards FK link (A.5).

These require a live Postgres (the disposable migrated pgvector) and are skipped when DATABASE_URL is
unset. They are order-independent: every ingest is idempotent, so they neither require nor perform
table cleanup.
"""

import json
import math
import os
from pathlib import Path

import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from backend.app.db.ingest_boards import board_artifact_to_orm, ingest_boards_if_absent
from backend.app.db.ingest_frame import (
    StaleFrameError,
    frame_artifact_to_orm,
    ingest_frame_if_absent,
)
from backend.app.db.models import BoardModel, MeasurementFrameModel

pytestmark = pytest.mark.skipif(
    not os.environ.get("DATABASE_URL"), reason="requires a live Postgres database"
)

_BOARDS_DIR = Path(__file__).resolve().parents[2] / "data" / "boards"
_FRAME_FILE = _BOARDS_DIR / "measurement_frame.json"


def _sidecar() -> dict:
    with open(_FRAME_FILE, encoding="utf-8") as f:
        return json.load(f)


def _target() -> str:
    return _sidecar()["frame_id"]


def test_frame_row_shape_columns_and_jsonb():
    from backend.app.db.session import session_scope

    with session_scope() as session:
        ingest_frame_if_absent(session, _FRAME_FILE)

    sidecar = _sidecar()
    encoder = sidecar["encoder"]
    with session_scope() as session:
        row = session.get(MeasurementFrameModel, _target())
        assert row is not None
        # encoder identity → columns, verbatim
        assert row.encoder_name == encoder["name"]
        assert row.encoder_revision == encoder["revision"]
        assert row.encoder_pooling == encoder["pooling"]
        assert row.encoder_normalize == encoder["normalize"]
        assert row.generator_version == sidecar["generator_version"]

        # gender_axis → the ONE pgvector column: 768 dims, unit L2 (the raw axis is normalized).
        axis = list(row.gender_axis)
        assert len(axis) == 768
        norm = math.sqrt(sum(float(x) * float(x) for x in axis))
        assert norm == pytest.approx(1.0, abs=1e-3)  # float32 pgvector storage

        # axis_construction JSONB carries the construction AND the nested mu_bar block.
        construction = row.axis_construction
        assert construction["method"] == "mean_difference"
        assert construction["weat_sets"] == [6, 7, 8]
        cref = construction["mu_bar"]["centering_reference"]
        assert cref["n_reference_items"] == sidecar["mu_bar"]["centering_reference"][
            "n_reference_items"
        ]
        assert cref["attribute_counts_by_pole"]["male"] == 19
        assert cref["attribute_counts_by_pole"]["female"] == 19
        assert cref["loaded_counts_by_pole"] == {
            "male": 46,
            "female": 40,
            "note": cref["loaded_counts_by_pole"]["note"],
        }


def test_ingest_frame_is_idempotent():
    from backend.app.db.session import session_scope

    with session_scope() as session:
        ingest_frame_if_absent(session, _FRAME_FILE)
    with session_scope() as session:
        inserted_again = ingest_frame_if_absent(session, _FRAME_FILE)
        assert inserted_again is False
    with session_scope() as session:
        count = session.execute(
            select(func.count())
            .select_from(MeasurementFrameModel)
            .where(MeasurementFrameModel.frame_id == _target())
        ).scalar_one()
        assert count == 1


def test_full_startup_order_frame_then_boards_links_all_28():
    """End-to-end A.1+A.4+A.5: frame first, then boards → 28 sealed rows whose FK resolves."""
    from backend.app.db.session import session_scope

    with session_scope() as session:
        ingest_frame_if_absent(session, _FRAME_FILE)
    with session_scope() as session:
        ingest_boards_if_absent(session, _BOARDS_DIR)

    target = _target()
    with session_scope() as session:
        sealed = session.execute(
            select(func.count())
            .select_from(BoardModel)
            .where(BoardModel.measurement_frame_id.isnot(None))
        ).scalar_one()
        assert sealed == 28

        # every sealed board points at the target frame, and the FK join resolves to the row.
        joined = session.execute(
            select(BoardModel.board_id)
            .join(
                MeasurementFrameModel,
                BoardModel.measurement_frame_id == MeasurementFrameModel.frame_id,
            )
            .where(MeasurementFrameModel.frame_id == target)
        ).all()
        assert len(joined) == 28

        distinct = session.execute(
            select(BoardModel.measurement_frame_id)
            .where(BoardModel.measurement_frame_id.isnot(None))
            .distinct()
        ).scalars().all()
        assert distinct == [target]


def test_board_referencing_missing_frame_is_rejected_by_fk():
    """The FK is enforced: a board with a dangling frame_id is REJECTED, never silently NULLed."""
    from backend.app.db.session import session_scope

    bogus = {
        "board_id": "probe-bogus-frame-000",
        "type": "probe",
        "measurement_frame_id": "0" * 64,  # no such frame row
        "grid": {"rows": 5, "cols": 5},
        "cards": [],
    }
    board, _ = board_artifact_to_orm(bogus)
    with pytest.raises(IntegrityError):
        with session_scope() as session:
            session.add(board)
            session.flush()


def test_existing_frame_with_different_content_stops(tmp_path):
    """An impossible hash/content disagreement raises StaleFrameError instead of overwriting."""
    from backend.app.db.session import session_scope

    with session_scope() as session:
        ingest_frame_if_absent(session, _FRAME_FILE)  # ensure the real row exists

    tampered = _sidecar()
    tampered["encoder"] = {**tampered["encoder"], "name": "different/model"}
    tampered_path = tmp_path / "measurement_frame.json"
    tampered_path.write_text(json.dumps(tampered), encoding="utf-8")

    with pytest.raises(StaleFrameError):
        with session_scope() as session:
            ingest_frame_if_absent(session, tampered_path)


def test_frame_artifact_to_orm_decodes_axis_without_db():
    """Pure mapping (no DB): the be-hex axis decodes to 768 floats and mu_bar nests into JSONB."""
    model = frame_artifact_to_orm(_sidecar())
    assert model.frame_id == _target()
    assert len(model.gender_axis) == 768
    assert "mu_bar" in model.axis_construction
