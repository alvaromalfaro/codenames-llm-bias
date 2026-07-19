"""Tests for create_run (game_runner.create_run) called directly."""
import os
import uuid

import pytest
from unittest.mock import MagicMock

from backend.app import config
from backend.app.core import game_runner, provenance
from backend.app.core.game_runner import (
    ModelDigestMismatchError, SeatSpec, create_run,
)

# A local model pinned in config plus the (bare-hex) digest the daemon would serve for a match.
_LOCAL_MODEL = "llama3.1:latest"
_SERVED_OK = config.EXPECTED_LOCAL_DIGESTS[_LOCAL_MODEL].split(":", 1)[1]
_SERVED_WRONG = "a" * 64

_LOCAL = SeatSpec("ollama", _LOCAL_MODEL)
_API = SeatSpec("openrouter", "m1")
_SPECS = (_LOCAL, _API)


# persist=False: deterministic id, no DB, no daemon
def test_persist_false_deterministic_run_id() -> None:
    rid, snap = create_run(seat_specs=_SPECS, master_seed=123, temperature=0.4,
                           template_fingerprint="tpl@v1", persist=False)
    # the same synthesized id the inline block produced: uuid5(_GAME_NAMESPACE, "run:{seed}")
    assert rid == str(uuid.uuid5(game_runner._GAME_NAMESPACE, "run:123"))
    rid2, _ = create_run(seat_specs=_SPECS, master_seed=123, temperature=0.4,
                         template_fingerprint="tpl@v1", persist=False)
    assert rid == rid2  # deterministic in master_seed
    assert snap is None  # pure dry-run: not persisting, not enforcing


def test_persist_false_no_enforce_builds_no_snapshot(monkeypatch) -> None:
    # The default dry-run path must add no daemon call: the snapshot is never built.
    spy = MagicMock(side_effect=AssertionError(
        "no snapshot may be built on a pure dry-run"))
    monkeypatch.setattr(provenance, "build_model_registry_snapshot", spy)
    rid, snap = create_run(seat_specs=_SPECS, master_seed=9, temperature=0.4,
                           template_fingerprint="t", persist=False, enforce_digests=False)
    assert snap is None
    spy.assert_not_called()


# the digest gate runs, and before any run row is created
def test_gate_aborts_before_any_persistence(monkeypatch) -> None:
    monkeypatch.setattr(provenance, "resolve_ollama_digest",
                        lambda model_name, host=None: _SERVED_WRONG)
    # persist=True would open a session to write the row; the gate must abort first, so a session
    # that explodes on use proves no run row is ever created on a mismatch.
    import backend.app.db.session as session_mod
    monkeypatch.setattr(session_mod, "session_scope",
                        MagicMock(side_effect=AssertionError("no run row may be created on abort")))
    with pytest.raises(ModelDigestMismatchError, match="expected"):
        create_run(seat_specs=_SPECS, master_seed=1, temperature=0.4,
                   template_fingerprint="t", persist=True, enforce_digests=True)


# DB-gated: persist=True creates exactly one run row with the right provenance
@pytest.mark.skipif(not os.environ.get("DATABASE_URL"),
                    reason="requires a live Postgres database")
def test_persist_true_creates_one_run_with_provenance(monkeypatch) -> None:
    monkeypatch.setattr(provenance, "resolve_ollama_digest",
                        lambda model_name, host=None: _SERVED_OK)
    monkeypatch.setattr(provenance, "git_code_version", lambda: "code@abcdef")
    rid, snap = create_run(seat_specs=_SPECS, master_seed=555, temperature=0.4,
                           template_fingerprint="tpl@fingerprint", persist=True,
                           enforce_digests=False)
    assert snap is not None  # persist path builds and stores the snapshot

    from backend.app.db.models import RunModel
    from backend.app.db.session import session_scope
    with session_scope() as session:
        rows = session.query(RunModel).filter_by(id=rid).all()
        assert len(rows) == 1  # exactly one run row
        run = rows[0]
        assert run.temperature == 0.4
        assert int(run.master_seed) == 555
        assert run.prompt_template_version == "tpl@fingerprint"
        assert run.code_version == "code@abcdef"
        assert run.model_registry_snapshot == snap
