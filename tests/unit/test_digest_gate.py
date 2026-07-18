"""Tests for the reproducibility digest gate (game_runner._enforce_local_digests).

The gate turns model_registry_snapshot from a record-only witness into an abort: when digest
enforcement is on (the batch path), a local seat whose served ollama digest does not match
config.EXPECTED_LOCAL_DIGESTS stops the run before any game is dispatched. All tests here are
hermetic - the served digest is supplied by monkeypatching provenance.resolve_ollama_digest, so no
ollama daemon and no database is touched.
"""
import re

import pytest
from unittest.mock import MagicMock

from backend.app import config
from backend.app.core import game_runner, provenance
from backend.app.core.game_runner import (
    ModelDigestMismatchError, SeatSpec, _enforce_local_digests, run_single_game,
)

# Reuse the validated 25-card board and mock-client helpers from the runner test module (same
# tests/unit package dir, on sys.path under pytest's default import mode).
from test_game_runner import _board, _make_client

# A local model that is pinned in config, plus the digest the daemon would serve for it. The served
# form is the bare hex (what resolve_ollama_digest returns verbatim from /api/tags); the config
# constant carries the `sha256:` prefix - so a passing match also exercises prefix-normalisation.
_LOCAL_MODEL = "llama3.1:latest"
_EXPECTED = config.EXPECTED_LOCAL_DIGESTS[_LOCAL_MODEL]
# bare hex, no prefix
_SERVED_OK = _EXPECTED.split(":", 1)[1]
# a clearly different digest
_SERVED_WRONG = "a" * 64

_LOCAL = SeatSpec("ollama", _LOCAL_MODEL)
_API = SeatSpec("openrouter", "m1")
_SPECS = (_LOCAL, _API)


# config sanity: the pinned constants are well-formed
def test_expected_digests_cover_both_local_models_and_are_wellformed():
    assert set(config.EXPECTED_LOCAL_DIGESTS) == {
        "llama3.1:latest", "mistral-small3.2:latest"}
    for tag, digest in config.EXPECTED_LOCAL_DIGESTS.items():
        assert re.fullmatch(r"sha256:[0-9a-f]{64}", digest), (tag, digest)


# unit: the gate helper over crafted snapshots
def _snapshot(*, served, identity_kind="ollama_digest", model=_LOCAL_MODEL):
    """A one-local-seat-plus-one-api-seat snapshot in the shape build_model_registry_snapshot emits."""
    return {
        "0": {"provider": "ollama", "model_name": model,
              "identity_kind": identity_kind, "identity": served, "note": "x"},
        "1": {"provider": "openrouter", "model_name": "m1",
              "identity_kind": "requested_model_string", "identity": "m1", "note": "x"},
    }


def test_gate_passes_on_matching_digest_prefix_agnostic():
    # served bare hex vs `sha256:`-prefixed expected -> normalised equal -> no raise.
    _enforce_local_digests(_snapshot(served=_SERVED_OK), enforce=True)


def test_gate_raises_on_mismatch():
    with pytest.raises(ModelDigestMismatchError, match="expected"):
        _enforce_local_digests(_snapshot(served=_SERVED_WRONG), enforce=True)


def test_gate_raises_on_unknown_local_model_under_enforcement():
    # not in EXPECTED_LOCAL_DIGESTS
    snap = _snapshot(served=_SERVED_OK, model="llama3.2:latest")
    with pytest.raises(ModelDigestMismatchError, match="no expected digest"):
        _enforce_local_digests(snap, enforce=True)


def test_gate_raises_on_unavailable_digest_when_enforcing():
    snap = _snapshot(served=None, identity_kind="ollama_digest_unavailable")
    with pytest.raises(ModelDigestMismatchError, match="unavailable"):
        _enforce_local_digests(snap, enforce=True)


def test_gate_is_noop_when_not_enforcing():
    # A mismatch and an unavailable digest both pass silently when enforcement is off (record-only).
    _enforce_local_digests(_snapshot(served=_SERVED_WRONG), enforce=False)
    _enforce_local_digests(
        _snapshot(served=None, identity_kind="ollama_digest_unavailable"), enforce=False)


def test_gate_skips_api_seats():
    # An openrouter-only snapshot has no local digest to certify -> passes even under enforcement.
    api_only = {"0": {"provider": "openrouter", "model_name": "x-ai/grok-4.3",
                      "identity_kind": "requested_model_string", "identity": "x-ai/grok-4.3"}}
    _enforce_local_digests(api_only, enforce=True)


# the gate through run_single_game (no DB, no daemon)
@pytest.mark.asyncio
async def test_run_aborts_on_mismatch_no_game_dispatched(monkeypatch):
    """enforce_digests=True with a served digest that differs: run_single_game raises before any
    client is built (no game dispatched) and before the writer is ever called (no run row)."""
    monkeypatch.setattr(game_runner, "_db_enabled", lambda: False)
    monkeypatch.setattr(provenance, "resolve_ollama_digest",
                        lambda model_name, host=None: _SERVED_WRONG)
    factory = MagicMock(side_effect=AssertionError(
        "no client may be built when the gate aborts"))
    persist_game = MagicMock(side_effect=AssertionError(
        "writer must not be called on abort"))
    monkeypatch.setattr(game_runner.writer, "persist_game", persist_game)

    with pytest.raises(ModelDigestMismatchError):
        await run_single_game(board=_board(), seat_specs=_SPECS, master_seed=1, temperature=0.4,
                              persist=False, enforce_digests=True, client_factory=factory)
    factory.assert_not_called()
    persist_game.assert_not_called()


@pytest.mark.asyncio
async def test_run_aborts_on_unavailable_digest_when_enforcing(monkeypatch):
    """enforce_digests=True with an unavailable digest (daemon unreachable) aborts the run."""
    monkeypatch.setattr(game_runner, "_db_enabled", lambda: False)
    monkeypatch.setattr(provenance, "resolve_ollama_digest",
                        lambda model_name, host=None: None)
    factory = MagicMock(side_effect=AssertionError(
        "no client may be built when the gate aborts"))
    with pytest.raises(ModelDigestMismatchError, match="unavailable"):
        await run_single_game(board=_board(), seat_specs=_SPECS, master_seed=1, temperature=0.4,
                              persist=False, enforce_digests=True, client_factory=factory)
    factory.assert_not_called()


@pytest.mark.asyncio
async def test_run_proceeds_on_matching_digest_when_enforcing(monkeypatch):
    """enforce_digests=True with a matching served digest lets the game play to completion - the gate
    passes (and prefix-normalisation holds: bare-hex served vs sha256:-prefixed config)."""
    monkeypatch.setattr(game_runner, "_db_enabled", lambda: False)
    monkeypatch.setattr(provenance, "resolve_ollama_digest",
                        lambda model_name, host=None: _SERVED_OK)
    clients = {0: _make_client("m0", lambda: ["LEMONADE"]),
               1: _make_client("m1", lambda: ["LEMONADE"])}
    res = await run_single_game(board=_board(), seat_specs=_SPECS, master_seed=777, temperature=0.4,
                                persist=False, enforce_digests=True,
                                client_factory=lambda i, spec: clients[i])
    assert res.status == "completed" and res.result == "loss_assassin"


@pytest.mark.asyncio
async def test_run_proceeds_on_unavailable_digest_when_not_enforcing(monkeypatch):
    """Interactive parity: enforce_digests=False (the default) leaves the snapshot record-only, so an
    unavailable digest does NOT abort - the game still plays."""
    monkeypatch.setattr(game_runner, "_db_enabled", lambda: False)
    monkeypatch.setattr(provenance, "resolve_ollama_digest",
                        lambda model_name, host=None: None)
    clients = {0: _make_client("m0", lambda: ["LEMONADE"]),
               1: _make_client("m1", lambda: ["LEMONADE"])}
    res = await run_single_game(board=_board(), seat_specs=_SPECS, master_seed=777, temperature=0.4,
                                persist=False, enforce_digests=False,
                                client_factory=lambda i, spec: clients[i])
    assert res.status == "completed" and res.result == "loss_assassin"
