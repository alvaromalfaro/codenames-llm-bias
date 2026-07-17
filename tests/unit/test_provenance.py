"""Tests for run-provenance helpers (backend.app.core.provenance) and the LLMService template
fingerprint. These run without a database; the end-to-end persisted-run provenance assertions live in
test_game_runner.py (DB-gated)."""
import re
import subprocess

import pytest

from backend.app.core import provenance
from backend.app.core.game_runner import SeatSpec
from backend.app.core.llm_service import LLMService


# template fingerprint
def test_template_fingerprint_deterministic():
    """Two services loading the same templates produce the same fingerprint (64-hex SHA-256)."""
    fp1 = LLMService().template_fingerprint()
    fp2 = LLMService().template_fingerprint()
    assert fp1 == fp2
    assert re.fullmatch(r"[0-9a-f]{64}", fp1)


def test_template_fingerprint_content_sensitive():
    """Mutating one loaded template text changes the fingerprint - it hashes content, not identity."""
    svc = LLMService()
    before = svc.template_fingerprint()
    svc._user_prompt_cg = svc._user_prompt_cg + "x"
    assert svc.template_fingerprint() != before


def test_template_fingerprint_reflects_default_fallback(monkeypatch):
    """The whole point of hashing loaded texts: a service that fell back to a ``_default_*`` template
    (because the file was 'missing') fingerprints differently than one that read the real file."""
    real = LLMService().template_fingerprint()

    # Simulate the CLUE_GIVER system template file being absent, forcing the _default_* fallback.
    real_open = open

    def fake_open(path, *args, **kwargs):
        if path == LLMService.SYSTEM_TEMP_CG_PATH:
            raise FileNotFoundError(path)
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", fake_open)
    fallback = LLMService()
    # The loaded text really is the hardcoded default now, not the file.
    assert fallback._system_prompt_cg == fallback._default_system_prompt_cg()
    assert fallback.template_fingerprint() != real


# model_registry_snapshot shape + asymmetry
def test_snapshot_encodes_strong_vs_best_effort_asymmetry():
    """A local (ollama) seat records a strong ``ollama_digest`` identity; an API (openrouter) seat
    records only the requested model string. Both seats present; the asymmetry is explicit in data."""
    specs = [SeatSpec("ollama", "m0"), SeatSpec("openrouter", "m1")]
    snap = provenance.build_model_registry_snapshot(
        specs, digest_resolver=lambda m: f"sha256:{m}-digest")

    assert set(snap) == {"0", "1"}
    local = snap["0"]
    assert local["provider"] == "ollama" and local["model_name"] == "m0"
    assert local["identity_kind"] == "ollama_digest"
    assert local["identity"] == "sha256:m0-digest"
    assert local["note"]

    api = snap["1"]
    assert api["provider"] == "openrouter" and api["model_name"] == "m1"
    assert api["identity_kind"] == "requested_model_string"
    assert api["identity"] == "m1"
    # The two seats carry different identity_kinds - the strong/best-effort distinction is in the data.
    assert local["identity_kind"] != api["identity_kind"]


def test_snapshot_digest_failure_records_null_identity_returning_normally():
    """Record-only: a resolver that returns None (daemon unreachable) yields a null identity + note
    without raising - a game must never abort over provenance."""
    specs = [SeatSpec("ollama", "m0"), SeatSpec("openrouter", "m1")]
    snap = provenance.build_model_registry_snapshot(
        specs, digest_resolver=lambda m: None)
    local = snap["0"]
    assert local["identity"] is None
    assert local["identity_kind"] == "ollama_digest_unavailable"
    assert local["note"]


def test_snapshot_digest_resolver_raising_is_swallowed():
    """Even a resolver that RAISES degrades to a null identity - the builder never propagates."""
    def boom(_model):
        raise RuntimeError("daemon down")

    specs = [SeatSpec("ollama", "m0")]
    snap = provenance.build_model_registry_snapshot(
        specs, digest_resolver=boom)
    assert snap["0"]["identity"] is None
    assert snap["0"]["identity_kind"] == "ollama_digest_unavailable"


def test_snapshot_default_resolver_honours_monkeypatch(monkeypatch):
    """With no explicit resolver, the builder resolves the module-level name at call time, so a
    monkeypatch of provenance.resolve_ollama_digest is honoured (the seam the DB path relies on)."""
    monkeypatch.setattr(provenance, "resolve_ollama_digest",
                        lambda m, host=None: "sha256:patched")
    snap = provenance.build_model_registry_snapshot([SeatSpec("ollama", "m0")])
    assert snap["0"]["identity"] == "sha256:patched"
    assert snap["0"]["identity_kind"] == "ollama_digest"


# code_version
def test_git_code_version_format():
    """In this repo git resolves: a short SHA, optionally suffixed ``-dirty``. Assert the format, not
    a specific SHA."""
    cv = provenance.git_code_version()
    assert cv is not None
    assert re.fullmatch(r"[0-9a-f]{7,40}(-dirty)?", cv)


def test_git_code_version_none_when_git_unavailable(monkeypatch):
    """git absent (e.g. running from a container image) -> None, never an exception."""
    def no_git(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(subprocess, "run", no_git)
    assert provenance.git_code_version() is None


def test_git_code_version_none_on_called_process_error(monkeypatch):
    """A non-repo directory (git returns non-zero) -> None."""
    def failing(*args, **kwargs):
        raise subprocess.CalledProcessError(128, args[0])

    monkeypatch.setattr(subprocess, "run", failing)
    assert provenance.git_code_version() is None
