"""Tests for run-provenance helpers (backend.app.core.provenance) and the LLMService template
fingerprint. These run without a database; the end-to-end persisted-run provenance assertions live in
test_game_runner.py (DB-gated)."""
import re
import subprocess
from types import SimpleNamespace

import httpx
import pytest

from backend.app import config
from backend.app.core import provenance
from backend.app.core.game_runner import SeatSpec
from backend.app.core.llm_service import LLMService
# The real lookup: conftest stands a fixed release in for it in every test, so no test reaches a
# daemon; the tests of the lookup itself call it here.
from backend.app.core.provenance import ollama_server_version as _real_ollama_server_version


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
    """The whole point of hashing loaded texts: the fingerprint follows the text a ``_default_*``
    fallback really loaded. The defaults are verbatim copies of the files, so falling back alone
    keeps the fingerprint; a default that drifted from its file changes it."""
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
    assert fallback.template_fingerprint() == real

    monkeypatch.setattr(LLMService, "_default_system_prompt_cg",
                        lambda self: "a default that drifted from its file")
    drifted = LLMService()
    assert drifted._system_prompt_cg == "a default that drifted from its file"
    assert drifted.template_fingerprint() != real


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


# the digest lookup waits a bounded time on the daemon
def test_digest_lookup_waits_the_configured_time(monkeypatch):
    """ollama's own default is no limit at all: a hung daemon would stop the batch before it began."""
    seen = {}

    class FakeClient:
        def __init__(self, host=None, **kwargs):
            seen.update(kwargs)

        def list(self):
            return SimpleNamespace(models=[SimpleNamespace(model="m0:latest", digest="sha256:d")])

    monkeypatch.setattr("ollama.Client", FakeClient)

    assert provenance.resolve_ollama_digest("m0") == "sha256:d"
    assert seen["timeout"].read == config.OLLAMA_TIMEOUT_S
    assert seen["timeout"].connect == config.OLLAMA_CONNECT_TIMEOUT_S


def test_digest_lookup_that_times_out_records_none(monkeypatch):
    """Record-only: a lookup cut by the limit is a null identity, never an exception (the batch's
    digest gate then refuses to start)."""
    class FakeClient:
        def __init__(self, host=None, **kwargs):
            pass

        def list(self):
            raise httpx.ReadTimeout("timed out")

    monkeypatch.setattr("ollama.Client", FakeClient)

    assert provenance.resolve_ollama_digest("m0") is None


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


# the Ollama daemon's release
class _Answer:
    def __init__(self, body, status=200):
        self.body, self.status_code = body, status

    def raise_for_status(self):
        if self.status_code != 200:
            raise httpx.HTTPStatusError("bad status", request=None, response=None)

    def json(self):
        return self.body


def test_the_server_version_is_read_from_the_daemon(monkeypatch) -> None:
    asked = []

    def get(url, timeout):
        asked.append(url)
        return _Answer({"version": "0.40.2"})
    monkeypatch.setattr(provenance.httpx, "get", get)
    assert _real_ollama_server_version("http://ollama:11434/") == "0.40.2"
    assert asked == ["http://ollama:11434/api/version"]


@pytest.mark.parametrize("answer", [
    httpx.ConnectError("refused"), _Answer({}), _Answer({"version": ""}), _Answer({}, status=500)])
def test_an_unreadable_server_version_is_null(monkeypatch, answer) -> None:
    def get(url, timeout):
        if isinstance(answer, Exception):
            raise answer
        return answer
    monkeypatch.setattr(provenance.httpx, "get", get)
    assert _real_ollama_server_version() is None


@pytest.mark.parametrize("host, env, url", [
    ("ollama:11434", None, "http://ollama:11434"),
    (None, "http://127.0.0.1:11434/", "http://127.0.0.1:11434"),
    (None, None, "http://localhost:11434"),
])
def test_the_daemon_url(monkeypatch, host, env, url) -> None:
    if env is None:
        monkeypatch.delenv("OLLAMA_HOST", raising=False)
    else:
        monkeypatch.setenv("OLLAMA_HOST", env)
    assert provenance.ollama_base_url(host) == url


# the environment snapshot
def test_the_environment_snapshot_records_what_the_run_depends_on() -> None:
    import hashlib
    import importlib.metadata
    import platform

    from backend.app.paths import MORPHOLEX_PATH

    snap = provenance.build_environment_snapshot([SeatSpec("ollama", "llama3.1:8b")])

    assert snap["python"] == platform.python_version()
    assert snap["packages"]["pydantic"] == importlib.metadata.version("pydantic")
    assert set(snap["packages"]) >= {"ollama", "openai", "nltk", "numpy", "sqlalchemy"}
    assert snap["ollama"]["version"] == config.EXPECTED_OLLAMA_VERSION  # conftest's stand-in
    validator = snap["clue_validator"]
    assert validator["morpholex_sha256"] == hashlib.sha256(MORPHOLEX_PATH.read_bytes()).hexdigest()
    assert validator["nltk"] == importlib.metadata.version("nltk")
    assert validator["wordnet_version"] == "3.0"
    assert re.fullmatch(r"[0-9a-f]{40}", snap["git"]["commit"])
    assert isinstance(snap["git"]["status"], list)


def test_a_run_without_a_local_seat_does_not_ask_the_daemon(monkeypatch) -> None:
    def never(host=None):  # pragma: no cover - must never run
        raise AssertionError("no local seat, no daemon")
    monkeypatch.setattr(provenance, "ollama_server_version", never)
    assert provenance.build_environment_snapshot([SeatSpec("openrouter", "m1")])["ollama"] is None


def test_the_environment_snapshot_never_raises(monkeypatch) -> None:
    """Record-only: without git, nvidia-smi or the MorphoLex file, it records nulls."""
    def no_command(*args, **kwargs):
        raise FileNotFoundError(args[0][0])
    monkeypatch.setattr(subprocess, "run", no_command)
    monkeypatch.setattr(provenance, "MORPHOLEX_PATH", provenance.Path("no/such/file.tsv"))

    snap = provenance.build_environment_snapshot([SeatSpec("ollama", "llama3.1:8b")])

    assert snap["git"] is None and snap["gpus"] is None
    assert snap["clue_validator"]["morpholex_sha256"] is None
