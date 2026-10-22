"""Run-provenance helpers: git code version, ollama model digest, the model-registry snapshot and
the environment snapshot.

These populate the ``run`` provenance columns (``code_version``, ``prompt_template_version``,
``model_registry_snapshot``, ``environment_snapshot``) so a persisted batch is reconstructible after
the fact. Everything here is RECORD-ONLY: a lookup that fails (git absent, ollama daemon
unreachable) must never abort or alter a game - it records a null value with an explanatory note and
logs a WARNING. The gates that do abort (the served digests, the Ollama version) live in game_runner
and read what these return.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import logging
import os
import platform
import subprocess
from pathlib import Path
from typing import Callable, Optional, Sequence

import httpx

from backend.app import config
from backend.app.core.llm.client_openrouter import provider_routing
from backend.app.paths import MORPHOLEX_PATH, REPO_ROOT

logger = logging.getLogger(__name__)

# The git cwd, so code_version is correct regardless of the process working directory.
_REPO_ROOT = REPO_ROOT


def git_code_version() -> Optional[str]:
    """Return the git HEAD short SHA, suffixed ``-dirty`` when the working tree has uncommitted
    changes (e.g. ``"a1b2c3d"`` vs ``"a1b2c3d-dirty"``). Returns ``None`` (and logs a WARNING) when
    git is unavailable or this is not a repo - provenance must never abort a run."""
    try:
        head = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        if not head:
            logger.warning("git_code_version: empty HEAD; recording None.")
            return None
        porcelain = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        return f"{head}-dirty" if porcelain.strip() else head
    except (OSError, subprocess.CalledProcessError) as e:
        logger.warning("git_code_version: git unavailable or not a repo (%s); recording None.", e)
        return None


def resolve_ollama_digest(model_name: str, host: Optional[str] = None) -> Optional[str]:
    """Resolve the ollama model digest pre-run by querying the local daemon - a STRONG identity that
    pins exactly which local weights served. Returns the digest string, or ``None`` on any failure
    (daemon unreachable, model not pulled, client error): record-only, never raises.

    The digest is exposed by the daemon's ``list`` endpoint (``ShowResponse`` carries no digest), so
    we list local models and match by name, tolerating the implicit ``:latest`` tag.
    """
    try:
        import httpx
        from ollama import Client

        # Bounded as the game's own calls are: without a limit a hung daemon would stop the batch
        # before its first game. A lookup that times out records null, which the digest gate
        # refuses.
        client = Client(
            host=host or os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
            timeout=httpx.Timeout(config.OLLAMA_TIMEOUT_S, connect=config.OLLAMA_CONNECT_TIMEOUT_S),
        )
        for model in client.list().models:
            name = getattr(model, "model", None)
            if (
                name == model_name
                or name == f"{model_name}:latest"
                or (":" not in model_name and name and name.split(":", 1)[0] == model_name)
            ):
                digest = getattr(model, "digest", None)
                if digest:
                    return digest
        logger.warning("resolve_ollama_digest: model %r not found in ollama list.", model_name)
        return None
    # record-only: any daemon/client failure degrades to a null identity.
    except Exception as e:
        logger.warning(
            "resolve_ollama_digest: lookup failed for %r (%s); recording null.", model_name, e
        )
        return None


# Notes attached to each snapshot entry, spelling out the strong/best-effort asymmetry in the data.
_NOTE_OLLAMA = (
    "Strong identity: model digest resolved from the ollama daemon pre-run, pinning the "
    "exact local weights."
)
_NOTE_OLLAMA_UNAVAILABLE = (
    "Best-effort: ollama digest lookup failed (daemon unreachable or model "
    "not pulled); only the requested model name is known."
)
_NOTE_OPENROUTER = (
    "Best-effort: API providers expose no pre-run resolvable identity. The requested model is "
    "pinned to the provider (and quantization) config names, if any ('routing'). Effective "
    "identity is captured per call in llm_call.resolved_model / system_fingerprint, and the "
    "provider that served the call in llm_call.raw_payload, which surfaces a provider silently "
    "swapping the model mid-experiment."
)


def _routed_identity(model_name: str, routing: dict) -> str:
    """What serves an API seat: its model, at the provider it is pinned to, and the quantization if
    one is pinned (e.g. ``mistralai/mistral-small-3.2-24b-instruct@DeepInfra/fp8``)."""
    identity = f"{model_name}@{routing['order'][0]}"
    quantizations = routing.get("quantizations")
    return f"{identity}/{quantizations[0]}" if quantizations else identity


def build_model_registry_snapshot(
    seat_specs: Sequence,
    *,
    digest_resolver: Optional[Callable[[str], Optional[str]]] = None,
) -> dict:
    """Build the ``model_registry_snapshot`` JSONB value: an asymmetric, honest record of which
    model served each seat, keyed by seat index (string keys, since JSON).

    Provider identity is not symmetrically resolvable before a batch:
    - ``ollama`` (local): the digest is queryable from the daemon ->
      ``identity_kind="ollama_digest"`` with the real digest (or ``"ollama_digest_unavailable"`` +
      null identity if the lookup failed).
    - ``openrouter`` (API): no pre-run identity exists. The model pinned to a provider by config
      (``routing``, see ``client_openrouter.provider_routing``) -> ``identity_kind=
      "requested_model_and_route"`` with ``model@provider[/quantization]``; one with no provider
      pinned -> ``"requested_model_string"`` with the requested model string, which the batch's
      gate refuses. Effective identity lives per call in ``llm_call``.

    The strong/best-effort distinction is explicit in ``identity_kind`` (data), not just prose.
    ``digest_resolver`` is injectable so tests can supply a deterministic digest without a daemon;
    when omitted it resolves the module-level ``resolve_ollama_digest`` at call time (so a
    monkeypatch of that name is honoured). Record-only: never raises on a lookup failure.
    """
    resolver = digest_resolver if digest_resolver is not None else resolve_ollama_digest
    snapshot: dict[str, dict] = {}
    for index, spec in enumerate(seat_specs):
        provider = spec.provider
        model_name = spec.model_name
        if provider == "openrouter":
            routing = provider_routing(model_name)
            entry = {
                "provider": provider,
                "model_name": model_name,
                "identity_kind": (
                    "requested_model_string" if routing is None else "requested_model_and_route"
                ),
                "identity": model_name
                if routing is None
                else _routed_identity(model_name, routing),
                "routing": routing,
                "note": _NOTE_OPENROUTER,
            }
        else:  # ollama / any local provider
            try:
                digest = resolver(model_name)
            # record-only: a raising resolver degrades to a null identity.
            except Exception as e:
                logger.warning(
                    "build_model_registry_snapshot: digest resolver raised for %r (%s); "
                    "recording null.",
                    model_name,
                    e,
                )
                digest = None
            if digest:
                entry = {
                    "provider": provider,
                    "model_name": model_name,
                    "identity_kind": "ollama_digest",
                    "identity": digest,
                    "note": _NOTE_OLLAMA,
                }
            else:
                entry = {
                    "provider": provider,
                    "model_name": model_name,
                    "identity_kind": "ollama_digest_unavailable",
                    "identity": None,
                    "note": _NOTE_OLLAMA_UNAVAILABLE,
                }
            # The context window and the answer cap every local request is sent with: ollama
            # would otherwise pick the context from the GPU's memory, so it is part of what ran.
            entry["options"] = {
                "num_ctx": config.OLLAMA_NUM_CTX,
                "num_predict": config.LLM_MAX_OUTPUT_TOKENS,
            }
        snapshot[str(index)] = entry
    return snapshot


def ollama_base_url(host: Optional[str] = None) -> str:
    """The Ollama daemon's base URL: ``host``, else ``OLLAMA_HOST``, else the default, with a
    scheme."""
    host = (host or os.environ.get("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
    return host if "://" in host else f"http://{host}"


def ollama_server_version(host: Optional[str] = None) -> Optional[str]:
    """The version the Ollama daemon reports (``/api/version``), or None when it cannot be read
    (daemon unreachable, unexpected answer). Bounded like a health check. Record-only: never
    raises."""
    url = f"{ollama_base_url(host)}/api/version"
    try:
        response = httpx.get(
            url,
            timeout=httpx.Timeout(
                config.OLLAMA_HEALTH_TIMEOUT_S, connect=config.OLLAMA_CONNECT_TIMEOUT_S
            ),
        )
        response.raise_for_status()
        version = response.json().get("version")
        return version if isinstance(version, str) and version else None
    except Exception as e:
        logger.warning("ollama_server_version: %s unreadable (%s); recording null.", url, e)
        return None


# The packages whose versions the environment snapshot records: what builds, sends and parses the
# requests, what the clue validator reads, and the database layer. uv.lock pins them all; the
# snapshot says what was actually installed.
_RECORDED_PACKAGES = (
    "ollama",
    "openai",
    "httpx",
    "pydantic",
    "nltk",
    "numpy",
    "sqlalchemy",
    "psycopg2-binary",
    "pgvector",
    "alembic",
)
# The NLTK corpora the clue validator reads, by the resource name nltk.data.find takes.
_NLTK_CORPORA = {"wordnet": "corpora/wordnet.zip", "omw-1.4": "corpora/omw-1.4.zip"}


def _file_sha256(path: Path) -> Optional[str]:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def _package_version(name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _gpus() -> Optional[list[str]]:
    """The NVIDIA GPUs ``nvidia-smi`` sees from this process (name, driver, memory), or None without
    it. The GPU that serves the models is the daemon's; this is the machine's view of it."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return [line.strip() for line in out.splitlines() if line.strip()] or None


def _validator_data() -> dict:
    """What the clue validator decides with: the MorphoLex segmentations it reads (by hash), and the
    NLTK release and WordNet corpora it lemmatises and looks words up with."""
    data: dict = {
        "morpholex_sha256": _file_sha256(MORPHOLEX_PATH),
        "nltk": _package_version("nltk"),
        "wordnet_version": None,
        "corpora_sha256": {},
    }
    try:
        import nltk
        from nltk.corpus import wordnet

        data["wordnet_version"] = wordnet.get_version()
        for name, resource in _NLTK_CORPORA.items():
            try:
                found = Path(str(nltk.data.find(resource)))
            except LookupError:
                found = None
            data["corpora_sha256"][name] = (
                _file_sha256(found) if found and found.is_file() else None
            )
    except Exception as e:
        logger.warning("environment snapshot: the WordNet data is unreadable (%s)", e)
    return data


def _git_state() -> Optional[dict]:
    """The full commit, the files that differ from it (untracked ones too, which the ``-dirty`` of
    code_version does not count), and the SHA-256 of the diff of the tracked ones, or None without
    git."""

    def git(*args: str) -> bytes:
        return subprocess.run(
            ["git", *args], cwd=_REPO_ROOT, capture_output=True, check=True, timeout=30
        ).stdout

    try:
        status = git("status", "--porcelain").decode("utf-8", "replace").splitlines()
        diff = git("diff", "HEAD", "--binary")
        return {
            "commit": git("rev-parse", "HEAD").decode().strip(),
            "status": status,
            "diff_sha256": hashlib.sha256(diff).hexdigest() if diff else None,
        }
    except (OSError, subprocess.SubprocessError) as e:
        logger.warning("environment snapshot: git unavailable (%s); recording null.", e)
        return None


def build_environment_snapshot(seat_specs: Sequence) -> dict:
    """Build the ``environment_snapshot`` JSONB value: the environment a run is minted in, beyond
    the code version and the models - the interpreter and the machine, the packages installed, the
    Ollama server when a seat is local, the GPU, the clue validator's data and the state of the
    working tree. Record-only: what cannot be read is recorded as null."""
    local = any(getattr(spec, "provider", None) == "ollama" for spec in seat_specs)
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": {name: _package_version(name) for name in _RECORDED_PACKAGES},
        "ollama": (
            {"host": ollama_base_url(), "version": ollama_server_version()} if local else None
        ),
        "gpus": _gpus(),
        "clue_validator": _validator_data(),
        "git": _git_state(),
    }
