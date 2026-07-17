"""Run-provenance helpers: git code version, ollama model digest, and the model-registry snapshot.

These populate the three ``run`` provenance columns (``code_version``, ``prompt_template_version``,
``model_registry_snapshot``) so a persisted batch is reconstructible after the fact. Everything here
is RECORD-ONLY: a lookup that fails (git absent, ollama daemon unreachable) must never abort or alter
a game - it records a null value with an explanatory note and logs a WARNING.
"""
from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import Callable, Optional, Sequence

logger = logging.getLogger(__name__)

# Repo root, derived from this file (backend/app/core/provenance.py -> repo root is 4 parents up).
# Used as the git cwd so code_version is correct regardless of the process working directory.
_REPO_ROOT = Path(__file__).resolve().parents[3]


def git_code_version() -> Optional[str]:
    """Return the git HEAD short SHA, suffixed ``-dirty`` when the working tree has uncommitted
    changes (e.g. ``"a1b2c3d"`` vs ``"a1b2c3d-dirty"``). Returns ``None`` (and logs a WARNING) when
    git is unavailable or this is not a repo - provenance must never abort a run."""
    try:
        head = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=_REPO_ROOT, capture_output=True, text=True, check=True,
        ).stdout.strip()
        if not head:
            logger.warning("git_code_version: empty HEAD; recording None.")
            return None
        porcelain = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=_REPO_ROOT, capture_output=True, text=True, check=True,
        ).stdout
        return f"{head}-dirty" if porcelain.strip() else head
    except (OSError, subprocess.CalledProcessError) as e:
        logger.warning(
            "git_code_version: git unavailable or not a repo (%s); recording None.", e)
        return None


def resolve_ollama_digest(model_name: str, host: Optional[str] = None) -> Optional[str]:
    """Resolve the ollama model digest pre-run by querying the local daemon - a STRONG identity that
    pins exactly which local weights served. Returns the digest string, or ``None`` on any failure
    (daemon unreachable, model not pulled, client error): record-only, never raises.

    The digest is exposed by the daemon's ``list`` endpoint (``ShowResponse`` carries no digest), so
    we list local models and match by name, tolerating the implicit ``:latest`` tag.
    """
    try:
        from ollama import Client

        client = Client(host=host or os.environ.get(
            "OLLAMA_HOST", "http://localhost:11434"))
        for model in client.list().models:
            name = getattr(model, "model", None)
            if name == model_name or name == f"{model_name}:latest" or (
                ":" not in model_name and name and name.split(":", 1)[
                    0] == model_name
            ):
                digest = getattr(model, "digest", None)
                if digest:
                    return digest
        logger.warning(
            "resolve_ollama_digest: model %r not found in ollama list.", model_name)
        return None
    # record-only: any daemon/client failure degrades to a null identity.
    except Exception as e:
        logger.warning("resolve_ollama_digest: lookup failed for %r (%s); recording null.",
                       model_name, e)
        return None


# Notes attached to each snapshot entry, spelling out the strong/best-effort asymmetry in the data.
_NOTE_OLLAMA = ("Strong identity: model digest resolved from the ollama daemon pre-run, pinning the "
                "exact local weights.")
_NOTE_OLLAMA_UNAVAILABLE = ("Best-effort: ollama digest lookup failed (daemon unreachable or model "
                            "not pulled); only the requested model name is known.")
_NOTE_OPENROUTER = ("Best-effort: API providers expose no pre-run resolvable identity. Effective "
                    "identity is captured per call in llm_call.resolved_model / system_fingerprint, "
                    "which surfaces a provider silently swapping the model mid-experiment.")


def build_model_registry_snapshot(
    seat_specs: Sequence,
    *,
    digest_resolver: Optional[Callable[[str], Optional[str]]] = None,
) -> dict:
    """Build the ``model_registry_snapshot`` JSONB value: an asymmetric, honest record of which model
    served each seat, keyed by seat index (string keys, since JSON).

    Provider identity is not symmetrically resolvable before a batch:
    - ``ollama`` (local): the digest is queryable from the daemon -> ``identity_kind="ollama_digest"``
      with the real digest (or ``"ollama_digest_unavailable"`` + null identity if the lookup failed).
    - ``openrouter`` (API): no pre-run identity exists -> ``identity_kind="requested_model_string"``
      with the requested model string; effective identity lives per call in ``llm_call``.

    The strong/best-effort distinction is explicit in ``identity_kind`` (data), not just prose.
    ``digest_resolver`` is injectable so tests can supply a deterministic digest without a daemon;
    when omitted it resolves the module-level ``resolve_ollama_digest`` at call time (so a monkeypatch
    of that name is honoured). Record-only: never raises on a lookup failure.
    """
    resolver = digest_resolver if digest_resolver is not None else resolve_ollama_digest
    snapshot: dict[str, dict] = {}
    for index, spec in enumerate(seat_specs):
        provider = spec.provider
        model_name = spec.model_name
        if provider == "openrouter":
            entry = {
                "provider": provider, "model_name": model_name,
                "identity_kind": "requested_model_string", "identity": model_name,
                "note": _NOTE_OPENROUTER,
            }
        else:  # ollama / any local provider
            try:
                digest = resolver(model_name)
            # record-only: a raising resolver degrades to a null identity.
            except Exception as e:
                logger.warning("build_model_registry_snapshot: digest resolver raised for %r (%s); "
                               "recording null.", model_name, e)
                digest = None
            if digest:
                entry = {
                    "provider": provider, "model_name": model_name,
                    "identity_kind": "ollama_digest", "identity": digest, "note": _NOTE_OLLAMA,
                }
            else:
                entry = {
                    "provider": provider, "model_name": model_name,
                    "identity_kind": "ollama_digest_unavailable", "identity": None,
                    "note": _NOTE_OLLAMA_UNAVAILABLE,
                }
        snapshot[str(index)] = entry
    return snapshot
