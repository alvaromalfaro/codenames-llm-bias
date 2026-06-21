"""External arbiter encoders - construction consensus and primary φ*.

Bare-word, prefix-free symmetric similarity: a single embed(word) primitive is used identically 
everywhere - candidate ranking, the consensus gate, and the downstream metrics. There are no 
prefix/carrier parameters; the consensus pool is restricted to prefix-free symmetric encoders 
(models that require query:/passage: prefixes, e.g. BGE or E5, are excluded).

Selection rules enforced here:
  * every arbiter is external to the evaluated models;
  * every arbiter is pinned by a Hugging Face revision;
  * the primary φ* belongs to the consensus set;
  * consensus encoders should come from distinct lineages (warned heuristically; no network calls).
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

# Models under evaluation - NONE may serve as an arbiter.
EVALUATED_MODELS: frozenset[str] = frozenset(
    {"llama-3.1", "mistral-small-3.2", "grok-4.3", "gemini-3-pro"}
)

# No baked-in default consensus: the caller MUST pass a pinned ConsensusSpec.
DEFAULT_CONSENSUS: ConsensusSpec | None = None


@dataclass(frozen=True, slots=True)
class ArbiterRef:
    """A pinned encoder reference: model id + Hugging Face revision."""

    model_id: str
    hf_revision: str

    def __str__(self) -> str:
        # The "model@rev" form recorded in board files.
        return f"{self.model_id}@{self.hf_revision}"


@dataclass(frozen=True, slots=True)
class ConsensusSpec:
    """The consensus encoder set with its designated primary φ*.

    Validated on construction: rejects an empty set, any missing HF revision, a primary outside the 
    set, or any evaluated model; warns on apparent same-lineage siblings (shared training data -> 
    false diversity).
    """

    encoders: tuple[ArbiterRef, ...]
    primary: ArbiterRef

    def __post_init__(self) -> None:
        if not self.encoders:
            raise ValueError(
                "consensus set is empty (requires N = 2-3 encoders)")
        for ref in self.encoders:
            if not ref.hf_revision:
                raise ValueError(
                    f"arbiter {ref.model_id!r} is not pinned by an HF revision (reproducibility)"
                )
        if self.primary not in self.encoders:
            raise ValueError(
                "primary φ* must belong to the consensus set")
        assert_external(self)
        _warn_if_same_lineage(self.encoders)


class Encoder(Protocol):
    """A bare-text -> vector encoder. The single seam for dependency injection.

    The real backend is SentenceTransformerEncoder; tests inject deterministic, network-free
    stubs implementing this same protocol.
    """

    def encode(self, text: str) -> NDArray[np.float64]: ...


@dataclass
class Arbiter:
    """A pinned encoder wrapped behind the bare-word embedding convention.

    The encoder is mandatory and explicit (no default factory): production wraps a
    SentenceTransformerEncoder via load_consensus; tests inject stubs.
    """

    ref: ArbiterRef
    encoder: Encoder

    def embed(self, word: str) -> NDArray[np.float64]:
        """Embed a single bare word - the one embedding primitive used everywhere.

        Lowercases the word and delegates to the encoder, returning the RAW model vector upcast to
        float64. No normalization here: cos normalizes internally, and the future e_gen step needs
        the raw geometry.
        """
        vec = self.encoder.encode(word.lower())
        return np.asarray(vec, dtype=np.float64)

    def cos(self, a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
        """Full cosine similarity in float64. Total: returns 0.0 if either vector has zero norm."""
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        norm_a = float(np.linalg.norm(a))
        norm_b = float(np.linalg.norm(b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))


class SentenceTransformerEncoder:
    """The ONLY encoder that touches sentence-transformers (and thus torch/transformers/network).

    sentence_transformers is imported lazily inside __init__ so that importing this module pulls no
    heavy dependencies and touches no network. Words are embedded bare and unnormalized, on CPU in
    eval mode, for deterministic, prefix-free, reproducible cosines.
    """

    def __init__(self, ref: ArbiterRef) -> None:
        from sentence_transformers import SentenceTransformer

        self.ref = ref
        self._model = SentenceTransformer(
            ref.model_id, revision=ref.hf_revision, device="cpu"
        )
        self._model.eval()

    def encode(self, text: str) -> NDArray[np.float64]:
        vec = self._model.encode(
            text, convert_to_numpy=True, normalize_embeddings=False
        )
        return np.asarray(vec, dtype=np.float64)


def load_consensus(spec: ConsensusSpec) -> list[Arbiter]:
    """Load each consensus encoder in-process via sentence-transformers, pinned by HF rev.

    The ONLY network-touching function and the ONLY place a SentenceTransformerEncoder is built.
    """
    return [Arbiter(ref=ref, encoder=SentenceTransformerEncoder(ref)) for ref in spec.encoders]


def assert_external(spec: ConsensusSpec) -> None:
    """Reject any arbiter that is an evaluated model.

    Using an evaluated model's embeddings to validate a dilemma it later plays is circular. This 
    guards against model identity, not the serving mechanism.
    """
    for ref in spec.encoders:
        model = ref.model_id.lower()
        if any(evaluated in model for evaluated in EVALUATED_MODELS):
            raise ValueError(
                f"arbiter {ref.model_id!r} is an evaluated model; using it to validate dilemmas it "
                "later plays is circular"
            )


def _lineage_key(model_id: str) -> str:
    """Heuristic lineage key - the model family token (e.g. 'all', 'gte', 'e5')."""
    name = model_id.split("/")[-1]
    return name.split("-")[0].lower()


def _warn_if_same_lineage(encoders: Sequence[ArbiterRef]) -> None:
    """Warn (no error) if two encoders look like same-lineage siblings."""
    seen: dict[str, str] = {}
    for ref in encoders:
        key = _lineage_key(ref.model_id)
        if key in seen:
            warnings.warn(
                f"arbiters {seen[key]!r} and {ref.model_id!r} look like same-lineage siblings "
                "(shared training data -> false diversity); consider replacing one with distinct "
                "lineage",
                stacklevel=3,
            )
        seen[key] = ref.model_id
