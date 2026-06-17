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

No default consensus is baked in (DEFAULT_CONSENSUS is None): the caller must supply a pinned 
ConsensusSpec.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass

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

    Validated on construction (no network): rejects an empty set, any missing HF revision, or a 
    primary outside the set; warns on apparent same-lineage siblings (shared training data -> false 
    diversity).
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
        _warn_if_same_lineage(self.encoders)


@dataclass
class Arbiter:
    """A loaded, pinned sentence-transformers encoder. Bare-word, prefix-free."""

    ref: ArbiterRef

    def embed(self, word: str) -> NDArray[np.float64]:
        """Embed a single bare word - the one embedding primitive used everywhere."""
        raise NotImplementedError

    def cos(self, a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
        """Cosine similarity between two embeddings."""
        raise NotImplementedError


def load_consensus(spec: ConsensusSpec) -> list[Arbiter]:
    """Load each consensus encoder in-process via sentence-transformers, pinned by HF rev."""
    raise NotImplementedError


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
