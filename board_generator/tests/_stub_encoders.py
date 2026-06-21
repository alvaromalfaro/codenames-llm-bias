"""Deterministic, network-free Encoder stubs for the offline arbiter tests.

These implement the same Encoder protocol as the real SentenceTransformerEncoder but never touch
sentence-transformers, torch, or the network, so the default test suite runs fully offline.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping

import numpy as np
from numpy.typing import NDArray


class HashEncoder:
    """Maps each text to a fixed-dim vector seeded from a stable hash of the text.

    Deterministic per text (same text -> identical vector), so it exercises determinism,
    idempotence, symmetry, and self-similarity without any model.
    """

    def __init__(self, dim: int = 16) -> None:
        self.dim = dim

    def encode(self, text: str) -> NDArray[np.float64]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], "big")
        rng = np.random.default_rng(seed)
        return rng.standard_normal(self.dim).astype(np.float64)


class ScriptedEncoder:
    """Maps specific texts to caller-provided vectors for exact hand-engineered geometry.

    Unknown texts raise KeyError so tests fail loudly rather than silently embedding noise.
    """

    def __init__(self, vectors: Mapping[str, NDArray[np.float64]]) -> None:
        self._vectors = {text: np.asarray(vec, dtype=np.float64) for text, vec in vectors.items()}

    def encode(self, text: str) -> NDArray[np.float64]:
        return self._vectors[text]
