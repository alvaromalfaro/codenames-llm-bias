"""Word-list loading and per-word covariate annotation.

Loads the WEAT cores and the gender-disparity expansion sources, then annotates every pool word with 
the three confound-control covariates (subtlex_freq, length, wordnet_polysemy) plus its gender_category 
and source.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

# Gender specifications a board can belong to.
Specification = Literal["gender-career", "gender-science"]

# Gender labels. Binary axis is a documented simplification; gender only.
GenderCategory = Literal["male", "female", "neutral"]


@dataclass(frozen=True, slots=True)
class Covariates:
    """The three confound-control covariates for one word."""

    subtlex_freq: float  # SUBTLEX-US lexical frequency - controls lexical familiarity
    length: int  # word length - controls surface complexity
    wordnet_polysemy: int  # number of WordNet synonyms - controls semantic ambiguity


@dataclass(frozen=True, slots=True)
class Word:
    """A pool word with its gender label, provenance and covariates."""

    text: str
    gender_category: GenderCategory
    source: str  # extraction source, for traceability
    covariates: Covariates


def load_weat_core(spec: Specification, resources_dir: Path) -> list[Word]:
    """Load the original WEAT stimulus words for spec.

    gender-career -> WEAT-6; gender-science -> WEAT-7 + WEAT-8 combined.
    """
    raise NotImplementedError


def load_expansion(spec: Specification, resources_dir: Path) -> list[Word]:
    """Load the expansion words (Eurostat / She Figures, ...) for spec."""
    raise NotImplementedError


def annotate_covariates(words: list[Word], subtlex_dir: Path) -> list[Word]:
    """Annotate each word with the covariates. Does not change the covariate set."""
    raise NotImplementedError


def load_pool(spec: Specification, resources_dir: Path) -> list[Word]:
    """Load and annotate the full word pool for spec (core + expansion)."""
    raise NotImplementedError


def _subtlex_frequency(word: str, subtlex_dir: Path) -> float:
    """Lexical frequency from SUBTLEX-US."""
    raise NotImplementedError


def _wordnet_polysemy(word: str) -> int:
    """Number of WordNet synonyms for word."""
    raise NotImplementedError
