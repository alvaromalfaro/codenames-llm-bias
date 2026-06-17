"""Covariate balancing of contrastable subsets.

Propensity score matching over the three covariates, followed by an equivalence check per 
contrastable pair (career-family and science-arts).

Default criterion: for each covariate the Mann-Whitney test is non-significant and Cohen's d < 0.2. 
Because "failing to reject H₀" is not evidence of equivalence, a TOST mode with an a-priori 
equivalence margin and per-covariate SMD reporting is also exposed. The criterion is configurable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from board_generator.lexicon import Word

# Equivalence criterion.
BalanceCriterion = Literal["mann_whitney_cohen", "tost"]

# The pairs that are balanced against each other.
ContrastablePair = Literal["career-family", "science-arts"]


@dataclass(frozen=True, slots=True)
class CovariateBalance:
    """Per-covariate equivalence result."""

    covariate: str
    mann_whitney_p: float
    cohen_d: float
    smd: float  # standardized mean difference
    tost_p: float | None  # populated only under the TOST criterion
    passed: bool


@dataclass(frozen=True, slots=True)
class PairBalance:
    """Balance verdict for one contrastable pair across all covariates."""

    pair: ContrastablePair
    covariates: list[CovariateBalance]
    passed: bool


@dataclass(frozen=True, slots=True)
class BalanceReport:
    """Bank-level balance_report.json payload."""

    pairs: list[PairBalance]
    criterion: BalanceCriterion
    seed: int


def propensity_score_match(
    group_a: list[Word], group_b: list[Word]
) -> tuple[list[Word], list[Word]]:
    """Match the two groups on (subtlex_freq, length, wordnet_polysemy) via PSM."""
    raise NotImplementedError


def check_balance(
    group_a: list[Word],
    group_b: list[Word],
    pair: ContrastablePair,
    criterion: BalanceCriterion = "mann_whitney_cohen",
    equivalence_margin: float = 0.2,
) -> PairBalance:
    """Assess covariate balance under criterion."""
    raise NotImplementedError


def _cohens_d(a: list[float], b: list[float]) -> float:
    """Cohen's d effect size between two covariate samples."""
    raise NotImplementedError


def _standardized_mean_difference(a: list[float], b: list[float]) -> float:
    """Standardized mean difference reported per covariate."""
    raise NotImplementedError
