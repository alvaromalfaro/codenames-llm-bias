"""Covariate balancing of contrastable subsets.

Two independent balancing exercises, one per gender specification, are derived from Word.weat_set: 
gender-career = words tagged weat-6; gender-science = words tagged weat-7 or weat-8. Neutral words 
(empty weat_set) are used in both boards contextually and are not balanced here. Within an exercise 
the binary PSM treatment is pole membership - 1 iff gender_category == "male" (career: 
Career=1/Family=0; science: STEM=1/Arts=0).

Pipeline per specification: low-anchor OOV imputation -> propensity-score matching -> per-covariate
equivalence check. The matching is deliberately greedy (1:1 nearest-neighbour on the propensity logit, 
no replacement, within a caliper), not optimal: greedy is deterministic and reproducible, which 
matters more here than squeezing out the last pair.

Equivalence is reported under both criteria: the SPEC-original criterion (Mann-Whitney 
non-significant and |d| < COHEN_D_THRESHOLD) and the a-priori TOST criterion (tost_p < alpha with a 
±TOST_MARGIN_SMD SMD-unit margin). The governing criterion is configurable and defaults to TOST. 
Nothing is tuned to pass: with small n TOST often fails to establish equivalence, and that is a 
valid, honest result that the report surfaces as-is.

All reported statistics are JSON-safe: any non-finite or undefined statistic is sanitized to None 
(never NaN/Inf), and an undefined equivalence statistic is treated as non-equivalent (conservative).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.weightstats import ttost_ind

from board_generator.lexicon import COVARIATE_KEYS, Specification, Word

# --- The three DISTINCT 0.2 thresholds. ---
# SPEC-original criterion: an effect this small on the mean difference is "negligible" (Cohen).
COHEN_D_THRESHOLD = 0.2
# A-priori TOST equivalence margin, expressed in SMD (standardized mean difference) units.
TOST_MARGIN_SMD = 0.2
# PSM caliper width = this fraction of the SD of the propensity logit (Austin 2011).
PSM_CALIPER_LOGIT_SD = 0.2

# Default significance level for Mann-Whitney, the |d| gate and TOST.
DEFAULT_ALPHA = 0.05
# Below this many matched pairs the balance still runs, but the report carries a warning.
DEFAULT_MIN_PAIRS = 8

# Equivalence criterion. Default is TOST.
BalanceCriterion = Literal["mann_whitney_cohen", "tost"]

# Human-readable label for the contrastable poles of each specification.
ContrastablePair = Literal["career-family", "science-arts"]

# The treatment pole (PSM treatment = 1) in every exercise.
TREATMENT_POLE = "male"

_SPEC_PAIR: dict[Specification, ContrastablePair] = {
    "gender-career": "career-family",
    "gender-science": "science-arts",
}


@dataclass(frozen=True, slots=True)
class CovariateBalance:
    """Per-covariate equivalence result on the matched sample.

    Statistic fields are None when undefined (e.g. a zero pooled SD, or too few points to estimate 
    variance) - never NaN/Inf. An undefined statistic counts as non-equivalent.
    """

    covariate: str
    smd: float | None  # standardized mean difference - the headline; None when undefined
    mann_whitney_p: float | None
    cohen_d: float | None
    tost_p: float | None
    tost_equivalent: bool  # TOST verdict (tost_p < alpha)
    spec_original_equivalent: bool  # MW non-significant AND |d| < COHEN_D_THRESHOLD
    passed: bool  # verdict under the governing criterion


@dataclass(frozen=True, slots=True)
class MatchCounts:
    """Matching tallies.

    The two dropped_* counts measure different things and do not sum to "total unmatched majority 
    words": dropped_by_group_excess is the structural surplus of the larger pole (|n_treat - n_ctrl|, 
    never matchable 1:1), while dropped_by_caliper counts minority-pole units with no opposite-pole 
    partner inside the caliper. pairs_kept is the usable count.
    """

    pairs_kept: int
    dropped_by_caliper: int
    dropped_by_group_excess: int
    imputed_count: int


@dataclass(frozen=True, slots=True)
class SpecificationBalance:
    """Balance verdict for one specification across all covariates."""

    specification: Specification
    contrastable_pair: ContrastablePair
    treatment_pole: str
    covariates: list[CovariateBalance]
    counts: MatchCounts
    # texts whose subtlex_freq was OOV and low-anchor imputed
    imputed_words: list[str]
    # set when pairs_kept < min_pairs (balance ran on little data)
    warning: str | None
    passed: bool  # all covariates pass under the governing criterion


@dataclass(frozen=True, slots=True)
class BalanceReport:
    """Bank-level balance_report.json payload. Pure data - JSON-serializable."""

    specifications: list[SpecificationBalance]
    criterion: BalanceCriterion
    seed: int
    alpha: float
    tost_margin: float
    caliper_sd: float


@dataclass(frozen=True, slots=True)
class MatchedSubset:
    """The matched, balanced Word subsets for one specification (for board composition)."""

    specification: Specification
    treatment: list[Word]
    control: list[Word]


@dataclass(frozen=True, slots=True)
class BalanceResult:
    """The serializable :class:BalanceReport plus the matched Word subsets (not JSON)."""

    report: BalanceReport
    matched: list[MatchedSubset]


def run_balancing(words: list[Word], seed: int, *, criterion: BalanceCriterion = "tost",
                  alpha: float = DEFAULT_ALPHA, tost_margin: float = TOST_MARGIN_SMD,
                  caliper_sd: float = PSM_CALIPER_LOGIT_SD, min_pairs: int = DEFAULT_MIN_PAIRS) -> BalanceResult:
    """Balance both specifications independently and assemble the report."""
    specs: list[SpecificationBalance] = []
    matched: list[MatchedSubset] = []

    for specification in ("gender-career", "gender-science"):
        pool = _select_specification_pool(words, specification)
        treatment = [w for w in pool if w.gender_category == TREATMENT_POLE]
        control = [w for w in pool if w.gender_category != TREATMENT_POLE]
        imputed_words = [
            w.text for w in pool if w.covariates["subtlex_freq"] is None]

        matched_treat, matched_ctrl, dropped_caliper, dropped_excess = propensity_score_match(
            treatment, control, caliper_sd=caliper_sd, seed=seed
        )
        covariates = check_balance(
            matched_treat,
            matched_ctrl,
            criterion=criterion,
            alpha=alpha,
            tost_margin=tost_margin,
        )
        counts = MatchCounts(
            pairs_kept=len(matched_treat),
            dropped_by_caliper=dropped_caliper,
            dropped_by_group_excess=dropped_excess,
            imputed_count=len(imputed_words),
        )
        warning = (
            None
            if len(matched_treat) >= min_pairs
            else (
                f"matched sample below min_pairs ({len(matched_treat)} < {min_pairs}); "
                "balance ran on little data"
            )
        )
        specs.append(
            SpecificationBalance(
                specification=specification,
                contrastable_pair=_SPEC_PAIR[specification],
                treatment_pole=TREATMENT_POLE,
                covariates=covariates,
                counts=counts,
                imputed_words=imputed_words,
                warning=warning,
                passed=bool(covariates) and all(c.passed for c in covariates),
            )
        )
        matched.append(
            MatchedSubset(
                specification=specification, treatment=matched_treat, control=matched_ctrl
            )
        )

    report = BalanceReport(
        specifications=specs,
        criterion=criterion,
        seed=seed,
        alpha=alpha,
        tost_margin=tost_margin,
        caliper_sd=caliper_sd,
    )
    return BalanceResult(report=report, matched=matched)


def propensity_score_match(treatment: list[Word], control: list[Word], *, caliper_sd: float = PSM_CALIPER_LOGIT_SD,
                           seed: int = 0) -> tuple[list[Word], list[Word], int, int]:
    """Greedy 1:1 caliper PSM on the propensity logit.

    Standardizes the three covariates, fits an L2 logistic propensity model (tolerant of separation 
    on tiny n), and greedily matches each minority-pole unit (in index order, ties broken by index) 
    to its nearest still-available opposite-pole unit whose logit is within caliper_sd x SD(logit). 
    Matching is without replacement.

    Returns (matched_treatment, matched_control, dropped_by_caliper, dropped_by_group_excess).
    """
    union = list(treatment) + list(control)
    n_treat, n_ctrl = len(treatment), len(control)
    dropped_by_group_excess = abs(n_treat - n_ctrl)
    if not treatment or not control:
        return [], [], 0, dropped_by_group_excess

    floor = _freq_floor(union)
    features = _covariate_rows(union, floor)
    is_treat = np.array([True] * n_treat + [False] * n_ctrl)

    standardized = StandardScaler().fit_transform(features)
    model = LogisticRegression(random_state=seed & 0xFFFFFFFF)
    model.fit(standardized, is_treat.astype(int))
    prob = np.clip(model.predict_proba(standardized)[:, 1], 1e-12, 1.0 - 1e-12)
    logit = np.log(prob / (1.0 - prob))
    # population SD (ddof=0) - deterministic
    caliper = caliper_sd * float(np.std(logit))

    treat_idx = list(range(n_treat))
    ctrl_idx = list(range(n_treat, n_treat + n_ctrl))
    minority, majority = (treat_idx, ctrl_idx) if n_treat <= n_ctrl else (
        ctrl_idx, treat_idx)

    available = set(majority)
    pairs: list[tuple[int, int]] = []
    for unit in minority:  # ascending index order (deterministic)
        best: int | None = None
        best_dist: float | None = None
        for candidate in sorted(available):  # ties -> lowest index wins
            dist = abs(float(logit[unit]) - float(logit[candidate]))
            if dist <= caliper and (best_dist is None or dist < best_dist):
                best, best_dist = candidate, dist
        if best is not None:
            available.discard(best)
            pairs.append((unit, best))

    matched_treat: list[Word] = []
    matched_ctrl: list[Word] = []
    for left, right in pairs:
        treat_i, ctrl_i = (left, right) if is_treat[left] else (right, left)
        matched_treat.append(union[treat_i])
        matched_ctrl.append(union[ctrl_i])

    dropped_by_caliper = len(minority) - len(pairs)
    return matched_treat, matched_ctrl, dropped_by_caliper, dropped_by_group_excess


def check_balance(treatment: list[Word], control: list[Word], *, criterion: BalanceCriterion = "tost",
                  alpha: float = DEFAULT_ALPHA, tost_margin: float = TOST_MARGIN_SMD) -> list[CovariateBalance]:
    """Per-covariate equivalence check on an already-matched sample."""
    floor = _freq_floor(list(treatment) + list(control))
    results: list[CovariateBalance] = []
    for key in COVARIATE_KEYS:
        treat_vals = _covariate_values(treatment, key, floor)
        ctrl_vals = _covariate_values(control, key, floor)
        results.append(
            _covariate_balance(key, treat_vals, ctrl_vals,
                               criterion, alpha, tost_margin)
        )
    return results


def _covariate_balance(covariate: str, treat: list[float], ctrl: list[float], criterion: BalanceCriterion,
                       alpha: float, tost_margin: float) -> CovariateBalance:
    """Compute the per-covariate statistics and the two equivalence verdicts."""
    smd = _standardized_mean_difference(treat, ctrl)
    cohen_d = _cohens_d(treat, ctrl)
    mann_whitney_p = _mann_whitney_p(treat, ctrl)
    tost_p = _tost_p(treat, ctrl, tost_margin)

    tost_equivalent = tost_p is not None and tost_p < alpha
    spec_original_equivalent = (
        mann_whitney_p is not None
        and mann_whitney_p >= alpha
        and cohen_d is not None
        and abs(cohen_d) < COHEN_D_THRESHOLD
    )
    passed = tost_equivalent if criterion == "tost" else spec_original_equivalent
    return CovariateBalance(
        covariate=covariate,
        smd=smd,
        mann_whitney_p=mann_whitney_p,
        cohen_d=cohen_d,
        tost_p=tost_p,
        tost_equivalent=tost_equivalent,
        spec_original_equivalent=spec_original_equivalent,
        passed=passed,
    )


def _standardized_mean_difference(a: list[float], b: list[float]) -> float | None:
    """SMD = (mean_a - mean_b) / pooled_SD; None when the pooled SD is undefined/zero."""
    pooled = _pooled_sd(a, b)
    if pooled is None or pooled == 0.0:
        return None
    return _finite_or_none((float(np.mean(a)) - float(np.mean(b))) / pooled)


def _cohens_d(a: list[float], b: list[float]) -> float | None:
    """Cohen's d with the n-weighted pooled SD; None when undefined/zero."""
    if len(a) < 2 or len(b) < 2:
        return None
    na, nb = len(a), len(b)
    var_a, var_b = float(np.var(a, ddof=1)), float(np.var(b, ddof=1))
    pooled = math.sqrt(((na - 1) * var_a + (nb - 1) * var_b) / (na + nb - 2))
    if pooled == 0.0 or not math.isfinite(pooled):
        return None
    return _finite_or_none((float(np.mean(a)) - float(np.mean(b))) / pooled)


def _pooled_sd(a: list[float], b: list[float]) -> float | None:
    """Simple pooled SD = sqrt((var_a + var_b)/2); None when variance is unestimable."""
    if len(a) < 2 or len(b) < 2:
        return None
    pooled = math.sqrt((float(np.var(a, ddof=1)) +
                       float(np.var(b, ddof=1))) / 2.0)
    return pooled if math.isfinite(pooled) else None


def _mann_whitney_p(a: list[float], b: list[float]) -> float | None:
    """Two-sided Mann-Whitney U p-value; None on degenerate input (e.g. all-identical)."""
    if not a or not b:
        return None
    try:
        _, p = mannwhitneyu(a, b, alternative="two-sided")
    except ValueError:
        return None
    return _finite_or_none(p)


def _tost_p(a: list[float], b: list[float], tost_margin: float) -> float | None:
    """TOST p-value with bounds ±(tost_margin x pooled_SD); None when undefined."""
    pooled = _pooled_sd(a, b)
    if pooled is None or pooled == 0.0:
        return None
    bound = tost_margin * pooled
    try:
        p, _, _ = ttost_ind(a, b, -bound, bound, usevar="pooled")
    except (ValueError, ZeroDivisionError):
        return None
    return _finite_or_none(p)


def _select_specification_pool(words: list[Word], specification: Specification) -> list[Word]:
    """Words routed to specification (neutral words, specification is None -> excluded).

    Routes by Word.specification rather than deriving from weat_set, so non-WEAT sources (She 
    Figures) that carry a specification but no weat_set are included. Text-sorted for a deterministic 
    order."""
    pool = [w for w in words if w.specification == specification]
    # deterministic order
    return sorted(pool, key=lambda w: w.text)


def _freq_floor(words: Sequence[Word]) -> float | None:
    """Lowest observed (non-None) Zipf - the conservative low anchor for OOV imputation."""
    observed: list[float] = []
    for w in words:
        freq = w.covariates["subtlex_freq"]
        if freq is not None:
            observed.append(float(freq))
    return min(observed) if observed else None


def _covariate_rows(words: Sequence[Word], floor: float | None) -> NDArray[np.float64]:
    """Build the COVARIATE_KEYS-ordered covariate matrix, imputing OOV freq with floor."""
    rows: list[list[float]] = []
    for w in words:
        rows.append(
            [
                _impute_freq(w.covariates["subtlex_freq"], floor),
                _required_float(w.covariates["length"]),
                _required_float(w.covariates["wordnet_polysemy"]),
            ]
        )
    return np.asarray(rows, dtype=np.float64)


def _covariate_values(words: Sequence[Word], key: str, floor: float | None) -> list[float]:
    """Extract one covariate's values, imputing OOV subtlex_freq with floor."""
    if key == "subtlex_freq":
        return [_impute_freq(w.covariates[key], floor) for w in words]
    return [_required_float(w.covariates[key]) for w in words]


def _impute_freq(value: float | None, floor: float | None) -> float:
    """Low-anchor OOV imputation: a missing freq becomes the pool's minimum observed Zipf."""
    if value is not None:
        return float(value)
    return floor if floor is not None else 0.0


def _required_float(value: float | None) -> float:
    """Narrow a never-OOV covariate (length/wordnet_polysemy) to float."""
    if value is None:
        raise ValueError("length and wordnet_polysemy are never OOV; got None")
    return float(value)


def _finite_or_none(value: float) -> float | None:
    """Sanitize a statistic: return it as float only if finite, else None (JSON-safe)."""
    result = float(value)
    return result if math.isfinite(result) else None
