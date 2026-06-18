"""Tests for covariate balancing.

Positive structural cases (counts) run against the real resources/words/ + SUBTLEX-US pool; the 
statistical verdicts and edge cases use small, hand-built Word pools so behaviour does not depend on 
the 41-word corpus or on WordNet/CSV loading. Both happy and unhappy paths are exercised: the 
important negative case asserts the report correctly reports NON-equivalence.
"""

from __future__ import annotations

import dataclasses
import inspect
import json
import warnings
from pathlib import Path

import pytest

from board_generator import balancing
from board_generator.balancing import (
    COHEN_D_THRESHOLD,
    PSM_CALIPER_LOGIT_SD,
    TOST_MARGIN_SMD,
    check_balance,
    run_balancing,
)
from board_generator.lexicon import LoadResult, Word, load_words

RESOURCES = Path(__file__).resolve().parents[1] / "resources"
REAL_WORDS = RESOURCES / "words"
REAL_SUBTLEX = RESOURCES / "frequencies" / "subtlex_us.csv"


def make_word(
    text: str,
    gender: str,
    freq: float | None,
    *,
    length: int | None = None,
    polysemy: int = 1,
    weat: tuple[str, ...] = ("weat-6",),
) -> Word:
    """Build a board-eligible Word directly, bypassing CSV/WordNet loading."""
    return Word(
        text=text,
        gender_category=gender,  # type: ignore[arg-type]
        word_kind="common",
        source="test",
        weat_set=weat,
        dom_pos=None,
        ambiguous_pos=False,
        covariates={
            "subtlex_freq": freq,
            "length": float(length if length is not None else len(text)),
            "wordnet_polysemy": float(polysemy),
        },
    )


@pytest.fixture(scope="module")
def real_load() -> LoadResult:
    with warnings.catch_warnings():  # proper nouns may be OOV; not under test here
        warnings.simplefilter("ignore")
        return load_words(REAL_WORDS, REAL_SUBTLEX)


def _spec(
    result_report: balancing.BalanceReport, specification: str
) -> balancing.SpecificationBalance:
    return next(s for s in result_report.specifications if s.specification == specification)


# Integration: real-pool counts (16-vs-9 science, 8-vs-8 career).
def test_real_pool_counts(real_load: LoadResult) -> None:
    result = run_balancing(real_load.words, seed=1234567)
    report = result.report

    science = _spec(report, "gender-science")
    # 16 male (STEM) vs 9 female (Arts) -> 7 majority-pole words can never be matched 1:1.
    assert science.counts.dropped_by_group_excess == 7
    assert science.counts.pairs_kept <= 9
    # All real pool words are present in SUBTLEX-US (proper nouns included) -> zero imputation.
    assert science.counts.imputed_count == 0

    career = _spec(report, "gender-career")
    # 8 male (Career) vs 8 female (Family) -> no structural surplus.
    assert career.counts.dropped_by_group_excess == 0
    assert career.counts.imputed_count == 0

    # The matched subsets are returned alongside the report, paired one-to-one.
    sci_subset = next(
        m for m in result.matched if m.specification == "gender-science")
    assert len(sci_subset.treatment) == len(
        sci_subset.control) == science.counts.pairs_kept


# NEGATIVE: a systematically shifted pole -> report shows non-equivalence.
# Driven through check_balance so PSM does not simply drop the whole imbalanced sample.
def test_systematic_shift_reports_non_equivalence() -> None:
    # Each covariate is shifted between poles but varies within a pole (so SMD is defined).
    treatment = [
        make_word(f"hi{i:02d}", "male", 6.0 + 0.05 * i,
                  length=12 + i % 4, polysemy=9 + i % 3)
        for i in range(12)
    ]
    control = [
        make_word(f"lo{i:02d}", "female", 2.0 + 0.05 * i,
                  length=4 + i % 4, polysemy=1 + i % 3)
        for i in range(12)
    ]

    covariates = check_balance(treatment, control, criterion="tost")
    by_name = {c.covariate: c for c in covariates}

    # Every covariate is shifted: large SMD, neither criterion can establish equivalence.
    for cov in covariates:
        assert cov.smd is not None and abs(cov.smd) > 0.8, cov
        assert cov.tost_equivalent is False
        assert cov.spec_original_equivalent is False
        assert cov.passed is False
    assert by_name["length"].smd is not None and by_name["length"].smd > 1.0


# POSITIVE: a balanced pool with enough n for TOST to have power -> equivalence holds.
def test_balanced_pool_with_power_reports_equivalence() -> None:
    # Identical distributions. At a 0.2-SD margin TOST needs large n to conclude equivalence
    # (the 90% CI half-width must fall below the margin) - this is the very T-2 power phenomenon, so
    # the synthetic pool is deliberately big; with small n TOST would (correctly) abstain.
    values = [4.0 + 0.05 * i for i in range(300)]
    treatment = [
        make_word(f"ta{i:03d}", "male", v, length=6, polysemy=3) for i, v in enumerate(values)
    ]
    control = [
        make_word(f"cb{i:03d}", "female", v, length=6, polysemy=3) for i, v in enumerate(values)
    ]

    covariates = check_balance(treatment, control, criterion="tost")
    freq = next(c for c in covariates if c.covariate == "subtlex_freq")
    assert freq.smd is not None and abs(freq.smd) < 0.05
    assert freq.tost_p is not None and freq.tost_equivalent is True
    assert freq.passed is True


# OOV imputation: low anchor (pool minimum Zipf) + a surviving freq_imputed flag.
def test_oov_word_is_low_anchor_imputed() -> None:
    pool = [
        make_word("alpha", "male", 5.0),
        make_word("bravo", "male", 3.5),
        make_word("oov", "male", None),  # OOV -> imputed
        make_word("delta", "female", 4.0),
        make_word("echo", "female", 2.5),  # the pool minimum observed Zipf
        make_word("foxtrot", "female", 4.5),
    ]
    result = run_balancing(pool, seed=7)
    career = _spec(result.report, "gender-career")
    assert career.imputed_words == ["oov"]
    assert career.counts.imputed_count == 1

    # The imputed value is the pool minimum observed Zipf (a conservative low anchor, not 0).
    floor = balancing._freq_floor(pool)
    assert floor == 2.5
    assert balancing._impute_freq(None, floor) == 2.5


# Caliper: an unmatchable outlier pole-member is dropped by the caliper, not silently matched.
def test_caliper_drops_unmatchable_outlier() -> None:
    treatment = [
        make_word("m0", "male", 5.0, length=5, polysemy=2),
        make_word("m1", "male", 5.0, length=5, polysemy=2),
        make_word("m2", "male", 5.0, length=5, polysemy=2),
        # extreme on every covariate
        make_word("out", "male", 7.0, length=40, polysemy=40),
    ]
    control = [make_word(f"f{i}", "female", 5.0,
                         length=5, polysemy=2) for i in range(4)]

    matched_t, matched_c, dropped_caliper, dropped_excess = balancing.propensity_score_match(
        treatment, control, seed=42
    )
    assert dropped_excess == 0  # equal pole sizes
    assert len(matched_t) == len(matched_c) == 3
    assert dropped_caliper == 1
    assert "out" not in {w.text for w in matched_t}


# Determinism: same input + seed -> identical report and identical matched subsets.
def test_deterministic(real_load: LoadResult) -> None:
    first = run_balancing(real_load.words, seed=99)
    second = run_balancing(real_load.words, seed=99)
    assert dataclasses.asdict(
        first.report) == dataclasses.asdict(second.report)
    for a, b in zip(first.matched, second.matched, strict=True):
        assert [w.text for w in a.treatment] == [w.text for w in b.treatment]
        assert [w.text for w in a.control] == [w.text for w in b.control]


# JSON safety: a degenerate (constant-covariate) pool must still serialize with no NaN/Inf.
def test_degenerate_pool_serializes_without_nan() -> None:
    pool = [
        make_word(f"c{i}", "male" if i < 3 else "female",
                  5.0, length=5, polysemy=2)
        for i in range(6)
    ]
    result = run_balancing(pool, seed=1)

    payload = dataclasses.asdict(result.report)
    # allow_nan=False raises if any NaN/Infinity slipped into a statistic field.
    text = json.dumps(payload, allow_nan=False)
    assert "NaN" not in text and "Infinity" not in text
    assert json.loads(text)["criterion"] == "tost"

    # Undefined statistics (zero pooled SD) are None and treated as non-equivalent.
    career = _spec(result.report, "gender-career")
    for cov in career.covariates:
        assert cov.smd is None
        assert cov.passed is False


# The three distinct 0.2 constants exist and are referenced by name; TOST is the default criterion.
def test_constants_and_defaults() -> None:
    assert COHEN_D_THRESHOLD == 0.2
    assert TOST_MARGIN_SMD == 0.2
    assert PSM_CALIPER_LOGIT_SD == 0.2
    # Three separately named constants (not one shared literal reused everywhere).
    names = {"COHEN_D_THRESHOLD", "TOST_MARGIN_SMD", "PSM_CALIPER_LOGIT_SD"}
    assert names <= set(vars(balancing))
    assert inspect.signature(
        run_balancing).parameters["criterion"].default == "tost"
    assert inspect.signature(
        check_balance).parameters["criterion"].default == "tost"
