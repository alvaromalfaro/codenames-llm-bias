"""Tests for covariate balancing.

Positive structural cases (counts) run against the real resources/words/ + SUBTLEX-US pool; the 
statistical verdicts and edge cases use small, hand-built Word pools so behaviour does not depend on 
the 41-word corpus or on WordNet/CSV loading. Both happy and unhappy paths are exercised: the 
important negative case asserts the report correctly reports NON-equivalence.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import inspect
import json
import warnings
from pathlib import Path

import pytest

from board_generator import balancing
from board_generator.balancing import (
    COHEN_D_THRESHOLD,
    PSM_CALIPER_LOGIT_SD,
    SMD_BALANCE_THRESHOLD,
    SMD_WELL_BALANCED_THRESHOLD,
    TOST_MARGIN_SMD,
    check_balance,
    run_balancing,
)
from board_generator.lexicon import LoadResult, Word, load_words

RESOURCES = Path(__file__).resolve().parents[1] / "resources"
REAL_WORDS = RESOURCES / "words"
REAL_SUBTLEX = RESOURCES / "frequencies" / "subtlex_us.csv"


def make_word(text: str, gender: str, freq: float | None, *, length: int | None = None,
              polysemy: int = 1, weat: tuple[str, ...] = ("weat-6",), specification: str = "gender-career",
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
        specification=specification
    )


@pytest.fixture(scope="module")
def real_load() -> LoadResult:
    with warnings.catch_warnings():  # proper nouns may be OOV; not under test here
        warnings.simplefilter("ignore")
        return load_words(REAL_WORDS, REAL_SUBTLEX)


def _spec(result_report: balancing.BalanceReport, specification: str
          ) -> balancing.SpecificationBalance:
    return next(s for s in result_report.specifications if s.specification == specification)


# Integration: real-pool counts. Expectations are derived from the selected pool, never
# hardcoded - the pools grow as expansions land.
def _pool(words: list[Word], specification: str) -> list[Word]:
    return [w for w in words if w.specification == specification]


def test_real_pool_counts(real_load: LoadResult) -> None:
    result = run_balancing(real_load.words, seed=1234567)
    report = result.report

    sci_pool = _pool(real_load.words, "gender-science")
    career_pool = _pool(real_load.words, "gender-career")

    science = _spec(report, "gender-science")
    n_male = sum(1 for w in sci_pool if w.gender_category == "male")
    n_female = len(sci_pool) - n_male
    # The majority-pole surplus can never be matched 1:1, derived from the actual pole sizes.
    assert science.counts.dropped_by_group_excess == abs(n_male - n_female)
    assert science.counts.pairs_kept <= min(n_male, n_female)
    # Imputation count is derived from the OOV words in the pool, not assumed zero.
    assert science.counts.imputed_count == sum(
        1 for w in sci_pool if w.covariates["subtlex_freq"] is None
    )

    career = _spec(report, "gender-career")
    n_male_c = sum(1 for w in career_pool if w.gender_category == "male")
    n_female_c = len(career_pool) - n_male_c
    assert career.counts.dropped_by_group_excess == abs(n_male_c - n_female_c)

    # Routing is by specification: a She Figures expansion word lands in science, not career, and
    # a known career word lands in career - behaviour the weat_set derivation could not give.
    sci_texts = {w.text for w in sci_pool}
    career_texts = {w.text for w in career_pool}
    assert "engineering" in sci_texts
    assert "engineering" not in career_texts
    assert "salary" in career_texts

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
    loaded = json.loads(text)
    assert loaded["criterion"] == "smd"
    # The new advisory threshold fields serialize alongside the existing ones.
    assert loaded["smd_threshold"] == SMD_BALANCE_THRESHOLD
    assert loaded["smd_well_threshold"] == SMD_WELL_BALANCED_THRESHOLD

    # Undefined statistics (zero pooled SD) are None and treated as non-equivalent.
    career = _spec(result.report, "gender-career")
    assert career.well_balanced is False
    for cov in career.covariates:
        assert cov.smd is None
        assert cov.passed is False
        assert cov.well_balanced is False


# GOVERNING SMD on the real pool: the curated pool is matched so that BOTH specifications balance
# (every |SMD| < 0.2) and pass. The fail path is exercised separately on a controlled fixture
# (test_systematic_shift_reports_non_equivalence); here we assert the real pool stays balanced.
def test_real_pool_smd_verdict(real_load: LoadResult) -> None:
    # default criterion == "smd"
    result = run_balancing(real_load.words, seed=1234567)
    report = result.report
    assert report.criterion == "smd"

    for specification in ("gender-science", "gender-career"):
        spec = _spec(report, specification)
        for cov in spec.covariates:
            assert cov.smd is not None and abs(cov.smd) < SMD_BALANCE_THRESHOLD, (
                specification,
                cov,
            )
        assert spec.passed is True


# The diagnostic script must inherit the governing SMD default, never override it back to TOST.
# (`scripts/` is not a package, so load the module from its path.)
def test_script_cli_default_is_smd() -> None:
    script_path = Path(__file__).resolve(
    ).parents[1] / "scripts" / "balance_report.py"
    spec = importlib.util.spec_from_file_location(
        "balance_report", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    args = module.build_parser().parse_args([])
    assert args.criterion == "smd"
    criterion_action = next(
        a for a in module.build_parser()._actions if a.dest == "criterion")
    assert "smd" in (criterion_action.choices or ())


# Graduated advisory flag: |SMD| in (0.1, 0.2) passes but is not well-balanced; < 0.1 is both;
# > 0.2 fails. SMDs are placed relative to the pool SD, never as hardcoded magic numbers.
def test_smd_graduated_flag() -> None:
    ctrl = [float(x) for x in range(10)]
    # treat shares ctrl's spread, so SMD = delta / pooled
    pooled = balancing._pooled_sd(ctrl, ctrl)
    assert pooled is not None and pooled > 0.0

    def verdict(target_smd: float) -> balancing.CovariateBalance:
        treat = [v + target_smd * pooled for v in ctrl]
        return balancing._covariate_balance(
            "length", treat, ctrl, "smd", balancing.DEFAULT_ALPHA, TOST_MARGIN_SMD
        )

    mid = verdict(0.15)  # 0.1 < |SMD| < 0.2
    assert mid.smd is not None and 0.1 < abs(mid.smd) < 0.2
    assert mid.passed is True and mid.well_balanced is False

    low = verdict(0.05)  # |SMD| < 0.1
    assert low.smd is not None and abs(low.smd) < 0.1
    assert low.passed is True and low.well_balanced is True

    high = verdict(0.35)  # |SMD| > 0.2
    assert high.smd is not None and abs(high.smd) > 0.2
    assert high.passed is False


# Undefined SMD (degenerate constant covariate) -> conservatively non-equivalent, not well-balanced.
def test_undefined_smd_fails() -> None:
    const = [5.0] * 6
    cov = balancing._covariate_balance(
        "length", const, const, "smd", balancing.DEFAULT_ALPHA, TOST_MARGIN_SMD
    )
    assert cov.smd is None
    assert cov.passed is False
    assert cov.well_balanced is False


# Regression: "tost" stays selectable and governs; secondary diagnostics remain present/populated.
def test_tost_still_selectable_and_governs(real_load: LoadResult) -> None:
    report = run_balancing(real_load.words, seed=1234567,
                           criterion="tost").report
    assert report.criterion == "tost"
    for spec in report.specifications:
        for cov in spec.covariates:
            # TOST and Mann–Whitney fields are still reported as secondary diagnostics.
            assert isinstance(cov.tost_equivalent, bool)
            assert isinstance(cov.spec_original_equivalent, bool)
            assert hasattr(cov, "tost_p") and hasattr(cov, "mann_whitney_p")
            # Under "tost" the governing verdict equals the (underpowered) TOST verdict.
            assert cov.passed == cov.tost_equivalent


# The distinct "small effect" constants exist and are referenced by name; SMD is the default.
def test_constants_and_defaults() -> None:
    assert SMD_BALANCE_THRESHOLD == 0.2
    assert SMD_WELL_BALANCED_THRESHOLD == 0.1
    assert COHEN_D_THRESHOLD == 0.2
    assert TOST_MARGIN_SMD == 0.2
    assert PSM_CALIPER_LOGIT_SD == 0.2
    # Each operationalization of "small effect" is separately named (not one shared literal).
    names = {
        "SMD_BALANCE_THRESHOLD",
        "SMD_WELL_BALANCED_THRESHOLD",
        "COHEN_D_THRESHOLD",
        "TOST_MARGIN_SMD",
        "PSM_CALIPER_LOGIT_SD",
    }
    assert names <= set(vars(balancing))
    # SMD governs by default; the significance-based criteria stay selectable.
    assert inspect.signature(
        run_balancing).parameters["criterion"].default == "smd"
    assert inspect.signature(
        check_balance).parameters["criterion"].default == "smd"
