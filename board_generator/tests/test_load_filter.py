"""Offline tests for the gender-load filter.

Everything runs on hand-engineered ScriptedEncoder geometries wrapped in a single φ* Arbiter (ρ_w
is a measurement - one fixed encoder, never the consensus); the real backend is never touched. The
gender axis is built along x, and every word vector is a 2-D unit vector vec(x) = [x, sqrt(1-x^2)],
so cos(word, e_gen) == x exactly and every load can be read by hand.
"""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest
from numpy.typing import NDArray

from board_generator.arbiter import Arbiter, ArbiterRef
from board_generator.balancing import BalanceReport
from board_generator.lexicon import GenderCategory, Specification, Word
from board_generator.load_filter import (
    AttributeWord,
    GenderPole,
    build_gender_axis,
    build_load_filter_report,
    build_mu_bar,
    build_sign_filter_report,
    calibrate_tau_load,
    filter_expansion,
    filter_expansion_sign,
    rho,
    signed_load_toward_pole,
)

from ._stub_encoders import ScriptedEncoder


def vec(x: float) -> NDArray[np.float64]:
    """A 2-D unit vector with x-component x, so cos(vec(x), e_gen=[1,0]) == x exactly."""
    return np.array([x, float(np.sqrt(max(0.0, 1.0 - x * x)))], dtype=np.float64)


def v(x: float, y: float) -> NDArray[np.float64]:
    """A raw 2-D vector (not unit) for hand-engineered centered geometry."""
    return np.array([x, y], dtype=np.float64)


def arbiter(vectors: dict[str, NDArray[np.float64]]) -> Arbiter:
    """Wrap a ScriptedEncoder over the given text->vector geometry as the single φ* arbiter."""
    return Arbiter(ref=ArbiterRef("stub/phi-star", "rev-test"), encoder=ScriptedEncoder(vectors))


def attr(word: str, pole: GenderPole) -> AttributeWord:
    return AttributeWord(word=word, gender_pole=pole, source="test", weat_set="weat-6")


def make_word(
    text: str,
    gender: GenderCategory,
    *,
    weat_set: tuple[str, ...],
    specification: Specification,
) -> Word:
    """Build a board-eligible loaded Word directly (core iff weat_set is non-empty)."""
    return Word(
        text=text,
        gender_category=gender,
        word_kind="common",
        source="test",
        weat_set=weat_set,
        dom_pos=None,
        ambiguous_pos=False,
        covariates={
            "subtlex_freq": float(len(text)),
            "length": float(len(text)),
            "wordnet_polysemy": 1.0,
        },
        specification=specification,
    )


# The shared male/female attribute geometry: male along +x, female along -x => e_gen == [1, 0].
_AXIS_VECTORS: dict[str, NDArray[np.float64]] = {
    "m_attr": vec(1.0),
    "f_attr": vec(-1.0),
}
_AXIS_ATTRS = [attr("m_attr", "male"), attr("f_attr", "female")]


# --- build_gender_axis


def test_build_gender_axis_is_normalized_mean_difference() -> None:
    phi = arbiter({"john": vec(1.0), "paul": vec(1.0),
                  "amy": vec(-1.0), "joan": vec(-1.0)})
    attrs = [
        attr("john", "male"),
        attr("paul", "male"),
        attr("amy", "female"),
        attr("joan", "female"),
    ]

    e_gen = build_gender_axis(attrs, phi)

    mean_male = (phi.embed("john") + phi.embed("paul")) / 2.0
    mean_female = (phi.embed("amy") + phi.embed("joan")) / 2.0
    expected = mean_male - mean_female
    expected = expected / np.linalg.norm(expected)
    assert e_gen == pytest.approx(expected)
    assert float(np.linalg.norm(e_gen)) == pytest.approx(1.0)
    assert e_gen == pytest.approx(np.array([1.0, 0.0]))


def test_build_gender_axis_dedups_repeated_pairs() -> None:
    phi = arbiter({"john": vec(1.0), "amy": vec(-1.0)})
    once = build_gender_axis(
        [attr("john", "male"), attr("amy", "female")], phi)
    # BROTHER/SON/HE-style repeats: a duplicate (word, pole) must not reweight the axis.
    twice = build_gender_axis(
        [attr("john", "male"), attr("john", "male"), attr("amy", "female")], phi
    )
    assert once == pytest.approx(twice)


def test_build_gender_axis_conflicting_pole_raises() -> None:
    phi = arbiter({"john": vec(1.0), "amy": vec(-1.0)})
    with pytest.raises(ValueError, match="conflicting poles"):
        build_gender_axis([attr("john", "male"), attr("john", "female")], phi)


# --- rho sign and signed_load_toward_pole


def test_rho_sign_follows_male_positive_axis() -> None:
    phi = arbiter({**_AXIS_VECTORS, "engineer": vec(0.7), "nurse": vec(-0.7)})
    e_gen = build_gender_axis(_AXIS_ATTRS, phi)

    assert rho(phi, e_gen, "engineer") > 0  # male-aligned
    assert rho(phi, e_gen, "nurse") < 0  # female-aligned


def test_signed_load_flips_for_female_pole() -> None:
    assert signed_load_toward_pole(0.6, "male") == pytest.approx(0.6)
    assert signed_load_toward_pole(-0.6, "female") == pytest.approx(0.6)


# --- calibrate_tau_load


def test_calibrate_tau_load_matches_numpy_quantile() -> None:
    phi = arbiter(
        {**_AXIS_VECTORS,
            "executive": vec(0.5), "secretary": vec(-0.5), "boss": vec(0.05)}
    )
    e_gen = build_gender_axis(_AXIS_ATTRS, phi)
    core = [
        make_word("executive", "male", weat_set=(
            "weat-6",), specification="gender-career"),
        make_word("secretary", "female", weat_set=(
            "weat-6",), specification="gender-career"),
        make_word("boss", "male", weat_set=("weat-6",),
                  specification="gender-career"),
    ]

    tau = calibrate_tau_load(core, phi, e_gen, 0.10)

    # signed-toward-own-pole: executive 0.5, secretary -(-0.5)=0.5, boss 0.05.
    assert tau == pytest.approx(
        float(np.quantile([0.5, 0.5, 0.05], 0.10, method="linear")))


def test_calibrate_tau_load_two_specs_independent() -> None:
    phi = arbiter(
        {
            **_AXIS_VECTORS,
            "executive": vec(0.5),
            "boss": vec(0.3),
            "math": vec(0.9),
            "physics": vec(0.7),
        }
    )
    e_gen = build_gender_axis(_AXIS_ATTRS, phi)
    career_core = [
        make_word("executive", "male", weat_set=(
            "weat-6",), specification="gender-career"),
        make_word("boss", "male", weat_set=("weat-6",),
                  specification="gender-career"),
    ]
    science_core = [
        make_word("math", "male", weat_set=("weat-7",),
                  specification="gender-science"),
        make_word("physics", "male", weat_set=("weat-7",),
                  specification="gender-science"),
    ]

    tau_career = calibrate_tau_load(career_core, phi, e_gen, 0.10)
    tau_science = calibrate_tau_load(science_core, phi, e_gen, 0.10)

    assert tau_career == pytest.approx(
        float(np.quantile([0.5, 0.3], 0.10, method="linear")))
    assert tau_science == pytest.approx(
        float(np.quantile([0.9, 0.7], 0.10, method="linear")))
    assert tau_career != tau_science


def test_calibrate_tau_load_empty_core_raises() -> None:
    phi = arbiter(_AXIS_VECTORS)
    e_gen = build_gender_axis(_AXIS_ATTRS, phi)
    with pytest.raises(ValueError, match="no WEAT core"):
        calibrate_tau_load([], phi, e_gen, 0.10)


# --- filter_expansion


def test_filter_expansion_admits_at_or_above_tau_rejects_below() -> None:
    phi = arbiter({**_AXIS_VECTORS, "above": vec(0.8), "below": vec(0.1)})
    e_gen = build_gender_axis(_AXIS_ATTRS, phi)
    expansion = [
        make_word("above", "male", weat_set=(), specification="gender-career"),
        make_word("below", "male", weat_set=(), specification="gender-career"),
    ]

    verdicts = {load.text: load.admitted for load in filter_expansion(
        expansion, 0.5, phi, e_gen)}

    assert verdicts == {"above": True, "below": False}


def test_filter_expansion_exact_tie_at_tau_admitted() -> None:
    # Non-strict '>=' with no margin: a signed load exactly equal to τ passes.
    phi = arbiter({**_AXIS_VECTORS, "tie": vec(0.5)})
    e_gen = build_gender_axis(_AXIS_ATTRS, phi)
    expansion = [make_word("tie", "male", weat_set=(),
                           specification="gender-career")]

    (verdict,) = filter_expansion(expansion, 0.5, phi, e_gen)

    assert verdict.signed_load == pytest.approx(0.5)
    assert verdict.admitted is True


def test_filter_expansion_rejects_weak_career_bridge_admits_strong() -> None:
    # The methodological case: a topically-correct but gender-weak bridge (ρ≈0) is rejected, while
    # a strongly-loaded expansion word is admitted.
    phi = arbiter({**_AXIS_VECTORS, "work": vec(0.0), "manager": vec(0.8)})
    e_gen = build_gender_axis(_AXIS_ATTRS, phi)
    expansion = [
        make_word("work", "male", weat_set=(), specification="gender-career"),
        make_word("manager", "male", weat_set=(),
                  specification="gender-career"),
    ]

    verdicts = {load.text: load.admitted for load in filter_expansion(
        expansion, 0.2, phi, e_gen)}

    assert verdicts == {"work": False, "manager": True}


# --- build_load_filter_report (end-to-end on a stub)


def _report_geometry() -> Arbiter:
    return arbiter(
        {
            **_AXIS_VECTORS,
            # career core (grandfathered); "boss" is a deliberately weak core word.
            "executive": vec(0.5),
            "secretary": vec(-0.5),
            "boss": vec(0.05),
            # career expansion
            "manager": vec(0.8),
            "work": vec(0.0),
            # science core
            "math": vec(0.6),
            "poetry": vec(-0.6),
            # science expansion
            "physics": vec(0.7),
            "dance": vec(-0.65),
        }
    )


def _report_pool() -> list[Word]:
    return [
        make_word("executive", "male", weat_set=(
            "weat-6",), specification="gender-career"),
        make_word("secretary", "female", weat_set=(
            "weat-6",), specification="gender-career"),
        make_word("boss", "male", weat_set=("weat-6",),
                  specification="gender-career"),
        make_word("manager", "male", weat_set=(),
                  specification="gender-career"),
        make_word("work", "male", weat_set=(), specification="gender-career"),
        make_word("math", "male", weat_set=("weat-7",),
                  specification="gender-science"),
        make_word("poetry", "female", weat_set=(
            "weat-7",), specification="gender-science"),
        make_word("physics", "male", weat_set=(),
                  specification="gender-science"),
        make_word("dance", "female", weat_set=(),
                  specification="gender-science"),
    ]


def test_build_load_filter_report_survivors_and_embedded_rebalance() -> None:
    phi = _report_geometry()
    report = build_load_filter_report(
        _report_pool(), _AXIS_ATTRS, phi, seed=1234567)

    by_spec = {s.specification: s for s in report.specifications}
    career = by_spec["gender-career"]
    science = by_spec["gender-science"]

    # Survivors = all core (incl. the weak "boss", grandfathered) + admitted expansion ("manager").
    assert set(career.survivors) == {
        "executive", "secretary", "boss", "manager"}
    assert "boss" in career.survivors  # weak core is NOT filtered
    assert "work" not in career.survivors  # weak expansion bridge rejected
    assert science.n_expansion_rejected == 0
    assert set(science.survivors) == {"math", "poetry", "physics", "dance"}

    # The embedded re-balance over the survivors is present.
    assert isinstance(report.rebalance, BalanceReport)
    assert {s.specification for s in report.rebalance.specifications} == {
        "gender-career",
        "gender-science",
    }
    assert report.arbiter_primary == "stub/phi-star@rev-test"
    assert report.gender_axis_dim == 2


def test_build_load_filter_report_serializes_with_allow_nan_false() -> None:
    phi = _report_geometry()
    report = build_load_filter_report(
        _report_pool(), _AXIS_ATTRS, phi, seed=1234567)

    # No NaN/Inf anywhere: a strict dump must succeed (the JSON-safety convention holds).
    payload = json.dumps(dataclasses.asdict(report), allow_nan=False)
    assert '"tau_load"' in payload


# --- determinism (I-8)


def test_load_filter_is_deterministic() -> None:
    phi = _report_geometry()
    first = build_load_filter_report(
        _report_pool(), _AXIS_ATTRS, phi, seed=1234567)
    second = build_load_filter_report(
        _report_pool(), _AXIS_ATTRS, phi, seed=1234567)

    assert build_gender_axis(_AXIS_ATTRS, phi) == pytest.approx(
        build_gender_axis(_AXIS_ATTRS, phi))
    first_by_spec = {s.specification: s for s in first.specifications}
    second_by_spec = {s.specification: s for s in second.specifications}
    for spec in ("gender-career", "gender-science"):
        assert first_by_spec[spec].tau_load == second_by_spec[spec].tau_load
        assert first_by_spec[spec].survivors == second_by_spec[spec].survivors


# --- centered mode: build_mu_bar, centered axis & rho (additive; raw path untouched above)


def test_build_mu_bar_dedups_lowercased_reference_set() -> None:
    # mean([2,0],[0,2]) = [1,1]; "A"/"a" collapse to one entry, so duplicates never shift μ̄.
    phi = arbiter({"a": v(2.0, 0.0), "b": v(0.0, 2.0)})

    base = build_mu_bar(["a", "b"], phi)
    with_dups = build_mu_bar(["A", "a", "b", "B", "a"], phi)

    assert base == pytest.approx(np.array([1.0, 1.0]))
    assert with_dups == pytest.approx(base)


def test_build_mu_bar_empty_reference_raises() -> None:
    phi = arbiter({"a": v(1.0, 0.0)})
    with pytest.raises(ValueError, match="reference set is empty"):
        build_mu_bar([], phi)


def test_centered_axis_and_rho_are_exact_with_known_mu_bar() -> None:
    # attrs along x => raw axis [1,0]; subtracting μ̄=[0,1] is offset-invariant => same axis.
    phi = arbiter(
        {"m_attr": v(3.0, 1.0), "f_attr": v(-3.0, 1.0), "word": v(2.0, 1.0)})
    attrs = [attr("m_attr", "male"), attr("f_attr", "female")]
    mu_bar = v(0.0, 1.0)

    axis_centered = build_gender_axis(attrs, phi, centered=True, mu_bar=mu_bar)
    assert axis_centered == pytest.approx(np.array([1.0, 0.0]))

    # φ̃("word") = [2,1] - [0,1] = [2,0] => cos with [1,0] is exactly 1.
    rho_centered = rho(phi, axis_centered, "word",
                       centered=True, mu_bar=mu_bar)
    rho_raw = rho(phi, build_gender_axis(attrs, phi), "word")
    assert rho_centered == pytest.approx(1.0)
    assert rho_raw == pytest.approx(2.0 / np.sqrt(5.0))
    # centering moves ρ (it is not offset-invariant)
    assert rho_centered != pytest.approx(rho_raw)


def test_centered_path_equals_raw_path_when_mu_bar_is_zero() -> None:
    phi = arbiter(
        {"m_attr": v(3.0, 1.0), "f_attr": v(-3.0, 1.0), "word": v(2.0, 1.0)})
    attrs = [attr("m_attr", "male"), attr("f_attr", "female")]
    zero = np.zeros(2, dtype=np.float64)

    axis_raw = build_gender_axis(attrs, phi)
    axis_centered_zero = build_gender_axis(
        attrs, phi, centered=True, mu_bar=zero)
    assert axis_centered_zero == pytest.approx(axis_raw)

    assert rho(phi, axis_raw, "word", centered=True, mu_bar=zero) == pytest.approx(
        rho(phi, axis_raw, "word")
    )


def test_centered_axis_requires_mu_bar() -> None:
    phi = arbiter(_AXIS_VECTORS)
    with pytest.raises(ValueError, match="requires mu_bar"):
        build_gender_axis(_AXIS_ATTRS, phi, centered=True)


def test_centered_rho_requires_mu_bar() -> None:
    phi = arbiter({**_AXIS_VECTORS, "word": vec(0.3)})
    e_gen = build_gender_axis(_AXIS_ATTRS, phi)
    with pytest.raises(ValueError, match="requires mu_bar"):
        rho(phi, e_gen, "word", centered=True)


# --- sign criterion: strict '>' admission on the centered axis


def test_filter_expansion_sign_is_strict_around_delta() -> None:
    # Probe strictness around a candidate's OWN signed load s, so the boundary is exact (no fragile
    # float tie from constructing a cosine that lands precisely on δ). Admit iff s > δ.
    phi = arbiter({**_AXIS_VECTORS, "cand": vec(0.6)})
    mu_bar = np.zeros(2, dtype=np.float64)
    axis_centered = build_gender_axis(
        _AXIS_ATTRS, phi, centered=True, mu_bar=mu_bar)
    expansion = [make_word("cand", "male", weat_set=(),
                           specification="gender-career")]

    (probe,) = filter_expansion_sign(expansion, 0.0, phi, axis_centered, mu_bar)
    s = probe.signed_load_centered

    def admitted(delta: float) -> bool:
        (load,) = filter_expansion_sign(
            expansion, delta, phi, axis_centered, mu_bar)
        return load.admitted

    # STRICT '>': just below δ admitted, exactly at δ rejected, just above δ rejected.
    assert admitted(float(np.nextafter(s, -np.inf))) is True
    assert admitted(s) is False
    assert admitted(float(np.nextafter(s, np.inf))) is False


def test_sign_filter_grandfathers_wrong_side_core_word() -> None:
    phi = arbiter(
        {
            **_AXIS_VECTORS,
            "executive": v(2.0, 1.0),  # career core, male
            # career core, male, but lands on the female side
            "wrongcore": v(-2.0, 1.0),
            "math": v(2.0, 1.0),  # science core, male
            "poetry": v(-2.0, 1.0),  # science core, female
        }
    )
    pool = [
        make_word("executive", "male", weat_set=(
            "weat-6",), specification="gender-career"),
        make_word("wrongcore", "male", weat_set=(
            "weat-6",), specification="gender-career"),
        make_word("math", "male", weat_set=("weat-7",),
                  specification="gender-science"),
        make_word("poetry", "female", weat_set=(
            "weat-7",), specification="gender-science"),
    ]

    report = build_sign_filter_report(
        pool, _AXIS_ATTRS, phi, seed=1234567, delta=0.0)
    career = next(
        s for s in report.specifications if s.specification == "gender-career")

    wrong = next(c for c in career.core if c.text == "wrongcore")
    assert wrong.grandfathered is True
    # a core word is never filtered, even on the wrong side
    assert wrong.admitted is True
    assert "wrongcore" in career.survivors


def test_sign_filter_admitted_set_is_antitone_in_delta() -> None:
    phi = arbiter(
        {
            **_AXIS_VECTORS,
            "w1": vec(0.1),
            "w2": vec(0.3),
            "w3": vec(0.5),
            "w4": vec(0.7),
        }
    )
    mu_bar = np.zeros(2, dtype=np.float64)
    axis_centered = build_gender_axis(
        _AXIS_ATTRS, phi, centered=True, mu_bar=mu_bar)
    expansion = [
        make_word("w1", "male", weat_set=(), specification="gender-career"),
        make_word("w2", "male", weat_set=(), specification="gender-career"),
        make_word("w3", "male", weat_set=(), specification="gender-career"),
        make_word("w4", "male", weat_set=(), specification="gender-career"),
    ]

    def admitted(delta: float) -> set[str]:
        return {
            load.text
            for load in filter_expansion_sign(expansion, delta, phi, axis_centered, mu_bar)
            if load.admitted
        }

    deltas = [0.0, 0.2, 0.4, 0.6, 0.8]
    sets = [admitted(d) for d in deltas]
    # Raising δ never admits more words: each admitted set is a subset of the previous one.
    for smaller_delta_set, larger_delta_set in zip(sets, sets[1:], strict=False):
        assert larger_delta_set <= smaller_delta_set


# --- sign-filter report: determinism and JSON-safety


def _sign_report_pool() -> list[Word]:
    return [
        make_word("executive", "male", weat_set=(
            "weat-6",), specification="gender-career"),
        make_word("secretary", "female", weat_set=(
            "weat-6",), specification="gender-career"),
        make_word("manager", "male", weat_set=(),
                  specification="gender-career"),
        make_word("work", "male", weat_set=(), specification="gender-career"),
        make_word("math", "male", weat_set=("weat-7",),
                  specification="gender-science"),
        make_word("poetry", "female", weat_set=(
            "weat-7",), specification="gender-science"),
        make_word("physics", "male", weat_set=(),
                  specification="gender-science"),
        make_word("dance", "female", weat_set=(),
                  specification="gender-science"),
    ]


def _sign_report_geometry() -> Arbiter:
    return arbiter(
        {
            **_AXIS_VECTORS,
            "executive": v(2.0, 1.0),
            "secretary": v(-2.0, 1.0),
            "manager": v(3.0, 1.0),
            "work": v(0.0, 1.0),
            "math": v(2.0, 1.0),
            "poetry": v(-2.0, 1.0),
            "physics": v(2.5, 1.0),
            "dance": v(-2.5, 1.0),
        }
    )


def test_sign_filter_report_is_deterministic() -> None:
    phi = _sign_report_geometry()
    pool = _sign_report_pool()
    first = build_sign_filter_report(
        pool, _AXIS_ATTRS, phi, seed=1234567, delta=0.0)
    second = build_sign_filter_report(
        pool, _AXIS_ATTRS, phi, seed=1234567, delta=0.0)

    # Same inputs + the same μ̄ construction => identical centered axis and partitions.
    assert first == second
    first_by_spec = {s.specification: s for s in first.specifications}
    second_by_spec = {s.specification: s for s in second.specifications}
    for spec in ("gender-career", "gender-science"):
        assert first_by_spec[spec].survivors == second_by_spec[spec].survivors
        assert [(c.text, c.admitted) for c in first_by_spec[spec].expansion] == [
            (c.text, c.admitted) for c in second_by_spec[spec].expansion
        ]


def test_sign_filter_report_serializes_with_allow_nan_false() -> None:
    phi = _sign_report_geometry()
    report = build_sign_filter_report(
        _sign_report_pool(), _AXIS_ATTRS, phi, seed=1234567, delta=0.0
    )

    payload = json.dumps(dataclasses.asdict(report), allow_nan=False)
    assert '"signed_load_centered"' in payload
    assert '"mu_bar_norm"' in payload
