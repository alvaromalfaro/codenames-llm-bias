"""Offline tests for the read-only axis diagnostics (anisotropy vs no-signal).

Everything runs on hand-engineered ScriptedEncoder geometries wrapped in a single φ* Arbiter (ρ_w is
a measurement - one fixed encoder); the real backend is never touched. The gender axis is built
along x throughout, so cosines can be read by hand.
"""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest
from numpy.typing import NDArray

from board_generator.arbiter import Arbiter, ArbiterRef
from board_generator.axis_diagnostics import (
    AxisDiagnostics,
    SpecificationDiagnostics,
    build_axis_diagnostics,
)
from board_generator.lexicon import GenderCategory, Specification, Word
from board_generator.load_filter import AttributeWord, GenderPole

from ._stub_encoders import ScriptedEncoder


def v(x: float, y: float) -> NDArray[np.float64]:
    """A raw 2-D vector (not unit) for hand-engineered geometry."""
    return np.array([x, y], dtype=np.float64)


def vec(x: float) -> NDArray[np.float64]:
    """A 2-D unit vector with x-component x, so cos(vec(x), e_gen=[1,0]) == x exactly."""
    return np.array([x, float(np.sqrt(max(0.0, 1.0 - x * x)))], dtype=np.float64)


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


def _spec(result: AxisDiagnostics, specification: Specification) -> SpecificationDiagnostics:
    return next(s for s in result.specifications if s.specification == specification)


# --- (1) clean separation: φ* really does separate male vs female on the cores


def test_clean_separation_large_d_small_permutation_p() -> None:
    phi = arbiter(
        {
            "m_attr": vec(1.0),
            "f_attr": vec(-1.0),
            "male1": vec(0.6),
            "male2": vec(0.7),
            "male3": vec(0.8),
            "male4": vec(0.9),
            "female1": vec(-0.6),
            "female2": vec(-0.7),
            "female3": vec(-0.8),
            "female4": vec(-0.9),
        }
    )
    attrs = [attr("m_attr", "male"), attr("f_attr", "female")]
    words = [
        make_word("male1", "male", weat_set=("weat-6",),
                  specification="gender-career"),
        make_word("male2", "male", weat_set=("weat-6",),
                  specification="gender-career"),
        make_word("male3", "male", weat_set=("weat-6",),
                  specification="gender-career"),
        make_word("male4", "male", weat_set=("weat-6",),
                  specification="gender-career"),
        make_word("female1", "female", weat_set=(
            "weat-6",), specification="gender-career"),
        make_word("female2", "female", weat_set=(
            "weat-6",), specification="gender-career"),
        make_word("female3", "female", weat_set=(
            "weat-6",), specification="gender-career"),
        make_word("female4", "female", weat_set=(
            "weat-6",), specification="gender-career"),
    ]

    result = build_axis_diagnostics(
        words, attrs, phi, seed=1234567, n_permutations=5000)
    career = _spec(result, "gender-career")

    assert career.effect_raw.cohen_d is not None
    assert career.effect_raw.cohen_d > 3.0  # cleanly separated
    assert career.effect_raw.permutation_p is not None
    assert career.effect_raw.permutation_p < 0.05  # separation is significant


# --- (2) anisotropy: a large shared offset hides the signal; centering recovers it


def _anisotropy_geometry() -> tuple[Arbiter, list[AttributeWord], list[Word]]:
    # Shared offset [10, 0] ALONG e_gen, plus pole-asymmetric base norms (varying y). Raw cosines
    # are squeezed together (poor separation); mean-centering removes the offset and separates them.
    phi = arbiter(
        {
            "m_attr": v(11.0, 0.0),  # base [1,0] + offset
            "f_attr": v(9.0, 0.0),  # base [-1,0] + offset
            "male1": v(11.0, 1.0),  # base [1,1]
            "male2": v(11.0, 5.0),  # base [1,5]
            "female1": v(9.0, 1.0),  # base [-1,1]
            "female2": v(9.0, 5.0),  # base [-1,5]
        }
    )
    attrs = [attr("m_attr", "male"), attr("f_attr", "female")]
    words = [
        make_word("male1", "male", weat_set=("weat-6",),
                  specification="gender-career"),
        make_word("male2", "male", weat_set=("weat-6",),
                  specification="gender-career"),
        make_word("female1", "female", weat_set=(
            "weat-6",), specification="gender-career"),
        make_word("female2", "female", weat_set=(
            "weat-6",), specification="gender-career"),
    ]
    return phi, attrs, words


def test_anisotropy_centering_recovers_separation() -> None:
    phi, attrs, words = _anisotropy_geometry()

    result = build_axis_diagnostics(
        words, attrs, phi, seed=1234567, n_permutations=2000)
    career = _spec(result, "gender-career")

    d_raw = career.effect_raw.cohen_d
    d_centered = career.effect_centered.cohen_d
    assert d_raw is not None and d_centered is not None
    # Raw axis looks weak (the offset hides the signal); centering recovers a clean separation.
    assert abs(d_raw) < 0.5
    assert abs(d_centered) > 1.0
    assert d_centered > d_raw
    # The centered poles land on opposite sides; the raw ones barely move apart.
    assert career.effect_centered.mean_rho_male is not None
    assert career.effect_centered.mean_rho_female is not None
    assert career.effect_centered.mean_rho_male > 0 > career.effect_centered.mean_rho_female


# --- (3) μ̄ is the mean over the deduplicated embedded set; centering is exact


def test_mu_bar_dedup_and_exact_centering() -> None:
    # "king" appears BOTH as an attribute and (upper-cased) as a board word -> one cache entry.
    phi = arbiter(
        {
            "king": v(4.0, 1.0),
            "queen": v(-4.0, 1.0),
            "actor": v(2.0, 1.0),
            "actress": v(-2.0, 1.0),
        }
    )
    attrs = [attr("king", "male"), attr("queen", "female")]
    words = [
        make_word("actor", "male", weat_set=("weat-6",),
                  specification="gender-career"),
        make_word("actress", "female", weat_set=(
            "weat-6",), specification="gender-career"),
        make_word("King", "male", weat_set=("weat-6",),
                  specification="gender-career"),
    ]

    result = build_axis_diagnostics(
        words, attrs, phi, seed=1, n_permutations=100)

    # Deduplicated by lowercased text: {king, queen, actor, actress} -> 4.
    assert result.n_embedded_items == 4
    # μ̄ = mean([4,1],[-4,1],[2,1],[-2,1]) = [0, 1]; if "King" were double-counted it would shift.
    assert result.mu_bar_norm == pytest.approx(1.0)

    career = _spec(result, "gender-career")
    actor_row = next(row for row in career.words if row.text == "actor")
    # Raw: cos([2,1],[1,0]) = 2/sqrt(5). Centered: cos([2,1]-[0,1]=[2,0],[1,0]) = 1 exactly.
    assert actor_row.rho_raw == pytest.approx(2.0 / np.sqrt(5.0))
    assert actor_row.rho_centered == pytest.approx(1.0)
    assert actor_row.signed_load_centered == pytest.approx(1.0)


# --- (4) determinism: same seed -> identical permutation p-values


def test_permutation_p_is_deterministic() -> None:
    phi, attrs, words = _anisotropy_geometry()

    first = build_axis_diagnostics(
        words, attrs, phi, seed=42, n_permutations=2000)
    second = build_axis_diagnostics(
        words, attrs, phi, seed=42, n_permutations=2000)

    assert first == second
    for spec in ("gender-career", "gender-science"):
        a = _spec(first, spec)
        b = _spec(second, spec)
        assert a.effect_raw.permutation_p == b.effect_raw.permutation_p
        assert a.effect_centered.permutation_p == b.effect_centered.permutation_p


# --- (5) JSON-safety: a strict dump with allow_nan=False must succeed


def test_serializes_with_allow_nan_false() -> None:
    phi, attrs, words = _anisotropy_geometry()
    result = build_axis_diagnostics(
        words, attrs, phi, seed=1234567, n_permutations=100)

    payload = json.dumps(dataclasses.asdict(result), allow_nan=False)
    assert '"mu_bar_norm"' in payload
    assert '"cohen_d"' in payload
