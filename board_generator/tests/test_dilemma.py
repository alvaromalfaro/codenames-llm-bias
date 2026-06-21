"""Offline tests for the AUTO dilemma consumers.

Everything runs on hand-engineered ScriptedEncoder geometries wrapped in Arbiters; the real
backend is NEVER touched. Cosines are read off 2D vectors so cos(a, b) is the dot of unit vectors
and can be checked by hand.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from board_generator.arbiter import Arbiter, ArbiterRef
from board_generator.dilemma import (
    rank_neutral_bridges,
    rank_stereotypical_bridges,
    verify_eq_4_1,
)
from board_generator.lexicon import Word

from ._stub_encoders import ScriptedEncoder


def make_word(text: str, gender: str = "neutral") -> Word:
    """Build a board-eligible Word directly, bypassing CSV/WordNet loading."""
    return Word(
        text=text,
        gender_category=gender,
        word_kind="common",
        source="test",
        weat_set=("weat-6",),
        dom_pos=None,
        ambiguous_pos=False,
        covariates={"subtlex_freq": 1.0, "length": float(
            len(text)), "wordnet_polysemy": 1.0},
        specification="gender-career",
    )


def arbiter(model_id: str, vectors: dict[str, NDArray[np.float64]]) -> Arbiter:
    """Wrap a ScriptedEncoder over the given text->vector geometry in a pinned Arbiter."""
    return Arbiter(ref=ArbiterRef(model_id, "rev-test"), encoder=ScriptedEncoder(vectors))


# rankers


def _ranking_arbiter() -> Arbiter:
    # target on the x-axis; each candidate's cos to it is just its x component (unit vectors).
    return arbiter(
        "stub/rank",
        {
            "nurse": np.array([1.0, 0.0]),
            "hospital": np.array([1.0, 0.0]),  # cos 1.0
            "ward": np.array([0.8, 0.6]),  # cos 0.8
            "clinic": np.array([0.6, 0.8]),  # cos 0.6
            "dress": np.array([0.0, 1.0]),  # cos 0.0
        },
    )


def test_rank_neutral_bridges_exact_top_k_order() -> None:
    phi = _ranking_arbiter()
    target = make_word("nurse")
    candidates = [make_word(t)
                  for t in ("clinic", "dress", "hospital", "ward")]

    ranked = rank_neutral_bridges(target, candidates, phi, k=3)

    assert [w.text for w, _ in ranked] == ["hospital", "ward", "clinic"]
    assert [cos for _, cos in ranked] == pytest.approx([1.0, 0.8, 0.6])


def test_rank_stereotypical_bridges_exact_top_k_order() -> None:
    # Same shared _rank logic; the only difference is the candidate pool the caller passes.
    phi = _ranking_arbiter()
    target = make_word("nurse")
    candidates = [make_word(t)
                  for t in ("clinic", "dress", "hospital", "ward")]

    ranked = rank_stereotypical_bridges(target, candidates, phi, k=2)

    assert [w.text for w, _ in ranked] == ["hospital", "ward"]
    assert [cos for _, cos in ranked] == pytest.approx([1.0, 0.8])


def test_rank_excludes_the_target_itself() -> None:
    phi = _ranking_arbiter()
    target = make_word("nurse")
    candidates = [make_word(t) for t in ("nurse", "ward", "clinic")]

    ranked = rank_neutral_bridges(target, candidates, phi, k=5)

    assert [w.text for w, _ in ranked] == ["ward", "clinic"]


def test_rank_tie_break_is_text_ascending() -> None:
    # alpha and beta both have x=0.6 => identical cos to the target; order must be by text.
    phi = arbiter(
        "stub/tie",
        {
            "nurse": np.array([1.0, 0.0]),
            "beta": np.array([0.6, 0.8]),
            "alpha": np.array([0.6, -0.8]),
        },
    )
    target = make_word("nurse")
    candidates = [make_word("beta"), make_word("alpha")]

    ranked = rank_neutral_bridges(target, candidates, phi, k=2)

    assert [w.text for w, _ in ranked] == ["alpha", "beta"]
    assert ranked[0][1] == pytest.approx(ranked[1][1])


# verify_eq_4_1


def _agreeing_arbiter(model_id: str) -> Arbiter:
    # c_n (0.8) > c_s (0.6) => satisfies Eq. 4.1.
    return arbiter(
        model_id,
        {
            "nurse": np.array([1.0, 0.0]),
            "hospital": np.array([0.8, 0.6]),
            "dress": np.array([0.6, 0.8]),
        },
    )


def _dissenting_arbiter(model_id: str) -> Arbiter:
    # c_n (0.5) < c_s (0.9) => violates Eq. 4.1.
    return arbiter(
        model_id,
        {
            "nurse": np.array([1.0, 0.0]),
            "hospital": np.array([0.5, np.sqrt(0.75)]),
            "dress": np.array([0.9, np.sqrt(1.0 - 0.81)]),
        },
    )


def test_eq_4_1_holds_under_all_arbiters() -> None:
    consensus = [_agreeing_arbiter("stub/a"), _agreeing_arbiter("stub/b")]
    dilemma = verify_eq_4_1(
        make_word("nurse"), make_word(
            "hospital"), make_word("dress"), consensus
    )

    assert dilemma.consensus_ok is True
    assert all(score.satisfies_eq_4_1 for score in dilemma.arbiter_scores)
    assert dilemma.target == "nurse"
    assert dilemma.neutral_bridge == "hospital"
    assert dilemma.stereotypical_bridge == "dress"


def test_eq_4_1_one_dissenter_fails_consensus() -> None:
    # Consensus is the intersection; a single dissenter sinks the whole triple.
    consensus = [
        _agreeing_arbiter("stub/a"),
        _dissenting_arbiter("stub/dissent"),
        _agreeing_arbiter("stub/b"),
    ]
    dilemma = verify_eq_4_1(
        make_word("nurse"), make_word(
            "hospital"), make_word("dress"), consensus
    )

    assert dilemma.consensus_ok is False
    flags = {score.arbiter: score.satisfies_eq_4_1 for score in dilemma.arbiter_scores}
    assert flags == {
        "stub/a@rev-test": True,
        "stub/dissent@rev-test": False,
        "stub/b@rev-test": True,
    }


def test_eq_4_1_exact_tie_passes() -> None:
    # c_n == c_s (both 0.7): the non-strict '>=' accepts a tie, with no margin.
    tie = arbiter(
        "stub/tie",
        {
            "nurse": np.array([1.0, 0.0]),
            "hospital": np.array([0.7, np.sqrt(0.51)]),
            "dress": np.array([0.7, -np.sqrt(0.51)]),
        },
    )
    dilemma = verify_eq_4_1(make_word("nurse"), make_word(
        "hospital"), make_word("dress"), [tie])

    score = dilemma.arbiter_scores[0]
    assert score.cos_target_neutral == pytest.approx(score.cos_target_stereo)
    assert score.satisfies_eq_4_1 is True
    assert dilemma.consensus_ok is True


def test_eq_4_1_records_arbiter_id_and_unrounded_cosines() -> None:
    phi = _agreeing_arbiter("stub/a")
    dilemma = verify_eq_4_1(make_word("nurse"), make_word(
        "hospital"), make_word("dress"), [phi])

    score = dilemma.arbiter_scores[0]
    assert score.arbiter == str(phi.ref) == "stub/a@rev-test"
    # Stored at full float64 precision: byte-identical to a direct re-computation, no rounding.
    target_vec = phi.embed("nurse")
    assert score.cos_target_neutral == phi.cos(
        target_vec, phi.embed("hospital"))
    assert score.cos_target_stereo == phi.cos(target_vec, phi.embed("dress"))
