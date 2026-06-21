"""Placeholders for the invariants that need the generators.

I-6's consensus GATE logic (verify_eq_4_1) is implemented, so its test is real below. The
remaining placeholders still depend on bank/board assembly that does not exist yet.
"""

from __future__ import annotations

import numpy as np
import pytest

from board_generator.arbiter import Arbiter, ArbiterRef
from board_generator.dilemma import verify_eq_4_1
from board_generator.lexicon import Word

from ._stub_encoders import ScriptedEncoder


@pytest.mark.skip(reason="Bank assembly (T=30, ~50/50 control/probe) not implemented")
def test_i1_bank_size_and_split() -> None: ...


@pytest.mark.skip(reason="Lexical composition (control=neutral, probe=valid dilemma) pending")
def test_i4_lexical_composition() -> None: ...


@pytest.mark.skip(reason="Role/gender independence test pending (see roles.py)")
def test_i5_role_gender_independence() -> None: ...


def _word(text: str) -> Word:
    return Word(
        text=text,
        gender_category="neutral",
        word_kind="common",
        source="test",
        weat_set=("weat-6",),
        dom_pos=None,
        ambiguous_pos=False,
        covariates={"subtlex_freq": 1.0, "length": float(
            len(text)), "wordnet_polysemy": 1.0},
        specification="gender-career",
    )


def _arbiter(model_id: str, neutral_x: float, stereo_x: float) -> Arbiter:
    """Arbiter whose cos(target, neutral)=neutral_x and cos(target, stereo)=stereo_x.

    The target sits on the x-axis, so each unit vector's cos to it is just its x component.
    """
    vectors = {
        "nurse": np.array([1.0, 0.0]),
        "hospital": np.array([neutral_x, np.sqrt(1.0 - neutral_x**2)]),
        "dress": np.array([stereo_x, np.sqrt(1.0 - stereo_x**2)]),
    }
    return Arbiter(ref=ArbiterRef(model_id, "rev-test"), encoder=ScriptedEncoder(vectors))


def test_i6_dilemma_consensus() -> None:
    """GATE logic: Eq. 4.1 must hold under EVERY consensus arbiter (intersection).

    This covers verify_eq_4_1's consensus AND. The BANK-LEVEL quantification (every assembled
    probe board carries a dilemma with consensus_ok == true) is deferred to the future board/cli
    increment, once boards can be assembled.
    """
    target, neutral, stereo = _word("nurse"), _word("hospital"), _word("dress")

    # Eq. 4.1 holds under all arbiters (c_n >= c_s each) -> accepted.
    holds = [_arbiter("stub/a", 0.8, 0.6), _arbiter("stub/b", 0.7, 0.7)]
    assert verify_eq_4_1(target, neutral, stereo, holds).consensus_ok is True

    # One dissenter (c_n < c_s) sinks the consensus even though the other passes.
    dissent = [_arbiter("stub/a", 0.8, 0.6),
               _arbiter("stub/dissent", 0.4, 0.9)]
    assert verify_eq_4_1(target, neutral, stereo,
                         dissent).consensus_ok is False


@pytest.mark.skip(reason="Byte-for-byte reproducibility pending (full generation flow)")
def test_i8_reproducible_bank() -> None: ...
