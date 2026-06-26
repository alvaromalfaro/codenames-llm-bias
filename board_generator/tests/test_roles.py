"""Role<->gender independence and the key-card audit block.

Descriptive Cramér's V (threshold 0.3), not a significance test (n=25, tiny cells).
Offline and deterministic.
"""

from __future__ import annotations

import random

from board_generator.lexicon import GenderCategory, Word
from board_generator.roles import (
    KeyCard,
    _cramers_v,
    assign_roles,
    build_keycard_audit,
    check_role_gender_independence,
    validate_keycard,
)


def _word(text: str, gender: GenderCategory) -> Word:
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


def test_cramers_v_undefined_for_single_gender_level() -> None:
    # All neutral -> one observed gender level -> undefined (guard: no NaN, no div-by-0).
    keycard = assign_roles(random.Random(1))
    genders = ["neutral"] * 25
    assert _cramers_v(keycard.role_a, genders) is None


def _gender_tracks_role() -> list[Word]:
    # legal_keycard.role_a is agent(0-8), bystander(9-21), assassin(22-24). Make gender track role
    # exactly (males = agents, females = everything else): a high Cramér's V on side A.
    words = [_word(f"w{i:02d}", "male") for i in range(9)]
    words += [_word(f"w{i:02d}", "female") for i in range(9, 25)]
    return words


def test_independence_fires_on_strong_association(legal_keycard: KeyCard) -> None:
    words = _gender_tracks_role()
    v_a = _cramers_v(legal_keycard.role_a, [w.gender_category for w in words])
    assert v_a is not None and v_a >= 0.3
    assert check_role_gender_independence(legal_keycard, words) is False


def test_independence_holds_when_gender_balanced(legal_keycard: KeyCard) -> None:
    # Gender assigned by index parity, uncorrelated with the role blocks -> low V on both faces.
    words = [_word(f"w{i:02d}", "male" if i % 2 == 0 else "female")
             for i in range(25)]
    assert check_role_gender_independence(legal_keycard, words) is True


def test_control_all_neutral_is_vacuously_independent(legal_keycard: KeyCard) -> None:
    words = [_word(f"w{i:02d}", "neutral") for i in range(25)]
    assert check_role_gender_independence(legal_keycard, words) is True


def test_build_keycard_audit_reports_legal_card() -> None:
    keycard = assign_roles(random.Random(424242))
    words = [_word(f"w{i:02d}", "neutral") for i in range(25)]
    audit = build_keycard_audit(keycard, words)
    assert audit.per_perspective == {
        "agent": 9, "bystander": 13, "assassin": 3}
    assert audit.overlap_ok is True
    assert audit.role_gender_independent is True
    assert validate_keycard(keycard) is True


def test_build_keycard_audit_flags_dependence(legal_keycard: KeyCard) -> None:
    words = _gender_tracks_role()
    audit = build_keycard_audit(legal_keycard, words)
    assert audit.overlap_ok is True
    assert audit.role_gender_independent is False


def _words_from_genders(genders: list[GenderCategory]) -> list[Word]:
    """25 position-aligned words carrying the given genders (text unique per index)."""
    return [_word(f"w{i:02d}", g) for i, g in enumerate(genders)]


def test_excluding_forced_dilemma_clears_spurious_dependence(legal_keycard: KeyCard) -> None:
    # legal_keycard role_b agents sit at {0,1,2,9,10,11,12,13,22}. Put the 3 forced dilemma words
    # (a distinct loaded gender, female) on role_b-agent positions 0,1,2; the 22 non-dilemma words
    # are a gender-blind male/neutral mix. Including the forced female-on-agent triple tips V over
    # the threshold; excluding it (the 22-word view) leaves the assignment independent.
    genders: list[GenderCategory] = ["female", "female", "female"]
    genders += ["male" if (i - 3) %
                2 == 0 else "neutral" for i in range(3, 25)]
    words = _words_from_genders(genders)
    dilemma_words = words[:3]

    # Forced dilemma included -> spurious association fires.
    assert check_role_gender_independence(legal_keycard, words) is False
    # Forced dilemma excluded -> the non-forced 22-word assignment is independent.
    assert (
        check_role_gender_independence(
            legal_keycard, words, [w.text for w in dilemma_words]
        )
        is True
    )
    audit = build_keycard_audit(legal_keycard, words, dilemma_words)
    assert audit.role_gender_independent is True


def test_exclusion_does_not_whitewash_real_dependence(legal_keycard: KeyCard) -> None:
    # Same forced dilemma on agent positions, but now the 22 NON-dilemma words genuinely track
    # role_b (female on every remaining agent position, male elsewhere). Excluding the dilemma must
    # not hide that real association.
    agents = {0, 1, 2, 9, 10, 11, 12, 13, 22}
    genders: list[GenderCategory] = [
        "female" if i in agents else "male" for i in range(25)
    ]
    words = _words_from_genders(genders)
    dilemma_words = words[:3]
    audit = build_keycard_audit(legal_keycard, words, dilemma_words)
    assert audit.role_gender_independent is False
