"""Board assembly: position randomization + Board object.

Covers placement (dilemma words on LLM-agent positions, gender-blind fill), gender-blindness of the
non-dilemma fill, determinism, key-card legality preservation, and covariate pass-through. Offline
and deterministic - no φ*/HF/network.
"""

from __future__ import annotations

import dataclasses
import random

import pytest

from board_generator.arbiter import ArbiterRef, ConsensusSpec
from board_generator.board import (
    Dilemma,
    WordEntry,
    assemble_board,
    randomize_positions,
    validate_board_grid,
)
from board_generator.lexicon import GenderCategory, Word
from board_generator.roles import (
    EXPECTED_JOINT,
    KeyCard,
    assign_roles,
    count_roles,
    joint_counts,
    validate_keycard,
)


def _word(text: str, gender: GenderCategory = "neutral", subtlex: float | None = 1.0) -> Word:
    """A board-eligible Word; subtlex=None models an OOV neutral."""
    return Word(
        text=text,
        gender_category=gender,
        word_kind="common",
        source="test",
        weat_set=("weat-6",),
        dom_pos=None,
        ambiguous_pos=False,
        covariates={"subtlex_freq": subtlex, "length": float(
            len(text)), "wordnet_polysemy": 1.0},
        specification="gender-career",
    )


def _consensus() -> ConsensusSpec:
    primary = ArbiterRef("sentence-transformers/all-mpnet-base-v2", "rev-a")
    other = ArbiterRef("Alibaba-NLP/gte-large-en-v1.5", "rev-b")
    return ConsensusSpec(encoders=(primary, other), primary=primary)


def _probe_words() -> tuple[list[Word], Dilemma]:
    """25 words (3 dilemma + 6 male + 6 female + 10 neutral) and a matching Dilemma block."""
    target = _word("nurse", "female")
    stereo = _word("dress", "female")
    neutral_bridge = _word("hospital", "neutral")
    loaded = [_word(f"male{i:02d}", "male") for i in range(6)]
    loaded += [_word(f"female{i:02d}", "female") for i in range(6)]
    neutral = [_word(f"neutral{i:02d}", "neutral") for i in range(10)]
    words = [target, stereo, neutral_bridge, *loaded, *neutral]
    assert len(words) == 25
    dilemma = Dilemma(
        target="nurse",
        neutral_bridge="hospital",
        stereotypical_bridge="dress",
        consensus_ok=True,
        arbiter_scores=[],
    )
    return words, dilemma


def _control_words() -> list[Word]:
    return [_word(f"word{i:02d}", "neutral") for i in range(25)]


def test_probe_dilemma_words_on_llm_agent_positions() -> None:
    words, dilemma = _probe_words()
    board = assemble_board(
        "probe-career-000", "probe", "gender-career", 20260626,
        words, _consensus(), dilemma,
    )

    dilemma_texts = {"nurse", "dress", "hospital"}
    for entry in board.words:
        if entry.text in dilemma_texts:
            assert entry.role_b == "agent"

    assert validate_board_grid(board) is True
    assert {e.text for e in board.words} == {w.text for w in words}
    assert sorted(e.index for e in board.words) == list(range(25))


def test_control_board_places_all_without_reserved_positions() -> None:
    words = _control_words()
    board = assemble_board(
        "control-career-000", "control", "gender-career", 7, words, _consensus(), None
    )
    assert validate_board_grid(board) is True
    assert board.dilemma is None
    assert {e.text for e in board.words} == {w.text for w in words}


def test_randomize_positions_aligns_with_keycard_roles() -> None:
    words, dilemma = _probe_words()
    keycard = assign_roles(random.Random(99))
    dilemma_words = [words[0], words[1], words[2]]  # nurse, dress, hospital
    placed = randomize_positions(
        words, keycard, dilemma_words, random.Random(99))
    assert len(placed) == 25
    assert len({w.text for w in placed}) == 25
    for i, word in enumerate(placed):
        if word.text in {"nurse", "dress", "hospital"}:
            assert keycard.role_b[i] == "agent"


def test_i5_fill_is_gender_blind_on_average() -> None:
    # Place 25 mixed-gender words with NO dilemma (pure gender-blind shuffle) across many seeds;
    # the position-role x gender association (Cramér's V on role_b) should stay low on average.
    from board_generator.roles import _cramers_v

    words = [_word(f"m{i}", "male") for i in range(9)]
    words += [_word(f"f{i}", "female") for i in range(8)]
    words += [_word(f"n{i}", "neutral") for i in range(8)]
    assert len(words) == 25
    keycard = assign_roles(random.Random(1))

    vs: list[float] = []
    for seed in range(200):
        placed = randomize_positions(words, keycard, None, random.Random(seed))
        v = _cramers_v(keycard.role_b, [w.gender_category for w in placed])
        assert v is not None
        vs.append(v)
    assert sum(vs) / len(vs) < 0.3


def test_i8_same_seed_byte_identical() -> None:
    words, dilemma = _probe_words()
    board_a = assemble_board(
        "probe-career-000", "probe", "gender-career", 555, words, _consensus(), dilemma
    )
    board_b = assemble_board(
        "probe-career-000", "probe", "gender-career", 555, words, _consensus(), dilemma
    )
    assert board_a == board_b


def test_i8_different_seed_changes_placement() -> None:
    words, _dilemma = _probe_words()
    # The key card is seed-derived too, so vary both via the seed the caller would use.
    placed_a = randomize_positions(
        words, assign_roles(random.Random(1)), words[:3], random.Random(1)
    )
    placed_b = randomize_positions(
        words, assign_roles(random.Random(2)), words[:3], random.Random(2)
    )
    assert [w.text for w in placed_a] != [w.text for w in placed_b]


def test_i8_positions_depend_on_roles_consuming_first() -> None:
    # rng sequential: assign_roles consumes first, then randomize_positions draws from the advanced
    # state. So the real placement must differ from a "fresh-stream" placement where positions are
    # drawn from a brand-new Random(seed).
    words, dilemma = _probe_words()
    seed = 4242
    board = assemble_board(
        "probe-career-000", "probe", "gender-career", seed, words, _consensus(), dilemma
    )
    real_order = [e.text for e in sorted(board.words, key=lambda e: e.index)]

    keycard = assign_roles(random.Random(seed))
    # nurse, dress, hospital (assemble_board's fixed dilemma order).
    dilemma_words = [words[0], words[1], words[2]]
    fresh = randomize_positions(
        words, keycard, dilemma_words, random.Random(seed))
    fresh_order = [w.text for w in fresh]

    assert real_order != fresh_order


def test_keycard_legality_preserved_through_assembly() -> None:
    words, dilemma = _probe_words()
    # Reconstruct the key card assemble_board builds internally: assign_roles(Random(seed)) consumes
    # a fresh Random(seed) identically to the first segment of assemble_board's single stream.
    keycard = assign_roles(random.Random(31337))
    board = assemble_board(
        "probe-career-001", "probe", "gender-career", 31337, words, _consensus(), dilemma
    )
    assert validate_keycard(keycard) is True
    counts_a, counts_b = count_roles(keycard)
    assert counts_a == {"agent": 9, "bystander": 13, "assassin": 3}
    assert counts_b == {"agent": 9, "bystander": 13, "assassin": 3}
    assert joint_counts(keycard) == EXPECTED_JOINT
    assert board.keycard_audit.per_perspective == {
        "agent": 9, "bystander": 13, "assassin": 3}
    assert board.keycard_audit.overlap_ok is True


def test_oov_covariate_passes_through_without_imputation() -> None:
    # One neutral word is OOV (subtlex_freq=None).
    words = _control_words()
    words[3] = _word("oovword", "neutral", subtlex=None)
    board = assemble_board(
        "control-career-001", "control", "gender-career", 11, words, _consensus(), None
    )
    entry = next(e for e in board.words if e.text == "oovword")
    assert entry.covariates["subtlex_freq"] is None


def test_assemble_rejects_missing_dilemma_word() -> None:
    words = _control_words()
    dilemma = Dilemma(
        target="absent",
        neutral_bridge="word00",
        stereotypical_bridge="word01",
        consensus_ok=True,
        arbiter_scores=[],
    )
    with pytest.raises(ValueError, match="absent"):
        assemble_board(
            "probe-career-002", "probe", "gender-career", 5, words, _consensus(), dilemma
        )


def test_assemble_rejects_duplicate_word() -> None:
    words = _control_words()
    words[1] = dataclasses.replace(
        words[1], text=words[0].text)  # duplicate text
    with pytest.raises(ValueError, match="grid validation"):
        assemble_board(
            "control-career-003", "control", "gender-career", 6, words, _consensus(), None
        )


def test_word_entry_index_matches_grid_position() -> None:
    # WordEntry.index must equal its grid slot, so the (role_a, role_b) pairing is correct.
    words = _control_words()
    # Reconstruct assemble_board's internal key card (assign_roles consumes a fresh Random(seed)
    # identically to the first segment of assemble_board's single stream) to assert role alignment.
    keycard = assign_roles(random.Random(8))
    board = assemble_board(
        "control-career-004", "control", "gender-career", 8, words, _consensus(), None
    )
    for i, entry in enumerate(sorted(board.words, key=lambda e: e.index)):
        assert entry.index == i
        assert entry.role_a == keycard.role_a[i]
        assert entry.role_b == keycard.role_b[i]
        assert isinstance(entry, WordEntry)


def test_keycard_fixture_shape_matches() -> None:
    # Guard: a legal key card's two faces share the 9/13/3 marginal used by per_perspective.
    keycard: KeyCard = assign_roles(random.Random(123))
    counts_a, counts_b = count_roles(keycard)
    assert counts_a == counts_b
