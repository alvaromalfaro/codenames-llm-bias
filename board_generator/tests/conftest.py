"""Shared, hand-built fixtures for the invariant tests."""

from __future__ import annotations

import pytest

from board_generator.arbiter import ArbiterRef, ConsensusSpec
from board_generator.board import ArbiterSet, Board, Grid, WordEntry
from board_generator.roles import EXPECTED_JOINT, KeyCard, KeycardAudit, Role


@pytest.fixture
def legal_keycard() -> KeyCard:
    """A double-sided key card reproducing the exact overlap scheme."""
    pairs: list[tuple[Role, Role]] = []
    for (role_a, role_b), count in EXPECTED_JOINT.items():
        pairs.extend([(role_a, role_b)] * count)
    return KeyCard(
        role_a=tuple(pair[0] for pair in pairs),
        role_b=tuple(pair[1] for pair in pairs),
    )


@pytest.fixture
def illegal_keycard(legal_keycard: KeyCard) -> KeyCard:
    """A key card that breaks perspective A's agent count (flips one agent -> bystander)."""
    role_a = list(legal_keycard.role_a)
    role_a[role_a.index("agent")] = "bystander"
    return KeyCard(role_a=tuple(role_a), role_b=legal_keycard.role_b)


@pytest.fixture
def control_board(legal_keycard: KeyCard) -> Board:
    """A legal 25-word control board (all neutral words, no dilemma)."""
    words = [
        WordEntry(
            text=f"WORD{i:02d}",
            index=i,
            role_a=legal_keycard.role_a[i],
            role_b=legal_keycard.role_b[i],
            gender_category="neutral",
            source="test",
            covariates={"subtlex_freq": 1.0,
                        "length": 6.0, "wordnet_polysemy": 1.0},
        )
        for i in range(25)
    ]
    return Board(
        board_id="control-career-000",
        type="control",
        specification="gender-career",
        seed=1234567,
        arbiters=ArbiterSet(consensus=["m@r"], primary="m@r"),
        grid=Grid(),
        words=words,
        dilemma=None,
        keycard_audit=KeycardAudit(
            per_perspective={"agent": 9, "bystander": 13, "assassin": 3},
            overlap_ok=True,
            role_gender_independent=True,
        ),
    )


@pytest.fixture
def valid_consensus_spec() -> ConsensusSpec:
    """A valid prefix-free, distinct-lineage consensus with φ* in the set."""
    primary = ArbiterRef("sentence-transformers/all-mpnet-base-v2", "rev-a")
    other = ArbiterRef("Alibaba-NLP/gte-large-en-v1.5", "rev-b")
    return ConsensusSpec(encoders=(primary, other), primary=primary)
