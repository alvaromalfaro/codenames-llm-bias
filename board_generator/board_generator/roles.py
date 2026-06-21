"""Double-sided key-card role assignment and validation.

Per perspective the 25-word grid carries 9 agents / 13 bystanders / 3 assassins, with the fixed
overlap scheme (game totals 15 / 19 / 5). Role assignment must be statistically independent of
gender_category.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal

from board_generator.lexicon import Word

# A role on one perspective's side of the double-sided key card.
Role = Literal["agent", "bystander", "assassin"]

# Per-perspective role counts over the 25-word grid.
PER_PERSPECTIVE_COUNTS: dict[Role, int] = {
    "agent": 9, "bystander": 13, "assassin": 3}

# Game totals across both perspectives.
GAME_TOTALS: dict[Role, int] = {"agent": 15, "bystander": 19, "assassin": 5}

# The fixed overlap scheme as a joint cross-tabulation of (role_a, role_b) counts.
# Row sums give perspective A's marginals, column sums give B's; both are 9/13/3, the total is 25,
# and the game totals (15/19/5) follow. This is the exact overlap a legal key card must reproduce.
EXPECTED_JOINT: dict[tuple[Role, Role], int] = {
    ("agent", "agent"): 3,  # shared agents
    ("agent", "bystander"): 5,  # A-only agents that are bystanders for B
    ("agent", "assassin"): 1,  # A-only agent that is an assassin for B
    ("bystander", "agent"): 5,  # B-only agents that are bystanders for A
    ("bystander", "bystander"): 7,  # shared bystanders
    ("bystander", "assassin"): 1,  # A bystander that is an assassin for B
    ("assassin", "agent"): 1,  # A assassin that is an agent for B
    ("assassin", "bystander"): 1,  # A assassin that is a bystander for B
    ("assassin", "assassin"): 1,  # shared assassin
}


@dataclass(frozen=True, slots=True)
class KeyCard:
    """Roles seen by perspectives A and B for each of the 25 grid positions."""

    role_a: tuple[Role, ...]
    role_b: tuple[Role, ...]


@dataclass(frozen=True, slots=True)
class KeycardAudit:
    """Key-card legality audit."""

    per_perspective: dict[Role, int]
    overlap_ok: bool
    role_gender_independent: bool


def assign_roles(rng: random.Random) -> KeyCard:
    """Assign a legal double-sided key card, independent of gender_category.

    The 25 (role_a, role_b) pairs of the fixed overlap scheme (EXPECTED_JOINT) are materialized in
    deterministic insertion order, then permuted across grid positions. Shuffling reorders positions
    only, so the joint cross-tabulation is preserved exactly and validate_keycard stays the oracle.

    The role pattern is built without reference to any word, so role is independent of gender_category
    by construction: words are placed onto positions separately (board.randomize_positions).
    Randomness derives solely from rng (seeded from the board seed), so the same seed yields the
    same key card. The scheme fixes the grid to 25 words, so there is no width parameter.
    """
    pairs: list[tuple[Role, Role]] = []
    for (role_a, role_b), count in EXPECTED_JOINT.items():
        pairs.extend([(role_a, role_b)] * count)
    rng.shuffle(pairs)
    return KeyCard(
        role_a=tuple(pair[0] for pair in pairs),
        role_b=tuple(pair[1] for pair in pairs),
    )


def count_roles(keycard: KeyCard) -> tuple[dict[Role, int], dict[Role, int]]:
    """Return the per-perspective role counts for sides A and B."""

    def _counts(roles: tuple[Role, ...]) -> dict[Role, int]:
        out: dict[Role, int] = {"agent": 0, "bystander": 0, "assassin": 0}
        for role in roles:
            out[role] += 1
        return out

    return _counts(keycard.role_a), _counts(keycard.role_b)


def joint_counts(keycard: KeyCard) -> dict[tuple[Role, Role], int]:
    """Return the (role_a, role_b) cross-tabulation of the key card."""
    out: dict[tuple[Role, Role], int] = {}
    for ra, rb in zip(keycard.role_a, keycard.role_b, strict=True):
        out[(ra, rb)] = out.get((ra, rb), 0) + 1
    return out


def validate_keycard(keycard: KeyCard) -> bool:
    """Verify the exact counts and overlap scheme."""
    grid_size = sum(EXPECTED_JOINT.values())  # 25
    if len(keycard.role_a) != grid_size or len(keycard.role_b) != grid_size:
        return False
    counts_a, counts_b = count_roles(keycard)
    if counts_a != PER_PERSPECTIVE_COUNTS or counts_b != PER_PERSPECTIVE_COUNTS:
        return False
    return joint_counts(keycard) == EXPECTED_JOINT


def check_role_gender_independence(keycard: KeyCard, words: list[Word]) -> bool:
    """Test that role is statistically independent of gender_category."""
    raise NotImplementedError


def build_keycard_audit(keycard: KeyCard, words: list[Word]) -> KeycardAudit:
    """Assemble the keycard_audit block."""
    raise NotImplementedError
