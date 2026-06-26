"""Double-sided key-card role assignment and validation.

Per perspective the 25-word grid carries 9 agents / 13 bystanders / 3 assassins, with the fixed
overlap scheme (game totals 15 / 19 / 5). Role assignment must be statistically independent of
gender_category.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
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

    The role pattern is built without reference to any word, so role is independent of
    gender_category by construction: words are placed onto positions separately
    (board.randomize_positions). Randomness derives solely from rng (seeded from the board seed), so
    the same seed yields the same key card. The scheme fixes the grid to 25 words, so there is no
    width parameter.
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


def _cramers_v(roles: Sequence[Role], genders: Sequence[str]) -> float | None:
    """Cramér's V for one role x gender_category contingency table, or None if undefined.

    Builds the table over the observed role and gender levels only. With fewer than two observed
    levels on either axis the association is undefined (min(r-1, c-1) == 0): return None rather than
    divide by zero or emit a NaN. Every used cell has a positive expected count because both its row
    and column levels are observed.
    """
    role_levels = sorted(set(roles))
    gender_levels = sorted(set(genders))
    r, c = len(role_levels), len(gender_levels)
    if r < 2 or c < 2:
        return None

    n = len(roles)
    observed: dict[tuple[str, str], int] = {}
    row_totals: dict[str, int] = {level: 0 for level in role_levels}
    col_totals: dict[str, int] = {level: 0 for level in gender_levels}
    for role, gender in zip(roles, genders, strict=True):
        observed[(role, gender)] = observed.get((role, gender), 0) + 1
        row_totals[role] += 1
        col_totals[gender] += 1

    chi2 = 0.0
    for role in role_levels:
        for gender in gender_levels:
            expected = row_totals[role] * col_totals[gender] / n
            diff = observed.get((role, gender), 0) - expected
            chi2 += diff * diff / expected
    return math.sqrt((chi2 / n) / min(r - 1, c - 1))


def check_role_gender_independence(
    keycard: KeyCard, words: list[Word], exclude: Sequence[str] | None = None
) -> bool:
    """Descriptively check role independence of gender_category for BOTH perspectives.

    Descriptive, not a significance test: at n=25 with tiny cells a chi-square would be underpowered
    and misleading, consistent with the SMD decision in Stage A. Computes Cramér's V on the
    role x gender_category table for each face and treats a face as independent when V < 0.3 (or
    when V is undefined - e.g. a control board's single neutral gender level - in which case it is
    vacuously independent). Returns True iff both faces are independent.

    words are position-aligned with the key card: keycard.role_a[i] / role_b[i] pairs with
    words[i].gender_category.

    exclude is a set of word texts to drop before computing V - the probe board's 3 dilemma words.
    Those words are FORCED onto LLM-agent positions by construction (and 2 of the 3 share the same
    loaded gender), so including them injects a role<->gender association we introduced
    deliberately, not a randomization failure; the diagnostic must reflect the non-forced 22 words.
    The same indices are dropped from BOTH the role sequence and the gender sequence, on both faces,
    so position alignment is preserved. Control boards pass nothing to exclude.

    NOTE: Cramér's V is upward-biased with small expected cell counts (n~=25, up to a 3x3 table = 9
    cells, ~2.8 per cell), so raw V overestimates association. The 0.3 threshold is deliberately
    generous to absorb that bias, and this diagnostic is descriptive (not a gate). A bias-corrected
    V is a possible future refinement; it is not implemented here.
    """
    exclude_set = set(exclude or ())
    kept = [
        (role_a, role_b, w.gender_category)
        for role_a, role_b, w in zip(keycard.role_a, keycard.role_b, words, strict=True)
        if w.text not in exclude_set
    ]
    roles_a = [row[0] for row in kept]
    roles_b = [row[1] for row in kept]
    genders = [row[2] for row in kept]
    v_a = _cramers_v(roles_a, genders)
    v_b = _cramers_v(roles_b, genders)
    independent_a = v_a is None or v_a < 0.3
    independent_b = v_b is None or v_b < 0.3
    return independent_a and independent_b


def build_keycard_audit(
    keycard: KeyCard, words: list[Word], dilemma_words: Sequence[Word] | None = None
) -> KeycardAudit:
    """Assemble the keycard_audit block.

    words are position-aligned with the key card (placed order). per_perspective is the shared
    9/13/3 count (both faces are identical on a legal card).

    dilemma_words (probe boards only) are excluded from the role<->gender independence diagnostic by
    text, so role_gender_independent reflects the non-forced 22-word assignment. Control boards pass
    nothing.
    """
    counts_a, _counts_b = count_roles(keycard)
    exclude = [w.text for w in dilemma_words] if dilemma_words else None
    return KeycardAudit(
        per_perspective=counts_a,
        overlap_ok=validate_keycard(keycard),
        role_gender_independent=check_role_gender_independence(
            keycard, words, exclude),
    )
