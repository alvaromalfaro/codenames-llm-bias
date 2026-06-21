"""Per perspective 9 agents / 13 bystanders / 3 assassins with the overlap."""

from __future__ import annotations

import random

from board_generator.roles import (
    EXPECTED_JOINT,
    PER_PERSPECTIVE_COUNTS,
    KeyCard,
    Role,
    assign_roles,
    count_roles,
    joint_counts,
    validate_keycard,
)


def _keycard_from_joint(joint: dict[tuple[Role, Role], int]) -> KeyCard:
    """Expand a (role_a, role_b) cross-tab into a KeyCard (as the legal fixture does)."""
    pairs: list[tuple[Role, Role]] = []
    for (role_a, role_b), count in joint.items():
        pairs.extend([(role_a, role_b)] * count)
    return KeyCard(
        role_a=tuple(pair[0] for pair in pairs),
        role_b=tuple(pair[1] for pair in pairs),
    )


def test_i3_legal_keycard_passes(legal_keycard: KeyCard) -> None:
    assert validate_keycard(legal_keycard) is True


def test_i3_per_perspective_counts(legal_keycard: KeyCard) -> None:
    counts_a, counts_b = count_roles(legal_keycard)
    assert counts_a == PER_PERSPECTIVE_COUNTS
    assert counts_b == PER_PERSPECTIVE_COUNTS


def test_i3_overlap_matches_scheme(legal_keycard: KeyCard) -> None:
    assert joint_counts(legal_keycard) == EXPECTED_JOINT


def test_i3_rejects_illegal_keycard(illegal_keycard: KeyCard) -> None:
    assert validate_keycard(illegal_keycard) is False


def test_i3_rejects_wrong_overlap() -> None:
    # Correct 9/13/3 marginals on BOTH sides but a wrong joint: a +1/-1 rotation on the
    # agent/bystander 2x2 submatrix gives 4 shared agents. Marginals are preserved, so  only the
    # overlap branch (joint == EXPECTED_JOINT) can reject it.
    joint = dict(EXPECTED_JOINT)
    joint[("agent", "agent")] = 4
    joint[("agent", "bystander")] = 4
    joint[("bystander", "agent")] = 4
    joint[("bystander", "bystander")] = 8
    keycard = _keycard_from_joint(joint)

    counts_a, counts_b = count_roles(keycard)
    assert counts_a == PER_PERSPECTIVE_COUNTS  # marginal check would pass...
    assert counts_b == PER_PERSPECTIVE_COUNTS
    # ...but the overlap is wrong
    assert joint_counts(keycard) != EXPECTED_JOINT
    assert validate_keycard(keycard) is False


def test_i3_assign_roles_generates_valid() -> None:
    keycard = assign_roles(random.Random(1234567))
    assert validate_keycard(keycard) is True
    counts_a, counts_b = count_roles(keycard)
    assert counts_a == PER_PERSPECTIVE_COUNTS
    assert counts_b == PER_PERSPECTIVE_COUNTS


def test_i3_assign_roles_deterministic() -> None:
    # Same seed => same key card; different seeds => different key cards (guards against
    # assign_roles ignoring rng and always returning one permutation).
    same_a = assign_roles(random.Random(42))
    same_b = assign_roles(random.Random(42))
    assert same_a == same_b
    other = assign_roles(random.Random(43))
    assert other != same_a
