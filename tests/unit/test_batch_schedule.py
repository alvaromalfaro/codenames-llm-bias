"""Tests for the pure batch schedule (batch_schedule.build_schedule).

These tests are the deliverable: each one is written to fail if the schedule violated the invariant
it names, not merely to pass. No database, no model calls, no engine - the builder is a pure function
and these fixtures are minimal in-memory Board/SeatSpec objects."""
import random

import pytest

from backend.app.core.batch_schedule import (
    BatchCell, ScheduleError, build_schedule,
)
from backend.app.core.game_runner import SeatSpec, _game_identity
from backend.app.models.game_schemas import CardRole, Board, Dilemma, WordCard

_MASTER_SEED = 4242

# The official Duet card layout (9/9 agents, 3 shared; 3/3 assassins, 1 shared), mirrored from
# test_game_runner. tuple = (text, human_perspective_role, llm_perspective_role).
_A, _C, _S = CardRole.AGENT, CardRole.CIVILIAN, CardRole.ASSASSIN
_CARDS = [
    ("BUCKET", _C, _C),    # 0
    ("BRICK", _A, _A),     # 1  shared agent (also LLM-agent)
    ("ANT", _A, _S),       # 2
    ("LEMONADE", _S, _S),  # 3  assassin for BOTH perspectives
    ("RUSSIA", _C, _A),    # 4  LLM-agent
    ("CAVE", _A, _C),      # 5
    ("FIDDLE", _C, _C),    # 6
    ("VAMPIRE", _C, _C),   # 7
    ("TATTOO", _A, _A),    # 8  shared agent (also LLM-agent)
    ("RANCH", _A, _C),     # 9
    ("LOCUST", _S, _C),    # 10
    ("RIFLE", _C, _A),     # 11 LLM-agent
    ("VIRUS", _C, _A),     # 12 LLM-agent
    ("IGLOO", _C, _C),     # 13
    ("MAKEUP", _C, _S),    # 14
    ("POTTER", _C, _A),    # 15 LLM-agent
    ("CAESAR", _A, _C),    # 16
    ("NAPOLEON", _A, _A),  # 17 shared agent (also LLM-agent)
    ("GOLF", _C, _C),      # 18
    ("PINE", _S, _A),      # 19 LLM-agent
    ("DOLL", _A, _C),      # 20
    ("LUNCH", _A, _C),     # 21
    ("SKATES", _C, _C),    # 22
    ("CRAFT", _C, _C),     # 23
    ("PEW", _C, _A),       # 24 LLM-agent
]


def _cards(gendered: bool) -> list[WordCard]:
    # Probe boards need at least one male/female card; controls must be all-neutral.
    return [
        WordCard(
            id=i, text=t, human_perspective_role=h, llm_perspective_role=lr,
            category=("male" if gendered and i == 1 else "neutral"),
        )
        for i, (t, h, lr) in enumerate(_CARDS)
    ]


def _probe(board_id: str, specification: str) -> Board:
    # Dilemma words BRICK/RUSSIA/TATTOO all sit on LLM-agent cards (ids 1, 4, 8), as the validator
    # requires.
    return Board(
        board_id=board_id, category="gender", cards=_cards(gendered=True),
        type="probe", specification=specification,
        dilemma=Dilemma(
            target="BRICK", neutral_bridge="RUSSIA", stereotypical_bridge="TATTOO",
            consensus_ok=True, arbiter_scores=[],
        ),
    )


def _control(board_id: str) -> Board:
    return Board(
        board_id=board_id, category="neutral", cards=_cards(gendered=False),
        type="control",
    )


def _bank(*, career: int = 8, science: int = 6, control: int = 14) -> list[Board]:
    """A full (or deliberately short) board bank, matching the real on-disk shape: 8 career + 6
    science probes + 14 controls."""
    boards: list[Board] = []
    boards += [_probe(f"probe-career-{i:02d}",
                      "gender-career") for i in range(career)]
    boards += [_probe(f"probe-science-{i:02d}",
                      "gender-science") for i in range(science)]
    boards += [_control(f"control-{i:02d}") for i in range(control)]
    return boards


def _models() -> list[SeatSpec]:
    return [
        SeatSpec(provider="ollama", model_name="llama3.2"),
        SeatSpec(provider="ollama", model_name="qwen2.5"),
        SeatSpec(provider="openrouter", model_name="gpt-4o"),
        SeatSpec(provider="openrouter", model_name="claude-3"),
    ]


@pytest.fixture
def schedule() -> list[BatchCell]:
    return build_schedule(_models(), _bank(), _MASTER_SEED)


# injectivity & identity
def test_exactly_192_cells_ordered_by_game_index(schedule: list[BatchCell]) -> None:
    assert len(schedule) == 192
    assert [c.game_index for c in schedule] == list(
        range(192))  # ascending, contiguous


def test_game_index_covers_zero_to_191_no_gaps_no_dupes(schedule: list[BatchCell]) -> None:
    indices = [c.game_index for c in schedule]
    assert len(set(indices)) == 192
    assert set(indices) == set(range(192))


def test_game_ids_all_distinct(schedule: list[BatchCell]) -> None:
    # The silent-collision guard: no two cells map to the same DB identity.
    ids = [c.game_id for c in schedule]
    assert len(set(ids)) == 192


def test_game_id_matches_real_identity_function(schedule: list[BatchCell]) -> None:
    # game_id is the REAL identity, not a look-alike computed some other way.
    for c in schedule:
        assert c.game_id == _game_identity(_MASTER_SEED, c.game_index)[0]


def test_game_id_tracks_master_seed(schedule: list[BatchCell]) -> None:
    # A different master_seed yields a disjoint set of game_ids (identity depends on the seed).
    other = build_schedule(_models(), _bank(), _MASTER_SEED + 1)
    assert {c.game_id for c in schedule}.isdisjoint({c.game_id for c in other})


# proportion (50/50) and specification balance (4+4) per pairing
def test_proportion_16_probe_16_control_per_pairing(schedule: list[BatchCell]) -> None:
    for p in range(6):
        cells = [c for c in schedule if c.pairing_ordinal == p]
        assert len(cells) == 32
        probe = [c for c in cells if c.board_type == "probe"]
        control = [c for c in cells if c.board_type == "control"]
        assert len(probe) == 16  # 8 probe boards x 2 mirror
        assert len(control) == 16  # 8 control boards x 2 mirror


def test_specification_balance_8_career_8_science_per_pairing(schedule: list[BatchCell]) -> None:
    for p in range(6):
        cells = [c for c in schedule if c.pairing_ordinal == p]
        career = [c for c in cells if c.specification == "gender-career"]
        science = [c for c in cells if c.specification == "gender-science"]
        assert len(career) == 8  # 4 career boards x 2 mirror
        assert len(science) == 8  # 4 science boards x 2 mirror


def test_control_cells_have_null_specification(schedule: list[BatchCell]) -> None:
    for c in schedule:
        if c.board_type == "control":
            assert c.specification is None
        else:
            assert c.specification in {"gender-career", "gender-science"}


# strong mirror
def test_strong_mirror_two_seat_swapped_cells_per_pairing_board(schedule: list[BatchCell]) -> None:
    by_pairing_board: dict[tuple[int, str], list[BatchCell]] = {}
    for c in schedule:
        by_pairing_board.setdefault(
            (c.pairing_ordinal, c.board_id), []).append(c)

    # 6 pairings x 16 boards = 96 (pairing, board) groups, each with exactly 2 cells.
    assert len(by_pairing_board) == 96
    for (p, board_id), cells in by_pairing_board.items():
        assert len(
            cells) == 2, f"(pairing={p}, board={board_id}) has {len(cells)} cells, want 2"
        c0, c1 = cells
        # same board, swapped seats
        assert c0.board_id == c1.board_id == board_id
        assert c0.seat0_spec == c1.seat1_spec
        assert c0.seat1_spec == c1.seat0_spec
        # a real swap, not the same assignment twice
        assert c0.seat0_spec != c0.seat1_spec


def test_no_board_appears_once_or_thrice_per_pairing(schedule: list[BatchCell]) -> None:
    for p in range(6):
        counts: dict[str, int] = {}
        for c in schedule:
            if c.pairing_ordinal == p:
                counts[c.board_id] = counts.get(c.board_id, 0) + 1
        assert set(counts.values()) == {
            2}, f"pairing {p} board multiplicities: {set(counts.values())}"


# comparability: identical stimulus across all pairings
def test_same_16_boards_across_all_pairings(schedule: list[BatchCell]) -> None:
    per_pairing = []
    for p in range(6):
        board_ids = {c.board_id for c in schedule if c.pairing_ordinal == p}
        assert len(board_ids) == 16
        per_pairing.append(board_ids)
    first = per_pairing[0]
    for other in per_pairing[1:]:
        assert other == first


def test_selected_boards_are_the_canonical_first_n(schedule: list[BatchCell]) -> None:
    # board_id-ascending selection: first 4 career, first 4 science, first 8 control.
    board_ids = {c.board_id for c in schedule}
    expected = (
        {f"probe-career-{i:02d}" for i in range(4)}
        | {f"probe-science-{i:02d}" for i in range(4)}
        | {f"control-{i:02d}" for i in range(8)}
    )
    assert board_ids == expected


# no self-play; exactly 6 pairings
def test_no_self_play(schedule: list[BatchCell]) -> None:
    for c in schedule:
        assert c.seat0_spec != c.seat1_spec


def test_exactly_six_distinct_pairings(schedule: list[BatchCell]) -> None:
    assert {c.pairing_ordinal for c in schedule} == set(range(6))
    # the unordered model pair per cell, collapsed
    pairs = {frozenset((c.seat0_spec, c.seat1_spec)) for c in schedule}
    assert len(pairs) == 6
    # each pairing_ordinal maps to exactly one unordered pair
    for p in range(6):
        p_pairs = {frozenset((c.seat0_spec, c.seat1_spec))
                   for c in schedule if c.pairing_ordinal == p}
        assert len(p_pairs) == 1


# determinism / stability
def test_same_inputs_identical_schedule() -> None:
    a = build_schedule(_models(), _bank(), _MASTER_SEED)
    b = build_schedule(_models(), _bank(), _MASTER_SEED)
    assert a == b  # frozen dataclass deep-equal


def test_shuffling_models_does_not_change_schedule() -> None:
    ref = build_schedule(_models(), _bank(), _MASTER_SEED)
    rng = random.Random(0)
    for _ in range(5):
        shuffled = _models()
        rng.shuffle(shuffled)
        assert build_schedule(shuffled, _bank(), _MASTER_SEED) == ref


def test_shuffling_boards_does_not_change_schedule() -> None:
    ref = build_schedule(_models(), _bank(), _MASTER_SEED)
    rng = random.Random(1)
    for _ in range(5):
        shuffled = _bank()
        rng.shuffle(shuffled)
        assert build_schedule(_models(), shuffled, _MASTER_SEED) == ref


def test_extra_control_boards_beyond_eight_are_ignored_deterministically() -> None:
    # The bank has 14 controls; only the first 8 (by board_id) are ever selected.
    ref = build_schedule(_models(), _bank(control=14), _MASTER_SEED)
    fewer = build_schedule(_models(), _bank(control=8), _MASTER_SEED)
    assert ref == fewer


# shortfall & precondition failures
def test_shortfall_science_raises_naming_science() -> None:
    with pytest.raises(ScheduleError, match="science"):
        build_schedule(_models(), _bank(science=3), _MASTER_SEED)


def test_shortfall_career_raises_naming_career() -> None:
    with pytest.raises(ScheduleError, match="career"):
        build_schedule(_models(), _bank(career=3), _MASTER_SEED)


def test_shortfall_control_raises_naming_control() -> None:
    with pytest.raises(ScheduleError, match="control"):
        build_schedule(_models(), _bank(control=7), _MASTER_SEED)


def test_wrong_model_count_raises() -> None:
    with pytest.raises(ScheduleError, match="exactly 4 models"):
        build_schedule(_models()[:3], _bank(), _MASTER_SEED)


def test_non_distinct_models_raise() -> None:
    dupe = _models()
    dupe[1] = dupe[0]  # same provider:model_name
    with pytest.raises(ScheduleError, match="distinct"):
        build_schedule(dupe, _bank(), _MASTER_SEED)
