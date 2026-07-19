"""The batch schedule.

This is the probe/control calendar, seat mirroring and pairing cross that ``run_pilot.py`` explicitly
is not. Given the models and a ``master_seed``, :func:`build_schedule` produces the exact list of
games to play - each with its pairing, board, seat assignment and ``game_index``.
Because a silent error here becomes a biased dataset discovered only at analysis time, identity is 
derived through the very same :func:`_game_identity` the runner and DB use, so the tests can prove 
no two cells collide on the real ``game_id``.

Design (fixed by prior decisions):
  - 4 models, all cross-model pairs, no self-play => C(4,2) = 6 pairings. A model's stable id is
    ``f"{provider}:{model_name}"`` (the local-only ``think`` flag is not identity). A pairing's key
    is ``tuple(sorted((id_a, id_b)))``; the 6 keys sorted lexicographically give the ordinal ``p``,
    so reordering the input models cannot change any ``p``.
  - Each pairing plays the SAME 16 boards - 4 career + 4 science + 8 control - the comparability
    guarantee (identical stimulus across pairings). Within each kind the canonical order is
    ``board_id`` ascending (a required, unique field: a total order, unlike the optional ``seed``);
    the first 4 / 4 / 8 are taken.
  - Each board is played twice per pairing: once with model_a in seat 0 and once with model_a in 
    seat 1, same ``board_id`` both times. So 16 boards x 2 = 32 games per pairing, 16 probe + 16 
    control (50/50), 8 career + 8 science (4+4).
  - ``game_index = p * 32 + k`` with ``k = j * 2 + o`` for board ordinal ``j`` in 0..15 (over the
    fixed order ``[career4, science4, control8]``) and orientation ``o`` in {0, 1}. Range 0..191.
"""
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations

from backend.app.core.game_runner import SeatSpec, _game_identity
from backend.app.models.game_schemas import Board

# The board budget per pairing (also the bank precondition, seed of 5c.c's check).
_N_CAREER = 4
_N_SCIENCE = 4
_N_CONTROL = 8
_BOARDS_PER_PAIRING = _N_CAREER + _N_SCIENCE + _N_CONTROL  # 16
_GAMES_PER_PAIRING = _BOARDS_PER_PAIRING * 2  # 32 (strong mirror)
_N_MODELS = 4
_N_PAIRINGS = 6  # C(4, 2)
_N_CELLS = _N_PAIRINGS * _GAMES_PER_PAIRING  # 192


class ScheduleError(ValueError):
    """A precondition of the schedule is not met: wrong model count, non-distinct models, or a bank
    that lacks enough boards of a required kind. Raised instead of silently emitting a short or
    skewed schedule."""


@dataclass(frozen=True)
class BatchCell:
    """One game in the batch: an opaque identity coordinate (``game_index`` / ``game_id``) plus the
    stimulus (board) and the seat assignment that give it meaning."""
    game_index: int          # opaque identity coordinate, 0..191
    # the real DB identity, from _game_identity(master_seed, game_index)
    game_id: str
    # p, 0..5 (canonical lexicographic order of pairing keys)
    pairing_ordinal: int
    # 0..31 within the pairing (= board_ordinal * 2 + orientation)
    k: int
    board_id: str
    board_type: str          # "probe" | "control"
    # "gender-career" | "gender-science" | None (control)
    specification: str | None
    seat0_spec: SeatSpec     # model playing seat 0 in this game
    seat1_spec: SeatSpec     # model playing seat 1 in this game


def _model_id(spec: SeatSpec) -> str:
    """A model's stable identity for pairing/ordering: provider + model, not the local-only think
    flag."""
    return f"{spec.provider}:{spec.model_name}"


def _select_boards(boards: Sequence[Board]) -> list[Board]:
    """Return the fixed 16-board list ``[career4, science4, control8]`` in canonical (board_id
    ascending) order, or raise :class:`ScheduleError` naming every kind that is short."""
    career = sorted(
        (b for b in boards if b.type ==
         "probe" and b.specification == "gender-career"),
        key=lambda b: b.board_id,
    )
    science = sorted(
        (b for b in boards if b.type ==
         "probe" and b.specification == "gender-science"),
        key=lambda b: b.board_id,
    )
    control = sorted(
        (b for b in boards if b.type == "control"),
        key=lambda b: b.board_id,
    )

    shortfalls = []
    if len(career) < _N_CAREER:
        shortfalls.append(
            f"career boards: need {_N_CAREER}, have {len(career)}")
    if len(science) < _N_SCIENCE:
        shortfalls.append(
            f"science boards: need {_N_SCIENCE}, have {len(science)}")
    if len(control) < _N_CONTROL:
        shortfalls.append(
            f"control boards: need {_N_CONTROL}, have {len(control)}")
    if shortfalls:
        raise ScheduleError("board bank shortfall - " + "; ".join(shortfalls))

    return career[:_N_CAREER] + science[:_N_SCIENCE] + control[:_N_CONTROL]


def build_schedule(
    models: Sequence[SeatSpec], boards: Sequence[Board], master_seed: int,
) -> list[BatchCell]:
    """Build the deterministic batch schedule.

    The same inputs always produce a byte-identical result, and reordering ``models`` or ``boards``
    does not change the output (canonical orders hold). ``boards`` is the already-loaded bank (the 
    caller reads disk; this function does not).

    Raises :class:`ScheduleError` if ``models`` is not exactly _N_MODELS distinct models, or the 
    bank lacks _N_CAREER career + _N_SCIENCE science + _N_CONTROL control boards.
    """
    if len(models) != _N_MODELS:
        raise ScheduleError(
            f"need exactly {_N_MODELS} models, got {len(models)}")
    by_id: dict[str, SeatSpec] = {_model_id(m): m for m in models}
    if len(by_id) != _N_MODELS:
        raise ScheduleError(
            f"models must be distinct by provider:model_name, got {len(by_id)} distinct of "
            f"{_N_MODELS}"
        )

    # the same 16 boards, for every pairing (comparability)
    selected = _select_boards(boards)

    # The 6 pairing keys, ordered lexicographically; p is the index into this order. model_a is the
    # lexicographically smaller model of the pair (key[0]).
    pairing_keys = sorted(tuple(sorted(pair))
                          for pair in combinations(by_id, 2))

    cells: list[BatchCell] = []
    for p, key in enumerate(pairing_keys):
        model_a, model_b = by_id[key[0]], by_id[key[1]]
        for j, board in enumerate(selected):
            board_type = board.type
            assert board_type is not None  # partitioned to probe/control in _select_boards
            for o in (0, 1):
                k = j * 2 + o
                game_index = p * _GAMES_PER_PAIRING + k
                game_id = _game_identity(master_seed, game_index)[0]
                seat0, seat1 = (model_a, model_b) if o == 0 else (
                    model_b, model_a)
                cells.append(
                    BatchCell(
                        game_index=game_index,
                        game_id=game_id,
                        pairing_ordinal=p,
                        k=k,
                        board_id=board.board_id,
                        board_type=board_type,
                        specification=board.specification,
                        seat0_spec=seat0,
                        seat1_spec=seat1,
                    )
                )

    # 6 pairings x 16 boards x 2 mirror orientations
    assert len(cells) == _N_CELLS
    return cells
