"""Seat -> perspective-column mapping, shared by the IAE and conc-SD metrics.

Codenames Duet is asymmetric: each card carries a role under each seat's perspective, stored as two
columns on ``card`` (``llm_perspective_role`` / ``human_perspective_role``). Getting the seat-to-
column mapping wrong silently measures the wrong player, so it lives in exactly one place.

Two rules are derived from this mapping. This module deliberately builds no sets:

  * **giver agent set** for seat ``s`` = the cards whose ``perspective_column(s)`` is ``'agent'``.
    These are the words the giver at seat ``s`` is trying to make its partner pick.
  * **guesser target-agent set** for seat ``s`` = the cards whose ``perspective_column(1 - s)`` is
    ``'agent'``. The guesser is hunting the *giver's* agents, which is the opposite seat's column.

WARNING - do not use ``engine.state.agents_remaining[s]`` (``backend/app/core/engine.py:297-303``)
as "seat s's own agents". That counter is **guesser-indexed**: it tracks how many agents seat ``s``
still has to find, which is the opposite seat's column, ``perspective_column(1 - s)``. Reading it as
the seat's own agent set inverts the perspective on every two-seat metric.
"""

from __future__ import annotations

LLM_SEAT: int = 0
HUMAN_SEAT: int = 1

_SEAT_COLUMNS: dict[int, str] = {
    LLM_SEAT: "llm_perspective_role",
    HUMAN_SEAT: "human_perspective_role",
}


def perspective_column(seat: int) -> str:
    """Return the ``card`` column holding ``seat``'s own view of each card's role.

    Seat 0 is the LLM, seat 1 the human. Any other seat is a programming error, not a default.
    """
    try:
        return _SEAT_COLUMNS[seat]
    except KeyError:
        raise ValueError(
            f"seat must be {LLM_SEAT} or {HUMAN_SEAT}, got {seat!r}") from None
