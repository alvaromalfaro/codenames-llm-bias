"""The batch orchestrator: play the game schedule under one run, with fault isolation.

This assembles :func:`build_schedule` (the deterministic calendar), :func:`create_run` (mint-once 
run identity + digest gate + provenance) and :func:`run_single_game` (the play unit) into the full 
batch. It owns exactly two distinct failure levels, kept separate:

Level 1 - BATCH PRECONDITIONS (fail loud, abort the whole batch before any game plays):
  a. bank precondition: the loaded bank has >= 4 career + 4 science + 8 control boards
     (:func:`build_schedule` raises :class:`ScheduleError`; we surface it, never swallow it);
  b. mint the run via :func:`create_run` with ``enforce_digests=True`` - the digest gate runs once
     here and a :class:`ModelDigestMismatchError` aborts before any game;
  c. collision check: none of the 192 deterministic ``game_id`` values may already exist in the DB
     (the "experiment already recorded" case, made a clean precondition instead of a mid-loop PK
     collision). Runs before minting so an already-recorded seed leaves no orphan run row.

Level 2 - PER-GAME FAULT ISOLATION (count-and-continue, inside the loop):
  - :func:`run_single_game` captures its own exceptions and returns a ``GameRunResult`` (status
    'completed'/'error'); it never propagates. So the loop INSPECTS ``result.status`` - it does not
    re-implement the error handling the runner already owns.
  - an 'error' result is recorded and the loop proceeds (count-and-continue);
  - a per-pairing CONSECUTIVE-failure threshold (default 5) aborts THAT pairing (skips its remaining
    cells) and continues with the next pairing - a systematic failure is a finding for one pairing,
    not a reason to kill the other five. The counter is per-pairing and resets on any completion.

The CLI wrapper lives in ``scripts/run_batch.py``; the orchestration lives here.
"""
from __future__ import annotations

import json
import logging
import os
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Optional

from sqlalchemy import select

from backend.app.core.batch_schedule import build_schedule
from backend.app.core.game_runner import (
    ClientFactory, SeatSpec, create_run, run_single_game,
)
from backend.app.core.llm.client import LLMClient
from backend.app.core.llm_service import LLMService
from backend.app.models.game_schemas import Board, CardRole
from backend.app.models.llm_schemas import (
    ClueJSONFormat, ConfidenceRankingJSONFormat, GuessJSONFormat, LLMRequest, LLMResponse,
)

logger = logging.getLogger("run_batch")

# Per-pairing consecutive-failure abort threshold (Level 2). A pairing with this many errors in a row
# is aborted (its remaining cells skipped); the batch continues with the next pairing.
_CONSECUTIVE_FAILURE_THRESHOLD = 5

# A nonsense, non-English clue for the dry-run mock: no WordNet synset, so it can never be a board
# word, a morphological form of one, or a compound component - it passes ClueValidator on any board.
_DRY_RUN_CLUE = "ZZXQJ"


class BatchPreconditionError(RuntimeError):
    """A Level-1 batch precondition failed (missing board, DB unavailable, or an already-recorded
    seed). Raised before any game plays; zero games run."""


def _model_id(spec: SeatSpec) -> str:
    """A model's stable identity: provider + model, not the local-only think flag (mirrors
    batch_schedule._model_id so pairing keys read identically)."""
    return f"{spec.provider}:{spec.model_name}"


def _classify_error(error: Optional[str]) -> str:
    """Best-effort classification of a GameRunResult.error string into an error kind, so a partial
    batch is data, not noise. Distinguishes a provider/transport failure from the model producing no
    legal play; falls back to 'collision' / 'other'. Deliberately modest - a coarse, honest bucket, 
    not a taxonomy."""
    if not error:
        return "other"
    low = error.lower()
    if any(t in low for t in
           ("duplicate key", "already exists", "unique", "primary key", "uniqueviolation")):
        return "collision"
    if any(t in low for t in
           ("timeout", "timed out", "connection", "unreachable", "refused", "rate limit",
            "429", "502", "503", "504", "network", "httpx", "socket", "provider")):
        return "provider"
    if any(t in low for t in
           ("no legal", "no playable", "illegal", "invalid clue", "could not parse", "no valid",
            "json", "validation", "malformed", "empty", "refus", "at least one guess")):
        return "model"
    return "other"


@dataclass
class PairingReport:
    """The outcome of one pairing's 32 games."""
    pairing_ordinal: int
    pairing_key: tuple[str, str]
    total_cells: int
    attempted: int = 0
    completed: int = 0
    errored: int = 0
    aborted: bool = False
    abort_reason: Optional[str] = None
    error_kinds: Counter = field(default_factory=Counter)

    @property
    def skipped(self) -> int:
        return self.total_cells - self.attempted

    def as_dict(self) -> dict:
        return {
            "pairing_ordinal": self.pairing_ordinal,
            "pairing_key": list(self.pairing_key),
            "total_cells": self.total_cells,
            "attempted": self.attempted,
            "completed": self.completed,
            "errored": self.errored,
            "skipped": self.skipped,
            "aborted": self.aborted,
            "abort_reason": self.abort_reason,
            "error_kinds": dict(self.error_kinds),
        }


@dataclass
class BatchReport:
    """The whole batch's outcome, structured so step-7 can read reduced effective-n off it: which
    pairings aborted and why, and the per-pairing completed/errored split."""
    run_id: str
    master_seed: int
    attempted: int
    completed: int
    errored: int
    pairings_aborted: int
    pairings: list[PairingReport]

    def as_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "master_seed": self.master_seed,
            "attempted": self.attempted,
            "completed": self.completed,
            "errored": self.errored,
            "pairings_aborted": self.pairings_aborted,
            "pairings": [p.as_dict() for p in self.pairings],
        }


def _check_no_collisions(game_ids: Sequence[str], *, master_seed: int) -> None:
    """Level-1c: fail if any of the schedule's deterministic game ids already exist. A single
    ``SELECT ... WHERE game.id IN (...)`` over the ids; raises before any provider call."""
    from backend.app.db.models import GameModel
    from backend.app.db.session import session_scope
    with session_scope() as session:
        existing = session.execute(
            select(GameModel.id).where(GameModel.id.in_(list(game_ids)))
        ).scalars().all()
    if existing:
        raise BatchPreconditionError(
            f"{len(existing)} of {len(game_ids)} games for master_seed={master_seed} are already "
            f"recorded - this seed's experiment is in the database (deterministic game.id "
            f"collision). Delete the prior run first, then re-run this master_seed. "
            f"(e.g. python scripts/run_pilot.py --delete-run <RUN_ID>; the run id is on the "
            f"existing game rows.) Colliding ids (first 3): {list(existing)[:3]}")


async def run_batch(
    *,
    models: Sequence[SeatSpec],
    boards: Sequence[Board],
    master_seed: int,
    temperature: float,
    persist: bool = True,
    enforce_digests: bool = True,
    client_factory: Optional[ClientFactory] = None,
    make_client_factory: Optional[Callable[[Board], ClientFactory]] = None,
    consecutive_failure_threshold: int = _CONSECUTIVE_FAILURE_THRESHOLD,
) -> BatchReport:
    """Play the full batch under one run and return a structured :class:`BatchReport`.

    Level-1 preconditions run first (schedule/bank, collision, mint+gate); if any fails it raises
    (:class:`ScheduleError`, :class:`BatchPreconditionError`, :class:`ModelDigestMismatchError`) and
    zero games play. Then the play loop runs with count-and-continue and the per-pairing
    consecutive-failure abort.

    ``client_factory`` (flat, ``(seat_index, spec) -> LLMClient``) is forwarded to every game; when
    ``None`` the runner's real default is used. ``make_client_factory`` takes precedence and builds a
    per-board factory (the dry-run path, whose mock must know each board's assassin). ``persist`` and
    ``enforce_digests`` default to the real batch path; the dry run passes both effectively off.
    """
    # Level 1a: the schedule (raises ScheduleError on a bank shortfall - surfaced, not swallowed)
    schedule = build_schedule(models, boards, master_seed)

    # every scheduled board must be resolvable from the loaded bank (Level 1).
    board_by_id: dict[str, Board] = {b.board_id: b for b in boards}
    missing = sorted({c.board_id for c in schedule} - board_by_id.keys())
    if missing:
        raise BatchPreconditionError(
            f"scheduled boards missing from the loaded bank: {missing}")

    if persist:
        if not os.environ.get("DATABASE_URL"):
            raise BatchPreconditionError(
                "run_batch(persist=True) requires DATABASE_URL; refusing to burn provider calls on "
                "a batch that cannot be persisted. Use the dry run for a no-DB check.")
        # Level 1c: collision check before minting, so an already-recorded seed leaves no orphan
        # run row.
        _check_no_collisions([c.game_id for c in schedule],
                             master_seed=master_seed)

    # Level 1b: mint the run + run the digest gate once (create_run raises on a bad/unavailable
    # local digest under enforcement). The run carries all four models' provenance.
    svc = LLMService(temperature=temperature)
    run_id, _snapshot = create_run(
        seat_specs=list(models), master_seed=master_seed, temperature=temperature,
        template_fingerprint=svc.template_fingerprint(), persist=persist,
        enforce_digests=enforce_digests)

    # per-pairing report scaffolding, in canonical pairing order.
    cells_per_pairing = Counter(c.pairing_ordinal for c in schedule)
    reports: dict[int, PairingReport] = {}
    for c in schedule:
        if c.pairing_ordinal not in reports:
            lo, hi = sorted((_model_id(c.seat0_spec), _model_id(c.seat1_spec)))
            reports[c.pairing_ordinal] = PairingReport(
                pairing_ordinal=c.pairing_ordinal, pairing_key=(lo, hi),
                total_cells=cells_per_pairing[c.pairing_ordinal])

    # Level 2: the play loop (game_index ascending), count-and-continue + per-pairing threshold.
    current_p: Optional[int] = None
    consecutive = 0
    aborted_pairings: set[int] = set()
    for cell in schedule:
        p = cell.pairing_ordinal
        if p != current_p:  # pairing boundary: the consecutive counter is PER-PAIRING
            current_p = p
            consecutive = 0
        if p in aborted_pairings:
            # remaining cells of an aborted pairing are skipped (never attempted)
            continue

        pr = reports[p]
        board = board_by_id[cell.board_id]
        factory = (make_client_factory(board) if make_client_factory is not None
                   else client_factory)
        extra = {} if factory is None else {"client_factory": factory}
        # The gate already ran once at mint; do not re-run it per game (enforce_digests=False).
        result = await run_single_game(
            board=board, seat_specs=[cell.seat0_spec, cell.seat1_spec],
            master_seed=master_seed, temperature=temperature, run_id=run_id,
            game_index=cell.game_index, persist=persist, enforce_digests=False, **extra)

        pr.attempted += 1
        if result.status == "completed":
            pr.completed += 1
            consecutive = 0  # reset on any completion -> the counter is consecutive, not cumulative
        else:
            pr.errored += 1
            pr.error_kinds[_classify_error(result.error)] += 1
            consecutive += 1
            if consecutive >= consecutive_failure_threshold:
                pr.aborted = True
                pr.abort_reason = (
                    f"{consecutive} consecutive errors in pairing {p} "
                    f"({pr.pairing_key[0]} vs {pr.pairing_key[1]}); "
                    f"{pr.total_cells - pr.attempted} remaining games skipped")
                aborted_pairings.add(p)
                logger.warning("pairing %s aborted: %s", p, pr.abort_reason)

    pairings = [reports[p] for p in sorted(reports)]
    report = BatchReport(
        run_id=run_id, master_seed=master_seed,
        attempted=sum(pr.attempted for pr in pairings),
        completed=sum(pr.completed for pr in pairings),
        errored=sum(pr.errored for pr in pairings),
        pairings_aborted=sum(1 for pr in pairings if pr.aborted),
        pairings=pairings,
    )
    logger.info("batch complete: %s", json.dumps(report.as_dict()))
    return report


# dry-run deterministic client (no DB, no daemon): checks the schedule + loop wiring end to end.
def shared_assassin_word(board: Board) -> str:
    """The one card that is an assassin from BOTH perspectives (the Duet layout guarantees exactly
    one shared assassin). Guessing it ends any game in ``loss_assassin`` regardless of which seat
    guesses first - the board-agnostic terminal the dry run relies on."""
    for card in board.cards:
        if (card.llm_perspective_role == CardRole.ASSASSIN
                and card.human_perspective_role == CardRole.ASSASSIN):
            return card.text
    raise BatchPreconditionError(
        f"board {board.board_id!r} has no shared assassin (not a valid Duet board).")


class DryRunClient(LLMClient):
    """A deterministic mock client that plays no real model: it gives a nonsense-safe clue and always
    guesses the board's shared assassin, so every game completes in two dispatches. Used only by the
    dry run to exercise schedule + loop wiring without a daemon or a database."""

    def __init__(self, assassin_word: str, model_name: str = "dry-run"):
        self._assassin = assassin_word
        self.model_name = model_name

    async def generate(self, request: LLMRequest, expected_format=None) -> LLMResponse:
        if expected_format is ClueJSONFormat:
            payload: dict = {"reasoning": "dry-run", "clue": _DRY_RUN_CLUE,
                             "count": 1, "targets": []}
        elif expected_format is GuessJSONFormat:
            payload = {"reasoning": "dry-run", "stop_reason": "done",
                       "proposals": [{"word": self._assassin, "confidence": 0.99}]}
        elif expected_format is ConfidenceRankingJSONFormat:
            payload = {"reasoning": "dry-run",
                       "rankings": [{"word": self._assassin, "confidence": 0.99}]}
        else:
            raise AssertionError(
                f"DryRunClient: unexpected expected_format {expected_format!r}")
        return LLMResponse(
            text=json.dumps(payload), model_used=self.model_name, latency_ms=0,
            raw_payload=payload, requested_seed=request.seed,
            requested_temperature=request.temperature)


def dry_run_client_factory(board: Board) -> ClientFactory:
    """Build a flat ``(seat_index, spec) -> DryRunClient`` factory bound to this board's shared
    assassin. Passed as ``make_client_factory`` so each game's mock knows its own board."""
    assassin = shared_assassin_word(board)
    return lambda seat_index, spec: DryRunClient(assassin, model_name=f"dry-run-seat{seat_index}")
