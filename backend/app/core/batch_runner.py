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
  d. bank ingestion: the measurement frame and then the boards under ``bank_dir`` are stored if
     absent (idempotent: what is stored is skipped), and every scheduled board must then be in the
     DB, since the games' rows point at it. A stored frame that contradicts the sidecar
     (``StaleFrameError``) aborts. Runs after the collision check and before minting.
  e. schema: the database must be at the head of the Alembic migrations, or every game would be
     played and then fail to be stored. Runs first of the database checks.

Level 2 - PER-GAME FAULT ISOLATION (count-and-continue, inside the loop):
  - :func:`run_single_game` captures its own exceptions and returns a ``GameRunResult`` (status
    'completed'/'error'); it never propagates. So the loop INSPECTS ``result.status`` - it does not
    re-implement the error handling the runner already owns.
  - an 'error' result is recorded and the loop proceeds (count-and-continue);
  - a per-pairing CONSECUTIVE-failure threshold (default 5) aborts THAT pairing (skips its remaining
    cells) and continues with the next pairing - a systematic failure is a finding for one pairing,
    not a reason to kill the other five. The counter is per-pairing and resets on any completion.
  - a game lost to the provider (a timeout, or a provider unavailable or rate limiting, still
    failing after the client's own retries) is the provider's failure, not the model's. The batch
    waits for both seats' providers to pass their health check, then plays the SAME game again: its
    error row is deleted first, and the same game_id and seeds make it the same game. It waits up to
    ``provider_wait_s`` (30 min); a provider that does not come back STOPS the batch there, the
    games after it unplayed and the reason on the report (``BatchReport.aborted_reason``), so an
    outage never reaches the threshold and never aborts pairings one after another. A game the
    provider still fails after ``provider_replays`` replays (2) stays an error and counts like any
    other: the provider answers, so the failure is that game's.

The CLI wrapper lives in ``scripts/run_batch.py``; the orchestration lives here.
"""
from __future__ import annotations

import asyncio
import functools
import json
import logging
import os
from collections import Counter
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from sqlalchemy import select

from backend.app.core.batch_schedule import BatchCell, build_schedule
from backend.app.core.game_runner import (
    ClientFactory, SeatSpec, create_run, default_client_factory, run_single_game,
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

# A game lost to the provider is played again once the provider answers again (Level 2): at most
# this many times per game, each time waiting at most _PROVIDER_WAIT_S for the provider to answer.
_PROVIDER_REPLAYS = 2
_PROVIDER_WAIT_S = 30 * 60
# The wait probes the provider at once, then after waits of 15 s, 30 s, 60 s, ..., never more than
# 5 min apart.
_PROVIDER_PROBE_BASE_S = 15
_PROVIDER_PROBE_MAX_S = 5 * 60

# The Alembic configuration (backend/alembic.ini), whose migrations define the schema this code
# writes.
_ALEMBIC_INI = Path(__file__).resolve().parents[2] / "alembic.ini"

# A nonsense, non-English clue for the dry-run mock: in neither MorphoLex nor WordNet, so it can never
# be a board word, a morphological or derived form of one, or a compound component - it passes
# ClueValidator on any board.
_DRY_RUN_CLUE = "ZZXQJ"


class BatchPreconditionError(RuntimeError):
    """A Level-1 batch precondition failed (missing board, DB unavailable, a schema behind the
    migrations, an already-recorded seed, or a bank that cannot be stored). Raised before any game
    plays; zero games run."""


def _model_id(spec: SeatSpec) -> str:
    """A model's stable identity: provider + model, not the local-only think flag (mirrors
    batch_schedule._model_id so pairing keys read identically)."""
    return f"{spec.provider}:{spec.model_name}"


# LLMError codes (models/llm_errors.py) by kind: what went wrong with the provider, or with the
# setup that reaches it, and what the model itself returned.
_PROVIDER_ERROR_CODES = frozenset(
    {"timeout", "rate_limit", "provider_unavailable", "auth", "model_not_provided",
     "model_not_found", "invalid_request"})
_MODEL_ERROR_CODES = frozenset(
    {"parse_error", "empty_response", "degenerate_response", "truncated_response", "refused"})
# The provider errors that say the provider failed, not the setup: the retriable ones. A game lost
# to one of them is played again once the provider answers (Level 2). The other provider errors (a
# model not pulled, a bad key, a rejected request) would fail the game the same way again.
_TRANSIENT_PROVIDER_CODES = frozenset({"timeout", "rate_limit", "provider_unavailable"})


def _classify_error(error: Optional[str], code: Optional[str] = None) -> str:
    """Best-effort classification of a failed game into an error kind, so a partial batch is data,
    not noise. Distinguishes a provider/transport failure from the model producing no legal play;
    falls back to 'collision' / 'other'. Deliberately modest - a coarse, honest bucket, not a
    taxonomy.

    An LLM error is classified by its ``code`` (``GameRunResult.error_code``). Its message cannot be
    read for it: the text of every LLMError names its provider ("provider=ollama"), so a model that
    loops would read as a provider failure. The message is read only for the other errors."""
    if code in _PROVIDER_ERROR_CODES:
        return "provider"
    if code in _MODEL_ERROR_CODES:
        return "model"
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
    """The whole batch's outcome, structured so the analysis can read reduced effective-n off it:
    which pairings aborted and why, and the per-pairing completed/errored split."""
    run_id: str
    master_seed: int
    attempted: int
    completed: int
    errored: int
    pairings_aborted: int
    pairings: list[PairingReport]
    # Why the batch stopped before its last game (a provider that did not come back), or None when
    # it played the whole schedule.
    aborted_reason: Optional[str] = None

    def as_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "master_seed": self.master_seed,
            "attempted": self.attempted,
            "completed": self.completed,
            "errored": self.errored,
            "pairings_aborted": self.pairings_aborted,
            "aborted_reason": self.aborted_reason,
            "pairings": [p.as_dict() for p in self.pairings],
        }


def _migration_heads() -> set[str]:
    """The head revision(s) of the Alembic migrations shipped with this code."""
    from alembic.config import Config
    from alembic.script import ScriptDirectory
    return set(ScriptDirectory.from_config(Config(str(_ALEMBIC_INI))).get_heads())


def _database_revisions() -> set[str]:
    """The revision(s) the database is stamped with; empty for a database that was never migrated."""
    from alembic.runtime.migration import MigrationContext
    from backend.app.db.session import get_engine
    with get_engine().connect() as connection:
        return set(MigrationContext.configure(connection).get_current_heads())


def _check_schema_at_head() -> None:
    """Level-1e: fail unless the database schema is at the head of the migrations. On an older schema
    every game would be played to the end and then fail to be stored (the writer fills every mapped
    column), and so would its error flush: the batch would burn GPU hours and keep nothing."""
    heads, current = _migration_heads(), _database_revisions()
    if current != heads:
        raise BatchPreconditionError(
            f"the database schema is at {sorted(current) or 'no revision (never migrated)'}, but "
            f"this code's migrations end at {sorted(heads)}: every game would be played and then "
            f"fail to be stored. Run `alembic -c backend/alembic.ini upgrade head` first.")


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
            f"(backend.app.db.writer.delete_run(<RUN_ID>), see 'Re-running a seed' in the README; "
            f"the run id is on the existing game rows.) "
            f"Colliding ids (first 3): {list(existing)[:3]}")


def _ingest_bank(bank_dir: Path, board_ids: Sequence[str]) -> None:
    """Level-1d: store the measurement frame, then the boards, and check every scheduled board is
    stored. The frame goes first: a sealed board's ``measurement_frame_id`` is a foreign key to it.

    Both ingestions skip what is already stored, so a second batch on the same database adds
    nothing. A board file that cannot be read or mapped is skipped with a warning by the ingestion
    itself; if it is a scheduled board, the check after it aborts the batch."""
    from backend.app.db.ingest_boards import ingest_boards_if_absent
    from backend.app.db.ingest_frame import StaleFrameError, ingest_frame_if_absent
    from backend.app.db.models import BoardModel
    from backend.app.db.session import session_scope
    frame_path = bank_dir / "measurement_frame.json"
    if not frame_path.exists():
        raise BatchPreconditionError(
            f"no measurement frame at {frame_path}: the sealed boards point at it, and the analysis "
            f"measures against it.")
    try:
        with session_scope() as session:
            frame_inserted = ingest_frame_if_absent(session, frame_path)
    except StaleFrameError as exc:
        raise BatchPreconditionError(
            f"the measurement frame under {bank_dir} contradicts the one already stored: {exc}"
        ) from exc
    with session_scope() as session:
        boards_inserted = ingest_boards_if_absent(session, bank_dir)
        stored = set(session.execute(
            select(BoardModel.board_id).where(BoardModel.board_id.in_(list(board_ids)))
        ).scalars().all())
    logger.info("bank ingestion: measurement frame %s, %s new board(s) from %s",
                "stored" if frame_inserted else "already stored", boards_inserted, bank_dir)
    missing = sorted(set(board_ids) - stored)
    if missing:
        raise BatchPreconditionError(
            f"scheduled boards missing from the database after ingesting {bank_dir}: {missing}")


async def _wait_for_provider(probe: Callable[[], Awaitable[bool]], *, max_wait_s: float,
                             sleep: Callable[[float], Awaitable[object]] = asyncio.sleep) -> bool:
    """Wait until ``probe`` says the provider answers again: True once it does, False if it still
    does not after ``max_wait_s`` seconds of waiting. Probes at once, then after each wait of an
    exponential backoff (``_PROVIDER_PROBE_BASE_S``, doubling, at most ``_PROVIDER_PROBE_MAX_S``),
    the last wait cut to the budget."""
    waited = 0.0
    delay = _PROVIDER_PROBE_BASE_S
    while not await probe():
        if waited >= max_wait_s:
            return False
        step = min(delay, _PROVIDER_PROBE_MAX_S, max_wait_s - waited)
        logger.warning("the provider does not answer; probing again in %.0fs "
                       "(waited %.0fs of %.0fs)", step, waited, max_wait_s)
        await sleep(step)
        waited += step
        delay *= 2
    return True


async def _provider_healthy(cell: BatchCell, factory: ClientFactory) -> bool:
    """Whether both seats' providers pass their health check, asked of clients built as the game
    builds them (``factory``) and closed afterwards."""
    clients = [factory(0, cell.seat0_spec), factory(1, cell.seat1_spec)]
    try:
        for client in clients:
            if not await client.health_check():
                return False
        return True
    finally:
        for client in clients:
            try:
                await client.close()
            except Exception:
                logger.debug("client close failed after a health check", exc_info=True)


def _forget_error_game(game_id: str, run_id: str) -> None:
    """Delete the error row of a game about to be played again, which would otherwise collide with
    the replay on its deterministic ``game.id``."""
    from backend.app.db.writer import delete_error_game
    delete_error_game(game_id, run_id)


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
    provider_wait_s: float = _PROVIDER_WAIT_S,
    provider_replays: int = _PROVIDER_REPLAYS,
    bank_dir: str | Path = "data/boards",
) -> BatchReport:
    """Play the full batch under one run and return a structured :class:`BatchReport`.

    Level-1 preconditions run first (schedule/bank, schema, collision, bank ingestion, mint+gate);
    if any fails it raises (:class:`ScheduleError`, :class:`BatchPreconditionError`,
    :class:`ModelDigestMismatchError`) and zero games play. Then the play loop runs with
    count-and-continue and the per-pairing consecutive-failure abort. A game lost to the provider
    is played again, up to ``provider_replays`` times, once the provider answers again; a provider
    that does not answer within ``provider_wait_s`` seconds stops the batch, and the report's
    ``aborted_reason`` says so.

    ``bank_dir`` is the directory ``boards`` was loaded from: with ``persist`` its measurement frame
    and boards are stored in the database before the run is minted.

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
        # Level 1e: the schema the games will be written to, before anything reads or writes it.
        _check_schema_at_head()
        # Level 1c: collision check before minting, so an already-recorded seed leaves no orphan
        # run row.
        _check_no_collisions([c.game_id for c in schedule],
                             master_seed=master_seed)
        # Level 1d: the frame and the boards into the database, before any row points at them.
        _ingest_bank(Path(bank_dir), sorted({c.board_id for c in schedule}))

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
    aborted_reason: Optional[str] = None
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
        replays = 0
        while True:
            # The gate already ran once at mint; do not re-run it per game (enforce_digests=False).
            result = await run_single_game(
                board=board, seat_specs=[cell.seat0_spec, cell.seat1_spec],
                master_seed=master_seed, temperature=temperature, run_id=run_id,
                game_index=cell.game_index, persist=persist, enforce_digests=False, **extra)
            if (result.status != "error" or result.error_code not in _TRANSIENT_PROVIDER_CODES
                    or replays >= provider_replays):
                break
            # Lost to the provider, not the model: wait for it to answer again, then play the same
            # game again.
            logger.warning("game %s (game_index %s) lost to the provider (%s); waiting for it to "
                           "answer again", result.game_id, cell.game_index, result.error_code)
            probe = functools.partial(_provider_healthy, cell, factory or default_client_factory)
            if not await _wait_for_provider(probe, max_wait_s=provider_wait_s):
                aborted_reason = (
                    f"the provider did not answer for {provider_wait_s / 60:g} min after "
                    f"game_index {cell.game_index} ({result.game_id}) failed with "
                    f"{result.error_code}; the games after it were not played")
                break
            if persist:
                _forget_error_game(result.game_id, run_id)
            replays += 1
            logger.warning("the provider answers again; playing game %s again (replay %s of %s)",
                           result.game_id, replays, provider_replays)

        pr.attempted += 1
        if result.status == "completed":
            pr.completed += 1
            consecutive = 0  # reset on any completion -> the counter is consecutive, not cumulative
        else:
            pr.errored += 1
            pr.error_kinds[_classify_error(result.error, result.error_code)] += 1
            if aborted_reason is not None:
                # An outage, not a finding about this pairing: it never reaches the threshold.
                logger.error("batch stopped: %s", aborted_reason)
                break
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
        aborted_reason=aborted_reason,
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
