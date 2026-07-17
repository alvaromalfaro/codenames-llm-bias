"""Headless single-game LLM-vs-LLM driver.

Plays one complete Codenames Duet game between two LLM seats with no HTTP layer. The driver owns
everything the interactive path leaves to the browser: explicit, reproducible seeding; a minimal
``run`` row; two real seat identities; and terminal persistence via the existing writer. It is the
single-game unit that the 180-game batch will call repeatedly.

The same ``master_seed`` played with the same (deterministic) clients yields a byte-identical game - 
same engine ``game_id``, same per-(seat, turn) derived seeds sent on each request, and the same 
recorder turn/reveal/proposal sequence. The engine, service and conductor stay seed-agnostic; all 
derivation lives here (see ``_derive`` and the seed scheme below).
"""
from __future__ import annotations

import hashlib
import logging
import os
import random
import uuid
from dataclasses import dataclass
from decimal import Decimal
from typing import Callable, Optional

from backend.app.core import provenance
from backend.app.core.engine import CodenamesDuetEngine
from backend.app.core.game_conductor import conduct_clue, conduct_guess, conduct_sd_guess
from backend.app.core.llm.client import LLMClient
from backend.app.core.llm.client_local import LLMClientLocal
from backend.app.core.llm.client_openrouter import LLMClientOpenRouter
from backend.app.core.llm_service import LLMService
from backend.app.db import writer
from backend.app.db.recorder import GameRecorder, SeatRecord
from backend.app.models.game_schemas import Board, GamePhase

logger = logging.getLogger(__name__)

# Fixed namespace for deterministic game/run ids (uuid5). A stable constant is what makes the id
# reproducible across re-runs; changing it would silently change every derived seed.
_GAME_NAMESPACE = uuid.UUID("6ba7b814-9dad-11d1-80b4-00c04fd430c8")

_MASK64 = (1 << 64) - 1
# The driver's transient-retry policy: each seat's client retries a retriable LLM error up to this
# many times (re-sending the identical request, no reseed) before it surfaces as terminal. Retry
# lives in the client, not the dispatch, so it is idempotent and loses no telemetry.
_CLIENT_MAX_RETRIES = 3
# safety cap so a non-progressing (mis-scripted) game cannot hang
_MAX_DISPATCHES = 2000


# seed derivation
def _derive(seed: int, *parts: bytes) -> int:
    """The single seed-derivation primitive: ``SHA-256(seed_as_8B_be || *parts) mod 2**64``.

    Deterministic and sensitive to every byte of every part, so different labels / seats / turns
    yield different seeds while identical inputs yield identical seeds. ``seed`` is masked to 64 bits
    before hashing so any caller-supplied master seed is accepted. Each integer part is fixed at an
    explicit byte width by the callers below (8-byte big-endian for seeds and turn numbers).
    """
    h = hashlib.sha256((seed & _MASK64).to_bytes(8, "big"))
    for part in parts:
        h.update(part)
    return int.from_bytes(h.digest(), "big") & _MASK64


def _seed_game(master_seed: int, game_id: str) -> int:
    """seed_game = SHA256(master_seed || game_id) mod 2**64."""
    return _derive(master_seed, game_id.encode("ascii"))


def _seed_engine(seed_game: int) -> int:
    """seed_engine = SHA256(seed_game || b"engine") mod 2**64 (drives the start-player pick)."""
    return _derive(seed_game, b"engine")


def _seed_play(seed_game: int, seat: int, turn: int) -> int:
    """seed_play(seat, turn) = SHA256(seed_game || f"seat{seat}" || turn_8B_be) mod 2**64."""
    return _derive(seed_game, f"seat{seat}".encode("ascii"), turn.to_bytes(8, "big"))


def _seed_meas(seed_game: int, seat: int, turn: int) -> int:
    """seed_meas(seat, turn) = SHA256(seed_game || f"meas{seat}" || turn_8B_be) mod 2**64."""
    return _derive(seed_game, f"meas{seat}".encode("ascii"), turn.to_bytes(8, "big"))


def _game_identity(master_seed: int, game_index: int) -> tuple[str, int, int]:
    """Return ``(game_id, seed_game, seed_engine)`` for this game.

    Game identity derives from ``master_seed + game_index`` only - the two things that define the
    experiment. ``run_id`` is deliberately not an input: it is persistence bookkeeping, not identity,
    and a DB-minted ``run.id`` is a random uuid4 that would break reproducibility if it fed the
    derivation. This function is pure and is the seam the tests use to predict the deterministic
    engine start-player.

    Because identity is deterministic, re-running the same ``(master_seed, game_index)`` yields the 
    same ``game_id``, so a second persist collides on ``game.id`` (primary key). This is intended 
    semantics meaning "this experiment is already in the database". Re-running requires deleting the 
    run first.
    """
    game_id = str(uuid.uuid5(_GAME_NAMESPACE, f"{master_seed}:{game_index}"))
    seed_game = _seed_game(master_seed, game_id)
    return game_id, seed_game, _seed_engine(seed_game)


# seat specs / result
@dataclass(frozen=True)
class SeatSpec:
    """The identity of one seat's model: provider + model, plus the local-only ``think`` flag."""
    provider: str          # "ollama" | "openrouter"
    model_name: str
    think: bool = False    # only meaningful for the local (ollama) client


@dataclass
class GameRunResult:
    """The outcome of one headless game."""
    game_id: str
    run_id: str
    seed_game: int
    status: str                    # "completed" | "error"
    # engine.state.result (e.g. "victory" / "loss_assassin" / ...)
    result: Optional[str]
    error: Optional[str] = None    # error message when status == "error"


# client construction / db gating
def _default_client_factory(seat_index: int, spec: SeatSpec) -> LLMClient:
    """Build one client for a seat from its spec, wired with the driver's transient-retry budget."""
    if spec.provider == "openrouter":
        return LLMClientOpenRouter(model_name=spec.model_name, max_retries=_CLIENT_MAX_RETRIES)
    return LLMClientLocal(model_name=spec.model_name, think=spec.think,
                          max_retries=_CLIENT_MAX_RETRIES)


def _db_enabled() -> bool:
    """Persistence (run row + persist_game) requires Postgres; gate on DATABASE_URL so the mock /
    reproducibility tests run in CI without a database."""
    return bool(os.environ.get("DATABASE_URL"))


ClientFactory = Callable[[int, SeatSpec], LLMClient]


# dispatch
async def _dispatch_phase(svc: LLMService, clients, engine: CodenamesDuetEngine,
                          recorder: GameRecorder, seed_game: int, flush) -> None:
    """Dispatch exactly one turn for the current phase to the acting seat's client, with the
    per-(seat, turn) derived seeds. The engine is the single source of truth for whose turn it is.

    Transient retry note: this is a plain dispatch - the client owns the bounded same-request retry 
    of retriable LLM errors, so no telemetry is lost and a succeeded call is never re-run. A
    retriable error that exhausts the client's budget, and every deterministic error, propagates
    straight to the game-level error boundary as terminal."""
    phase = engine.state.current_phase
    turn = engine.state.turn_number
    game_id = engine.state.game_id
    if phase == GamePhase.GIVING_CLUE:
        cg = engine.state.clue_giver
        logger.info("dispatch clue: game_id=%s turn=%s seat=%s",
                    game_id, turn, cg)
        await conduct_clue(svc, clients[cg], engine, recorder, player_id=cg,
                           seed=_seed_play(seed_game, cg, turn))
    elif phase == GamePhase.GUESSING:
        g = engine.state.guesser
        logger.info("dispatch guess: game_id=%s turn=%s seat=%s",
                    game_id, turn, g)
        await conduct_guess(svc, clients[g], engine, recorder, player_id=g, flush=flush,
                            on_reveal=None, seed=_seed_play(seed_game, g, turn),
                            measurement_seed=_seed_meas(seed_game, g, turn))
    elif phase == GamePhase.SUDDEN_DEATH_LLM:
        logger.info(
            "dispatch sudden-death seat0: game_id=%s turn=%s", game_id, turn)
        await conduct_sd_guess(svc, clients[0], engine, recorder, player_id=0, flush=flush,
                               on_reveal=None, seed=_seed_play(seed_game, 0, turn),
                               measurement_seed=_seed_meas(seed_game, 0, turn))
    elif phase == GamePhase.SUDDEN_DEATH_HUMAN:
        logger.info(
            "dispatch sudden-death seat1: game_id=%s turn=%s", game_id, turn)
        await conduct_sd_guess(svc, clients[1], engine, recorder, player_id=1, flush=flush,
                               on_reveal=None, seed=_seed_play(seed_game, 1, turn),
                               measurement_seed=_seed_meas(seed_game, 1, turn))
    else:
        raise RuntimeError(
            f"Unexpected non-terminal phase {phase!r} in the dispatch loop.")


# entry point
async def run_single_game(*, board: Board, seat_specs, master_seed: int, temperature: float,
                          run_id: Optional[str] = None, game_index: int = 0, persist: bool = True,
                          client_factory: ClientFactory = _default_client_factory) -> GameRunResult:
    """Play ONE complete LLM-vs-LLM Codenames Duet game and persist it.

    Game identity is deterministic in ``master_seed + game_index`` only (see ``_game_identity``);
    ``run_id`` is persistence bookkeeping and never feeds identity. Consequently re-running the same
    ``(master_seed, game_index)`` re-derives the same ``game_id``, and persisting a second time
    collides on the ``game.id`` primary key - the intended "this experiment is already recorded"
    signal. To re-run, delete the run first.

    Args:
        board: the board to play (its row must already exist in the DB for persistence to succeed).
        seat_specs: a 2-element sequence of ``SeatSpec`` (index == seat).
        master_seed: the batch/game master seed; all per-call seeds derive from it.
        temperature: required and explicit - raised if None (never falls back to the default).
        run_id: an existing run id to attach to; when None a minimal ``run`` row is created (DB only).
        game_index: index of this game within its run, folded into the deterministic game id.
        persist: when True (default) the game is written to Postgres and a missing ``DATABASE_URL``
            is a hard error raised up front - refusing to burn provider calls on an unpersistable
            game. Pass False for a dry run (mock / reproducibility tests) that touches no database.
        client_factory: builds a client per (seat_index, spec); overridden by tests with mocks.
    """
    # temperature must be explicit; do not fall back to any default.
    if temperature is None:
        raise ValueError(
            "run_single_game requires an explicit temperature; the 0.7 default is never used.")
    if len(seat_specs) != 2:
        raise ValueError(
            "seat_specs must have exactly two entries (one per seat).")

    # 1fail loud before any client/engine work so a missing DATABASE_URL wastes zero provider calls.
    # persist=False opts out entirely (no DB touched).
    if persist and not _db_enabled():
        raise RuntimeError(
            "run_single_game(persist=True) requires DATABASE_URL; refusing to burn provider calls "
            "on a game that cannot be persisted. Pass persist=False for a dry run.")

    # deterministic game id + game/engine seeds (independent of any DB-minted run_id).
    game_id, seed_game, seed_engine = _game_identity(master_seed, game_index)

    # The service carries the explicit temperature into every request (overriding its 0.7 default).
    # Built here (before the run row) so its template fingerprint can seed the run's provenance
    # without loading the templates twice.
    svc = LLMService(temperature=temperature)

    # minimal run row, committed first in its own short-lived session (FK: run before game). When a
    # run row is created here we also fill its provenance (which models served, which prompt texts
    # were sent, which code ran); an existing run_id is left untouched - the batch owns its
    # provenance. Provenance is record-only: no lookup may abort the game (see backend...provenance).
    if run_id is None:
        if persist:
            from backend.app.db.models import RunModel
            from backend.app.db.session import session_scope
            with session_scope() as session:
                run = RunModel(
                    master_seed=Decimal(int(master_seed)), temperature=temperature,
                    model_registry_snapshot=provenance.build_model_registry_snapshot(seat_specs),
                    prompt_template_version=svc.template_fingerprint(),
                    code_version=provenance.git_code_version(),
                )
                session.add(run)
                session.flush()
                run_id = run.id
        else:
            # No DB: synthesize a deterministic run id so results/tests are still reproducible.
            run_id = str(uuid.uuid5(_GAME_NAMESPACE, f"run:{master_seed}"))

    logger.info(
        "run_single_game start: run_id=%s game_id=%s master_seed=%s temperature=%s seats=%s",
        run_id, game_id, master_seed, temperature,
        [(s.provider, s.model_name, s.think) for s in seat_specs])

    # two clients, seeded engine (injected game_id), two-seat recorder.
    clients = [client_factory(0, seat_specs[0]),
               client_factory(1, seat_specs[1])]
    engine = CodenamesDuetEngine(
        board, rng=random.Random(seed_engine), game_id=game_id)
    # Both seats carry the constant per-game temperature (it has a referent); requested_seed is left
    # NULL because the runner seeds per (seat, turn) - see GameSeatModel and _observe_seat0_sampling.
    recorder = GameRecorder(
        game_id=game_id, board_id=board.board_id, start_player=engine.state.clue_giver,
        seats=[SeatRecord(0, seat_specs[0].provider, seat_specs[0].model_name,
                          requested_temperature=temperature, requested_seed=None),
               SeatRecord(1, seat_specs[1].provider, seat_specs[1].model_name,
                          requested_temperature=temperature, requested_seed=None)],
    )
    recorder.run_id = run_id
    recorder.derived_seed = seed_game

    def _flush_if_over(eng: CodenamesDuetEngine, rec: GameRecorder) -> None:
        """Terminal flush trigger passed to the conductors. Persists once at game-over (idempotent
        via ``rec.flushed``). Unlike the interactive path this does not swallow a persist error - it
        propagates to the game-level error boundary."""
        if not eng.state.is_game_over or rec.flushed:
            return
        rec.set_outcome(eng.state.result, eng.state.timer_tokens)
        if persist:
            writer.persist_game(rec, status="completed")

    try:
        dispatches = 0
        while not engine.state.is_game_over:
            if dispatches >= _MAX_DISPATCHES:
                raise RuntimeError(
                    f"Game {game_id} exceeded {_MAX_DISPATCHES} dispatches without terminating.")
            dispatches += 1
            await _dispatch_phase(svc, clients, engine, recorder, seed_game, _flush_if_over)

        # Defensive terminal flush: the conductor's per-reveal flush already fired at game-over, but
        # re-asserting here (idempotent) guarantees persistence even for terminal transitions that do
        # not end on a reveal.
        _flush_if_over(engine, recorder)
        logger.info("run_single_game completed: game_id=%s result=%s",
                    game_id, engine.state.result)
        return GameRunResult(game_id=game_id, run_id=run_id, seed_game=seed_game,
                             status="completed", result=engine.state.result)
    except Exception as exc:
        # Terminal error: either deterministic, or a retriable error whose client-side retry budget
        # was exhausted. The boundary sees only terminals - transient retry lives in the client.
        # Flush the partial game as status='error', then return an error result.
        logger.error("run_single_game failed: game_id=%s run_id=%s error=%s",
                     game_id, run_id, exc, exc_info=True)
        _error_flush(engine, recorder, game_id, persist=persist)
        return GameRunResult(game_id=game_id, run_id=run_id, seed_game=seed_game,
                             status="error", result=engine.state.result, error=str(exc))
    finally:
        for client in clients:
            try:
                await client.close()
            except Exception:
                logger.debug("client close failed for game_id=%s",
                             game_id, exc_info=True)


def _error_flush(engine: CodenamesDuetEngine, recorder: GameRecorder, game_id: str, *,
                 persist: bool) -> None:
    """Hardened partial-game flush as status='error'. Never raises: if the error-flush itself fails
    it is logged richly so it cannot mask the original error that triggered it."""
    if recorder.flushed:
        return
    try:
        recorder.set_outcome(engine.state.result, engine.state.timer_tokens)
        if persist:
            writer.persist_game(recorder, status="error")
    except Exception:
        logger.error(
            "hardened error-flush failed for game_id=%s; original error is preserved.",
            game_id, exc_info=True)
