"""Atomic terminal flush of one completed game to Postgres.

``persist_game`` reads a fully-accumulated ``GameRecorder`` and writes the whole game in ONE
transaction via ``session_scope``. Status is written directly as its terminal value 
(``'completed'`` / ``'error'``); ``'in_progress'`` is never written, so the absence of a game row
means the game never completed.

The writer RAISES on failure and does not swallow - swallowing is the caller's choice (the
interactive routes swallow so a flush failure never changes the HTTP response; a future headless
runner will not). ``recorder.flushed`` is set only after the transaction commits successfully.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from sqlalchemy import delete, func, select

from backend.app.db.models import (
    LlmCallModel,
    RunModel,
    TurnModel,
    ClueModel,
    ClueTargetModel,
    GuessProposalItemModel,
    GuessProposalModel,
    RevealEventModel,
    BoardModel,
    WordCardModel,
    GameModel,
    GameSeatModel,
)
from backend.app.db.recorder import GameRecorder, TurnRecord
from backend.app.db.session import session_scope

logger = logging.getLogger(__name__)


def _new_llm_call(record, *, game_id: str, turn_id: int, seat_index: int) -> LlmCallModel:
    """Map an in-memory ``LLMCallRecord`` to an ``llm_call`` row (rendered_prompt -> JSONB)."""
    return LlmCallModel(
        game_id=game_id,
        turn_id=turn_id,
        seat_index=seat_index,
        role=record.role,
        retry_index=record.retry_index,
        provider=record.provider,
        model_used=record.model_used,
        resolved_model=record.resolved_model,
        system_fingerprint=record.system_fingerprint,
        requested_temperature=record.requested_temperature,
        requested_seed=record.requested_seed,
        latency_ms=record.latency_ms,
        prompt_tokens=record.prompt_tokens,
        completion_tokens=record.completion_tokens,
        finish_reason=record.finish_reason,
        request_id=record.request_id,
        raw_payload=record.raw_payload,
        rendered_prompt=[m.model_dump() for m in record.rendered_prompt],
    )


def _write_play(session, gp, *, game_id: str, turn_id: int, guesser_seat: int,
                word_map: dict[str, int]) -> list[GuessProposalItemModel]:
    """Insert one ``kind='play'`` guess_proposal (+ its llm_call and ordered items) for ``guesser_seat``.

    Returns the item rows in position order so the caller can index-align reveal backfill.
    """
    play_call_id = None
    if gp.llm_call is not None:
        call = _new_llm_call(gp.llm_call, game_id=game_id,
                             turn_id=turn_id, seat_index=guesser_seat)
        session.add(call)
        session.flush()
        play_call_id = call.id
    proposal = GuessProposalModel(
        turn_id=turn_id,
        llm_call_id=play_call_id,
        guesser_seat=guesser_seat,
        kind="play",
        reasoning=gp.reasoning,
        stop_reason=gp.stop_reason,
    )
    session.add(proposal)
    session.flush()
    items: list[GuessProposalItemModel] = []
    for position, (word, confidence) in enumerate(zip(gp.proposals, gp.confidence)):
        item = GuessProposalItemModel(
            guess_proposal_id=proposal.id,
            position=position,
            word=word,
            confidence=confidence,
            resolved_card_id=word_map.get(word.lower()),
        )
        session.add(item)
        items.append(item)
    session.flush()
    return items


def _write_measurement(session, cr, *, game_id: str, turn_id: int, guesser_seat: int,
                       word_map: dict[str, int]) -> None:
    """Insert one ``kind='measurement'`` guess_proposal (+ its llm_call and items) for ``guesser_seat``.

    Measurement items never reference a reveal (they are an out-of-band ranking, not plays).
    """
    meas_call_id = None
    if cr.llm_call is not None:
        call = _new_llm_call(cr.llm_call, game_id=game_id,
                             turn_id=turn_id, seat_index=guesser_seat)
        session.add(call)
        session.flush()
        meas_call_id = call.id
    proposal = GuessProposalModel(
        turn_id=turn_id,
        llm_call_id=meas_call_id,
        guesser_seat=guesser_seat,
        kind="measurement",
        reasoning=cr.reasoning,
        stop_reason=None,
    )
    session.add(proposal)
    session.flush()
    for position, ranked in enumerate(cr.rankings):
        session.add(
            GuessProposalItemModel(
                guess_proposal_id=proposal.id,
                position=position,
                word=ranked.word,
                confidence=ranked.confidence,
                resolved_card_id=word_map.get(ranked.word.lower()),
                reveal_event_id=None,
            )
        )
    session.flush()


def _write_reveals(session, reveals, *, turn_id: int) -> list[RevealEventModel]:
    """Insert a turn's reveal_event rows in order; return them aligned to ``reveals``."""
    rows: list[RevealEventModel] = []
    for position, reveal in enumerate(reveals):
        row = RevealEventModel(
            turn_id=turn_id,
            position_in_turn=position,
            card_id=reveal.card_id,
            acting_seat=reveal.acting_seat,
            result_role=reveal.result_role,
            ended_turn=reveal.ended_turn,
            ended_game=reveal.ended_game,
            timer_tokens_after=reveal.timer_tokens_after,
        )
        session.add(row)
        rows.append(row)
    if rows:
        session.flush()
    return rows


def _write_turn(session, turn: TurnRecord, *, game_id: str, word_map: dict[str, int]) -> None:
    """Insert one turn and all its children, honoring FK insert order within the turn."""
    turn_row = TurnModel(
        game_id=game_id,
        turn_number=turn.turn_number,
        clue_giver_seat=turn.clue_giver_seat,
        phase=turn.phase,
    )
    session.add(turn_row)
    session.flush()

    # clue (+ its llm_call attempts)
    if turn.clue is not None:
        clue_calls = [
            _new_llm_call(rec, game_id=game_id, turn_id=turn_row.id,
                          seat_index=turn.clue_giver_seat)
            for rec in turn.clue.llm_calls
        ]
        if clue_calls:
            session.add_all(clue_calls)
            session.flush()
        # The accepted attempt is the last one appended; rejected attempts stay unlinked.
        accepted_call_id = clue_calls[-1].id if clue_calls else None

        clue_row = ClueModel(
            turn_id=turn_row.id,
            llm_call_id=accepted_call_id,
            clue_word=turn.clue.clue_word,
            count=turn.clue.count,
            reasoning=turn.clue.reasoning,
            targets_raw=list(turn.clue.targets_raw),
        )
        session.add(clue_row)
        session.flush()

        for position, target in enumerate(turn.clue.targets_resolved):
            giver_role = target.giver_role.value if target.giver_role is not None else None
            session.add(
                ClueTargetModel(
                    clue_id=clue_row.id,
                    position=position,
                    word=target.word,
                    card_id=target.card_id,
                    giver_role=giver_role,
                    revealed_at_clue=target.revealed_at_clue,
                )
            )

    if turn.phase == "sudden_death":
        # Sudden death is one collective turn on which both seats may guess/measure. Emit a
        # guess_proposal row per (seat, kind); UNIQUE(turn_id, kind, guesser_seat) admits both seats.
        play_items_by_seat: dict[int, list[GuessProposalItemModel]] = {}
        for seat, gp in sorted(turn.sd_play_by_seat.items()):
            play_items_by_seat[seat] = _write_play(
                session, gp, game_id=game_id, turn_id=turn_row.id,
                guesser_seat=seat, word_map=word_map)
        for seat, cr in sorted(turn.sd_measurement_by_seat.items()):
            _write_measurement(session, cr, game_id=game_id, turn_id=turn_row.id,
                               guesser_seat=seat, word_map=word_map)

        reveal_rows = _write_reveals(
            session, turn.reveals, turn_id=turn_row.id)
        # Per-seat index-aligned backfill: a reveal by seat s indexes seat s's play items only.
        # CONTRACT: reveal.proposal_index is 0-based relative to acting_seat's own SD play proposal,
        # never a global index over turn.reveals - the SD conductor resets it per seat. A global
        # counter would overshoot len(items) and drop the backfill.
        backfilled = False
        for reveal, row in zip(turn.reveals, reveal_rows):
            idx = reveal.proposal_index
            items = play_items_by_seat.get(reveal.acting_seat, [])
            if idx is not None and 0 <= idx < len(items):
                items[idx].reveal_event_id = row.id
                backfilled = True
            elif idx is not None and idx >= len(items):
                # Out-of-range index breaks the per-seat contract (e.g. a global counter leaking in);
                # skip rather than raise but surface it.
                logger.warning(
                    "SD reveal proposal_index %d out of range for game %s turn %d seat %d "
                    "(%d play items); reveal_event backfill skipped",
                    idx, game_id, turn.turn_number, reveal.acting_seat, len(
                        items),
                )
        if backfilled:
            session.flush()
    else:
        # Normal turn: the guesser is the non-clue-giver (a game identity, not a hardcoded seat).
        guesser_seat = 1 - turn.clue_giver_seat
        play_items: list[GuessProposalItemModel] = []
        if turn.play_proposal is not None:
            play_items = _write_play(
                session, turn.play_proposal, game_id=game_id, turn_id=turn_row.id,
                guesser_seat=guesser_seat, word_map=word_map)
        if turn.measurement is not None:
            _write_measurement(session, turn.measurement, game_id=game_id, turn_id=turn_row.id,
                               guesser_seat=guesser_seat, word_map=word_map)

        reveal_rows = _write_reveals(
            session, turn.reveals, turn_id=turn_row.id)
        # index-aligned backfill of play items' reveal_event_id; the unreached / off-board tail
        # (items with no reveal) keeps reveal_event_id NULL.
        for reveal, row in zip(turn.reveals, reveal_rows):
            idx = reveal.proposal_index
            if idx is not None and 0 <= idx < len(play_items):
                play_items[idx].reveal_event_id = row.id
        if play_items:
            session.flush()


def persist_game(recorder: GameRecorder, *, status: str) -> None:
    """Write ``recorder``'s whole game to Postgres in one transaction.

    :param status: terminal ``game_status`` to seal (``'completed'`` or ``'error'``). Never
        ``'in_progress'``.
    :raises ValueError: if the referenced board row is absent (minimal policy: do not ingest here).
    """
    with session_scope() as session:
        board_present = session.execute(
            select(BoardModel.board_id).where(
                BoardModel.board_id == recorder.board_id)
        ).first()
        if board_present is None:
            raise ValueError(
                f"Cannot persist game {recorder.game_id}: board {recorder.board_id!r} is not in the "
                "database. Ingest the board before persisting games that reference it."
            )

        # Case-insensitive word -> card_id map (mirrors Board.get_card_id_by_word).
        word_rows = session.execute(
            select(WordCardModel.text, WordCardModel.card_id).where(
                WordCardModel.board_id == recorder.board_id)
        ).all()
        word_map = {text.lower(): card_id for text, card_id in word_rows}

        game = GameModel(
            id=recorder.game_id,
            run_id=recorder.run_id,
            board_id=recorder.board_id,
            derived_seed=recorder.derived_seed,
            start_player=recorder.start_player,
            game_status=status,
            result=recorder.result,
            timer_tokens_final=recorder.timer_tokens_final,
            completed_at=func.now(),
        )
        session.add(game)
        session.flush()

        for seat in recorder.seats:
            session.add(
                GameSeatModel(
                    game_id=game.id,
                    seat_index=seat.seat_index,
                    model_ref=seat.model_ref,
                    provider=seat.provider,
                    requested_temperature=seat.requested_temperature,
                    requested_seed=seat.requested_seed,
                )
            )
        session.flush()

        for turn in recorder.turns:
            _write_turn(session, turn, game_id=game.id, word_map=word_map)

    # Only latch after the transaction has committed successfully.
    recorder.flushed = True


@dataclass(frozen=True)
class DeleteRunResult:
    """Outcome of :func:`delete_run`.

    :param found: whether a ``run`` row with the given id existed (and was deleted).
    :param games_deleted: how many ``game`` rows belonged to that run (counted before the delete);
        the whole game subtree goes with them via DB cascade.
    """

    found: bool
    games_deleted: int


def delete_run(run_id: str) -> DeleteRunResult:
    """Delete a run and, via DB cascade, all its games and their entire subtree.

    Removing a run is how a re-run gets unblocked: game identity is deterministic in
    ``(master_seed, game_index)`` (a uuid5 ``game.id``), so re-persisting the same experiment
    collides on the primary key; the run must be deleted first. It is also how throwaway pilot
    batches are cleaned up. The delete leans entirely on ``ON DELETE CASCADE`` from ``game.run_id`` - 
    deleting the ``run`` row tears down every ``game``, ``game_seat``, ``turn``,
    ``clue``, ``clue_target``, ``llm_call``, ``guess_proposal``, ``guess_proposal_item`` and
    ``reveal_event`` beneath it. Shared ingest data (``board`` / ``word_card``) is not run-owned and
    is never touched.

    Idempotent: deleting a run that does not exist is a no-op returning ``found=False`` (re-running
    the pilot -> inspect -> delete -> re-run cycle stays safe to repeat). Genuine database errors
    propagate - the writer does not swallow.

    :returns: a :class:`DeleteRunResult` with ``found`` and the ``games_deleted`` count.
    """
    with session_scope() as session:
        run_present = session.execute(
            select(RunModel.id).where(RunModel.id == run_id)
        ).first()
        if run_present is None:
            return DeleteRunResult(found=False, games_deleted=0)

        games_deleted = session.execute(
            select(func.count()).select_from(GameModel).where(
                GameModel.run_id == run_id)
        ).scalar_one()

        session.execute(delete(RunModel).where(RunModel.id == run_id))

    return DeleteRunResult(found=True, games_deleted=int(games_deleted))
