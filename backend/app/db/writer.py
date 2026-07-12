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

from sqlalchemy import func, select

from backend.app.db.models import (
    LlmCallModel,
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

# The LLM occupies seat 0 and is the only seat that issues model calls.
_LLM_SEAT = 0


def _new_llm_call(record, *, game_id: str, turn_id: int, seat_index: int = _LLM_SEAT) -> LlmCallModel:
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
            _new_llm_call(rec, game_id=game_id, turn_id=turn_row.id)
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

    # play proposal (kind='play')
    play_items: list[GuessProposalItemModel] = []
    if turn.play_proposal is not None:
        gp = turn.play_proposal
        play_call_id = None
        if gp.llm_call is not None:
            call = _new_llm_call(
                gp.llm_call, game_id=game_id, turn_id=turn_row.id)
            session.add(call)
            session.flush()
            play_call_id = call.id
        play_proposal = GuessProposalModel(
            turn_id=turn_row.id,
            llm_call_id=play_call_id,
            guesser_seat=_LLM_SEAT,
            kind="play",
            reasoning=gp.reasoning,
            stop_reason=gp.stop_reason,
        )
        session.add(play_proposal)
        session.flush()
        for position, (word, confidence) in enumerate(zip(gp.proposals, gp.confidence)):
            item = GuessProposalItemModel(
                guess_proposal_id=play_proposal.id,
                position=position,
                word=word,
                confidence=confidence,
                resolved_card_id=word_map.get(word.lower()),
            )
            session.add(item)
            play_items.append(item)
        session.flush()

    # measurement proposal (kind='measurement'); items never reference a reveal
    if turn.measurement is not None:
        cr = turn.measurement
        meas_call_id = None
        if cr.llm_call is not None:
            call = _new_llm_call(
                cr.llm_call, game_id=game_id, turn_id=turn_row.id)
            session.add(call)
            session.flush()
            meas_call_id = call.id
        meas_proposal = GuessProposalModel(
            turn_id=turn_row.id,
            llm_call_id=meas_call_id,
            guesser_seat=_LLM_SEAT,
            kind="measurement",
            reasoning=cr.reasoning,
            stop_reason=None,
        )
        session.add(meas_proposal)
        session.flush()
        for position, ranked in enumerate(cr.rankings):
            session.add(
                GuessProposalItemModel(
                    guess_proposal_id=meas_proposal.id,
                    position=position,
                    word=ranked.word,
                    confidence=ranked.confidence,
                    resolved_card_id=word_map.get(ranked.word.lower()),
                    reveal_event_id=None,
                )
            )

    # reveal events
    reveal_rows: list[RevealEventModel] = []
    for position, reveal in enumerate(turn.reveals):
        row = RevealEventModel(
            turn_id=turn_row.id,
            position_in_turn=position,
            card_id=reveal.card_id,
            acting_seat=reveal.acting_seat,
            result_role=reveal.result_role,
            ended_turn=reveal.ended_turn,
            ended_game=reveal.ended_game,
            timer_tokens_after=reveal.timer_tokens_after,
        )
        session.add(row)
        reveal_rows.append(row)
    if reveal_rows:
        session.flush()

    # index-aligned backfill of play items' reveal_event_id
    # Each reveal carries the index of the play-proposal item it came from; the unreached / off-board
    # tail (items with no reveal) keeps reveal_event_id NULL.
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
