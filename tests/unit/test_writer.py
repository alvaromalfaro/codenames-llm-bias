"""Tests for the atomic terminal flush (writer.persist_game).

These require a live Postgres and are skipped when DATABASE_URL is unset, matching test_db_ingest.
Each test inserts its own board (unique id) so runs are independent and need no cleanup.
"""
import os
import uuid

import pytest
from sqlalchemy import select

from backend.app.models.game_schemas import (
    CardRole, ClueEntry, ConfidenceRanking, RankedCard, ResolvedTarget,
)
from backend.app.models.llm_schemas import ClueProposal, GuessProposal, LLMCallRecord, LLMMessage

pytestmark = pytest.mark.skipif(
    not os.environ.get("DATABASE_URL"), reason="requires a live Postgres database"
)


class _FakeOpenRouterClient:
    model_name = "or-model"


def _call(role, retry_index=0, temperature=0.5, seed=7):
    return LLMCallRecord(
        role=role,
        retry_index=retry_index,
        rendered_prompt=[LLMMessage(role="user", content="prompt text")],
        requested_temperature=temperature,
        requested_seed=seed,
        prompt_tokens=10,
        completion_tokens=5,
    )


def _insert_board(session):
    """Insert a minimal board + three word cards; return (board_id, {word: card_id})."""
    from backend.app.db.models import BoardModel, WordCardModel

    board_id = f"test-writer-{uuid.uuid4()}"
    session.add(BoardModel(board_id=board_id, type="control"))
    cards = [
        (0, "ALPHA", CardRole.AGENT),
        (1, "BETA", CardRole.CIVILIAN),
        (2, "GAMMA", CardRole.ASSASSIN),
        (3, "DELTA", CardRole.AGENT),
    ]
    for card_id, text, role in cards:
        session.add(WordCardModel(
            board_id=board_id, card_id=card_id, text=text,
            llm_perspective_role=role.value, human_perspective_role=role.value,
        ))
    session.flush()
    return board_id, {"ALPHA": 0, "BETA": 1, "GAMMA": 2, "DELTA": 3}


def _recorder(board_id):
    from backend.app.db.recorder import GameRecorder
    return GameRecorder(
        game_id=str(uuid.uuid4()),
        board_id=board_id,
        start_player=0,
        llm_client=_FakeOpenRouterClient(),
    )


def test_persist_normal_game_maps_all_rows():
    from backend.app.db import writer
    from backend.app.db.models import (
        ClueModel, ClueTargetModel, GameModel, GameSeatModel, GuessProposalItemModel,
        GuessProposalModel, LlmCallModel, RevealEventModel, TurnModel,
    )
    from backend.app.db.session import session_scope

    with session_scope() as session:
        board_id, _ = _insert_board(session)

    rec = _recorder(board_id)
    # Clue accepted on the second attempt: attempt 0 rejected, attempt 1 accepted.
    clue_proposal = ClueProposal(
        clue="battle", count=2, reasoning="military",
        llm_calls=[_call("clue_giver", 0), _call("clue_giver", 1)],
    )
    clue_entry = ClueEntry(
        clue="battle", count=2, clue_giver=0, turn_number=0,
        targets=["ALPHA"],
        targets_resolved=[ResolvedTarget(
            word="ALPHA", card_id=0, giver_role=CardRole.AGENT, revealed_at_clue=False)],
    )
    rec.record_clue(clue_entry, proposal=clue_proposal)
    # Three-item play proposal, but only two reveals -> item 2 stays unbacked (tail NULL).
    rec.record_play_proposal(GuessProposal(
        proposals=["ALPHA", "BETA", "GAMMA"], confidence=[0.9, 0.4, 0.2],
        reasoning="r", stop_reason="s", llm_call=_call("guesser"),
    ))
    rec.record_measurement(ConfidenceRanking(
        reasoning="m",
        rankings=[RankedCard(word="ALPHA", confidence=0.8),
                  RankedCard(word="BETA", confidence=0.3)],
        llm_call=_call("measurement"),
    ))
    rec.record_reveal(card_id=0, result_str="agent", timer_tokens_after=9,
                      ended_game=False, proposal_index=0, acting_seat=0)
    rec.record_reveal(card_id=1, result_str="civilian", timer_tokens_after=8,
                      ended_game=False, proposal_index=1, acting_seat=0)
    rec.set_outcome("loss_time", 8)

    writer.persist_game(rec, status="completed")
    assert rec.flushed is True

    gid = rec.game_id
    with session_scope() as session:
        game = session.get(GameModel, gid)
        assert game is not None
        assert game.game_status == "completed"
        assert game.run_id is None
        assert game.result == "loss_time"
        assert game.start_player == 0

        seats = session.execute(select(GameSeatModel).where(
            GameSeatModel.game_id == gid)).scalars().all()
        assert {s.seat_index for s in seats} == {0, 1}
        seat0 = next(s for s in seats if s.seat_index == 0)
        assert seat0.provider == "openrouter" and seat0.model_ref == "or-model"
        assert seat0.requested_temperature == 0.5 and seat0.requested_seed == 7
        assert next(s for s in seats if s.seat_index == 1).provider == "human"

        turns = session.execute(select(TurnModel).where(
            TurnModel.game_id == gid)).scalars().all()
        assert len(turns) == 1
        turn = turns[0]
        assert turn.phase == "normal" and turn.clue_giver_seat == 0

        # Clue + its accepted-link; both attempts persisted, rejected one unlinked.
        clue = session.execute(select(ClueModel).where(
            ClueModel.turn_id == turn.id)).scalar_one()
        assert clue.clue_word == "battle" and clue.targets_raw == ["ALPHA"]
        calls = session.execute(select(LlmCallModel).where(
            LlmCallModel.turn_id == turn.id, LlmCallModel.role == "clue_giver")).scalars().all()
        assert {c.retry_index for c in calls} == {0, 1}
        accepted = next(c for c in calls if c.retry_index == 1)
        rejected = next(c for c in calls if c.retry_index == 0)
        assert clue.llm_call_id == accepted.id
        assert clue.llm_call_id != rejected.id

        targets = session.execute(select(ClueTargetModel).where(
            ClueTargetModel.clue_id == clue.id)).scalars().all()
        assert len(targets) == 1
        assert targets[0].word == "ALPHA" and targets[0].card_id == 0
        assert targets[0].giver_role == "agent"

        # Both play and measurement proposals coexist on the same turn (UNIQUE(turn_id, kind)).
        proposals = session.execute(select(GuessProposalModel).where(
            GuessProposalModel.turn_id == turn.id)).scalars().all()
        by_kind = {p.kind: p for p in proposals}
        assert set(by_kind) == {"play", "measurement"}

        play_items = session.execute(select(GuessProposalItemModel).where(
            GuessProposalItemModel.guess_proposal_id == by_kind["play"].id
        ).order_by(GuessProposalItemModel.position)).scalars().all()
        assert [i.word for i in play_items] == ["ALPHA", "BETA", "GAMMA"]
        assert [i.confidence for i in play_items] == [0.9, 0.4, 0.2]
        # Index-aligned backfill: items 0,1 linked to reveals; item 2 (unreached) stays NULL.
        assert play_items[0].reveal_event_id is not None
        assert play_items[1].reveal_event_id is not None
        assert play_items[2].reveal_event_id is None
        assert play_items[0].resolved_card_id == 0

        meas_items = session.execute(select(GuessProposalItemModel).where(
            GuessProposalItemModel.guess_proposal_id == by_kind["measurement"].id)).scalars().all()
        assert len(meas_items) == 2
        assert all(i.reveal_event_id is None for i in meas_items)

        # Play-proposal call role, measurement call role.
        play_call = session.get(LlmCallModel, by_kind["play"].llm_call_id)
        assert play_call.role == "guesser"
        meas_call = session.get(
            LlmCallModel, by_kind["measurement"].llm_call_id)
        assert meas_call.role == "measurement"

        reveals = session.execute(select(RevealEventModel).where(
            RevealEventModel.turn_id == turn.id
        ).order_by(RevealEventModel.position_in_turn)).scalars().all()
        assert [r.result_role for r in reveals] == ["agent", "civilian"]
        assert [r.ended_turn for r in reveals] == [False, True]


def test_persist_sudden_death_game():
    from backend.app.db import writer
    from backend.app.db.models import ClueModel, GuessProposalModel, TurnModel
    from backend.app.db.session import session_scope

    with session_scope() as session:
        board_id, _ = _insert_board(session)

    rec = _recorder(board_id)
    rec.record_sd_measurement(ConfidenceRanking(
        rankings=[RankedCard(word="ALPHA", confidence=0.7)],
        llm_call=_call("measurement_sd"),
    ), clue_giver_seat=1)
    rec.record_sd_play_proposal(GuessProposal(
        proposals=["ALPHA"], confidence=[0.9], reasoning="r", stop_reason="s",
        llm_call=_call("guesser_sd"),
    ), clue_giver_seat=1)
    rec.record_sd_reveal(clue_giver_seat=1, card_id=0, result_str="victory_sd",
                         timer_tokens_after=0, ended_game=True, proposal_index=0, acting_seat=0)
    rec.set_outcome("victory", 0)

    writer.persist_game(rec, status="completed")

    with session_scope() as session:
        turn = session.execute(select(TurnModel).where(
            TurnModel.game_id == rec.game_id)).scalar_one()
        assert turn.phase == "sudden_death"
        assert turn.clue_giver_seat == 1
        # No clue on a sudden-death turn.
        assert session.execute(select(ClueModel).where(
            ClueModel.turn_id == turn.id)).first() is None
        proposals = session.execute(select(GuessProposalModel).where(
            GuessProposalModel.turn_id == turn.id)).scalars().all()
        assert {p.kind for p in proposals} == {"play", "measurement"}


def test_persist_single_seat_sd_attributes_guesser_seat():
    """The SD play/measurement proposals are attributed to the recorded guesser seat, not a
    hardcoded seat 0 (seat 1 here)."""
    from backend.app.db import writer
    from backend.app.db.models import GuessProposalModel, TurnModel
    from backend.app.db.session import session_scope

    with session_scope() as session:
        board_id, _ = _insert_board(session)

    rec = _recorder(board_id)
    rec.record_sd_measurement(ConfidenceRanking(
        rankings=[RankedCard(word="ALPHA", confidence=0.7)],
        llm_call=_call("measurement_sd"),
    ), clue_giver_seat=0, guesser_seat=1)
    rec.record_sd_play_proposal(GuessProposal(
        proposals=["ALPHA"], confidence=[0.9], reasoning="r", stop_reason="s",
        llm_call=_call("guesser_sd"),
    ), clue_giver_seat=0, guesser_seat=1)
    rec.record_sd_reveal(clue_giver_seat=0, card_id=0, result_str="victory_sd",
                         timer_tokens_after=0, ended_game=True, proposal_index=0, acting_seat=1)
    rec.set_outcome("victory", 0)

    writer.persist_game(rec, status="completed")

    with session_scope() as session:
        turn = session.execute(select(TurnModel).where(
            TurnModel.game_id == rec.game_id)).scalar_one()
        proposals = session.execute(select(GuessProposalModel).where(
            GuessProposalModel.turn_id == turn.id)).scalars().all()
        assert {p.kind for p in proposals} == {"play", "measurement"}
        assert all(p.guesser_seat == 1 for p in proposals)


def test_persist_two_seat_sd_persists_both():
    """Both seats guessing + being measured on the ONE sudden-death turn now persist: four
    guess_proposal rows (2 play + 2 measurement) with distinct guesser seats under a single
    turn(phase='sudden_death'), permitted by UNIQUE(turn_id, kind, guesser_seat)."""
    from backend.app.db import writer
    from backend.app.db.models import GuessProposalModel, TurnModel
    from backend.app.db.session import session_scope

    with session_scope() as session:
        board_id, _ = _insert_board(session)

    rec = _recorder(board_id)
    for seat in (0, 1):
        rec.record_sd_play_proposal(GuessProposal(
            proposals=["ALPHA"], confidence=[0.9], reasoning="r", stop_reason="s",
            llm_call=_call("guesser_sd"),
        ), clue_giver_seat=1, guesser_seat=seat)
        rec.record_sd_measurement(ConfidenceRanking(
            rankings=[RankedCard(word="ALPHA", confidence=0.7)],
            llm_call=_call("measurement_sd"),
        ), clue_giver_seat=1, guesser_seat=seat)
    rec.set_outcome("victory", 0)

    writer.persist_game(rec, status="completed")
    assert rec.flushed is True

    with session_scope() as session:
        # Exactly one sudden-death turn holds all four proposals.
        turn = session.execute(select(TurnModel).where(
            TurnModel.game_id == rec.game_id)).scalar_one()
        assert turn.phase == "sudden_death"
        proposals = session.execute(select(GuessProposalModel).where(
            GuessProposalModel.turn_id == turn.id)).scalars().all()
        assert len(proposals) == 4
        assert {(p.kind, p.guesser_seat) for p in proposals} == {
            ("play", 0), ("play", 1), ("measurement", 0), ("measurement", 1)}


def test_persist_normal_guesser_seat_derived():
    """Normal-play guesser_seat is derived as 1 - clue_giver_seat (the guesser is the non-clue-giver),
    for either clue giver."""
    from backend.app.db import writer
    from backend.app.db.models import GuessProposalModel, TurnModel
    from backend.app.db.session import session_scope

    for clue_giver in (0, 1):
        with session_scope() as session:
            board_id, _ = _insert_board(session)

        rec = _recorder(board_id)
        clue_entry = ClueEntry(
            clue="battle", count=1, clue_giver=clue_giver, turn_number=0,
            targets=["ALPHA"],
            targets_resolved=[ResolvedTarget(
                word="ALPHA", card_id=0, giver_role=CardRole.AGENT, revealed_at_clue=False)],
        )
        rec.record_clue(clue_entry, proposal=ClueProposal(
            clue="battle", count=1, reasoning="r", llm_calls=[_call("clue_giver")]))
        rec.record_play_proposal(GuessProposal(
            proposals=["ALPHA"], confidence=[0.9], reasoning="r", stop_reason="s",
            llm_call=_call("guesser"),
        ))
        rec.set_outcome("loss_time", 0)

        writer.persist_game(rec, status="completed")

        with session_scope() as session:
            turn = session.execute(select(TurnModel).where(
                TurnModel.game_id == rec.game_id)).scalar_one()
            play = session.execute(select(GuessProposalModel).where(
                GuessProposalModel.turn_id == turn.id,
                GuessProposalModel.kind == "play")).scalar_one()
            assert play.guesser_seat == 1 - clue_giver


def test_persist_two_seat_sd_backfill_is_per_seat():
    """SD reveal.proposal_index is 0-based per acting seat's own play proposal, not a global counter.

    Both seats play; seat 1 reveals TWO cards, the second at per-seat index 1. That index-1 reveal is
    the discriminating case: a global counter over turn.reveals would hand it index 2 (its ordinal
    across all reveals), overshoot seat 1's 2-item proposal, and leave item[1] NULL - failing this
    test. Under the correct per-seat contract, item[1] is backfilled and no reveal leaks across seats.
    """
    from backend.app.db import writer
    from backend.app.db.models import GuessProposalItemModel, GuessProposalModel, TurnModel
    from backend.app.db.session import session_scope

    with session_scope() as session:
        board_id, _ = _insert_board(session)

    rec = _recorder(board_id)
    # Seat 0 proposes [ALPHA, BETA] and plays only ALPHA (index 0); BETA (index 1) stays unplayed.
    rec.record_sd_play_proposal(GuessProposal(
        proposals=["ALPHA", "BETA"], confidence=[0.9, 0.5], reasoning="r", stop_reason="s",
        llm_call=_call("guesser_sd"),
    ), clue_giver_seat=0, guesser_seat=0)
    # Seat 1 proposes [GAMMA, DELTA] and plays BOTH (per-seat indices 0 then 1).
    rec.record_sd_play_proposal(GuessProposal(
        proposals=["GAMMA", "DELTA"], confidence=[0.8, 0.4], reasoning="r", stop_reason="s",
        llm_call=_call("guesser_sd"),
    ), clue_giver_seat=0, guesser_seat=1)
    # Reveals in turn order: seat 0 @ idx0, seat 1 @ idx0, seat 1 @ idx1. The per-seat indices reset
    # per seat, so seat 1's second reveal is idx 1 (not its global ordinal of 2).
    rec.record_sd_reveal(clue_giver_seat=0, card_id=0, result_str="agent",
                         timer_tokens_after=1, ended_game=False, proposal_index=0, acting_seat=0)
    rec.record_sd_reveal(clue_giver_seat=0, card_id=2, result_str="agent",
                         timer_tokens_after=1, ended_game=False, proposal_index=0, acting_seat=1)
    rec.record_sd_reveal(clue_giver_seat=0, card_id=3, result_str="victory_sd",
                         timer_tokens_after=0, ended_game=True, proposal_index=1, acting_seat=1)
    rec.set_outcome("victory", 0)

    writer.persist_game(rec, status="completed")

    with session_scope() as session:
        turn = session.execute(select(TurnModel).where(
            TurnModel.game_id == rec.game_id)).scalar_one()
        plays = {p.guesser_seat: p for p in session.execute(select(GuessProposalModel).where(
            GuessProposalModel.turn_id == turn.id,
            GuessProposalModel.kind == "play")).scalars().all()}

        def items(seat):
            return session.execute(select(GuessProposalItemModel).where(
                GuessProposalItemModel.guess_proposal_id == plays[seat].id
            ).order_by(GuessProposalItemModel.position)).scalars().all()

        seat0_items, seat1_items = items(0), items(1)
        # Seat 0 played only item 0; its tail (item 1) stays NULL.
        assert seat0_items[0].reveal_event_id is not None
        assert seat0_items[1].reveal_event_id is None
        # Seat 1 played BOTH items; item[1] at per-seat index 1 is the discriminating assertion.
        assert seat1_items[0].reveal_event_id is not None
        assert seat1_items[1].reveal_event_id is not None
        # No cross-seat leakage: each item is linked to a reveal by its own acting seat.
        seat0_reveal_ids = {seat0_items[0].reveal_event_id}
        seat1_reveal_ids = {
            seat1_items[0].reveal_event_id, seat1_items[1].reveal_event_id}
        assert seat0_reveal_ids.isdisjoint(seat1_reveal_ids)


def test_persist_raises_on_missing_board():
    from backend.app.db import writer

    rec = _recorder(f"nonexistent-{uuid.uuid4()}")
    rec.set_outcome("victory", 5)
    with pytest.raises(ValueError, match="not in the database"):
        writer.persist_game(rec, status="completed")
    assert rec.flushed is False
