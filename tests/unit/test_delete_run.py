"""Tests for writer.delete_run: run deletion + DB cascade (migration 0005).

Require a live Postgres and are skipped when DATABASE_URL is unset, matching test_writer.py. Each
test inserts its own board / run / game (unique ids) so runs are independent and need no cleanup.
The full game graph is built through the real writer (writer.persist_game) so the cascade is
exercised against a realistic tree - a normal turn AND a two-seat sudden-death turn, with non-null
reveal_event_id backfill on both.
"""
import os
import uuid

import pytest
from sqlalchemy import func, select

from backend.app.core.game_runner import _game_identity
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
        role=role, retry_index=retry_index,
        rendered_prompt=[LLMMessage(role="user", content="prompt text")],
        requested_temperature=temperature, requested_seed=seed,
        prompt_tokens=10, completion_tokens=5,
    )


def _insert_board(session):
    """Insert a minimal board + four word cards; return board_id."""
    from backend.app.db.models import BoardModel, WordCardModel

    board_id = f"test-delete-run-{uuid.uuid4()}"
    session.add(BoardModel(board_id=board_id, type="control"))
    for card_id, text, role in [
        (0, "ALPHA", CardRole.AGENT), (1, "BETA", CardRole.CIVILIAN),
        (2, "GAMMA", CardRole.ASSASSIN), (3, "DELTA", CardRole.AGENT),
    ]:
        session.add(WordCardModel(
            board_id=board_id, card_id=card_id, text=text,
            llm_perspective_role=role.value, human_perspective_role=role.value))
    session.flush()
    return board_id


def _insert_run(session):
    from backend.app.db.models import RunModel

    run_id = str(uuid.uuid4())
    session.add(RunModel(id=run_id, master_seed=12345,
                         temperature=0.5, regime_label="test-delete-run"))
    session.flush()
    return run_id


def _recorder(*, game_id, board_id, run_id):
    from backend.app.db.recorder import GameRecorder

    rec = GameRecorder(game_id=game_id, board_id=board_id,
                       start_player=0, llm_client=_FakeOpenRouterClient())
    rec.run_id = run_id
    rec.derived_seed = 999
    return rec


def _record_full_game(rec):
    """Drive a rec through a normal turn AND a two-seat sudden-death turn with backfilled reveals."""
    # Normal turn: clue (2 attempts) + 3-item play (2 reveals -> tail NULL) + measurement.
    rec.record_clue(
        ClueEntry(clue="battle", count=2, clue_giver=0, turn_number=0, targets=["ALPHA"],
                  targets_resolved=[ResolvedTarget(word="ALPHA", card_id=0,
                                                   giver_role=CardRole.AGENT, revealed_at_clue=False)]),
        proposal=ClueProposal(clue="battle", count=2, reasoning="military",
                              llm_calls=[_call("clue_giver", 0), _call("clue_giver", 1)]))
    rec.record_play_proposal(GuessProposal(
        proposals=["ALPHA", "BETA", "GAMMA"], confidence=[0.9, 0.4, 0.2],
        reasoning="r", stop_reason="s", llm_call=_call("guesser")))
    rec.record_measurement(ConfidenceRanking(
        reasoning="m", rankings=[RankedCard(word="ALPHA", confidence=0.8),
                                 RankedCard(word="BETA", confidence=0.3)],
        llm_call=_call("measurement")))
    rec.record_reveal(card_id=0, result_str="agent", timer_tokens_after=9,
                      ended_game=False, proposal_index=0, acting_seat=0)
    rec.record_reveal(card_id=1, result_str="civilian", timer_tokens_after=8,
                      ended_game=False, proposal_index=1, acting_seat=0)
    # Two-seat sudden-death turn: both seats play + are measured, seat-0 reveal backfilled.
    for seat in (0, 1):
        rec.record_sd_play_proposal(GuessProposal(
            proposals=["DELTA"], confidence=[0.9], reasoning="r", stop_reason="s",
            llm_call=_call("guesser_sd")), clue_giver_seat=1, guesser_seat=seat)
        rec.record_sd_measurement(ConfidenceRanking(
            rankings=[RankedCard(word="DELTA", confidence=0.7)],
            llm_call=_call("measurement_sd")), clue_giver_seat=1, guesser_seat=seat)
    rec.record_sd_reveal(clue_giver_seat=1, card_id=3, result_str="victory_sd",
                         timer_tokens_after=0, ended_game=True, proposal_index=0, acting_seat=0)
    rec.set_outcome("victory", 0)


def _persist_full_game(*, game_id, board_id, run_id):
    from backend.app.db import writer

    rec = _recorder(game_id=game_id, board_id=board_id, run_id=run_id)
    _record_full_game(rec)
    writer.persist_game(rec, status="completed")
    return rec


def _row_counts_for_game(session, game_id):
    """Return {table: count} across the whole game subtree, filtered to one game."""
    from backend.app.db.models import (
        ClueModel, ClueTargetModel, GameModel, GameSeatModel, GuessProposalItemModel,
        GuessProposalModel, LlmCallModel, RevealEventModel, TurnModel,
    )

    turn_ids = session.execute(select(TurnModel.id).where(
        TurnModel.game_id == game_id)).scalars().all()
    clue_ids = session.execute(select(ClueModel.id).where(
        ClueModel.turn_id.in_(turn_ids))).scalars().all() if turn_ids else []
    gp_ids = session.execute(select(GuessProposalModel.id).where(
        GuessProposalModel.turn_id.in_(turn_ids))).scalars().all() if turn_ids else []

    def _count(model, col, ids):
        if not ids:
            return 0
        return session.execute(select(func.count()).select_from(model).where(
            col.in_(ids))).scalar_one()

    return {
        "game": session.execute(select(func.count()).select_from(GameModel).where(
            GameModel.id == game_id)).scalar_one(),
        "game_seat": session.execute(select(func.count()).select_from(GameSeatModel).where(
            GameSeatModel.game_id == game_id)).scalar_one(),
        "turn": len(turn_ids),
        "clue": len(clue_ids),
        "clue_target": _count(ClueTargetModel, ClueTargetModel.clue_id, clue_ids),
        "llm_call": session.execute(select(func.count()).select_from(LlmCallModel).where(
            LlmCallModel.game_id == game_id)).scalar_one(),
        "guess_proposal": len(gp_ids),
        "guess_proposal_item": _count(
            GuessProposalItemModel, GuessProposalItemModel.guess_proposal_id, gp_ids),
        "reveal_event": _count(RevealEventModel, RevealEventModel.turn_id, turn_ids),
    }


def test_delete_run_cascades_whole_subtree():
    from backend.app.db import writer
    from backend.app.db.session import session_scope

    with session_scope() as session:
        board_id = _insert_board(session)
        run_id = _insert_run(session)

    game_id = str(uuid.uuid4())
    _persist_full_game(game_id=game_id, board_id=board_id, run_id=run_id)

    # Precondition: every table populated, and reveal_event_id backfill is non-null somewhere
    # (so the SET-NULL cascade path is actually exercised).
    with session_scope() as session:
        before = _row_counts_for_game(session, game_id)
        from backend.app.db.models import GuessProposalItemModel, GuessProposalModel, TurnModel
        turn_ids = session.execute(select(TurnModel.id).where(
            TurnModel.game_id == game_id)).scalars().all()
        gp_ids = session.execute(select(GuessProposalModel.id).where(
            GuessProposalModel.turn_id.in_(turn_ids))).scalars().all()
        backfilled = session.execute(select(func.count()).select_from(GuessProposalItemModel).where(
            GuessProposalItemModel.guess_proposal_id.in_(gp_ids),
            GuessProposalItemModel.reveal_event_id.is_not(None))).scalar_one()
    assert all(v > 0 for v in before.values()), before
    assert backfilled > 0

    result = writer.delete_run(run_id)
    assert result.found is True
    assert result.games_deleted == 1

    # Every table for that game is now empty.
    with session_scope() as session:
        after = _row_counts_for_game(session, game_id)
        from backend.app.db.models import RunModel
        run_gone = session.execute(select(RunModel.id).where(
            RunModel.id == run_id)).first()
    assert after == {k: 0 for k in before}, after
    assert run_gone is None


def test_delete_run_preserves_shared_ingest_data():
    """board / word_card are shared ingest data, not run-owned - they survive the delete."""
    from backend.app.db import writer
    from backend.app.db.models import BoardModel, WordCardModel
    from backend.app.db.session import session_scope

    with session_scope() as session:
        board_id = _insert_board(session)
        run_id = _insert_run(session)

    _persist_full_game(game_id=str(uuid.uuid4()), board_id=board_id, run_id=run_id)
    writer.delete_run(run_id)

    with session_scope() as session:
        board = session.get(BoardModel, board_id)
        n_cards = session.execute(select(func.count()).select_from(WordCardModel).where(
            WordCardModel.board_id == board_id)).scalar_one()
    assert board is not None
    assert n_cards == 4


def test_delete_run_counts_multiple_games():
    from backend.app.db import writer
    from backend.app.db.session import session_scope

    with session_scope() as session:
        board_id = _insert_board(session)
        run_id = _insert_run(session)

    for _ in range(3):
        _persist_full_game(game_id=str(uuid.uuid4()), board_id=board_id, run_id=run_id)

    result = writer.delete_run(run_id)
    assert result.found is True
    assert result.games_deleted == 3


def test_delete_run_empty_run():
    """A run with no games deletes fine: found, zero games."""
    from backend.app.db import writer
    from backend.app.db.models import RunModel
    from backend.app.db.session import session_scope

    with session_scope() as session:
        run_id = _insert_run(session)

    result = writer.delete_run(run_id)
    assert result.found is True
    assert result.games_deleted == 0

    with session_scope() as session:
        assert session.get(RunModel, run_id) is None


def test_delete_run_nonexistent_is_noop():
    """Deleting a run that does not exist is an idempotent no-op (no raise)."""
    from backend.app.db import writer

    result = writer.delete_run(str(uuid.uuid4()))
    assert result.found is False
    assert result.games_deleted == 0


def test_delete_run_enables_rerun_of_same_identity():
    """The collision-as-a-feature payoff: a deterministic (master_seed, game_index) game persisted,
    deleted via delete_run, then persisted AGAIN under the same game.id - proving deterministic
    identity + delete_run give a working pilot -> inspect -> delete -> re-run cycle."""
    from sqlalchemy.exc import IntegrityError

    from backend.app.db import writer
    from backend.app.db.models import GameModel, RunModel
    from backend.app.db.session import session_scope

    # A unique master_seed per invocation keeps the deterministic game_id from colliding with a row
    # left by a prior run of this test (mirrors test_game_runner._unique_game_index); the identity is
    # still reused WITHIN this test to demonstrate the same-identity collision.
    master_seed, game_index = uuid.uuid4().int % 2**31, 0
    game_id, _, _ = _game_identity(master_seed, game_index)

    with session_scope() as session:
        board_id = _insert_board(session)
        run_id = _insert_run(session)

    # First persist of the deterministic identity: succeeds.
    _persist_full_game(game_id=game_id, board_id=board_id, run_id=run_id)

    # Re-persisting the SAME identity without deleting collides on the game.id primary key.
    with pytest.raises(IntegrityError):
        _persist_full_game(game_id=game_id, board_id=board_id, run_id=run_id)

    # Delete the run (unblocks the re-run), then re-create the run row and re-persist the SAME id.
    result = writer.delete_run(run_id)
    assert result.found is True and result.games_deleted == 1

    with session_scope() as session:
        assert session.get(GameModel, game_id) is None
        session.add(RunModel(id=run_id, master_seed=master_seed,
                             temperature=0.5, regime_label="test-delete-run-rerun"))

    rec = _persist_full_game(game_id=game_id, board_id=board_id, run_id=run_id)
    assert rec.flushed is True
    with session_scope() as session:
        assert session.get(GameModel, game_id) is not None

    # Clean up the re-persisted game so a re-run of the whole suite stays collision-free.
    writer.delete_run(run_id)
