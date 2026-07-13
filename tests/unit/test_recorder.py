"""Tests for the in-memory per-game accumulator (GameRecorder).

These run without a database: the recorder is pure in-memory and imports no DB modules.
"""
import pytest

from backend.app.db.recorder import GameRecorder, result_role_of
from backend.app.models.game_schemas import ClueEntry, ConfidenceRanking, RankedCard, ResolvedTarget
from backend.app.models.llm_schemas import GuessProposal, LLMCallRecord, LLMMessage


class _FakeOpenRouterClient:
    model_name = "or-model"


class _FakeLocalClient:
    model_name = "local-model"


def _recorder(client=None) -> GameRecorder:
    return GameRecorder(
        game_id="game-1",
        board_id="board-1",
        start_player=1,
        llm_client=client if client is not None else _FakeOpenRouterClient(),
    )


def _clue_entry(clue="battle", clue_giver=0, targets=None, resolved=None) -> ClueEntry:
    return ClueEntry(
        clue=clue,
        count=2,
        clue_giver=clue_giver,
        turn_number=0,
        targets=targets or [],
        targets_resolved=resolved or [],
    )


def _call(role="clue_giver", retry_index=0, temperature=None, seed=None) -> LLMCallRecord:
    return LLMCallRecord(
        role=role,
        retry_index=retry_index,
        rendered_prompt=[LLMMessage(role="user", content="hi")],
        requested_temperature=temperature,
        requested_seed=seed,
    )


def _guess_proposal(call=None) -> GuessProposal:
    return GuessProposal(
        proposals=["ALPHA", "BETA"],
        confidence=[0.9, 0.4],
        reasoning="r",
        stop_reason="done",
        llm_call=call,
    )


# creation / seats
def test_creation_captures_seats_and_start_player():
    rec = _recorder(_FakeOpenRouterClient())
    assert rec.start_player == 1
    assert rec.flushed is False
    seat0, seat1 = rec.seats
    assert seat0.seat_index == 0 and seat0.provider == "openrouter" and seat0.model_ref == "or-model"
    assert seat1.seat_index == 1 and seat1.provider == "human" and seat1.model_ref is None


def test_local_client_maps_to_ollama_provider():
    rec = _recorder(_FakeLocalClient())
    assert rec.seats[0].provider == "ollama"
    assert rec.seats[0].model_ref == "local-model"


# result-role mapping
@pytest.mark.parametrize("result_str,expected", [
    ("agent", "agent"),
    ("victory", "agent"),
    ("victory_sd", "agent"),
    ("assassin", "assassin"),
    ("loss_assassin_sd", "assassin"),
    ("civilian", "civilian"),
    ("loss_civilian_sd", "civilian"),
])
def test_result_role_mapping(result_str, expected):
    assert result_role_of(result_str) == expected


def test_result_role_rejects_unknown():
    with pytest.raises(ValueError):
        result_role_of("nonsense")


# clue / proposal / measurement / reveal accumulation
def test_record_clue_with_llm_calls_opens_normal_turn():
    rec = _recorder()
    calls = [_call(retry_index=0), _call(retry_index=1)]
    # A ClueProposal-like object carrying attempts + reasoning.
    from backend.app.models.llm_schemas import ClueProposal
    proposal = ClueProposal(clue="battle", count=2,
                            reasoning="because", llm_calls=calls)
    entry = _clue_entry(targets=["ALPHA"], resolved=[
                        ResolvedTarget(word="ALPHA", card_id=3)])

    rec.record_clue(entry, proposal=proposal)

    assert len(rec.turns) == 1
    turn = rec.turns[0]
    assert turn.phase == "normal"
    assert turn.turn_number == 0
    assert turn.clue_giver_seat == 0
    assert turn.clue.clue_word == "battle"
    assert turn.clue.reasoning == "because"
    assert turn.clue.targets_raw == ["ALPHA"]
    assert len(turn.clue.targets_resolved) == 1
    assert [c.retry_index for c in turn.clue.llm_calls] == [0, 1]


def test_human_clue_has_no_calls_and_empty_targets():
    rec = _recorder()
    rec.record_clue(_clue_entry(clue_giver=1), proposal=None)
    turn = rec.turns[0]
    assert turn.clue.llm_calls == []
    assert turn.clue.targets_raw == []
    assert turn.clue.reasoning is None


def test_play_proposal_and_reveal_index_alignment():
    rec = _recorder()
    rec.record_clue(_clue_entry(), proposal=None)
    rec.record_play_proposal(_guess_proposal(_call(role="guesser")))
    # Two reveals from proposal items 0 and 1.
    rec.record_reveal(card_id=5, result_str="agent", timer_tokens_after=9,
                      ended_game=False, proposal_index=0, acting_seat=0)
    rec.record_reveal(card_id=6, result_str="civilian", timer_tokens_after=8,
                      ended_game=False, proposal_index=1, acting_seat=0)
    reveals = rec.turns[0].reveals
    assert [r.proposal_index for r in reveals] == [0, 1]
    assert [r.result_role for r in reveals] == ["agent", "civilian"]
    assert [r.ended_turn for r in reveals] == [False, True]


def test_record_measurement_none_is_noop():
    rec = _recorder()
    rec.record_clue(_clue_entry(), proposal=None)
    rec.record_measurement(None)
    assert rec.turns[0].measurement is None


def test_record_measurement_sets_ranking():
    rec = _recorder()
    rec.record_clue(_clue_entry(), proposal=None)
    ranking = ConfidenceRanking(
        reasoning="r",
        rankings=[RankedCard(word="ALPHA", confidence=0.7)],
        llm_call=_call(role="measurement"),
    )
    rec.record_measurement(ranking)
    assert rec.turns[0].measurement is ranking


def test_seat0_sampling_captured_on_first_observation():
    rec = _recorder()
    from backend.app.models.llm_schemas import ClueProposal
    proposal = ClueProposal(
        clue="battle", count=2,
        llm_calls=[_call(temperature=0.3, seed=42)],
    )
    rec.record_clue(_clue_entry(), proposal=proposal)
    assert rec.seats[0].requested_temperature == 0.3
    assert rec.seats[0].requested_seed == 42


# sudden death
def test_sudden_death_single_turn_no_clue():
    rec = _recorder()
    ranking = ConfidenceRanking(rankings=[RankedCard(word="X", confidence=0.5)],
                                llm_call=_call(role="measurement_sd"))
    rec.record_sd_measurement(ranking, clue_giver_seat=0)
    rec.record_sd_play_proposal(_guess_proposal(
        _call(role="guesser_sd")), clue_giver_seat=0)
    rec.record_sd_reveal(clue_giver_seat=0, card_id=7, result_str="victory_sd",
                         timer_tokens_after=0, ended_game=True, proposal_index=0, acting_seat=0)
    # Exactly one sudden-death turn, no clue, with measurement + proposal + reveal.
    sd_turns = [t for t in rec.turns if t.phase == "sudden_death"]
    assert len(sd_turns) == 1
    sd = sd_turns[0]
    assert sd.clue is None
    assert sd.clue_giver_seat == 0
    assert sd.measurement is ranking
    assert sd.play_proposal is not None
    assert len(sd.reveals) == 1
    assert sd.reveals[0].result_role == "agent"
    assert sd.reveals[0].ended_game is True


def test_ensure_sudden_death_turn_is_idempotent():
    rec = _recorder()
    first = rec.ensure_sudden_death_turn(0)
    second = rec.ensure_sudden_death_turn(1)
    assert first is second
    assert len([t for t in rec.turns if t.phase == "sudden_death"]) == 1


def test_sd_records_tag_explicit_guesser_seat():
    """The SD record methods store the guesser seat (in SD both seats guess, so it is not derivable
    from clue_giver_seat and must be carried explicitly)."""
    rec = _recorder()
    rec.record_sd_measurement(
        ConfidenceRanking(rankings=[RankedCard(word="X", confidence=0.5)],
                          llm_call=_call(role="measurement_sd")),
        clue_giver_seat=0, guesser_seat=1)
    rec.record_sd_play_proposal(
        _guess_proposal(_call(role="guesser_sd")), clue_giver_seat=0, guesser_seat=1)

    sd = [t for t in rec.turns if t.phase == "sudden_death"][0]
    assert sd.sd_guesser_seats == {1}


def test_sd_records_accumulate_both_seats_for_writer_detection():
    """When both seats reach SD on the one SD turn, the recorder retains both seat tags so the writer
    can detect the (schema-blocked) two-seat case."""
    rec = _recorder()
    rec.record_sd_play_proposal(
        _guess_proposal(_call(role="guesser_sd")), clue_giver_seat=0, guesser_seat=0)
    rec.record_sd_play_proposal(
        _guess_proposal(_call(role="guesser_sd")), clue_giver_seat=0, guesser_seat=1)

    sd = [t for t in rec.turns if t.phase == "sudden_death"][0]
    assert sd.sd_guesser_seats == {0, 1}


def test_sd_records_default_guesser_seat_is_llm():
    """The default guesser seat is the LLM (0), keeping the interactive path's positional calls."""
    rec = _recorder()
    rec.record_sd_play_proposal(
        _guess_proposal(_call(role="guesser_sd")), clue_giver_seat=1)
    sd = [t for t in rec.turns if t.phase == "sudden_death"][0]
    assert sd.sd_guesser_seats == {0}


# outcome / flushed latch

def test_set_outcome_and_flushed_latch():
    rec = _recorder()
    assert rec.result is None and rec.timer_tokens_final is None
    rec.set_outcome("victory", 4)
    assert rec.result == "victory"
    assert rec.timer_tokens_final == 4
    # The latch is a plain flag the writer sets on commit; it starts False.
    assert rec.flushed is False
    rec.flushed = True
    assert rec.flushed is True
