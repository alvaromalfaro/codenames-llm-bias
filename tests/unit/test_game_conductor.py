import json

import pytest
from unittest.mock import MagicMock, AsyncMock

from backend.app.core.engine import CodenamesDuetEngine
from backend.app.core.llm_service import LLMService
from backend.app.core.llm.client import LLMClient
from backend.app.core.game_conductor import (
    conduct_clue, conduct_guess, conduct_sd_guess,
)
from backend.app.db.recorder import GameRecorder
from backend.app.models.game_schemas import (
    Board, WordCard, CardRole, GamePhase, ClueEntry,
)
from backend.app.models.llm_schemas import (
    GuessJSONFormat, ConfidenceRankingJSONFormat, ClueJSONFormat,
)


# mocks / builders
def _mock_response(text: str) -> MagicMock:
    resp = MagicMock()
    resp.text = text
    resp.model_used = "test_model"
    resp.latency_ms = 0
    resp.raw_payload = json.loads(text)
    # Typed telemetry (None) so the audit-carrier build gets real values, not child mocks.
    resp.usage = None
    resp.finish_reason = None
    resp.provider = None
    resp.request_id = None
    resp.resolved_model = None
    resp.system_fingerprint = None
    resp.requested_temperature = None
    resp.requested_seed = None
    return resp


def _mock_client(texts: list[str]) -> MagicMock:
    """A mock LLMClient whose ``generate`` returns the given canned responses in order."""
    client = MagicMock(spec=LLMClient)
    client.model_name = "test_model"
    client.generate = AsyncMock(side_effect=[_mock_response(t) for t in texts])
    return client


def _guess_json(words, confidences=None, reasoning="r", stop_reason="done") -> str:
    confidences = confidences or [0.9] * len(words)
    items = ", ".join(
        f'{{"word": "{w}", "confidence": {c}}}' for w, c in zip(words, confidences))
    return f'{{"reasoning": "{reasoning}", "stop_reason": "{stop_reason}", "proposals": [{items}]}}'


def _rankings_json(pairs=(("BRICK", 0.9), ("DESK", 0.1)), reasoning="r") -> str:
    items = ", ".join(
        f'{{"word": "{w}", "confidence": {c}}}' for w, c in pairs)
    return f'{{"reasoning": "{reasoning}", "rankings": [{items}]}}'


def _clue_json(clue="OCEAN", count=1, reasoning="r", targets=None) -> str:
    targets = targets if targets is not None else []
    return json.dumps({"reasoning": reasoning, "clue": clue, "count": count, "targets": targets})


# The official Codenames Duet layout from the shared conftest (a Board validator requires exactly 9
# agents per seat, 3 shared). Roles are (word, human_perspective_role, llm_perspective_role). Cards
# referenced by these tests: BRICK(1) shared agent; CAVE(5) seat-0 agent; RUSSIA(4)/RIFLE(11) seat-1
# agents; DESK/BUCKET civilians.
_A, _C, _S = CardRole.AGENT, CardRole.CIVILIAN, CardRole.ASSASSIN
_CARDS = [
    ("BUCKET", _C, _C),    # 0
    ("BRICK", _A, _A),     # 1  shared agent
    ("ANT", _A, _S),       # 2
    ("LEMONADE", _S, _S),  # 3
    ("RUSSIA", _C, _A),    # 4  seat-1 agent
    ("CAVE", _A, _C),      # 5  seat-0 agent
    ("FIDDLE", _C, _C),    # 6
    ("VAMPIRE", _C, _C),   # 7
    ("TATTOO", _A, _A),    # 8  shared agent
    ("RANCH", _A, _C),     # 9  seat-0 agent
    ("LOCUST", _S, _C),    # 10
    ("RIFLE", _C, _A),     # 11 seat-1 agent
    ("VIRUS", _C, _A),     # 12 seat-1 agent
    ("IGLOO", _C, _C),     # 13
    ("MAKEUP", _C, _S),    # 14
    ("POTTER", _C, _A),    # 15 seat-1 agent
    ("CAESAR", _A, _C),    # 16 seat-0 agent
    ("NAPOLEON", _A, _A),  # 17 shared agent
    ("GOLF", _C, _C),      # 18
    ("PINE", _S, _A),      # 19 seat-1 agent
    ("DOLL", _A, _C),      # 20 seat-0 agent
    ("LUNCH", _A, _C),     # 21 seat-0 agent
    ("SKATES", _C, _C),    # 22
    ("CRAFT", _C, _C),     # 23
    ("PEW", _C, _A),       # 24 seat-1 agent
]


def _board() -> Board:
    return Board(
        board_id="tb", category="neutral",
        cards=[WordCard(id=i, text=t, human_perspective_role=h,
                        llm_perspective_role=l, category="neutral")
               for i, (t, h, l) in enumerate(_CARDS)],
    )


def _recorder(client) -> GameRecorder:
    return GameRecorder(game_id="g", board_id="tb", start_player=0, llm_client=client)


def _guessing_engine(guesser: int, agents=(5, 5)) -> CodenamesDuetEngine:
    eng = CodenamesDuetEngine(_board())
    st = eng.state
    st.current_phase = GamePhase.GUESSING
    st.guesser = guesser
    st.clue_giver = 1 - guesser
    st.turn_number = 1
    st.agents_remaining = list(agents)
    st.guesses_made_this_turn = 0
    st.current_clue = ClueEntry(clue="OCEAN", count=1,
                                clue_giver=1 - guesser, turn_number=1)
    return eng


def _sd_engine(phase: GamePhase, clue_giver: int, agents=(2, 3)) -> CodenamesDuetEngine:
    eng = CodenamesDuetEngine(_board())
    st = eng.state
    st.current_phase = phase
    st.clue_giver = clue_giver
    st.guesser = 1 - clue_giver
    st.agents_remaining = list(agents)
    st.sd_measurement_pending = True
    st.sudden_death = None
    return eng


# conduct_clue
@pytest.mark.asyncio
async def test_conduct_clue_seat0_drives_engine_and_records():
    eng = CodenamesDuetEngine(_board())
    eng.state.current_phase = GamePhase.GIVING_CLUE
    eng.state.clue_giver = 0
    eng.state.guesser = 1
    eng.state.turn_number = 1
    client = _mock_client([_clue_json(clue="OCEAN", count=1)])
    rec = _recorder(client)
    service = LLMService()

    proposal = await conduct_clue(service, client, eng, rec, player_id=0)

    assert proposal.clue == "OCEAN"
    assert eng.state.current_phase == GamePhase.GUESSING
    assert eng.state.current_clue.clue == "OCEAN"
    # One normal turn opened, carrying the clue and the model attempt.
    assert len(rec.turns) == 1
    assert rec.turns[-1].clue.clue_word == "OCEAN"
    assert rec.turns[-1].clue_giver_seat == 0
    assert len(rec.turns[-1].clue.llm_calls) == 1


# conduct_guess
@pytest.mark.asyncio
async def test_conduct_guess_seat0_equivalence():
    """Seat-0 guess drives engine + recorder: measurement recorded after the play proposal and 
    before the resolve loop; per-reveal proposal_index / acting_seat; on_reveal fires per reveal in 
    order; flush fires per reveal; all-agent proposal passes the turn."""
    eng = _guessing_engine(guesser=0, agents=(5, 5))
    client = _mock_client([_guess_json(["BRICK", "CAVE"]), _rankings_json()])
    rec = _recorder(client)
    # open the turn (clue already given)
    rec.record_clue(eng.state.current_clue, proposal=None)
    service = LLMService()

    seen: list = []
    flush = MagicMock()

    reveals = await conduct_guess(service, client, eng, rec, player_id=0,
                                  flush=flush, on_reveal=lambda cid, r, c: seen.append((cid, r, c)))

    # Return + hook agree, in order.
    assert [(cid, r) for cid, r, _ in reveals] == [(1, "agent"), (5, "agent")]
    assert seen == reveals

    turn = rec.turns[-1]
    assert turn.play_proposal.proposals == ["BRICK", "CAVE"]
    assert turn.measurement is not None
    assert [(rv.proposal_index, rv.acting_seat)
            for rv in turn.reveals] == [(0, 0), (1, 0)]

    # Two LLM calls: the play proposal, then the measurement (before the resolve loop, which makes
    # no LLM calls). This is the measurement-before-resolve ordering.
    assert client.generate.call_count == 2
    assert client.generate.call_args_list[0].kwargs["expected_format"] is GuessJSONFormat
    assert client.generate.call_args_list[1].kwargs["expected_format"] is ConfidenceRankingJSONFormat

    # Flush fired once per reveal.
    assert flush.call_count == 2
    for call in flush.call_args_list:
        assert call.args == (eng, rec)

    # All proposals were agents -> turn passed.
    assert eng.state.current_phase == GamePhase.GIVING_CLUE


@pytest.mark.asyncio
async def test_conduct_guess_seat1_drives_seat1_guess_and_measurement():
    """The web never drives seat 1 through conduction; verify it records with acting_seat=1."""
    eng = _guessing_engine(guesser=1, agents=(5, 5))
    client = _mock_client([_guess_json(["RUSSIA", "RIFLE"]),
                           _rankings_json(pairs=(("RUSSIA", 0.8), ("RIFLE", 0.7)))])
    rec = _recorder(client)
    rec.record_clue(eng.state.current_clue, proposal=None)
    service = LLMService()

    reveals = await conduct_guess(service, client, eng, rec, player_id=1,
                                  flush=MagicMock(), on_reveal=None)

    assert [(cid, r) for cid, r, _ in reveals] == [(4, "agent"), (11, "agent")]
    turn = rec.turns[-1]
    assert turn.play_proposal.proposals == ["RUSSIA", "RIFLE"]
    assert turn.measurement is not None
    assert [(rv.proposal_index, rv.acting_seat)
            for rv in turn.reveals] == [(0, 1), (1, 1)]


@pytest.mark.asyncio
async def test_conduct_guess_flush_sees_game_over_on_terminal_reveal():
    # APPLE (shared) empties both -> victory
    eng = _guessing_engine(guesser=0, agents=(1, 1))
    client = _mock_client([_guess_json(["BRICK"]), _rankings_json()])
    rec = _recorder(client)
    rec.record_clue(eng.state.current_clue, proposal=None)
    service = LLMService()

    flush_states: list[bool] = []
    flush = MagicMock(side_effect=lambda e,
                      r: flush_states.append(e.state.is_game_over))

    reveals = await conduct_guess(service, client, eng, rec, player_id=0,
                                  flush=flush, on_reveal=None)

    assert [(cid, r) for cid, r, _ in reveals] == [(1, "victory")]
    assert flush.call_count == 1
    assert flush_states == [True]  # flush observed the game-over state
    assert eng.state.is_game_over


@pytest.mark.asyncio
async def test_conduct_guess_reraises_proposal_error():
    """A raising propose_guess is re-raised unchanged (same type/message) so the caller can map it 
    to its 400; the play proposal is NOT recorded and ``proposal`` is never touched."""
    eng = _guessing_engine(guesser=0, agents=(5, 5))
    client = MagicMock(spec=LLMClient)
    client.model_name = "test_model"
    rec = _recorder(client)
    rec.record_clue(eng.state.current_clue, proposal=None)
    mock_service = MagicMock()
    mock_service.propose_guess = AsyncMock(side_effect=ValueError("boom"))

    with pytest.raises(ValueError, match="boom"):
        await conduct_guess(mock_service, client, eng, rec, player_id=0,
                            flush=MagicMock(), on_reveal=None)

    # The swallow is gone: no play proposal was recorded for this turn.
    assert rec.turns[-1].play_proposal is None


@pytest.mark.asyncio
async def test_conduct_guess_measurement_failure_does_not_propagate():
    """A measurement failure is still swallowed - the proposal succeeds, the measurement raises, 
    conduct_guess completes normally and records the play proposal with measurement absent."""
    eng = _guessing_engine(guesser=0, agents=(5, 5))
    client = _mock_client([_guess_json(["BRICK", "CAVE"])])
    rec = _recorder(client)
    rec.record_clue(eng.state.current_clue, proposal=None)
    service = LLMService()
    service.measure_and_attach_confidence_ranking = AsyncMock(
        side_effect=ValueError("measurement boom"))

    reveals = await conduct_guess(service, client, eng, rec, player_id=0,
                                  flush=MagicMock(), on_reveal=None)

    assert [(cid, r) for cid, r, _ in reveals] == [(1, "agent"), (5, "agent")]
    turn = rec.turns[-1]
    assert turn.play_proposal.proposals == ["BRICK", "CAVE"]
    assert turn.measurement is None


# conduct_sd_guess
@pytest.mark.asyncio
async def test_conduct_sd_guess_seat0_records_measurement_and_reveal():
    eng = _sd_engine(GamePhase.SUDDEN_DEATH_LLM, clue_giver=1, agents=(2, 3))
    # Seat-0 SD: measurement first (pending), then the SD proposal.
    client = _mock_client([_rankings_json(), _guess_json(["CAVE"])])
    rec = _recorder(client)
    service = LLMService()

    reveals = await conduct_sd_guess(service, client, eng, rec, player_id=0,
                                     flush=MagicMock(), on_reveal=None)

    assert [(cid, r) for cid, r, _ in reveals] == [(5, "agent")]
    sd = rec.turns[-1]
    assert sd.phase == "sudden_death"
    # clue_giver held at SD entry
    assert sd.clue_giver_seat == 1
    assert 0 in sd.sd_play_by_seat
    assert 0 in sd.sd_measurement_by_seat
    assert [(rv.proposal_index, rv.acting_seat)
            for rv in sd.reveals] == [(0, 0)]
    # Per-seat store read (not the mirror): seat 0's ranking was attached and recorded.
    assert eng.state.sudden_death.rankings_by_seat[0] is sd.sd_measurement_by_seat[0]


@pytest.mark.asyncio
async def test_conduct_sd_guess_seat1_per_seat_proposal_index():
    """Seat 1's SUDDEN_DEATH_HUMAN turn, driven by the real service through its seat-symmetric gate:
    records SD proposal/measurement with guesser_seat=1 and a per-seat proposal_index (a first
    hallucinated word is skipped, so its own item 1 is the reveal)."""
    eng = _sd_engine(GamePhase.SUDDEN_DEATH_HUMAN, clue_giver=0, agents=(0, 3))
    service = LLMService()
    # Measurement (pending) first, then the SD proposal - the same order as the seat-0 SD test.
    client = _mock_client(
        [_rankings_json(), _guess_json(["NONWORD", "RUSSIA"])])
    rec = _recorder(client)

    reveals = await conduct_sd_guess(service, client, eng, rec, player_id=1,
                                     flush=MagicMock(), on_reveal=None)

    assert [(cid, r) for cid, r, _ in reveals] == [
        (4, "agent")]  # RUSSIA, at its own index 1
    sd = rec.turns[-1]
    assert 1 in sd.sd_play_by_seat
    assert 1 in sd.sd_measurement_by_seat
    assert [(rv.proposal_index, rv.acting_seat)
            for rv in sd.reveals] == [(1, 1)]
    assert eng.state.sudden_death.rankings_by_seat[1] is sd.sd_measurement_by_seat[1]


@pytest.mark.asyncio
async def test_two_seat_sudden_death_populates_both_seats():
    """Both seats conducting SD in one game (seat 0 in SUDDEN_DEATH_LLM, then the engine hands off to
    seat 1 in SUDDEN_DEATH_HUMAN) populate sd_play_by_seat / sd_measurement_by_seat for both seats on
    the single SD turn."""
    eng = _sd_engine(GamePhase.SUDDEN_DEATH_LLM, clue_giver=1, agents=(1, 3))
    service = LLMService()

    # Seat 0: real service (SUDDEN_DEATH_LLM). Revealing BRICK empties seat 0 -> engine hands off to
    # SUDDEN_DEATH_HUMAN and re-arms the measurement flag for seat 1.
    client0 = _mock_client([_rankings_json(), _guess_json(["CAVE"])])
    rec = _recorder(client0)
    await conduct_sd_guess(service, client0, eng, rec, player_id=0,
                           flush=MagicMock(), on_reveal=None)

    assert eng.state.current_phase == GamePhase.SUDDEN_DEATH_HUMAN
    assert eng.state.sd_measurement_pending is True

    # Seat 1: the real service too (SUDDEN_DEATH_HUMAN), measurement (re-armed) then proposal.
    client1 = _mock_client([_rankings_json(), _guess_json(["RUSSIA"])])
    await conduct_sd_guess(service, client1, eng, rec, player_id=1,
                           flush=MagicMock(), on_reveal=None)

    # One collective SD turn carrying BOTH seats' proposals and measurements.
    assert len(rec.turns) == 1
    sd = rec.turns[-1]
    assert set(sd.sd_play_by_seat.keys()) == {0, 1}
    assert set(sd.sd_measurement_by_seat.keys()) == {0, 1}
    assert [(rv.proposal_index, rv.acting_seat)
            for rv in sd.reveals] == [(0, 0), (0, 1)]


@pytest.mark.asyncio
async def test_conduct_sd_guess_reraises_proposal_error():
    """A raising propose_guess_sd is printed and re-raised, so the caller can map it to its 400."""
    eng = _sd_engine(GamePhase.SUDDEN_DEATH_HUMAN, clue_giver=0, agents=(0, 3))
    # skip measurement; isolate the proposal error
    eng.state.sd_measurement_pending = False
    eng.state.sudden_death = None
    service = LLMService()
    mock_service = MagicMock()
    mock_service.propose_guess_sd = AsyncMock(side_effect=ValueError("boom"))
    client = MagicMock(spec=LLMClient)
    client.model_name = "test_model"
    rec = _recorder(client)

    with pytest.raises(ValueError, match="boom"):
        await conduct_sd_guess(mock_service, client, eng, rec, player_id=1,
                               flush=MagicMock(), on_reveal=None)


# LLMService seat-symmetric SD phase gate (direct)
@pytest.mark.asyncio
async def test_sd_gate_admits_seat0_in_sudden_death_llm():
    """Unchanged behavior: seat 0 is valid in SUDDEN_DEATH_LLM for both SD entry points."""
    state = _sd_engine(GamePhase.SUDDEN_DEATH_LLM,
                       clue_giver=1, agents=(2, 3)).state
    service = LLMService()

    proposal = await service.propose_guess_sd(
        _mock_client([_guess_json(["CAVE"])]), state, player_id=0)
    assert proposal.proposals == ["CAVE"]

    ranking = await service.elicit_confidence_ranking_sd(
        _mock_client([_rankings_json()]), state, player_id=0)
    assert ranking.rankings  # parsed a non-empty ranking


@pytest.mark.asyncio
async def test_sd_gate_admits_seat1_in_sudden_death_human():
    """New behavior: seat 1 is valid in SUDDEN_DEATH_HUMAN for both SD entry points."""
    state = _sd_engine(GamePhase.SUDDEN_DEATH_HUMAN,
                       clue_giver=0, agents=(0, 3)).state
    service = LLMService()

    proposal = await service.propose_guess_sd(
        _mock_client([_guess_json(["RUSSIA"])]), state, player_id=1)
    assert proposal.proposals == ["RUSSIA"]

    ranking = await service.elicit_confidence_ranking_sd(
        _mock_client([_rankings_json()]), state, player_id=1)
    assert ranking.rankings


@pytest.mark.asyncio
async def test_sd_gate_rejects_seat_phase_mismatch():
    """Seat/phase mismatch raises before any client call: seat 1 in LLM phase, seat 0 in HUMAN phase."""
    service = LLMService()
    # gate raises before generate() is reached
    client = MagicMock(spec=LLMClient)
    client.model_name = "test_model"

    llm_state = _sd_engine(GamePhase.SUDDEN_DEATH_LLM,
                           clue_giver=0, agents=(2, 3)).state
    with pytest.raises(ValueError):
        await service.propose_guess_sd(client, llm_state, player_id=1)
    with pytest.raises(ValueError):
        await service.elicit_confidence_ranking_sd(client, llm_state, player_id=1)

    human_state = _sd_engine(GamePhase.SUDDEN_DEATH_HUMAN,
                             clue_giver=1, agents=(2, 3)).state
    with pytest.raises(ValueError):
        await service.propose_guess_sd(client, human_state, player_id=0)
    with pytest.raises(ValueError):
        await service.elicit_confidence_ranking_sd(client, human_state, player_id=0)
    client.generate.assert_not_called()


@pytest.mark.asyncio
async def test_sd_gate_rejects_non_sd_phase():
    """A non-SD phase (GUESSING) is rejected for both SD entry points, either seat."""
    state = _guessing_engine(guesser=0).state
    service = LLMService()
    client = MagicMock(spec=LLMClient)
    client.model_name = "test_model"

    with pytest.raises(ValueError):
        await service.propose_guess_sd(client, state, player_id=0)
    with pytest.raises(ValueError):
        await service.elicit_confidence_ranking_sd(client, state, player_id=1)
