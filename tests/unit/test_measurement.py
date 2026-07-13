"""
Tests for the out-of-band confidence-ranking measurement call (CIT / sudden-death metrics).

The measurement is strictly additive and side-effect-free: it must never mutate game state, never
receive the clue-giver's intended target set S, and be elicited at the correct pre-resolution instant.
"""
import json

import pytest
from unittest.mock import MagicMock, AsyncMock

from backend.app.core.engine import CodenamesDuetEngine
from backend.app.core.llm_service import LLMService
from backend.app.core.llm.client import LLMClient
from backend.app.models.game_schemas import (
    Board, GameState, GamePhase, ClueEntry, ResolvedTarget,
    ConfidenceRanking, RankedCard, SuddenDeathEntry,
)


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


def _mock_client(text: str) -> MagicMock:
    client = MagicMock(spec=LLMClient)
    client.model_name = "test_model"
    client.generate = AsyncMock(return_value=_mock_response(text))
    return client


def _rankings_json(pairs, reasoning="r") -> str:
    items = ", ".join(
        f'{{"word": "{w}", "confidence": {c}}}' for w, c in pairs)
    return f'{{"reasoning": "{reasoning}", "rankings": [{items}]}}'


def _state_snapshot(state: GameState) -> dict:
    """Snapshot every game-state field a mutation would touch (excludes measurement-only fields)."""
    return {
        "current_phase": state.current_phase,
        "clue_giver": state.clue_giver,
        "guesser": state.guesser,
        "turn_number": state.turn_number,
        "timer_tokens": state.timer_tokens,
        "agents_remaining": list(state.agents_remaining),
        "is_game_over": state.is_game_over,
        "result": state.result,
        "clue_history_len": len(state.clue_history),
        "cards": [
            (c.revealed, list(c.revealed_by), list(c.time_marker_by))
            for c in state.board.cards
        ],
    }


def _guessing_engine(valid_board_data: dict) -> CodenamesDuetEngine:
    board = Board(**valid_board_data)
    engine = CodenamesDuetEngine(board)
    engine.state.clue_giver = 1
    engine.state.guesser = 0
    engine.state.current_phase = GamePhase.GUESSING
    engine.state.current_clue = ClueEntry(
        clue="battle", count=3, clue_giver=1,
        turn_number=engine.state.turn_number)
    return engine


def _sd_engine(valid_board_data: dict) -> CodenamesDuetEngine:
    board = Board(**valid_board_data)
    engine = CodenamesDuetEngine(board)
    engine.state.current_phase = GamePhase.SUDDEN_DEATH_LLM
    engine.state.sd_measurement_pending = True
    return engine


# Format / parse

def test_build_confidence_ranking_well_formed():
    """A well-formed measurement response parses into a ConfidenceRanking, order preserved."""
    text = _rankings_json([("BRICK", 0.9), ("ANT", 0.2), ("RUSSIA", 0.55)])
    ranking = LLMService()._build_confidence_ranking(_mock_response(text))

    assert isinstance(ranking, ConfidenceRanking)
    assert [(r.word, r.confidence) for r in ranking.rankings] == [
        ("BRICK", 0.9), ("ANT", 0.2), ("RUSSIA", 0.55)]
    assert ranking.reasoning == "r"
    assert ranking.raw_payload == json.loads(text)


def test_build_confidence_ranking_missing_cards():
    """A response that omits some cards parses the present ones without error (record-only)."""
    text = _rankings_json([("BRICK", 0.9)])  # only 1 of 25 cards scored
    ranking = LLMService()._build_confidence_ranking(_mock_response(text))

    assert len(ranking.rankings) == 1
    assert ranking.rankings[0] == RankedCard(word="BRICK", confidence=0.9)


def test_build_confidence_ranking_clamps_confidence():
    """Out-of-range confidences are clamped into [0, 1] rather than rejected."""
    text = _rankings_json([("BRICK", 1.5), ("ANT", -0.3), ("RUSSIA", 0.4)])
    ranking = LLMService()._build_confidence_ranking(_mock_response(text))

    assert [r.confidence for r in ranking.rankings] == [1.0, 0.0, 0.4]


def test_build_confidence_ranking_skips_empty_words():
    """Entries with an empty word are skipped (derivable malformation, no flag fields)."""
    text = '{"reasoning": "r", "rankings": [{"word": "  ", "confidence": 0.5}, {"word": "BRICK", "confidence": 0.8}]}'
    ranking = LLMService()._build_confidence_ranking(_mock_response(text))

    assert [r.word for r in ranking.rankings] == ["BRICK"]


def test_build_confidence_ranking_json_error():
    """A non-JSON response raises ValueError (same contract as the guess proposal parser)."""
    resp = MagicMock()
    resp.text = "not json at all"
    resp.raw_payload = {}
    with pytest.raises(ValueError):
        LLMService()._build_confidence_ranking(resp)


@pytest.mark.asyncio
async def test_elicit_confidence_ranking_end_to_end(game_state_guessing):
    """elicit_confidence_ranking drives the client with the measurement format and returns the parse."""
    text = _rankings_json([("BRICK", 0.9), ("ANT", 0.1)])
    client = _mock_client(text)

    ranking = await LLMService().elicit_confidence_ranking(
        client, game_state_guessing, player_id=0)

    assert isinstance(ranking, ConfidenceRanking)
    assert [r.word for r in ranking.rankings] == ["BRICK", "ANT"]
    client.generate.assert_awaited_once()

    # Audit carrier: the measurement call is captured with role 'measurement' and the sent messages.
    assert ranking.llm_call is not None
    assert ranking.llm_call.role == "measurement"
    sent_request = client.generate.await_args_list[0][0][0]
    assert ranking.llm_call.rendered_prompt == sent_request.messages


@pytest.mark.asyncio
async def test_sd_measurement_records_llm_call(valid_board_data):
    """The sudden-death measurement is captured with role 'measurement_sd'."""
    engine = _sd_engine(valid_board_data)
    ranking = await LLMService().elicit_confidence_ranking_sd(
        _mock_client(_rankings_json([("BRICK", 0.9)])), engine.state, player_id=0)
    assert ranking.llm_call is not None
    assert ranking.llm_call.role == "measurement_sd"


@pytest.mark.asyncio
async def test_captured_measurement_prompt_excludes_targets(valid_board_data):
    """Persistence-evidence guardrail: the intended target set S never appears in the captured
    rendered_prompt of a measurement call - the exact bytes the writer will persist to
    ``llm_call.rendered_prompt`` for roles 'measurement'/'measurement_sd'."""
    sentinel = "ZZ_SENTINEL_TARGET_ZZ"

    # Standard measurement: stamp the sentinel onto the live clue's S.
    engine = _guessing_engine(valid_board_data)
    engine.state.current_clue.targets = [sentinel]
    engine.state.current_clue.targets_resolved = [
        ResolvedTarget(word=sentinel)]
    ranking = await LLMService().elicit_confidence_ranking(
        _mock_client(_rankings_json([("BRICK", 0.9)])), engine.state, player_id=0)
    assert ranking.llm_call.role == "measurement"
    for message in ranking.llm_call.rendered_prompt:
        assert sentinel not in message.content

    # Sudden-death measurement: stamp the sentinel onto a historical clue's S.
    sd_engine = _sd_engine(valid_board_data)
    sd_engine.state.clue_history.append(ClueEntry(
        clue="ocean", count=2, clue_giver=1, turn_number=1,
        targets=[sentinel], targets_resolved=[ResolvedTarget(word=sentinel)],
    ))
    sd_ranking = await LLMService().elicit_confidence_ranking_sd(
        _mock_client(_rankings_json([("BRICK", 0.9)])), sd_engine.state, player_id=0)
    assert sd_ranking.llm_call.role == "measurement_sd"
    for message in sd_ranking.llm_call.rendered_prompt:
        assert sentinel not in message.content


@pytest.mark.asyncio
async def test_elicit_confidence_ranking_wrong_phase(game_state_cg):
    """Measurement respects the same phase guard as the guesser (GUESSING only)."""
    with pytest.raises(ValueError):
        await LLMService().elicit_confidence_ranking(
            _mock_client(_rankings_json([("BRICK", 0.5)])), game_state_cg, player_id=0)


# No game-state mutation

@pytest.mark.asyncio
async def test_standard_measurement_does_not_mutate_state(valid_board_data):
    engine = _guessing_engine(valid_board_data)
    client = _mock_client(_rankings_json([("BRICK", 0.9), ("ANT", 0.2)]))

    before = _state_snapshot(engine.state)
    await LLMService().measure_and_attach_confidence_ranking(client, engine, player_id=0)
    after = _state_snapshot(engine.state)

    assert before == after
    # The only visible change is the additive measurement field on the live clue.
    assert engine.state.current_clue.confidence_ranking is not None


@pytest.mark.asyncio
async def test_sd_measurement_does_not_mutate_state(valid_board_data):
    engine = _sd_engine(valid_board_data)
    client = _mock_client(_rankings_json([("BRICK", 0.9)]))

    before = _state_snapshot(engine.state)
    await LLMService().measure_and_attach_confidence_ranking_sd(client, engine, player_id=0)
    after = _state_snapshot(engine.state)

    # No card revealed, no timer/phase/agents change - measurement is side-effect-free.
    assert before == after
    assert engine.state.sudden_death.confidence_ranking is not None


# Out of band (does not leak into or influence the play call)

@pytest.mark.asyncio
async def test_measurement_does_not_influence_play_prompt(valid_board_data):
    """The play-guess prompt is byte-identical before and after a measurement call."""
    engine = _guessing_engine(valid_board_data)
    service = LLMService()

    before = service._build_guess_request(
        engine.state, "test_model", player_id=0)
    await service.measure_and_attach_confidence_ranking(
        _mock_client(_rankings_json([("BRICK", 0.9)])), engine, player_id=0)
    after = service._build_guess_request(
        engine.state, "test_model", player_id=0)

    assert [m.content for m in before.messages] == [
        m.content for m in after.messages]


def test_measurement_request_is_standalone(valid_board_data):
    """The measurement request carries only a system + user message (no shared conversation)."""
    engine = _guessing_engine(valid_board_data)
    request = LLMService()._build_measurement_request(
        engine.state, "test_model", player_id=0)

    roles = [m.role for m in request.messages]
    assert roles == ["system", "user"]


# S never reaches the measurement prompt (standard + sudden death)

def test_targets_never_reach_standard_measurement_prompt(game_state_guessing):
    sentinel = "ZZ_SENTINEL_TARGET_ZZ"

    current_clue = game_state_guessing.current_clue
    current_clue.targets = [sentinel]
    current_clue.targets_resolved = [ResolvedTarget(word=sentinel)]

    history_clue = ClueEntry(
        clue="ocean", count=2, clue_giver=1, turn_number=0,
        targets=[sentinel], targets_resolved=[ResolvedTarget(word=sentinel)],
    )
    game_state_guessing.clue_history.insert(0, history_clue)

    request = LLMService()._build_measurement_request(
        game_state_guessing, "test_model", player_id=0)

    for message in request.messages:
        assert sentinel not in message.content


def test_targets_never_reach_sd_measurement_prompt(valid_board_data):
    sentinel = "ZZ_SENTINEL_TARGET_ZZ"
    board = Board(**valid_board_data)
    state = GameState(game_id="g", board=board)
    state.current_phase = GamePhase.SUDDEN_DEATH_LLM
    state.clue_history.append(ClueEntry(
        clue="ocean", count=2, clue_giver=1, turn_number=1,
        targets=[sentinel], targets_resolved=[ResolvedTarget(word=sentinel)],
    ))

    request = LLMService()._build_measurement_sd_request(
        state, "test_model", player_id=0)

    for message in request.messages:
        assert sentinel not in message.content


# Timing

@pytest.mark.asyncio
async def test_standard_measurement_taken_from_pre_resolution_state(valid_board_data):
    """
    The standard measurement is taken from the same state as the play-guess request: game state is
    identical at both instants, and the ranking lands on the LIVE current clue (not yet archived).
    """
    engine = _guessing_engine(valid_board_data)
    live_clue = engine.state.current_clue

    play_instant = _state_snapshot(engine.state)
    await LLMService().measure_and_attach_confidence_ranking(
        _mock_client(_rankings_json([("BRICK", 0.9)])), engine, player_id=0)
    measurement_instant = _state_snapshot(engine.state)

    assert play_instant == measurement_instant
    # Pre-resolution: the clue is still live (same object, not archived to history).
    assert engine.state.current_clue is live_clue
    assert live_clue.confidence_ranking is not None
    assert len(engine.state.clue_history) == 0


@pytest.mark.asyncio
async def test_sd_measurement_taken_before_first_resolution(valid_board_data):
    """The sudden-death ranking is attached before any card is revealed, and clears the flag."""
    engine = _sd_engine(valid_board_data)
    revealed_before = [c.revealed for c in engine.state.board.cards]

    await LLMService().measure_and_attach_confidence_ranking_sd(
        _mock_client(_rankings_json([("BRICK", 0.9)])), engine, player_id=0)

    revealed_after = [c.revealed for c in engine.state.board.cards]
    assert revealed_before == revealed_after  # no selection happened
    assert engine.state.sudden_death.confidence_ranking is not None
    assert engine.state.sd_measurement_pending is False


# Sudden death is per-game (one per game), not per-turn

def test_sudden_death_record_is_per_game_single(valid_board_data):
    state = GameState(game_id="g", board=Board(**valid_board_data))
    # Default: no record, and it is a single optional - not a list.
    assert state.sudden_death is None
    assert not isinstance(state.sudden_death, list)

    engine = CodenamesDuetEngine(Board(**valid_board_data))
    r1 = ConfidenceRanking(rankings=[RankedCard(word="BRICK", confidence=0.5)])
    r2 = ConfidenceRanking(rankings=[RankedCard(word="ANT", confidence=0.6)])
    engine.attach_sudden_death_ranking(r1)
    engine.attach_sudden_death_ranking(r2)

    # Still exactly one record - the second attach overwrites, it does not append.
    assert isinstance(engine.state.sudden_death, SuddenDeathEntry)
    assert engine.state.sudden_death.confidence_ranking is r2


# Coverage - the standard measurement prompt lists every unrevealed card

def test_standard_measurement_prompt_lists_all_unrevealed_cards(game_state_guessing):
    request = LLMService()._build_measurement_request(
        game_state_guessing, "test_model", player_id=0)
    user_content = request.messages[-1].content

    for card in game_state_guessing.board.cards:
        assert card.text in user_content


# Engine attach units + SD flag

def test_attach_confidence_ranking_requires_live_clue(valid_board_data):
    engine = CodenamesDuetEngine(Board(**valid_board_data))
    engine.state.current_clue = None
    with pytest.raises(ValueError):
        engine.attach_confidence_ranking(ConfidenceRanking())


def test_attach_confidence_ranking_sets_field(valid_board_data):
    engine = _guessing_engine(valid_board_data)
    ranking = ConfidenceRanking(
        rankings=[RankedCard(word="BRICK", confidence=0.5)])
    engine.attach_confidence_ranking(ranking)
    assert engine.state.current_clue.confidence_ranking is ranking


def test_switch_roles_sets_sd_pending_flag(valid_board_data):
    """Entering sudden death (timer exhausted, agents remaining) flags the measurement as pending."""
    engine = CodenamesDuetEngine(Board(**valid_board_data))
    engine.state.timer_tokens = 0
    engine.state.current_clue = ClueEntry(
        clue="battle", count=1, clue_giver=engine.state.clue_giver, turn_number=1)

    assert engine.state.sd_measurement_pending is False
    engine._switch_roles()

    assert engine.state.current_phase in (
        GamePhase.SUDDEN_DEATH_LLM, GamePhase.SUDDEN_DEATH_HUMAN)
    assert engine.state.sd_measurement_pending is True


# Seat symmetry: the guess/measurement/SD builders are driven by the guesser seat, using the
# per-seat reveal predicate (player_id not in card.revealed_by) that mirrors resolve_guess's guard.

def _seat1_guessing_engine(valid_board_data: dict) -> CodenamesDuetEngine:
    """A guessing-phase engine with seat 1 as the guesser (the future LLM-vs-LLM seat)."""
    board = Board(**valid_board_data)
    engine = CodenamesDuetEngine(board)
    engine.state.clue_giver = 0
    engine.state.guesser = 1
    engine.state.current_phase = GamePhase.GUESSING
    engine.state.current_clue = ClueEntry(
        clue="battle", count=3, clue_giver=0, turn_number=engine.state.turn_number)
    return engine


def _board_words_in(text: str, words: set[str]) -> set[str]:
    """The board words listed (as '- WORD') in a rendered prompt."""
    return {w for w in words if f"- {w}" in text}


def test_seat1_guess_builder_uses_per_seat_reveal_predicate(valid_board_data):
    """The seat-1 guess builder offers cards by seat 1's own revealed_by, mirroring the engine's
    per-seat resolve_guess guard: a card revealed only by seat 0 is still shown to seat 1, and a
    card seat 1 revealed is hidden. A shared `not card.revealed` reading would wrongly hide the
    seat-0-only card (both are card.revealed here)."""
    engine = _seat1_guessing_engine(valid_board_data)
    cards = engine.state.board.cards
    cards[0].revealed = True
    cards[0].revealed_by = [1]   # BUCKET: seat-1 reveal -> hidden from seat 1
    cards[1].revealed = True
    # BRICK: seat-0-only reveal -> still shown to seat 1
    cards[1].revealed_by = [0]

    user_prompt = LLMService()._build_guess_request(
        engine.state, "test_model", player_id=1).messages[-1].content

    # discriminating card: per-seat keeps it; shared would drop it
    assert "BRICK" in user_prompt
    assert "BUCKET" not in user_prompt   # seat-1 reveal excluded


def test_seat1_measurement_builder_matches_seat1_guess_words(valid_board_data):
    """The seat-1 measurement builder observes the identical unrevealed board-word set as the seat-1
    guess builder (the measurement must mirror the guess for the same seat)."""
    engine = _seat1_guessing_engine(valid_board_data)
    cards = engine.state.board.cards
    cards[0].revealed = True
    cards[0].revealed_by = [1]
    cards[1].revealed = True
    cards[1].revealed_by = [0]

    svc = LLMService()
    board_words = {c.text for c in cards}
    guess = svc._build_guess_request(
        engine.state, "m", player_id=1).messages[-1].content
    meas = svc._build_measurement_request(
        engine.state, "m", player_id=1).messages[-1].content

    assert _board_words_in(
        guess, board_words) == _board_words_in(meas, board_words)
    assert "BRICK" in _board_words_in(meas, board_words)
    assert "BUCKET" not in _board_words_in(meas, board_words)


def test_seat1_sd_builders_use_seat1_agents_and_reveals(valid_board_data):
    """The seat-1 SD guess + measurement builders both report seat 1's remaining agent count and the
    same seat-1 unrevealed-word set (they mirror each other for the second SD seat)."""
    board = Board(**valid_board_data)
    engine = CodenamesDuetEngine(board)
    engine.state.current_phase = GamePhase.SUDDEN_DEATH_HUMAN
    engine.state.agents_remaining = [3, 5]
    cards = engine.state.board.cards
    cards[0].revealed = True
    cards[0].revealed_by = [1]   # hidden from seat 1
    cards[1].revealed = True
    cards[1].revealed_by = [0]   # shown to seat 1

    svc = LLMService()
    board_words = {c.text for c in cards}
    guess = svc._build_guess_sd_request(
        engine.state, "m", player_id=1).messages[-1].content
    meas = svc._build_measurement_sd_request(
        engine.state, "m", player_id=1).messages[-1].content

    # agents_remaining[1], not seat 0's 3
    assert "5 agent" in guess and "5 agent" in meas
    assert "3 agent" not in guess
    assert _board_words_in(
        guess, board_words) == _board_words_in(meas, board_words)
    assert "BRICK" in _board_words_in(guess, board_words)
    assert "BUCKET" not in _board_words_in(guess, board_words)


def test_both_seats_reach_sudden_death_hold_both_rankings(valid_board_data):
    """A game where both seats reach sudden death holds BOTH seats' rankings, each captured at that
    seat's own pre-first-selection instant: the pending flag is re-armed at the SUDDEN_DEATH_LLM ->
    SUDDEN_DEATH_HUMAN handoff, and rankings_by_seat carries both."""
    engine = CodenamesDuetEngine(Board(**valid_board_data))
    engine.state.current_phase = GamePhase.SUDDEN_DEATH_LLM
    engine.state.agents_remaining = [1, 1]
    engine.state.sd_measurement_pending = True

    r0 = ConfidenceRanking(
        rankings=[RankedCard(word="RUSSIA", confidence=0.8)])
    engine.attach_sudden_death_ranking(r0, player_id=0)
    assert engine.state.sd_measurement_pending is False

    # Seat 0 reveals its last SD agent (CAVE id 5: human=AGENT, llm=CIVILIAN -> seat-0-only).
    assert engine.resolve_guess(5, player_id=0) == "agent"
    assert engine.state.agents_remaining[0] == 0
    assert engine.state.current_phase == GamePhase.SUDDEN_DEATH_HUMAN
    # Re-armed at the handoff so seat 1 is measured at its own pre-first-selection instant.
    assert engine.state.sd_measurement_pending is True

    r1 = ConfidenceRanking(rankings=[RankedCard(word="BRICK", confidence=0.9)])
    engine.attach_sudden_death_ranking(r1, player_id=1)
    assert engine.state.sd_measurement_pending is False

    sd = engine.state.sudden_death
    assert sd.rankings_by_seat[0] is r0
    assert sd.rankings_by_seat[1] is r1
    assert sd.confidence_ranking is r1   # backward-compat mirror = most recent


def test_targets_never_reach_seat1_measurement_prompt(valid_board_data):
    """Guardrail holds for seat 1: the clue-giver's intended target set S never reaches the
    seat-1 measurement prompt, from either the live clue or history."""
    sentinel = "ZZ_SENTINEL_TARGET_ZZ"
    engine = _seat1_guessing_engine(valid_board_data)
    engine.state.current_clue.targets = [sentinel]
    engine.state.current_clue.targets_resolved = [
        ResolvedTarget(word=sentinel)]
    engine.state.clue_history.insert(0, ClueEntry(
        clue="ocean", count=2, clue_giver=0, turn_number=0,
        targets=[sentinel], targets_resolved=[ResolvedTarget(word=sentinel)]))

    request = LLMService()._build_measurement_request(
        engine.state, "test_model", player_id=1)

    for message in request.messages:
        assert sentinel not in message.content
