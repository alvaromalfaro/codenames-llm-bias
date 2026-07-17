"""Seat-parameterized, HTTP-agnostic game-conduction layer.

This layer orchestrates the engine, service and recorder; it renders no HTML, imports no Jinja
/ FastAPI, and performs no persistence. Two hooks let the caller keep its own concerns:

- ``flush(engine, recorder)`` is invoked at each post-reveal point (the interactive path passes its
  terminal-flush trigger; the runner injects its own persistence).
- ``on_reveal(card_id, result, card)`` is invoked at the exact mid-loop instant (after the reveal is
  recorded and flushed, before the loop's break decision) so the caller can render each reveal
  against the live, intermediate ``engine.state`` - a byte-for-byte match with the prior inline
  rendering. It is optional (the runner passes ``None``).
"""
from __future__ import annotations

import logging
from typing import Callable, Optional

from backend.app.models.game_schemas import WordCard
from backend.app.models.llm_schemas import ClueProposal

logger = logging.getLogger(__name__)

# A reveal, as returned to the caller: (card_id, resolve_guess result string, card).
Reveal = tuple[int, str, WordCard]
FlushHook = Callable[[object, object], None]
RevealHook = Callable[[int, str, WordCard], None]


async def conduct_clue(service, client, engine, recorder, *, player_id: int,
                       seed: Optional[int] = None) -> ClueProposal:
    """Conduct one LLM clue-giving turn for ``player_id``.

    Returns the ``ClueProposal`` (the interactive caller renders from ``engine.state`` instead).
    """
    proposal = await service.propose_clue(
        client, engine.state, engine.clue_validator, player_id=player_id, seed=seed)

    engine.receive_clue(proposal.clue, proposal.count,
                        player_id=player_id, raw_payload=proposal.raw_payload,
                        targets=proposal.targets)

    # Record the clue with all its model attempts (accepted + rejected) from the proposal.
    recorder.record_clue(engine.state.current_clue, proposal=proposal)

    print(
        f"LLM proposed clue: {proposal.clue} ({proposal.count}) with reasoning: {proposal.reasoning}")

    return proposal


async def conduct_guess(service, client, engine, recorder, *, player_id: int,
                        flush: FlushHook, on_reveal: Optional[RevealHook] = None,
                        seed: Optional[int] = None,
                        measurement_seed: Optional[int] = None) -> list[Reveal]:
    """Conduct one LLM normal-play guessing turn for ``player_id``.

    Returns the ordered list of resolved ``(card_id, result, card)`` reveals.

    This never returns with ``engine.state.current_phase == GUESSING``. Either the loop broke on a 
    non-agent reveal (the engine advanced the phase), or the ``for/else`` passed the turn (the 
    engine advanced the phase), or an exception propagated. This is what keeps the phase-driven
    driver loop from re-dispatching GUESSING for the same turn and overwriting this turn's records.
    """
    proposal = await service.propose_guess(client, engine.state, player_id=player_id, seed=seed)

    # Record the ordered play proposal (kind='play') for this turn.
    recorder.record_play_proposal(proposal)

    # Out-of-band measurement: elicit the confidence ranking over all unrevealed cards at the
    # pre-resolution instant (same state as the play-guess request above), before the resolve loop.
    # Strictly additive and side-effect-free; a failure here must never break game play.
    try:
        await service.measure_and_attach_confidence_ranking(
            client, engine, player_id=player_id, seed=measurement_seed)
    except (ValueError, PermissionError) as e:
        print(f"Error during LLM confidence-ranking measurement: {str(e)}")

    # Record the measurement (no-op if it failed above).
    recorder.record_measurement(engine.state.current_clue.confidence_ranking)

    reveals: list[Reveal] = []
    for idx, word in enumerate(proposal.proposals):
        card_id = engine.state.board.get_card_id_by_word(word)
        if card_id is None:
            # LLM hallucinated a word not on the board
            continue

        try:
            result = engine.resolve_guess(card_id, player_id)
        except (ValueError, PermissionError) as e:
            # The engine refused this item (an already-revealed / time-marked card - models re-propose
            # played cards because the guesser prompt carries the clue history). Treat it like a
            # hallucinated word: skip it (its proposal item keeps reveal_event_id NULL) and continue,
            # so a malformed item never costs the turn and never leaves the loop before the for/else
            # passes the turn. Breaking here would return with the phase still GUESSING and let the
            # driver re-dispatch the same turn, silently overwriting this turn's proposal record.
            logger.warning(
                "Skipping unplayable guess %r (seat %s): %s", word, player_id, e)
            continue

        # Index-aligned reveal capture: idx is the position in the play proposal's items.
        recorder.record_reveal(
            card_id=card_id,
            result_str=result,
            timer_tokens_after=engine.state.timer_tokens,
            ended_game=engine.state.is_game_over,
            proposal_index=idx,
            acting_seat=player_id,
        )
        flush(engine, recorder)

        print(f"LLM proposed guess: {word}")

        card = engine.state.board.cards[card_id]
        if on_reveal is not None:
            on_reveal(card_id, result, card)
        reveals.append((card_id, result, card))

        if result != "agent":
            # civilian, assassin, or victory - turn or game ended
            break
    else:
        # Loop exhausted with no turn-ending reveal: every item was an agent, unmappable, or
        # unplayable. Pass the turn to advance the phase. If pass_turn raises here, the loop resolved
        # zero guesses (guesses_made_this_turn == 0) - Duet requires >=1 guess before a pass, so this
        # is a state the rules do not contemplate (a model that produced no playable card). Let the
        # error propagate to the caller's boundary (headless: flush status='error'; web: 400) rather
        # than swallow it, which would leave the phase at GUESSING and spin the driver.
        engine.pass_turn(player_id)

    return reveals


async def conduct_sd_guess(service, client, engine, recorder, *, player_id: int,
                           flush: FlushHook, on_reveal: Optional[RevealHook] = None,
                           seed: Optional[int] = None,
                           measurement_seed: Optional[int] = None) -> list[Reveal]:
    """Conduct one sudden-death guessing turn for ``player_id`` (either SD seat).

    Returns the ordered list of resolved ``(card_id, result, card)`` reveals.
    """
    # The sudden-death turn has no clue; record the clue_giver held at SD entry as its clue_giver_seat.
    sd_clue_giver = engine.state.clue_giver

    # Out-of-band measurement: on entry to this seat's sudden-death turn, if the engine flagged the
    # sudden-death transition, elicit and attach the confidence ranking once, before any selection.
    # Strictly additive; a failure here must never break game play.
    if engine.state.sd_measurement_pending:
        try:
            await service.measure_and_attach_confidence_ranking_sd(
                client, engine, player_id=player_id, seed=measurement_seed)
        except (ValueError, PermissionError) as e:
            print(
                f"Error during LLM sudden-death confidence-ranking measurement: {str(e)}")

    # Record the SD measurement (no-op if it failed or was already taken this game). Read the
    # authoritative per-seat store; for seat 0 this is the same object as the scalar mirror.
    if engine.state.sudden_death is not None:
        recorder.record_sd_measurement(
            engine.state.sudden_death.rankings_by_seat.get(player_id),
            sd_clue_giver, guesser_seat=player_id)

    try:
        proposal = await service.propose_guess_sd(client, engine.state, player_id=player_id, seed=seed)
    except (ValueError, PermissionError) as e:
        print(f"Error during LLM sudden death guess proposal: {str(e)}")
        raise

    # Record the SD play proposal (kind='play') on the sudden-death turn.
    recorder.record_sd_play_proposal(
        proposal, sd_clue_giver, guesser_seat=player_id)

    reveals: list[Reveal] = []
    for idx, word in enumerate(proposal.proposals):
        card_id = engine.state.board.get_card_id_by_word(word)
        if card_id is None:
            continue

        try:
            result = engine.resolve_guess(card_id, player_id)
        except (ValueError, PermissionError):
            break

        # Per-seat proposal_index: idx is 0-based within THIS seat's own SD proposal (each seat
        # conducts its own loop), satisfying the writer's per-seat backfill contract.
        recorder.record_sd_reveal(
            clue_giver_seat=sd_clue_giver,
            card_id=card_id,
            result_str=result,
            timer_tokens_after=engine.state.timer_tokens,
            ended_game=engine.state.is_game_over,
            proposal_index=idx,
            acting_seat=player_id,
        )
        flush(engine, recorder)

        print(f"LLM sudden death guess: {word}")

        card = engine.state.board.cards[card_id]
        if on_reveal is not None:
            on_reveal(card_id, result, card)
        reveals.append((card_id, result, card))

        if result != "agent":
            break

    return reveals
