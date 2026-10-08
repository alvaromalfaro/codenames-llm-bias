from typing import Optional
import uuid
import random
from pydantic import ValidationError
from backend.app.models.game_schemas import (
    Board, GamePhase, GameState, CardRole, ClueEntry, WordCard, ResolvedTarget,
    ConfidenceRanking, SuddenDeathEntry,
)
from backend.app.core.clue_validator import ClueValidator

# The sudden-death phase in which each seat (0 = LLM, 1 = human) is the one guessing.
_SD_PHASE_BY_SEAT = {0: GamePhase.SUDDEN_DEATH_LLM,
                     1: GamePhase.SUDDEN_DEATH_HUMAN}


class CodenamesDuetEngine:
    """
    The CodenamesDuetEngine class is responsible for managing the core game logic of Codenames Duet.
    It handles the game state, player interactions, and enforces the rules of the game. The engine
    maintains the current board configuration, tracks revealed cards, and determines win/loss
    conditions.
    """

    def __init__(self, board: Board, rng: random.Random | None = None,
                 game_id: str | None = None):
        """
        Initializes the CodenamesDuetEngine with a given board configuration and player identifiers.

        :param board: The Board object representing the game board configuration.
        :param rng: Optional injected random.Random instance driving the engine's randomness (the
            start-player pick). Injecting a seeded RNG makes the game reproducible and keeps the
            randomness per-instance, which is safe for concurrently interleaved games. When omitted,
            a fresh unseeded random.Random() is used, reproducing the previous behaviour.
        :param game_id: Optional externally supplied game id. The headless runner injects a
            deterministic id so seed derivation and persistence are reproducible; when omitted a fresh
            uuid4 is generated, reproducing the previous (interactive) behaviour.
        """
        self._rng = rng if rng is not None else random.Random()
        start_player = self._rng.choice([0, 1])
        self.state = GameState(
            game_id=game_id if game_id is not None else str(uuid.uuid4()),
            board=board.model_copy(deep=True),
            clue_giver=start_player,
            guesser=1 - start_player
        )
        self.clue_validator = ClueValidator(board.cards)

    def receive_clue(self, clue: str, count: int, player_id: int, raw_payload: Optional[dict] = None,
                     targets: Optional[list[str]] = None) -> ClueEntry:
        """
        Processes a clue provided by the clue-giving player.

        A clue that breaks the validity rules (§8.1) is accepted all the same, as the rules say
        (§8.4): one token is discarded from the reserve as a penalty, and the guesser guesses as if
        the clue were valid, so the turn usually spends another token when it ends. The reason is
        recorded on the clue (``ClueEntry.invalid_reason``). Only a malformed clue - blank, or with
        a count below 1 - is rejected, since it is not a clue at all.

        If the penalty takes the last token, sudden death starts at once (§7.3): the reserve is
        empty, so no turn is left to guess on, and the clue goes to the history as information both
        seats already have. The rules do not cover this case (§14).

        :param clue: The clue word provided by the clue-giving player.
        :param count: The number of cards the clue relates to.
        :param player_id: The identifier of the player providing the clue.
        :param raw_payload: Optional raw payload from the LLM response.
        :param targets: Optional intended target set S (the board words the clue-giver means its
            clue to activate). Recorded raw and as a clue-time resolved snapshot on the ClueEntry
            for measurement only. It never affects clue legality and is never transmitted to the
            guesser. A malformed or empty list is captured as-is, never rejected.

        :return: The accepted ClueEntry. Callers should read it from here: when the penalty starts
            sudden death, ``state.current_clue`` is already None.

        :raises ValueError: If the clue is malformed or if the game is not in the GIVING_CLUE phase.
        :raises PermissionError: If a player other than the clue giver attempts to provide a clue.
        """
        if self.state.current_phase != GamePhase.GIVING_CLUE:
            raise ValueError(
                "Clues can only be given during the GIVING_CLUE phase.")
        if player_id != self.state.clue_giver:
            raise PermissionError("Only the clue giver can provide a clue.")

        targets = targets or []
        try:
            clue_entry = ClueEntry(
                clue=clue.strip(),
                count=count,
                clue_giver=player_id,
                turn_number=self.state.turn_number,
                targets=targets,
                targets_resolved=self._resolve_targets(targets, player_id),
                raw_payload=raw_payload
            )
        except ValidationError as e:
            raise ValueError(str(e)) from e
        valid, reason = self.clue_validator.is_valid(clue_entry)
        if not valid:
            clue_entry.invalid_reason = reason

        # Store the clue
        self.state.current_clue = clue_entry
        self.state.guesses_made_this_turn = 0

        # Transition to the guessing phase
        self.state.current_phase = GamePhase.GUESSING

        if not valid:
            # Penalty for an invalid clue (§8.4): one token is discarded, on top of the one the turn
            # spends when it ends.
            self.state.timer_tokens -= 1
            self.state.penalty_tokens += 1
            # With the reserve empty there is no turn left: _switch_roles archives the clue and
            # starts sudden death.
            if self.state.timer_tokens == 0:
                self._switch_roles()

        return clue_entry

    def _resolve_targets(self, targets: list[str], player_id: int) -> list[ResolvedTarget]:
        """
        Resolves the clue-giver's intended target set S against the authoritative board, from the
        CLUE-GIVER's perspective and using the current (clue-time) reveal state. This is a
        measurement snapshot only; it is never used for game rules and never reaches the guesser.

        Each emitted word is matched case-insensitively against the board. A word that maps to a
        card carries its ``card_id``, the card's role from the giver's perspective (LLM role when
        ``player_id == 0``, otherwise the human role), and the card's reveal state at this moment.
        An unmappable word yields an all-``None`` snapshot (the diagnostic that S was malformed).

        :param targets: The intended target words exactly as emitted by the clue-giver.
        :param player_id: The seat of the clue-giver (0 = LLM, 1 = human).

        :return: One ResolvedTarget per emitted word, in order.
        """
        board = self.state.board
        resolved: list[ResolvedTarget] = []
        for word in targets:
            card = next(
                (c for c in board.cards if c.text.lower() == word.lower()), None)
            if card is None:
                resolved.append(ResolvedTarget(word=word))
                continue
            giver_role = (card.llm_perspective_role if player_id == 0
                          else card.human_perspective_role)
            resolved.append(ResolvedTarget(
                word=word,
                card_id=card.id,
                giver_role=giver_role,
                revealed_at_clue=card.revealed,
            ))
        return resolved

    def attach_confidence_ranking(self, ranking: ConfidenceRanking) -> None:
        """
        Attaches an out-of-band confidence ranking to the current turn's ClueEntry, at the
        pre-resolution instant (the clue is live, not yet archived). This mirrors how the resolved
        target snapshot is attached to the same record: it is measurement-only, never used for game
        rules, and never reaches the guesser.

        :param ranking: The parsed confidence ranking over the unrevealed cards for this turn.

        :raises ValueError: If there is no live current clue to attach the ranking to.
        """
        if self.state.current_clue is None:
            raise ValueError(
                "No current clue to attach a confidence ranking to.")
        self.state.current_clue.confidence_ranking = ranking

    def attach_sudden_death_ranking(self, ranking: ConfidenceRanking, player_id: int = 0) -> None:
        """
        Attaches the out-of-band sudden-death confidence ranking to the per-game SuddenDeathEntry,
        creating that record on first use, and consumes the pending flag so each seat's measurement
        happens exactly once (the flag is re-armed at the SD seat handoff in _reveal_agent).

        The ranking is stored per guesser seat in ``rankings_by_seat`` (the authoritative store);
        ``confidence_ranking`` is also updated as a backward-compat mirror of the most recent attach.

        :param ranking: The parsed confidence ranking over the guesser's remaining agents at that
            seat's sudden-death entry.
        :param player_id: The seat of the guesser whose ranking this is (0 = LLM, 1 = human).
        """
        if self.state.sudden_death is None:
            self.state.sudden_death = SuddenDeathEntry()
        self.state.sudden_death.rankings_by_seat[player_id] = ranking
        self.state.sudden_death.confidence_ranking = ranking
        self.state.sd_measurement_pending = False

    def resolve_guess(self, card_id: int, player_id: int) -> str:
        """
        Processes a guess made by the guessing player.

        The card must still be reachable for ``player_id`` (``WordCard.is_guessable_by``): a word
        covered by an agent card or by two time tokens is out of play for BOTH seats, and a word
        already marked by this seat's own time token is out of play for this seat only - the other
        seat may still touch it.

        :param card_id: The ``id`` of the card being guessed.
        :param player_id: The identifier of the player making the guess.

        :return: A string indicating the result of the guess ("agent", "agent_turn_end",
            "assassin", "civilian", "victory", or a sudden-death result). "agent_turn_end" is a hit
            that left the guesser with no pending words, which ends the turn (§6.8).

        :raises ValueError: If no card carries that id, if the card is covered (agent card or two
            time tokens), or if it is already marked by this player's own time token.
        :raises PermissionError: If a player other than the guesser attempts to make a guess, the 
            game is not in the GUESSING or SUDDEN_DEATH phase, or if the guesser has already
            revealed all of their agents in the SUDDEN_DEATH phase.
        """
        _sd_phases = [GamePhase.SUDDEN_DEATH_HUMAN, GamePhase.SUDDEN_DEATH_LLM]
        if self.state.current_phase not in [GamePhase.GUESSING] + _sd_phases:
            raise PermissionError(
                "Guesses can only be made during the GUESSING, SUDDEN_DEATH_HUMAN, or SUDDEN_DEATH_LLM phase.")

        if self.state.current_phase == GamePhase.GUESSING and player_id != self.state.guesser:
            raise PermissionError("Only the guesser can make guesses.")

        if self.state.current_phase == GamePhase.SUDDEN_DEATH_HUMAN and player_id != 1:
            raise PermissionError(
                "Only the human can guess during SUDDEN_DEATH_HUMAN phase.")

        if self.state.current_phase == GamePhase.SUDDEN_DEATH_LLM and player_id != 0:
            raise PermissionError(
                "Only the LLM can guess during SUDDEN_DEATH_LLM phase.")

        if self.state.current_phase in _sd_phases and self.state.pending_words[player_id] == 0:
            raise PermissionError(
                "The player has already revealed all of their agents and cannot make more guesses.")

        card = self.state.board.get_card_by_id(card_id)
        if card is None:
            raise ValueError(
                f"There is no card with id {card_id} on this board.")

        # Coverage is a property of the word, not of the seat asking: once an agent card - or a
        # second time token - sits on a word, NEITHER player may touch it again. Reading the per-seat
        # bookkeeping (revealed_by) here instead would let the seat that did not cover the word
        # re-guess it and have it resolved against the new clue giver's face.
        if card.revealed:
            raise ValueError(
                "This card is covered by an agent card and cannot be guessed.")

        if len(card.time_marker_by) == 2:
            raise ValueError(
                "This card is covered by two time tokens and cannot be guessed.")

        if player_id in card.time_marker_by:
            raise ValueError(
                "This card is currently marked by a time token and cannot be guessed.")

        # Determine the role of the card for the guessing player and update the number of guesses made
        card_role = card.human_perspective_role if player_id == 0 else card.llm_perspective_role
        self.state.guesses_made_this_turn += 1

        # Resolve the guess based on the current game phase and return result
        if self.state.current_phase in [GamePhase.SUDDEN_DEATH_HUMAN, GamePhase.SUDDEN_DEATH_LLM]:
            return self._resolve_guess_sudden_death(card, card_role, player_id)

        return self._resolve_guess_normal(card, card_role)

    def pass_turn(self, player_id: int):
        """
        Allows the guessing player to pass their turn to the clue-giving player.

        :param player_id: The identifier of the player attempting to pass their turn.

        :raises PermissionError: If a player other than the guesser attempts to pass their turn or 
            if the game is not in the GUESSING phase.
        :raises ValueError: If the guesser attempts to pass their turn without making at least one 
            guess.
        """
        if self.state.current_phase != GamePhase.GUESSING:
            raise PermissionError(
                "Turns can only be passed during the GUESSING phase.")
        if player_id != self.state.guesser:
            raise PermissionError("Only the guesser can pass the turn.")
        if self.state.guesses_made_this_turn < 1:
            raise ValueError(
                "The guesser must make at least one guess before passing.")

        # The guesser takes a token from the reserve and keeps it, check face up (§6.5)
        self.state.timer_tokens -= 1
        self.state.check_tokens += 1

        self._switch_roles()

    def concede_sudden_death(self, player_id: int) -> str:
        """
        Ends sudden death as a loss because the seat guessing stops with words still pending.

        The rules only describe two ways out of sudden death - every guess a hit (victory) or any
        miss (loss) - and say nothing about stopping (§7.3, §14). No clue will ever come again, and
        the game can only be won once this seat's words are found, so stopping can never lead to a
        win: it is treated as conceding, and both players lose without touching a card. Without this
        action a seat that will not, or cannot, propose a playable card leaves the game stuck in its
        sudden-death phase.

        :param player_id: The seat whose sudden-death turn it is.

        :return: "loss_stopped_sd".

        :raises PermissionError: If the game is not in a sudden-death phase, or if ``player_id`` is
            not the seat guessing in it.
        """
        if self.state.current_phase not in _SD_PHASE_BY_SEAT.values():
            raise PermissionError(
                "Sudden death can only be conceded during the SUDDEN_DEATH_HUMAN or SUDDEN_DEATH_LLM phase.")
        if self.state.current_phase != _SD_PHASE_BY_SEAT.get(player_id):
            raise PermissionError(
                "Only the player guessing in sudden death can concede it.")

        self._finish_game(result="loss_stopped_sd")
        return "loss_stopped_sd"

    def _resolve_guess_normal(self, card: WordCard, card_role: CardRole) -> str:
        """
        Resolves a guess during the normal guessing phase. If the guessed card is an agent, it is
        revealed. If it's a civilian, the time token is placed and the turn ends. If it's an
        assassin, the game ends immediately with a loss.

        :param card: The WordCard object representing the guessed card.
        :param card_role: The role of the guessed card for the guessing player.

        :return: A string indicating the result of the guess ("agent", "agent_turn_end",
            "assassin", "civilian", "victory").
        """
        if card_role == CardRole.AGENT:
            return self._reveal_agent(card, guessed_by=self.state.guesser)
        elif card_role == CardRole.ASSASSIN:
            self._finish_game(result="loss_assassin")
            return "assassin"
        else:
            card.time_marker_by.append(self.state.guesser)
            if len(card.time_marker_by) == 2:
                self.clue_validator.remove_word(card.text)
            self.state.timer_tokens -= 1
            self.state.bystander_tokens += 1
            self._switch_roles()

            return "civilian"

    def _resolve_guess_sudden_death(self, card: WordCard, card_role: CardRole, player_id: int) -> str:
        """
        In the sudden death phase, both players are effectively guessers and make guesses on their
        own cards. If a player guesses an agent, it is revealed as normal. If they guess a civilian
        or assassin, the game ends immediately with a loss.

        :param card: The WordCard object representing the guessed card.
        :param card_role: The role of the guessed card for the guessing player.
        :param player_id: The identifier of the player making the guess.
        """
        if card_role == CardRole.AGENT:
            return self._reveal_agent(card, guessed_by=player_id)
        else:
            self._finish_game(result=f"loss_{card_role.value}_sd")
            return f"loss_{card_role.value}_sd"

    def _reveal_agent(self, card: WordCard, guessed_by: int):
        """
        Reveals an agent card and updates the game state accordingly. Checks for win conditions
        after revealing the card; a win in normal play spends the turn's token as a check (§10). In
        normal play, if the guesser has no pending words left after the reveal, its turn ends on the
        spot as a voluntary stop (§6.8).

        :param card: The WordCard object representing the guessed card.
        :param guessed_by: The identifier of the player who made the guess that revealed the agent

        :return: A string indicating the result of the guess ("agent", "agent_turn_end" when the
            reveal also ended the turn, "victory" or "victory_sd").
        """
        # Reveal the card
        card.revealed = True
        card.revealed_by.append(guessed_by)
        self.clue_validator.remove_word(card.text)

        # The word stops being pending for the guesser, and also for the giver if it is a shared agent
        # (green on both sides).
        self.state.pending_words[guessed_by] -= 1
        if card.llm_perspective_role == card.human_perspective_role == CardRole.AGENT:
            self.state.pending_words[1 - guessed_by] -= 1
            card.revealed_by.append(1 - guessed_by)

        # Check for win condition
        if self.state.pending_words[0] == 0 and self.state.pending_words[1] == 0:
            in_sd = self.state.current_phase in [
                GamePhase.SUDDEN_DEATH_HUMAN, GamePhase.SUDDEN_DEATH_LLM]
            # The winning turn also spends a token (§10). The rules do not say whether it counts as
            # a check (§14); here it does, since the turn ends on a hit, just as a voluntary stop.
            # In sudden death the reserve is already empty, so there is nothing to spend.
            if not in_sd:
                self.state.timer_tokens -= 1
                self.state.check_tokens += 1
            res = "victory_sd" if in_sd else "victory"
            self._finish_game(result=res)
            return res

        # The guesser just covered the last word it had to find (§6.8): the giver must tell it there
        # is nothing left, and every word it could still touch is beige or black on the giver's face.
        # The turn ends here as a voluntary stop - one token, exactly as pass_turn - and
        # _switch_roles hands every remaining guess to the partner.
        if (self.state.current_phase == GamePhase.GUESSING
                and self.state.pending_words[guessed_by] == 0):
            self.pass_turn(guessed_by)
            return "agent_turn_end"

        # LLM just found their last agent in SUDDEN_DEATH_LLM -> hand off to Human (seat 1); this is
        # the only handoff of the fixed LLM -> human order (see _switch_roles). Re-arm the
        # measurement flag so seat 1's sudden-death confidence ranking is elicited once, at its own
        # pre-first-selection instant (consumed at that seat's SD proposal entry).
        if (self.state.current_phase == GamePhase.SUDDEN_DEATH_LLM
                and self.state.pending_words[0] == 0):
            self.state.current_phase = GamePhase.SUDDEN_DEATH_HUMAN
            if self.state.pending_words[1] > 0:
                self.state.sd_measurement_pending = True

        return "agent"

    def _finish_game(self, result: str):
        """
        Finishes the game by setting the game over flag, updating the current phase to GAME_OVER, 
        and storing the result.

        :param result: A string indicating the result of the game: "victory", "loss_assassin", or a
            sudden-death ending: "victory_sd", "loss_civilian_sd", "loss_assassin_sd",
            "loss_stopped_sd".
        """
        self.state.is_game_over = True
        self.state.current_phase = GamePhase.GAME_OVER
        self.state.result = result

    def _switch_roles(self):
        """
        Hands over to the next turn: the roles of the clue giver and guesser alternate (§6.1) unless
        the seat that would guess next has no pending words left, in which case the roles are kept -
        the seat with nothing left to guess gives every remaining clue and the other is always the
        guesser (§6.8). Also resets the current clue and count and updates the turn number. If the
        timer tokens have run out and there are still pending words, transitions to the
        SUDDEN_DEATH phase.
        """
        next_giver, next_guesser = self.state.guesser, self.state.clue_giver
        # Handing the guess to a seat with nothing pending would force it to touch a word that is
        # beige or black on the giver's face (it cannot pass without a guess). Both seats can never
        # be at 0 here: that is a victory, which ends the game without switching roles.
        if self.state.pending_words[next_guesser] == 0:
            next_giver, next_guesser = next_guesser, next_giver
        self.state.clue_giver, self.state.guesser = next_giver, next_guesser
        self.state.current_phase = GamePhase.GIVING_CLUE
        self.state.guesses_made_this_turn = 0
        self.state.turn_number += 1

        if self.state.current_clue is not None:
            self.state.clue_history.append(self.state.current_clue)
        self.state.current_clue = None

        # If the timer tokens have run out and there are still pending words, transition to the
        # sudden death phase - LLM goes first unless it has nothing pending. When both seats have
        # words pending the rules let them guess in any order (§7.3); fixing the order LLM -> human
        # is a deliberate deviation that gives every seat a single sudden-death play: the human may
        # only guess once the LLM has found all its words. A seat that stops before that concedes
        # (concede_sudden_death).
        if self.state.timer_tokens <= 0 and self._any_pending_words():
            if self.state.pending_words[0] == 0:
                self.state.current_phase = GamePhase.SUDDEN_DEATH_HUMAN
            else:
                self.state.current_phase = GamePhase.SUDDEN_DEATH_LLM
            # Flag the pre-selection instant so the out-of-band sudden-death confidence ranking is
            # elicited exactly once, before the first sudden-death guess. Detected here; consumed at
            # the LLM's sudden-death proposal entry.
            self.state.sd_measurement_pending = True

    def _any_pending_words(self) -> bool:
        """
        Checks if either player still has pending words.

        :return: True if either player still has words to find, False otherwise.
        """
        return any(pending > 0 for pending in self.state.pending_words)
