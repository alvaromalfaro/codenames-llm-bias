from typing import Optional
import uuid
import random
from pydantic import ValidationError
from backend.app.models.game_schemas import Board, GamePhase, GameState, CardRole, ClueEntry, WordCard
from backend.app.core.clue_validator import ClueValidator


class CodenamesDuetEngine:
    """
    The CodenamesDuetEngine class is responsible for managing the core game logic of Codenames Duet.
    It handles the game state, player interactions, and enforces the rules of the game. The engine
    maintains the current board configuration, tracks revealed cards, and determines win/loss
    conditions.
    """

    def __init__(self, board: Board):
        """
        Initializes the CodenamesDuetEngine with a given board configuration and player identifiers.

        :param board: The Board object representing the game board configuration.
        """
        start_player = random.choice([0, 1])
        self.state = GameState(
            # TODO: The game id will come from the database when it's implemented
            game_id=str(uuid.uuid4()),
            board=board,
            clue_giver=start_player,
            guesser=1 - start_player
        )
        self.clue_validator = ClueValidator(board.cards)

    def receive_clue(self, clue: str, count: int, player_id: int, raw_payload: Optional[dict] = None):
        """
        Processes a clue provided by the clue-giving player.

        :param clue: The clue word provided by the clue-giving player.
        :param count: The number of cards the clue relates to.
        :param player_id: The identifier of the player providing the clue.
        :param raw_payload: Optional raw payload from the LLM response.

        :raises ValueError: If the clue is invalid or if it's not the clue-giving player's turn.
        :raises PermissionError: If a player other than the clue giver attempts to provide a clue.
        """
        if self.state.current_phase != GamePhase.GIVING_CLUE:
            raise ValueError(
                "Clues can only be given during the GIVING_CLUE phase.")
        if player_id != self.state.clue_giver:
            raise PermissionError("Only the clue giver can provide a clue.")

        try:
            clue_entry = ClueEntry(
                clue=clue,
                count=count,
                clue_giver=player_id,
                turn_number=self.state.turn_number,
                raw_payload=raw_payload
            )
        except ValidationError as e:
            raise ValueError(str(e)) from e
        valid, reason = self.clue_validator.is_valid(clue_entry)
        if not valid:
            raise ValueError(f"Invalid clue: {reason}")

        # Store the clue
        self.state.current_clue = clue_entry
        self.state.guesses_made_this_turn = 0

        # Transition to the guessing phase
        self.state.current_phase = GamePhase.GUESSING

    def resolve_guess(self, card_id: int, player_id: int) -> str:
        """
        Processes a guess made by the guessing player.

        :param card_id: The identifier of the card being guessed.
        :param player_id: The identifier of the player making the guess.

        :return: A string indicating the result of the guess ("agent", "assassin", "civilian",
            "victory").

        :raises ValueError: If the card is already revealed, marked by a time token, or if it's not 
            the guessing player's turn.
        :raises PermissionError: If a player other than the guesser attempts to make a guess, the 
            game is not in the GUESSING or SUDDEN_DEATH phase, or if the guesser has already
            revealed all of their agents in the SUDDEN_DEATH phase.
        """
        if self.state.current_phase not in [GamePhase.GUESSING, GamePhase.SUDDEN_DEATH]:
            raise PermissionError(
                "Guesses can only be made during the GUESSING or SUDDEN_DEATH phase.")

        if self.state.current_phase == GamePhase.GUESSING and player_id != self.state.guesser:
            raise PermissionError("Only the guesser can make guesses.")

        if self.state.current_phase == GamePhase.SUDDEN_DEATH and self.state.agents_remaining[player_id] == 0:
            raise PermissionError(
                "The guesser has already revealed all of their agents and cannot make more guesses.")

        card = self.state.board.cards[card_id]
        if player_id in card.revealed_by:
            raise ValueError("This card has already been revealed.")

        if player_id in card.time_marker_by:
            raise ValueError(
                "This card is currently marked by a time token and cannot be guessed.")

        # Determine the role of the card for the guessing player and update the number of guesses made
        card_role = card.human_perspective_role if player_id == 0 else card.llm_perspective_role
        self.state.guesses_made_this_turn += 1

        # Resolve the guess based on the current game phase and return result
        if self.state.current_phase == GamePhase.SUDDEN_DEATH:
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

        # Decrement the timer token
        self.state.timer_tokens -= 1

        self._switch_roles()

    def _resolve_guess_normal(self, card: WordCard, card_role: CardRole) -> str:
        """
        Resolves a guess during the normal guessing phase. If the guessed card is an agent, it is
        revealed. If it's a civilian, the time token is placed and the turn ends. If it's an
        assassin, the game ends immediately with a loss.

        :param card: The WordCard object representing the guessed card.
        :param card_role: The role of the guessed card for the guessing player.

        :return: A string indicating the result of the guess ("agent", "assassin", "civilian", 
            "victory").
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
        after revealing the card.

        :param card: The WordCard object representing the guessed card.
        :param guessed_by: The identifier of the player who made the guess that revealed the agent

        :return: A string indicating the result of the guess ("agent" or "victory").
        """
        # Reveal the card
        card.revealed = True
        card.revealed_by.append(guessed_by)
        self.clue_validator.remove_word(card.text)

        # Update the count of remaining agents for the guessing player (or both if it's a shared agent)
        self.state.agents_remaining[guessed_by] -= 1
        if card.llm_perspective_role == card.human_perspective_role == CardRole.AGENT:
            self.state.agents_remaining[1 - guessed_by] -= 1
            card.revealed_by.append(1 - guessed_by)

        # Check for win condition
        if self.state.agents_remaining[guessed_by] == self.state.agents_remaining[1 - guessed_by] == 0:
            res = "victory" if self.state.current_phase != GamePhase.SUDDEN_DEATH else "victory_sd"
            self._finish_game(result=res)
            return res

        return "agent"

    def _finish_game(self, result: str):
        """
        Finishes the game by setting the game over flag, updating the current phase to GAME_OVER, 
        and storing the result.

        :param result: A string indicating the result of the game ("victory", "loss_assassin", 
            "loss_civilian", etc.).
        """
        self.state.is_game_over = True
        self.state.current_phase = GamePhase.GAME_OVER
        self.state.result = result

    def _switch_roles(self):
        """
        Switches the roles of the clue giver and guesser, resets the current clue and count, and 
        updates the turn number. If the timer tokens have run out and there are still agents
        remaining, transitions to the SUDDEN_DEATH phase.
        """
        self.state.clue_giver, self.state.guesser = self.state.guesser, self.state.clue_giver
        self.state.current_phase = GamePhase.GIVING_CLUE
        self.state.guesses_made_this_turn = 0
        self.state.turn_number += 1

        if self.state.current_clue is not None:
            self.state.clue_history.append(self.state.current_clue)
        self.state.current_clue = None

        # If the timer tokens have run out and there are still agents remaining, transition to the
        # sudden death phase
        if self.state.timer_tokens <= 0 and self._any_agents_remaining():
            self.state.current_phase = GamePhase.SUDDEN_DEATH

    def _any_agents_remaining(self) -> bool:
        """
        Checks if there are any agents remaining for either player.

        :return: True if there are any agents remaining for either player, False otherwise.
        """
        return any(agents > 0 for agents in self.state.agents_remaining)
