from typing import Optional
import uuid
import random
from backend.app.models.game_schemas import Board, GamePhase, GameState, CardRole, ClueEntry


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

        :param self: The instance of the CodenamesDuetEngine class.
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

    def receive_clue(self, clue: str, count: int, player_id: int, raw_payload: Optional[dict] = None):
        """
        Processes a clue provided by the clue-giving player.

        :param self: The instance of the CodenamesDuetEngine class.
        :param clue: The clue word provided by the clue-giving player.
        :param count: The number of cards the clue relates to.
        :param player_id: The identifier of the player providing the clue.

        :raises ValueError: If the clue is invalid or if it's not the clue-giving player's turn.
        :raises PermissionError: If a player other than the clue giver attempts to provide a clue.
        """
        if self.state.current_phase != GamePhase.GIVING_CLUE:
            raise ValueError(
                "Clues can only be given during the GIVING_CLUE phase.")
        if player_id != self.state.clue_giver:
            raise PermissionError("Only the clue giver can provide a clue.")

        self._validate_clue(clue, count)

        # Store the clue
        self.state.current_clue = ClueEntry(
            clue=clue,
            count=count,
            clue_giver=player_id,
            turn_number=self.state.turn_number,
            raw_payload=raw_payload
        )
        self.state.guesses_made_this_turn = 0

        # Transition to the guessing phase
        self.state.current_phase = GamePhase.GUESSING

    def resolve_guess(self, card_id: int, player_id: int) -> str:
        """
        Processes a guess made by the guessing player.

        :param self: The instance of the CodenamesDuetEngine class.
        :param card_id: The identifier of the card being guessed.
        :param player_id: The identifier of the player making the guess.

        :return: A string indicating the result of the guess ("agent", "assassin", "civilian",
            "victory").

        :raises ValueError: If the card is already revealed, marked by a time token, or if it's not 
            the guessing player's turn.
        :raises PermissionError: If a player other than the guesser attempts to make a guess.
        """
        if self.state.current_phase != GamePhase.GUESSING:
            raise ValueError(
                "Guesses can only be made during the GUESSING phase.")
        if player_id != self.state.guesser:
            raise PermissionError("Only the guesser can make guesses.")

        card = self.state.board.cards[card_id]
        if card.revealed:
            raise ValueError("This card has already been revealed.")

        if card.has_time_marker and card.time_marker_by == self.state.guesser:
            raise ValueError(
                "This card is currently marked by a time token and cannot be guessed.")

        # Determine the role of the card for the guessing player and update the number of guesses made
        partner_role = card.llm_role if self.state.clue_giver == 0 else card.human_role
        self.state.guesses_made_this_turn += 1

        if partner_role == CardRole.AGENT:
            # Reveal the card
            card.revealed = True
            card.revealed_by = self.state.guesser

            # Update the count of remaining agents for the guessing player (or both if it's a shared agent)
            self.state.agents_remaining[self.state.guesser] -= 1
            if card.llm_role == card.human_role == CardRole.AGENT:
                self.state.agents_remaining[1 - self.state.guesser] -= 1

            # Check for win condition
            if self.state.agents_remaining[self.state.guesser] == self.state.agents_remaining[1 - self.state.guesser] == 0:
                self.state.is_game_over = True
                self.state.current_phase = GamePhase.GAME_OVER
                self.state.result = "victory"
                return "victory"

            return "agent"
        elif partner_role == CardRole.ASSASSIN:
            self.state.is_game_over = True
            self.state.current_phase = GamePhase.GAME_OVER
            self.state.result = "loss_assassin"

            return "assassin"
        else:
            card.has_time_marker = True
            card.time_marker_by = self.state.guesser
            self.state.timer_tokens -= 1
            self._switch_roles()

            return "civilian"

    def pass_turn(self, player_id: int):
        """
        Allows the guessing player to pass their turn to the clue-giving player.

        :param player_id: The identifier of the player attempting to pass their turn.

        :raises PermissionError: If a player other than the guesser attempts to pass their turn.
        :raises ValueError: If the guesser attempts to pass their turn without making at least one guess.
        """
        if player_id != self.state.guesser:
            raise PermissionError("Only the guesser can pass their turn.")
        if self.state.guesses_made_this_turn < 1:
            raise ValueError(
                "The guesser must make at least one guess before passing.")

        # Decrement the timer token
        self.state.timer_tokens -= 1

        self._switch_roles()

    def _switch_roles(self):
        """
        Switches the roles of the clue giver and guesser, resets the current clue and count, and checks
        for loss conditions related to timer tokens.
        """
        self.state.clue_giver, self.state.guesser = self.state.guesser, self.state.clue_giver
        self.state.current_phase = GamePhase.GIVING_CLUE
        self.state.guesses_made_this_turn = 0
        self.state.turn_number += 1

        if self.state.current_clue is not None:
            self.state.clue_history.append(self.state.current_clue)
        self.state.current_clue = None

        # Check for loss condition due to timer tokens running out
        if self.state.timer_tokens <= 0:
            self.state.is_game_over = True
            self.state.current_phase = GamePhase.GAME_OVER
            self.state.result = "loss_time"

    def _validate_clue(self, clue: str, count: int) -> None:
        """
        Validates the provided clue against the current board state and game rules.

        :param clue: The clue word provided by the clue-giving player.
        :param count: The number of cards the clue relates to.
        """
        normalized_clue = clue.strip().lower()

        if not normalized_clue:
            raise ValueError("Clue cannot be empty.")

        if count < 1:
            raise ValueError("Clue count must be at least 1.")

        board_words = {card.text.strip().lower()
                       for card in self.state.board.cards}
        if normalized_clue in board_words:
            raise ValueError(
                "Clue cannot be the same as any word on the board.")

        # TODO: Implement more robust clue validation (check agains a dictionary, filter lexemes, etc.)
