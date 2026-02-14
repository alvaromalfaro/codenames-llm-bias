import uuid
import random
from app.models.game_schemas import Board, GamePhase, GameState, CardRole


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

    def receive_clue(self, clue: str, count: int, player_id: int):
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

        # Store the clue
        self.state.current_clue = clue
        self.state.current_count = count
        self.state.guesses_made_this_turn = 0

        # Decrement the timer token
        self.state.timer_tokens -= 1

        # Transition to the guessing phase
        self.state.current_phase = GamePhase.GUESSING

    def resolve_guess(self, card_id: int, player_id: int) -> str:
        pass

    def pass_turn(self, player_id: str):
        """
        Allows the guessing player to pass their turn to the clue-giving player.

        :param player_id: The identifier of the player attempting to pass their turn.

        raises PermissionError: If a player other than the guesser attempts to pass their turn.
        raises ValueError: If the guesser attempts to pass their turn without making at least one guess.
        """
        if player_id != self.state.guesser:
            raise PermissionError("Only the guesser can pass their turn.")
        if self.state.guesses_made_this_turn < 1:
            raise ValueError(
                "The guesser must make at least one guess before passing.")

        self._switch_roles()

    def _switch_roles(self):
        """
        Switches the roles of the clue giver and guesser, resets the current clue and count, and checks
        for loss conditions related to timer tokens.
        """
        self.state.clue_giver, self.state.guesser = self.state.guesser, self.state.clue_giver
        self.state.current_phase = GamePhase.GIVING_CLUE
        self.state.guesses_made_this_turn = 0

        # Check for loss condition due to timer tokens running out
        if self.state.timer_tokens <= 0:
            self.state.is_game_over = True
            self.state.current_phase = GamePhase.GAME_OVER
            self.state.result = "loss_time"
