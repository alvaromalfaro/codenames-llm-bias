import random
from enum import Enum
from pydantic import BaseModel, Field, model_validator
from typing import Optional


class CardRole(str, Enum):
    """
    The role of a card in the game.
        - AGENT: A card that must be guessed by the other player.
        - ASSASSIN: A card that causes the end of the game and the loss of the game if guessed by 
            the other player.
        - CIVILIAN: A card that causes the end of turn and the loss of a time point if guessed by 
            the other player.
    """
    AGENT = "agent"
    ASSASSIN = "assassin"
    CIVILIAN = "civilian"


class WordCard(BaseModel):
    """
    A card in the game, defined by its text (the word on the card) and its role for both the LLM and
    the human player. It also has a state to indicate whether it has been revealed and by whom.
    """
    id: int  # Unique identifier for the card
    text: str  # The word on the card

    # The role of the card (agent, assassin, civilian)
    role_llm: CardRole
    role_human: CardRole

    # Card state
    revealed: bool = False
    revealed_by: Optional[str] = None  # "llm" or "human"

    # Bias category (optional, can be used for analysis)
    category: Optional[str] = None


class Board(BaseModel):
    board_id: str  # Unique identifier for the board
    category: str  # Bias category
    # List of cards on the board (exactly 25)
    cards: list[WordCard] = Field(..., min_length=25, max_length=25)

    @model_validator(mode="after")
    def rules_validation(self) -> "Board":
        """
        Validates the rules of the game for a duet game:
        - There must be exactly 9 agent cards for both LLM and human players (3 shared between them).
        - There must be exactly 3 assassin cards (1 shared between LLM and human players, 1 unique to LLM, 1 unique to human).
        - The rest of the cards will be civilian cards.

        :param self: The instance of the Board class to validate.
        :return: The validated Board instance.
        :rtype: Board
        """

        cards = self.cards

        # Agent cards (9 for both LLM and human players, with 3 shared between them)
        agents_llm = set(
            card.id for card in cards if card.role_llm == CardRole.AGENT)
        agents_human = set(
            card.id for card in cards if card.role_human == CardRole.AGENT)

        if len(agents_llm) != 9 or len(agents_human) != 9 or len(agents_llm.intersection(agents_human)) != 3:
            raise ValueError(
                "There must be exactly 9 agent cards for both LLM and human players (3 shared between them)."
            )

        # Assassin cards (3 for both LLM and human players, with 1 shared between them)
        assassins_llm = set(
            card.id for card in cards if card.role_llm == CardRole.ASSASSIN)
        assassins_human = set(
            card.id for card in cards if card.role_human == CardRole.ASSASSIN)

        if len(assassins_llm) != 3 or len(assassins_human) != 3 or len(assassins_llm.intersection(assassins_human)) != 1:
            raise ValueError(
                "There must be exactly 3 assassin cards (1 shared between LLM and human players, 1 unique to LLM, 1 unique to human)."
            )

        if len(assassins_human.intersection(agents_llm)) != 1:
            raise ValueError(
                "One of the human's assassin cards must be one of the LLM's agent cards."
            )

        if len(assassins_llm.intersection(agents_human)) != 1:
            raise ValueError(
                "One of the LLM's assassin cards must be one of the human's agent cards."
            )

        return self


class GamePhase(str, Enum):
    # The phase where the clue-giving player provides a clue and a count to the guessing player.
    GIVING_CLUE = "giving_clue"
    # The phase where the guessing player makes guesses based on the clue provided by the clue-giving player.
    GUESSING = "guessing"
    # The phase when the game has ended, either by win, loss, or other termination conditions.
    GAME_OVER = "game_over"


class GameState(BaseModel):
    game_id: str  # Unique identifier for the game
    board: Board  # The board configuration for the game

    current_phase: GamePhase = GamePhase.GIVING_CLUE
    clue_giver: int = 1
    guesser: int = 0

    # Current clue data
    current_clue: Optional[str] = None
    current_clue_count: Optional[int] = None

    # Guess tracking
    guesses_made_this_turn: int = 0

    # Timer tokens for the game
    timer_tokens: int = 9

    # Finalization state
    is_game_over: bool = False
    # None, "victory", "loss_assassin", "loss_time"
    result: Optional[str] = None
