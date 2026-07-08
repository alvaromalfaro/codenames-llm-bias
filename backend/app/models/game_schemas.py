from enum import Enum
from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing import Literal, Optional


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


class Covariates(BaseModel):
    """Per-word covariates emitted by the board generator (used for balance diagnostics)."""
    model_config = ConfigDict(extra="forbid")

    subtlex_freq: Optional[float] = None  # SUBTLEX log-frequency of the word
    length: Optional[int] = None  # Character length of the word
    wordnet_polysemy: Optional[int] = None  # Number of WordNet senses


class WordCard(BaseModel):
    """
    A card in the game, defined by its text (the word on the card) and its role for both the LLM and
    the human player. It also has a state to indicate whether it has been revealed and by whom.
    """
    model_config = ConfigDict(extra="forbid")

    id: int  # Unique identifier for the card
    text: str = Field(min_length=1, pattern=r"^\S+$")  # The word on the card

    # The role of the card (agent, assassin, civilian)
    llm_perspective_role: CardRole
    human_perspective_role: CardRole

    # Card state
    revealed: bool = False
    revealed_by: list[int] = []  # 0: "llm" or 1: "human"

    # Time marker state
    time_marker_by: list[int] = []  # 0: "llm" or 1: "human"

    # Bias category (male | female | neutral)
    category: Optional[str] = None

    # Board-generator artifact metadata (absent in the minimal runtime shape)
    # provenance: duet | weat | eige | eurostat | own-criterion | ...
    source: Optional[str] = None
    weat_set: list[str] = []  # e.g. ["weat-6"]; [] if none
    covariates: Optional[Covariates] = None


class Grid(BaseModel):
    """Board dimensions as emitted by the generator."""
    model_config = ConfigDict(extra="forbid")

    rows: int
    cols: int


class Arbiters(BaseModel):
    """Semantic arbiter models used by the generator's consensus."""
    model_config = ConfigDict(extra="forbid")

    consensus: list[str]  # e.g. ["model@rev", ...]
    primary: str


class ArbiterScore(BaseModel):
    """Per-arbiter cosine proximities of the dilemma target to each bridge."""
    model_config = ConfigDict(extra="forbid")

    arbiter: str
    cos_target_neutral: float
    cos_target_stereo: float
    satisfies_eq_4_1: bool


class Dilemma(BaseModel):
    """
    The dilemma triple placed on the LLM's agent cells (probe boards only): a target word plus a
    neutral bridge and a stereotypical bridge of comparable semantic proximity.
    """
    model_config = ConfigDict(extra="forbid")

    target: str
    neutral_bridge: str
    stereotypical_bridge: str
    consensus_ok: bool
    arbiter_scores: list[ArbiterScore]


class PerPerspective(BaseModel):
    """Card counts per key-card role, as audited by the generator."""
    model_config = ConfigDict(extra="forbid")

    agent: int
    bystander: int
    assassin: int


class KeycardAudit(BaseModel):
    """Generator self-audit that the key card is valid and role/gender independent."""
    model_config = ConfigDict(extra="forbid")

    per_perspective: PerPerspective
    overlap_ok: bool
    role_gender_independent: bool


class Board(BaseModel):
    model_config = ConfigDict(extra="forbid")

    board_id: str  # Unique identifier for the board
    category: str  # Bias category
    # List of cards on the board (exactly 25)
    cards: list[WordCard] = Field(..., min_length=25, max_length=25)

    # Board-generator artifact metadata (absent in the minimal runtime shape; control boards
    # legitimately have dilemma/specification == None).
    type: Optional[Literal["probe", "control"]] = None
    # gender specification (e.g. "gender-career"); null for control
    specification: Optional[str] = None
    # board RNG seed (reproducibility of generation)
    seed: Optional[int] = None
    grid: Optional[Grid] = None
    arbiters: Optional[Arbiters] = None
    dilemma: Optional[Dilemma] = None  # present on probe, null on control
    keycard_audit: Optional[KeycardAudit] = None

    def get_card_id_by_word(self, text: str) -> Optional[int]:
        for card in self.cards:
            if card.text.lower() == text.lower():
                return card.id

        return None

    @model_validator(mode="after")
    def rules_validation(self) -> "Board":
        """
        Validates the rules of the game for a duet game:
        - There must be exactly 9 agent cards for both LLM and human players (3 shared between them).
        - There must be exactly 3 assassin cards (1 shared between LLM and human players, 1 unique to LLM, 1 unique to human).
        - The rest of the cards will be civilian cards.

        :return: The validated Board instance.
        """

        cards = self.cards

        # Agent cards (9 for both LLM and human players, with 3 shared between them)
        agents_llm = set(
            card.id for card in cards if card.llm_perspective_role == CardRole.AGENT)
        agents_human = set(
            card.id for card in cards if card.human_perspective_role == CardRole.AGENT)

        if len(agents_llm) != 9 or len(agents_human) != 9 or len(agents_llm.intersection(agents_human)) != 3:
            raise ValueError(
                "There must be exactly 9 agent cards for both LLM and human players (3 shared between them)."
            )

        # Assassin cards (3 for both LLM and human players, with 1 shared between them)
        assassins_llm = set(
            card.id for card in cards if card.llm_perspective_role == CardRole.ASSASSIN)
        assassins_human = set(
            card.id for card in cards if card.human_perspective_role == CardRole.ASSASSIN)

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

    @model_validator(mode="after")
    def type_coherence_validation(self) -> "Board":
        """
        Validates coherence between the artifact's ``type`` and the rest of the board metadata.
        The minimal runtime shape (``example_board.json``) has ``type is None`` and is exempt.

        - probe   => dilemma and specification present, and at least one gendered card.
        - control => dilemma and specification absent, and every card is neutral.
        - a present dilemma => its three words each sit on an LLM-agent card (the dilemma is
          placed on the LLM's agent cells).

        :return: The validated Board instance.
        """
        if self.type is None:
            return self

        if self.type == "probe":
            if self.dilemma is None or self.specification is None:
                raise ValueError(
                    "A probe board must have a dilemma and a specification."
                )
            if not any(card.category in {"male", "female"} for card in self.cards):
                raise ValueError(
                    "A probe board must have at least one male or female card."
                )
        elif self.type == "control":
            if self.dilemma is not None or self.specification is not None:
                raise ValueError(
                    "A control board must not have a dilemma or a specification."
                )
            if any(card.category != "neutral" for card in self.cards):
                raise ValueError(
                    "Every card on a control board must be neutral."
                )

        if self.dilemma is not None:
            dilemma_words = (
                self.dilemma.target,
                self.dilemma.neutral_bridge,
                self.dilemma.stereotypical_bridge,
            )
            for word in dilemma_words:
                card = next(
                    (c for c in self.cards if c.text.lower() == word.lower()), None)
                if card is None:
                    raise ValueError(
                        f"Dilemma word '{word}' does not appear on any card."
                    )
                if card.llm_perspective_role != CardRole.AGENT:
                    raise ValueError(
                        f"Dilemma word '{word}' must sit on an LLM-agent card."
                    )

        return self


class GamePhase(str, Enum):
    # The phase where the clue-giving player provides a clue and a count to the guessing player.
    GIVING_CLUE = "giving_clue"
    # The phase where the guessing player makes guesses based on the clue provided by the
    # clue-giving player.
    GUESSING = "guessing"
    # The phase when the game has ended, either by win, loss, or other termination conditions.
    GAME_OVER = "game_over"
    # Endgame phase when the timer tokens have run out: human guesses their remaining agents first.
    SUDDEN_DEATH_HUMAN = "sudden_death_human"
    # After the human finishes, the LLM guesses its remaining agents.
    SUDDEN_DEATH_LLM = "sudden_death_llm"


class ResolvedTarget(BaseModel):
    """
    A clue-time snapshot of one word from the clue-giver's intended target set S, resolved against
    the authoritative board. Because reveal state changes during play, this snapshot must be
    computed at the moment the clue is given, not reconstructed later. Captured for measurement
    only; never transmitted to the guesser.

    An unmappable word (not on the board) yields ``card_id``/``giver_role``/``revealed_at_clue`` all
    ``None`` - the snapshot itself is the malformation diagnostic (no separate flag fields).
    """
    word: str  # The target word exactly as emitted by the clue-giver
    # Board card id, via Board.get_card_id_by_word; None if unmappable
    card_id: Optional[int] = None
    # The target's role from the CLUE-GIVER's perspective (llm if player_id==0 else human role);
    # None if unmappable.
    giver_role: Optional[CardRole] = None
    # The card's reveal state at the moment the clue was given; None if unmappable.
    revealed_at_clue: Optional[bool] = None


class ClueEntry(BaseModel):
    # Clue must be a non-empty string without spaces (more complex clue validation will be
    # implemented in the game engine).
    clue: str = Field(min_length=1, pattern=r"^\S+$")
    # Clue count must be a positive integer. Codenames Duet allows a clue count of 0, but for
    # simplicity, we will require at least 1.
    count: int = Field(ge=1)
    # The player who gave the clue
    clue_giver: int = Field(ge=0, le=1)  # 0: "llm" or 1: "human"
    # The turn number when the clue was given (for historical tracking)
    turn_number: int = 0
    # The intended target set S, exactly as emitted by the clue-giver (measurement only).
    targets: list[str] = Field(default_factory=list)
    # Clue-time resolved snapshot of S against the authoritative board (measurement only).
    targets_resolved: list[ResolvedTarget] = Field(default_factory=list)
    # LLM response payload
    raw_payload: Optional[dict] = None


class GameState(BaseModel):
    game_id: str  # Unique identifier for the game
    board: Board  # The board configuration for the game

    # Current game state. Initialized to GIVING_CLUE phase. A default initial value is provided here
    # for the clue giver and guesser, but these will be set randomly at the start of the game in the
    # game engine.
    current_phase: GamePhase = GamePhase.GIVING_CLUE
    clue_giver: int = 1  # 0: "llm" or 1: "human"
    guesser: int = 0  # 0: "llm" or 1: "human"
    turn_number: int = 1

    # Current clue data
    current_clue: Optional[ClueEntry] = None

    # Guess tracking
    guesses_made_this_turn: int = 0

    # Timer tokens for the game
    timer_tokens: int = 9

    # LLM and human agents remaining (for win condition tracking)
    # [LLM agents remaining, Human agents remaining]
    agents_remaining: list[int] = Field(default_factory=lambda: [9, 9])

    # Clue history
    clue_history: list[ClueEntry] = Field(default_factory=list)

    # Finalization state
    is_game_over: bool = False
    # None, "victory", "loss_assassin", "loss_time"
    result: Optional[str] = None
