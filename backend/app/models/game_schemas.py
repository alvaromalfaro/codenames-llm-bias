from enum import Enum
from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing import Literal, Optional

from backend.app.models.llm_schemas import LLMCallRecord


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
    the human player, plus the two independent pieces of physical state the rules put on a word: the
    agent card that may cover it and the time tokens that may sit on it.

    The two state families answer different questions and must not be conflated (see
    ``is_covered`` vs ``revealed_by``):

    - ``revealed`` / ``time_marker_by`` describe what physically sits on the word, and therefore
      whether ANYONE may still touch it. This is what ``is_covered`` and ``is_guessable_by`` are 
      derived from.
    - ``revealed_by`` describes for WHICH seats the word has stopped being a pending word, i.e.
      whose clue-giving no longer needs to target it. It is bookkeeping for the clue-giver side and
      says nothing about who may touch the card.
    """
    model_config = ConfigDict(extra="forbid")

    id: int  # Unique identifier for the card
    text: str = Field(min_length=1, pattern=r"^\S+$")  # The word on the card

    # The role of the card (agent, assassin, civilian)
    llm_perspective_role: CardRole
    human_perspective_role: CardRole

    # Card state: an agent card covers this word. Set for either seat's correct guess, so it is
    # seat-independent - a covered word is covered for BOTH players.
    revealed: bool = False
    # Seats for which this word has stopped being pending: the guesser who found it, plus the other
    # seat when the word is a shared agent. 0: "llm" or 1: "human". NOT a guess guard.
    revealed_by: list[int] = []

    # Time marker state: the seats that touched this word and hit an innocent bystander. A token
    # carries a direction (clue giver -> guesser) and the seat stored here is the guesser, i.e. the
    # seat that may no longer touch this word. Two tokens (one per direction) cover it.
    time_marker_by: list[int] = []  # 0: "llm" or 1: "human"

    @property
    def is_covered(self) -> bool:
        """
        Whether this word is covered and therefore out of play for BOTH players: an agent card sits
        on it, or two time tokens do - one from each direction.

        A covered word can never be guessed again by either seat, and it stops being visible for
        clue validation. The seat that covered it is irrelevant.

        :return: True if the word is covered, False otherwise.
        """
        return self.revealed or len(self.time_marker_by) == 2

    def is_guessable_by(self, player_id: int) -> bool:
        """
        Whether ``player_id`` may still touch this word.

        A word is out of reach for a seat when it is covered (agent card, or two time tokens), or
        when that seat already touched it and hit an innocent bystander - its own time token points
        at it, and the word is still an innocent bystander on the other face.

        This is the single predicate behind the engine's guess guard, the words offered to an LLM
        guesser, and the cards the UI lets a human guesser click.

        :param player_id: The seat asking (0 = LLM, 1 = human).

        :return: True if that seat may still guess this word, False otherwise.
        """
        return not self.is_covered and player_id not in self.time_marker_by

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
    # Optional association to a measurement frame.
    measurement_frame_id: Optional[str] = None

    def get_card_id_by_word(self, text: str) -> Optional[int]:
        for card in self.cards:
            if card.text.lower() == text.lower():
                return card.id

        return None

    def get_card_by_id(self, card_id: int) -> Optional[WordCard]:
        """
        Looks a card up by its declared ``id`` rather than by list position, so a caller holding an
        id from outside the process (an HTTP form field, an LLM proposal) can never address the
        wrong card: an unknown id yields None instead of an IndexError, and a negative id can never
        wrap around to a real card the way ``cards[card_id]`` would.

        :param card_id: The card's ``id`` field.

        :return: The matching WordCard, or None when no card carries that id.
        """
        for card in self.cards:
            if card.id == card_id:
                return card

        return None

    @model_validator(mode="after")
    def rules_validation(self) -> "Board":
        """
        Validates the rules of the game for a duet game:
        - The card ids must be 0..24, in board order (so they are unique and ``id`` is the grid
          position, as the database's ``card_id BETWEEN 0 AND 24`` expects).
        - The 25 words must be distinct, ignoring case (§3), since words are matched to cards
          case-insensitively (``get_card_id_by_word``, the writer's word -> card_id map).
        - There must be exactly 9 agent cards for both LLM and human players (3 shared between them).
        - There must be exactly 3 assassin cards (1 shared between LLM and human players, 1 unique to LLM, 1 unique to human).
        - The rest of the cards will be civilian cards.

        The id and word checks run first: the role checks below collect card ids into sets, so a
        duplicated id would otherwise slip through them or surface as a misleading count error.

        :return: The validated Board instance.
        """

        cards = self.cards

        # Card ids (0..24, in board order)
        ids = [card.id for card in cards]
        if ids != list(range(len(cards))):
            raise ValueError(
                f"Card ids must be 0..24 in board order; got {ids}."
            )

        # Distinct words, ignoring case
        words = [card.text.lower() for card in cards]
        repeated_words = sorted({word for word in words if words.count(word) > 1})

        if repeated_words:
            raise ValueError(
                f"The 25 words on the board must be distinct (ignoring case); repeated: {repeated_words}."
            )

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
    # Endgame phase when the timer tokens have run out with words still pending, human guessing.
    # Entered once the LLM has nothing left to find: straight away if it had nothing at the start of
    # sudden death, or from SUDDEN_DEATH_LLM when the LLM finds its last agent.
    SUDDEN_DEATH_HUMAN = "sudden_death_human"
    # Endgame phase, LLM guessing. Always entered first when the LLM has words pending: the engine
    # fixes the order LLM -> human (the rules allow any order, §7.3).
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


class RankedCard(BaseModel):
    """
    One entry of an out-of-band confidence-ranking measurement: a board word and the model's
    confidence that it belongs to the set being measured (the clue's targets, or the guesser's
    remaining agents in sudden death). Confidence is clamped to [0, 1] by the parser before this
    model is constructed.
    """
    word: str  # The unrevealed board word being scored
    confidence: float = Field(ge=0.0, le=1.0)


class ConfidenceRanking(BaseModel):
    """
    A parsed confidence-ranking measurement over the unrevealed cards at a fixed pre-resolution
    instant. This is an out-of-band signal captured for the CIT / sudden-death bias metrics; it is
    never used for game rules and never influences play. Missing cards (the model omitted some) are
    a derivable malformation - the list simply holds fewer entries, with no flag fields.
    """
    reasoning: Optional[str] = None
    # Ordered (word, confidence) entries, one per unrevealed card the model scored.
    rankings: list[RankedCard] = Field(default_factory=list)
    # The raw LLM payload for the measurement call (in-memory only).
    raw_payload: Optional[dict] = None
    # In-memory audit carrier for the measurement call that produced this ranking. Never sent to a
    # provider; consumed only by the persistence write-path.
    llm_call: Optional[LLMCallRecord] = None


class ClueEntry(BaseModel):
    # Clue must be a non-empty string without spaces (more complex clue validation will be
    # implemented in the game engine).
    clue: str = Field(min_length=1, pattern=r"^\S+$")
    # Clue count must be a positive integer. This is a deliberate deviation from the Duet rules,
    # which allow a count of 0 (rules §6.2, §8.3): a "zero clue" means "avoid the words related to
    # this one", and the guesser must still make at least one guess. Supporting it would mean
    # explaining that inverted meaning in both prompts (when to give a zero clue, and that "count 0"
    # does not mean "do not guess"). That makes the prompts longer and the task more complex, which
    # makes the models more likely to hallucinate. Applies to both seats, LLM and human; a count of
    # 0 is rejected here. See "Deviations from the Duet rules" in the README.
    count: int = Field(ge=1)
    # The player who gave the clue
    clue_giver: int = Field(ge=0, le=1)  # 0: "llm" or 1: "human"
    # The turn number when the clue was given (for historical tracking)
    turn_number: int = 0
    # The intended target set S, exactly as emitted by the clue-giver (measurement only).
    targets: list[str] = Field(default_factory=list)
    # Clue-time resolved snapshot of S against the authoritative board (measurement only).
    targets_resolved: list[ResolvedTarget] = Field(default_factory=list)
    # Out-of-band confidence ranking over all unrevealed cards, elicited at the pre-resolution
    # instant of this turn (measurement only; parallel to targets_resolved). None until measured.
    confidence_ranking: Optional[ConfidenceRanking] = None
    # LLM response payload
    raw_payload: Optional[dict] = None


class SuddenDeathEntry(BaseModel):
    """
    Per-GAME sudden-death record. There is exactly one sudden-death state per game, so this is a
    single object hanging off GameState (not a per-turn list). It holds the out-of-band confidence
    rankings elicited on entry to each seat's sudden-death guessing, before that seat's first
    selection.

    Both seats can reach sudden death sequentially (the SUDDEN_DEATH_LLM -> SUDDEN_DEATH_HUMAN
    handoff may put a second LLM on seat 1 in an LLM-vs-LLM run), so the authoritative store is
    per-(game, seat): ``rankings_by_seat`` maps the guesser seat -> its ranking. ``confidence_ranking``
    is retained as a backward-compat mirror of the most-recently-attached ranking (the single-seat
    interactive path only ever measures seat 0).
    """
    # Authoritative per-seat store: guesser seat (0 = LLM, 1 = human) -> its SD ranking.
    rankings_by_seat: dict[int, ConfidenceRanking] = Field(
        default_factory=dict)
    # Backward-compat mirror of the most-recently-attached ranking (see class docstring).
    confidence_ranking: Optional[ConfidenceRanking] = None


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

    # Per-game sudden-death record (measurement only); None until the sudden-death phase is entered
    # and its confidence ranking is attached. Exactly one per game - not a per-turn list.
    sudden_death: Optional[SuddenDeathEntry] = None
    # Set by the engine when the game transitions into sudden death, and consumed once when the LLM's
    # sudden-death confidence ranking is elicited and attached (before the first sudden-death guess).
    sd_measurement_pending: bool = False

    # Finalization state
    is_game_over: bool = False
    # None, "victory", "loss_assassin", or a sudden-death ending: "victory_sd", "loss_civilian_sd",
    # "loss_assassin_sd", "loss_stopped_sd" (the guessing seat conceded with words still pending)
    result: Optional[str] = None
