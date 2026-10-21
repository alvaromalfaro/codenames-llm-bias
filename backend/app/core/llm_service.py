import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional, TypeVar

from pydantic import BaseModel
from backend.app import config
from backend.app.paths import PROMPT_TEMPLATES_DIR
from backend.app.core.llm.client import LLMClient
from backend.app.models.llm_schemas import (
    ClueProposal,
    GuessProposal,
    LLMRequest,
    LLMResponse,
    LLMMessage,
    LLMCallRecord,
    ClueJSONFormat,
    GuessJSONFormat,
    ConfidenceRankingJSONFormat,
)
from backend.app.models.game_schemas import (
    ClueEntry,
    GameState,
    GamePhase,
    CardRole,
    ConfidenceRanking,
    RankedCard,
)

if TYPE_CHECKING:
    from backend.app.core.engine import CodenamesDuetEngine

logger = logging.getLogger(__name__)


def text_digest(text: str) -> str:
    """A short, stable id for a prompt text: the first 12 hex digits of its SHA-256. The call log
    names the constant messages of a request (system prompt, one-shot example) by this id instead of
    repeating them; whoever configures the log can print each template once with the same id."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


# The prompt templates and one-shot examples the service sends, by the key the fingerprint knows
# each by, as files of data/prompt_templates. All of them must be there: the service refuses to
# start without one, rather than send a prompt that is not the versioned file.
TEMPLATE_FILES = {
    "system_prompt_cg": "SYSTEM_TEMPLATE_CLUE_GIVER.txt",
    "user_prompt_cg": "USER_TEMPLATE_CLUE_GIVER.txt",
    "system_prompt_gg": "SYSTEM_TEMPLATE_GUESSER.txt",
    "user_prompt_gg": "USER_TEMPLATE_GUESSER.txt",
    "system_prompt_sd_gg": "SYSTEM_TEMPLATE_SUDDEN_DEATH_GUESSER.txt",
    "user_prompt_sd_gg": "USER_TEMPLATE_SUDDEN_DEATH_GUESSER.txt",
    "system_prompt_meas_gg": "SYSTEM_TEMPLATE_MEASUREMENT_GUESSER.txt",
    "user_prompt_meas_gg": "USER_TEMPLATE_MEASUREMENT_GUESSER.txt",
    "system_prompt_meas_sd": "SYSTEM_TEMPLATE_MEASUREMENT_SD.txt",
    "user_prompt_meas_sd": "USER_TEMPLATE_MEASUREMENT_SD.txt",
    "one_shot_user_cg": "ONE_SHOT_USER_CLUE_GIVER.txt",
    "one_shot_assistant_cg": "ONE_SHOT_ASSISTANT_CLUE_GIVER.txt",
    "one_shot_user_gg": "ONE_SHOT_USER_GUESSER.txt",
    "one_shot_assistant_gg": "ONE_SHOT_ASSISTANT_GUESSER.txt",
}


def load_templates(directory: Path | None = None) -> dict[str, str]:
    """Every file of ``TEMPLATE_FILES`` under ``directory`` (data/prompt_templates by default), by
    its key. Read as text, so a CRLF checkout reads as LF and the fingerprint is the same on every
    system.

    :raises FileNotFoundError: naming every file that is missing.
    """
    directory = directory or PROMPT_TEMPLATES_DIR
    missing = [name for name in TEMPLATE_FILES.values() if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(f"prompt templates missing from {directory}: {', '.join(missing)}")
    return {
        key: (directory / name).read_text(encoding="utf-8") for key, name in TEMPLATE_FILES.items()
    }


# The response schema each kind of call sends with its request: the provider holds the answer to it
# (Ollama constrains the decoding with it, OpenRouter passes it on), so it is part of what the model
# is asked, and the template fingerprint covers it. pydantic puts each format's docstring in the
# schema as its description, so editing one changes the fingerprint too.
RESPONSE_FORMATS: dict[str, type[BaseModel]] = {
    "clue": ClueJSONFormat,
    "guess": GuessJSONFormat,
    "measurement": ConfidenceRankingJSONFormat,
}


@dataclass(frozen=True)
class GuesserView:
    """What a seat sees when it guesses, read once from the state: the clues its partner gave it
    before this turn's, and the words it may still touch, in board order. A play request and the
    measurement taken at the same instant are both built from it, so they see the same game.

    The clue history is listed in two ways, the ones the prompts have always shown: changing either
    changes what the models are sent.
    """

    clues: tuple[ClueEntry, ...]
    words: tuple[str, ...]

    @classmethod
    def of(cls, game_state: GameState, seat: int) -> "GuesserView":
        return cls(
            clues=tuple(e for e in game_state.clue_history if e.clue_giver != seat),
            words=tuple(c.text for c in game_state.board.cards if c.is_guessable_by(seat)),
        )

    def previous_clues(self) -> str:
        """The clues received, as the normal-phase prompts list them."""
        lines = [f"- Turn: {e.turn_number}, Clue: {e.clue}, Count: {e.count}" for e in self.clues]
        return "\n".join(lines) or "No previous clues."

    def clue_history(self) -> str:
        """The clues received, as the sudden-death prompts list them."""
        lines = [f"- Turn {e.turn_number}: Clue '{e.clue}' (count {e.count})" for e in self.clues]
        return "\n".join(lines) or "No clues were given."

    def words_remaining(self) -> str:
        return "\n".join(f"- {word}" for word in self.words)


_Format = TypeVar("_Format", bound=BaseModel)


def _answer(response: LLMResponse, fmt: type[_Format]) -> _Format:
    """The answer of ``response`` as ``fmt``: as the client validated it, or, from a client that
    hands over no parse of its own, validated here, as the client would have.

    :raises pydantic.ValidationError: a ValueError, when the text does not match ``fmt``.
    """
    if isinstance(response.parsed, fmt):
        return response.parsed
    return fmt.model_validate_json(response.text)


# The seat allowed to guess in each sudden-death phase, mirroring the engine's own invariant
# (resolve_guess raises unless seat 1 acts in SUDDEN_DEATH_HUMAN / seat 0 in SUDDEN_DEATH_LLM).
_SD_PHASE_BY_SEAT = {0: GamePhase.SUDDEN_DEATH_LLM, 1: GamePhase.SUDDEN_DEATH_HUMAN}


def _require_sd_seat_phase(game_state: GameState, player_id: int, action: str) -> None:
    """Guard: seat 0 may act only in SUDDEN_DEATH_LLM, seat 1 only in SUDDEN_DEATH_HUMAN."""
    expected = _SD_PHASE_BY_SEAT.get(player_id)
    if expected is None or game_state.current_phase != expected:
        raise ValueError(
            f"Cannot {action} for seat {player_id} in phase {game_state.current_phase}: "
            f"seat 0 is valid only in SUDDEN_DEATH_LLM and seat 1 only in SUDDEN_DEATH_HUMAN."
        )


class LLMService:
    def __init__(
        self,
        temperature: float = 0.7,
        max_tokens: int = config.LLM_MAX_OUTPUT_TOKENS,
        timeout_s: int = 30,
    ):
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s
        # Read once, here: every game of a batch is played with the texts the run's fingerprint
        # records (see template_fingerprint).
        texts = load_templates()
        self._system_prompt_cg = texts["system_prompt_cg"]
        self._user_prompt_cg = texts["user_prompt_cg"]
        self._system_prompt_gg = texts["system_prompt_gg"]
        self._user_prompt_gg = texts["user_prompt_gg"]
        self._system_prompt_sd_gg = texts["system_prompt_sd_gg"]
        self._user_prompt_sd_gg = texts["user_prompt_sd_gg"]
        self._system_prompt_meas_gg = texts["system_prompt_meas_gg"]
        self._user_prompt_meas_gg = texts["user_prompt_meas_gg"]
        self._system_prompt_meas_sd = texts["system_prompt_meas_sd"]
        self._user_prompt_meas_sd = texts["user_prompt_meas_sd"]
        self._one_shot_user_cg = texts["one_shot_user_cg"]
        self._one_shot_assistant_cg = texts["one_shot_assistant_cg"]
        self._one_shot_user_gg = texts["one_shot_user_gg"]
        self._one_shot_assistant_gg = texts["one_shot_assistant_gg"]

    def _loaded_template_texts(self) -> dict[str, str]:
        """Read-only accessor: stable template key -> the text the service actually holds in memory.

        Covers every template text the service loads and sends - the 10 prompt templates AND the 4
        one-shot example texts - so the fingerprint reflects the real payloads.
        """
        return {
            "system_prompt_cg": self._system_prompt_cg,
            "user_prompt_cg": self._user_prompt_cg,
            "system_prompt_gg": self._system_prompt_gg,
            "user_prompt_gg": self._user_prompt_gg,
            "system_prompt_sd_gg": self._system_prompt_sd_gg,
            "user_prompt_sd_gg": self._user_prompt_sd_gg,
            "system_prompt_meas_gg": self._system_prompt_meas_gg,
            "user_prompt_meas_gg": self._user_prompt_meas_gg,
            "system_prompt_meas_sd": self._system_prompt_meas_sd,
            "user_prompt_meas_sd": self._user_prompt_meas_sd,
            "one_shot_user_cg": self._one_shot_user_cg,
            "one_shot_assistant_cg": self._one_shot_assistant_cg,
            "one_shot_user_gg": self._one_shot_user_gg,
            "one_shot_assistant_gg": self._one_shot_assistant_gg,
        }

    @staticmethod
    def _response_schema_texts() -> dict[str, str]:
        """Read-only accessor: stable key -> the JSON schema each kind of call sends
        (``RESPONSE_FORMATS``), as canonical JSON: sorted keys, no whitespace."""
        return {
            f"response_schema_{kind}": json.dumps(
                fmt.model_json_schema(), sort_keys=True, separators=(",", ":")
            )
            for kind, fmt in RESPONSE_FORMATS.items()
        }

    def template_fingerprint(self) -> str:
        """Deterministic SHA-256 hex fingerprint of the LOADED template texts (not the directory)
        and of the response schemas the requests carry.

        Hashing the loaded texts - rather than the files on disk - is the whole point: the run row
        records exactly which prompt texts were sent, even if a file changes after the service read
        it. The response schemas are hashed as they are sent, so a limit added to a format, or a
        change to its docstring, changes the fingerprint as a changed prompt would. Stable across
        processes: the texts are sorted by key and each contributes ``key\\0text\\0`` to the digest.
        """
        h = hashlib.sha256()
        texts = {**self._loaded_template_texts(), **self._response_schema_texts()}
        for key, text in sorted(texts.items()):
            h.update(key.encode("utf-8"))
            h.update(b"\0")
            h.update(text.encode("utf-8"))
            h.update(b"\0")
        return h.hexdigest()

    @staticmethod
    def _call_record(request: LLMRequest, response: LLMResponse, role: str) -> LLMCallRecord:
        """Build the in-memory audit carrier from the paired request/response.

        The service only fills a field it already has in hand; it never touches persistence. The
        rendered prompt is the messages as sent (``request.messages``); all sampling telemetry is
        copied verbatim from the ``LLMResponse``.
        """
        usage = response.usage
        return LLMCallRecord(
            role=role,
            rendered_prompt=list(request.messages),
            resolved_model=response.resolved_model,
            system_fingerprint=response.system_fingerprint,
            requested_temperature=response.requested_temperature,
            requested_seed=response.requested_seed,
            prompt_tokens=usage.prompt_tokens if usage else None,
            completion_tokens=usage.completion_tokens if usage else None,
            latency_ms=response.latency_ms,
            finish_reason=response.finish_reason,
            model_used=response.model_used,
            provider=response.provider,
            request_id=response.request_id,
            raw_payload=response.raw_payload,
        )

    @staticmethod
    async def _generate(
        llm_client: LLMClient, request: LLMRequest, expected_format, role: str, player_id: int
    ) -> LLMResponse:
        """
        Sends one request and logs the exchange: every model call of the game goes through here.

        DEBUG records what was sent and what came back: the constant messages (system prompt,
        one-shot example) by their ``text_digest``, the last message - the rendered game state - in
        full, then the raw response text and its telemetry. A call that fails is logged as an error
        before the exception propagates.

        :param llm_client: The client to send the request with.
        :param request: The request to send.
        :param expected_format: The response schema the client validates against.
        :param role: What the call is for ("clue_giver", "guesser", "measurement", ...).
        :param player_id: The seat making the call.

        :return: The client's response.
        """
        *constant, last = request.messages
        logger.debug(
            "llm request role=%s seat=%s model=%s seed=%s temperature=%s constant_messages=[%s]\n"
            "--- %s message ---\n%s\n--- end of message ---",
            role,
            player_id,
            request.model,
            request.seed,
            request.temperature,
            ", ".join(f"{m.role}:{text_digest(m.content)}" for m in constant),
            last.role,
            last.content,
        )
        try:
            response = await llm_client.generate(request, expected_format=expected_format)
        except Exception:
            logger.error(
                "llm call failed role=%s seat=%s model=%s seed=%s",
                role,
                player_id,
                request.model,
                request.seed,
                exc_info=True,
            )
            raise
        usage = response.usage
        logger.debug(
            "llm response role=%s seat=%s model=%s latency_ms=%s prompt_tokens=%s "
            "completion_tokens=%s finish_reason=%s\n--- response ---\n%s\n--- end of response ---",
            role,
            player_id,
            response.model_used,
            response.latency_ms,
            usage.prompt_tokens if usage else None,
            usage.completion_tokens if usage else None,
            response.finish_reason,
            response.text,
        )
        return response

    async def propose_clue(
        self,
        llm_client: LLMClient,
        game_state: GameState,
        player_id: int = 0,
        seed: Optional[int] = None,
    ) -> ClueProposal:
        """
        Proposes a clue for the current game state. This method checks that the game is in the
        correct phase and that the seat is the clue giver before building the request, sending it to
        the LLM, and processing the response.

        The clue is not checked against the validity rules here, and an invalid one is not
        regenerated: as in the rules, the engine accepts it with a penalty token (§8.4).

        :param llm_client: The LLM client to use for generating responses.
        :param game_state: The current state of the game.
        :param player_id: The ID of the player proposing the clue (0 or 1).

        :return: An instance of ClueProposal containing the proposed clue and count from the LLM.
        """
        if game_state.current_phase != GamePhase.GIVING_CLUE:
            raise ValueError("Cannot propose a clue when the game is not in the GIVING_CLUE phase.")

        if game_state.clue_giver != player_id:
            raise ValueError("The player must be the clue giver to propose a clue.")

        request = self._build_clue_request(game_state, llm_client.model_name, player_id, seed=seed)

        response = await self._generate(
            llm_client, request, ClueJSONFormat, "clue_giver", player_id
        )
        proposal = self._build_clue_proposal(response)
        proposal.llm_calls = [self._call_record(request, response, "clue_giver")]

        return proposal

    async def propose_guess(
        self,
        llm_client: LLMClient,
        game_state: GameState,
        player_id: int = 0,
        seed: Optional[int] = None,
    ) -> GuessProposal:
        """
        Proposes guesses for the current game state. This method checks that the game is in the
        correct phase and that the seat is the guesser before building the request, sending it to
        the LLM, and processing the response.

        :param llm_client: The LLM client to use for generating responses.
        :param game_state: The current state of the game.
        :param player_id: The ID of the player proposing the guesses (0 or 1).

        :return: An instance of GuessProposal containing the proposed guesses from the LLM.
        """
        if game_state.current_phase != GamePhase.GUESSING:
            raise ValueError("Cannot propose a guess when the game is not in the GUESSING phase.")

        if game_state.guesser != player_id:
            raise ValueError("The player must be the guesser to propose a guess.")

        if (
            game_state.current_clue is None
            or game_state.current_clue.turn_number != game_state.turn_number
        ):
            raise ValueError("Cannot propose a guess when there is no clue available.")

        # Build the LLM request for proposing a guess
        request = self._build_guess_request(game_state, llm_client.model_name, player_id, seed=seed)

        # Send the request to the LLM client and get the response.
        response = await self._generate(llm_client, request, GuessJSONFormat, "guesser", player_id)

        # Process the response and convert it into a GuessProposal.
        guess_proposal = self._build_guess_proposal(response)
        guess_proposal.llm_call = self._call_record(request, response, "guesser")

        return guess_proposal

    async def propose_guess_sd(
        self,
        llm_client: LLMClient,
        game_state: GameState,
        player_id: int = 0,
        seed: Optional[int] = None,
    ) -> GuessProposal:
        """
        Proposes guesses for the SUDDEN_DEATH_LLM phase. No clue is available; the LLM must
        identify its remaining agents from memory of the full game.

        :param llm_client: The LLM client to use for generating responses.
        :param game_state: The current state of the game.
        :param player_id: The ID of the player proposing the guesses (0 or 1).

        :return: An instance of GuessProposal containing the proposed guesses from the LLM.
        """
        _require_sd_seat_phase(game_state, player_id, "propose a sudden death guess")

        request = self._build_guess_sd_request(
            game_state, llm_client.model_name, player_id, seed=seed
        )
        response = await self._generate(
            llm_client, request, GuessJSONFormat, "guesser_sd", player_id
        )
        guess_proposal = self._build_guess_proposal(response)
        guess_proposal.llm_call = self._call_record(request, response, "guesser_sd")
        return guess_proposal

    async def elicit_confidence_ranking(
        self,
        llm_client: LLMClient,
        game_state: GameState,
        player_id: int = 0,
        seed: Optional[int] = None,
    ) -> ConfidenceRanking:
        """
        Elicits an out-of-band confidence ranking over all unrevealed cards for the current clue,
        parallel to propose_guess but for measurement. This is a separate call that does not touch
        the game-play guess request, is not appended to any conversation history, and never sees the
        clue-giver's intended target set S. It is elicited at the pre-resolution instant (same game
        state as the play-guess request) and returns the parsed ranking; attaching it is the
        engine's job.

        :param llm_client: The LLM client to use for generating the response.
        :param game_state: The current state of the game (guessing phase, before resolution).
        :param player_id: The seat of the guesser (0 or 1).

        :return: The parsed ConfidenceRanking over the unrevealed cards.
        """
        if game_state.current_phase != GamePhase.GUESSING:
            raise ValueError(
                "Cannot elicit a confidence ranking when the game is not in the GUESSING phase."
            )

        if game_state.guesser != player_id:
            raise ValueError("The player must be the guesser to elicit a confidence ranking.")

        if (
            game_state.current_clue is None
            or game_state.current_clue.turn_number != game_state.turn_number
        ):
            raise ValueError("Cannot elicit a confidence ranking when there is no clue available.")

        request = self._build_measurement_request(
            game_state, llm_client.model_name, player_id, seed=seed
        )
        response = await self._generate(
            llm_client, request, ConfidenceRankingJSONFormat, "measurement", player_id
        )
        ranking = self._build_confidence_ranking(response)
        ranking.llm_call = self._call_record(request, response, "measurement")
        return ranking

    async def elicit_confidence_ranking_sd(
        self,
        llm_client: LLMClient,
        game_state: GameState,
        player_id: int = 0,
        seed: Optional[int] = None,
    ) -> ConfidenceRanking:
        """
        Elicits the out-of-band sudden-death confidence ranking over all unrevealed cards, parallel
        to propose_guess_sd. Measurement only; never sees S. Elicited on entry to the sudden-death
        phase, before the first selection.

        :param llm_client: The LLM client to use for generating the response.
        :param game_state: The current state of the game (SUDDEN_DEATH_LLM phase).
        :param player_id: The seat of the guesser (0 or 1).

        :return: The parsed ConfidenceRanking over the unrevealed cards.
        """
        _require_sd_seat_phase(game_state, player_id, "elicit a sudden death confidence ranking")

        request = self._build_measurement_sd_request(
            game_state, llm_client.model_name, player_id, seed=seed
        )
        response = await self._generate(
            llm_client, request, ConfidenceRankingJSONFormat, "measurement_sd", player_id
        )
        ranking = self._build_confidence_ranking(response)
        ranking.llm_call = self._call_record(request, response, "measurement_sd")
        return ranking

    async def measure_and_attach_confidence_ranking(
        self,
        llm_client: LLMClient,
        engine: "CodenamesDuetEngine",
        player_id: int = 0,
        seed: Optional[int] = None,
    ) -> ConfidenceRanking:
        """
        Composed measurement entry point: the service elicits the standard confidence ranking and
        the engine attaches it to the current turn's record.

        :param llm_client: The LLM client to use for generating the response.
        :param engine: The game engine whose current-turn record receives the ranking.
        :param player_id: The seat of the guesser (0 or 1).

        :return: The parsed ConfidenceRanking that was attached.
        """
        ranking = await self.elicit_confidence_ranking(
            llm_client, engine.state, player_id, seed=seed
        )
        engine.attach_confidence_ranking(ranking)
        return ranking

    async def measure_and_attach_confidence_ranking_sd(
        self,
        llm_client: LLMClient,
        engine: "CodenamesDuetEngine",
        player_id: int = 0,
        seed: Optional[int] = None,
    ) -> ConfidenceRanking:
        """
        Composed sudden-death measurement entry point (headless-invocable): the service elicits the
        sudden-death confidence ranking and the engine attaches it to the per-game SuddenDeathEntry,
        consuming the pending flag.

        :param llm_client: The LLM client to use for generating the response.
        :param engine: The game engine whose SuddenDeathEntry receives the ranking.
        :param player_id: The seat of the guesser (0 or 1).

        :return: The parsed ConfidenceRanking that was attached.
        """
        ranking = await self.elicit_confidence_ranking_sd(
            llm_client, engine.state, player_id, seed=seed
        )
        engine.attach_sudden_death_ranking(ranking, player_id)
        return ranking

    def _build_clue_request(
        self, game_state: GameState, model: str, player_id: int, seed: Optional[int] = None
    ) -> LLMRequest:
        """
        Builds an LLMRequest for proposing a clue based on the current game state. This method
        extracts relevant information from the game state, formats it into a user prompt, and
        constructs the list of messages for the LLM request.

        :param game_state: The current state of the game.
        :param model: The LLM model to use for generating the clue.
        :param player_id: The ID of the player proposing the clue (0 or 1).

        :return: An instance of LLMRequest containing the messages and parameters for the LLM
            generation.
        """
        # The board as this seat sees it on the table (seat-parameterized, so either seat of an
        # LLM-vs-LLM game gets its own side of the key card).
        board = self._clue_giver_board(game_state, player_id)

        def listed(key: str) -> str:
            return "\n".join(board[key]) or "None."

        # Format the user prompt with the current game state information
        user_prompt = self._user_prompt_cg.format(
            turn_number=game_state.turn_number,
            agent_words=listed("agents"),
            assassin_words=listed("assassins"),
            civilian_words=listed("civilians"),
            other_words=listed("other_visible"),
            revealed_words="\n".join(board["revealed"]) or "No words revealed yet.",
        )

        # Build the list of messages for the LLM request
        messages = [LLMMessage(role="system", content=self._system_prompt_cg)]
        if self._one_shot_user_cg and self._one_shot_assistant_cg:
            messages += [
                LLMMessage(role="user", content=self._one_shot_user_cg),
                LLMMessage(role="assistant", content=self._one_shot_assistant_cg),
            ]
        messages.append(LLMMessage(role="user", content=user_prompt))

        return LLMRequest(
            messages=messages,
            model=model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout_s=self.timeout_s,
            seed=seed,
        )

    def _build_clue_proposal(self, response: LLMResponse) -> ClueProposal:
        """
        Processes the LLMResponse to extract the proposed clue and count, and constructs a
        ClueProposal instance.

        :param response: The response from the LLM containing the generated clue proposal.

        :return: An instance of ClueProposal containing the proposed clue, count, reasoning, and raw
            payload from the LLM response.
        """
        answer = _answer(response, ClueJSONFormat)
        logger.info(
            "clue proposal: clue=%r count=%s targets=%s reasoning=%r",
            answer.clue,
            answer.count,
            answer.targets,
            answer.reasoning,
        )
        # The intended target set S is captured as it came, for measurement only: an empty list is
        # recorded, never a reason to draw again.
        return ClueProposal(
            clue=answer.clue.strip(),
            count=answer.count,
            reasoning=answer.reasoning.strip(),
            targets=answer.targets,
            raw_payload=response.raw_payload,
        )

    def _build_guess_request(
        self, game_state: GameState, model: str, player_id: int, seed: Optional[int] = None
    ) -> LLMRequest:
        """
        Builds an LLMRequest for proposing guesses based on the current game state. This method
        extracts relevant information from the game state, formats it into a user prompt, and
        constructs the list of messages for the LLM request. The prompt carries the guesser's own
        side of the key card (see _format_own_key).

        :param game_state: The current state of the game.
        :param model: The LLM model to use for generating the guess.
        :param player_id: The ID of the player proposing the guess (0 or 1).

        :return: An instance of LLMRequest containing the messages and parameters for the LLM
            generation.
        """
        # Extract relevant information from the game state
        turn_number = game_state.turn_number
        if game_state.current_clue is None:
            raise ValueError("Cannot build a guess request when there is no clue available.")
        view = GuesserView.of(game_state, player_id)
        user_prompt = self._user_prompt_gg.format(
            turn_number=turn_number,
            clue=game_state.current_clue.clue,
            count=game_state.current_clue.count,
            previous_clues_history=view.previous_clues(),
            own_key=self._format_own_key(game_state, player_id),
            words_remaining=view.words_remaining(),
        )

        # Build the list of messages for the LLM request
        messages = [LLMMessage(role="system", content=self._system_prompt_gg)]
        if self._one_shot_user_gg and self._one_shot_assistant_gg:
            messages += [
                LLMMessage(role="user", content=self._one_shot_user_gg),
                LLMMessage(role="assistant", content=self._one_shot_assistant_gg),
            ]
        messages.append(LLMMessage(role="user", content=user_prompt))

        return LLMRequest(
            messages=messages,
            model=model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout_s=self.timeout_s,
            seed=seed,
        )

    def _build_guess_sd_request(
        self, game_state: GameState, model: str, player_id: int, seed: Optional[int] = None
    ) -> LLMRequest:
        """
        Builds an LLMRequest for sudden death guessing. No current clue is available; the LLM
        receives full clue history and must identify all remaining agents from memory. It is not
        told how many are left: the rules never say it (§9), and a player cannot work it out from
        its own key side. It does get that key side, as in the standard guess prompt (see
        _format_own_key). Seat-parameterized: the clues and the words on offer come from the
        seat's ``GuesserView``, as in _build_measurement_sd_request.

        :param game_state: The current state of the game (sudden death).
        :param model: The LLM model to use.
        :param player_id: The seat of the guesser (0 or 1).
        """
        view = GuesserView.of(game_state, player_id)
        user_prompt = self._user_prompt_sd_gg.format(
            clue_history=view.clue_history(),
            own_key=self._format_own_key(game_state, player_id),
            words_remaining=view.words_remaining(),
        )

        messages = [LLMMessage(role="system", content=self._system_prompt_sd_gg)]
        messages.append(LLMMessage(role="user", content=user_prompt))
        return LLMRequest(
            messages=messages,
            model=model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout_s=self.timeout_s,
            seed=seed,
        )

    def _build_guess_proposal(self, response: LLMResponse) -> GuessProposal:
        """
        Processes the LLMResponse to extract the proposed guesses, confidence scores, reasoning, and
        stop reason, and constructs a GuessProposal instance.

        :param response: The response from the LLM containing the generated guess proposal.

        :return: An instance of GuessProposal containing the proposed guesses, confidence scores,
            reasoning, stop reason, and raw payload from the LLM response.
        """
        answer = _answer(response, GuessJSONFormat)
        words = [proposal.word.strip() for proposal in answer.proposals]
        confidence = [proposal.confidence for proposal in answer.proposals]
        logger.info(
            "guess proposal: [%s] reasoning=%r stop_reason=%r",
            ", ".join(f"{word} ({conf})" for word, conf in zip(words, confidence, strict=True)),
            answer.reasoning,
            answer.stop_reason,
        )
        return GuessProposal(
            proposals=words,
            confidence=confidence,
            reasoning=answer.reasoning.strip(),
            stop_reason=answer.stop_reason.strip(),
            raw_payload=response.raw_payload,
        )

    def _build_measurement_request(
        self, game_state: GameState, model: str, player_id: int, seed: Optional[int] = None
    ) -> LLMRequest:
        """
        Builds the out-of-band measurement request for the standard guessing phase. It reads the
        public inputs the guesser sees - the current clue and count, and the previous clues and the
        words on offer from the same ``GuesserView`` - so the measurement observes the game state
        the play-guess request does. The clue-giver's intended
        target set S is never read here (only ``current_clue.clue``/``.count`` and history
        clue/count), preserving the guardrail that S never reaches the guesser side. It also leaves
        out the guesser's own key side, which the play request carries: the measurement scores what
        the clue points to, and the key side would mix the risk of each word into that score.

        :param game_state: The current state of the game.
        :param model: The LLM model to use.
        :param player_id: The seat of the guesser (0 or 1).

        :return: The LLMRequest for the measurement call.
        """
        if game_state.current_clue is None:
            raise ValueError("Cannot build a measurement request when there is no clue available.")
        view = GuesserView.of(game_state, player_id)
        user_prompt = self._user_prompt_meas_gg.format(
            clue=game_state.current_clue.clue,
            count=game_state.current_clue.count,
            previous_clues_history=view.previous_clues(),
            words_remaining=view.words_remaining(),
        )

        messages = [
            LLMMessage(role="system", content=self._system_prompt_meas_gg),
            LLMMessage(role="user", content=user_prompt),
        ]

        return LLMRequest(
            messages=messages,
            model=model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout_s=self.timeout_s,
            seed=seed,
        )

    def _build_measurement_sd_request(
        self, game_state: GameState, model: str, player_id: int, seed: Optional[int] = None
    ) -> LLMRequest:
        """
        Builds the out-of-band measurement request for the sudden-death phase. Mirrors
        _build_guess_sd_request: it does not say how many agents are left (§9), and it is
        seat-parameterized, from the same ``GuesserView`` of that seat. Like
        _build_measurement_request, it leaves out the guesser's own key side.

        :param game_state: The current state of the game (sudden death).
        :param model: The LLM model to use.
        :param player_id: The seat of the guesser (0 or 1).

        :return: The LLMRequest for the sudden-death measurement call.
        """
        view = GuesserView.of(game_state, player_id)
        user_prompt = self._user_prompt_meas_sd.format(
            clue_history=view.clue_history(),
            words_remaining=view.words_remaining(),
        )

        messages = [
            LLMMessage(role="system", content=self._system_prompt_meas_sd),
            LLMMessage(role="user", content=user_prompt),
        ]
        return LLMRequest(
            messages=messages,
            model=model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout_s=self.timeout_s,
            seed=seed,
        )

    def _build_confidence_ranking(self, response: LLMResponse) -> ConfidenceRanking:
        """
        Parses an LLMResponse into a ConfidenceRanking. Permissive and record-only: entries with a
        blank word are skipped, and a response that omits some cards simply yields fewer entries
        (the missing-card malformation is derivable, so no flag fields). A confidence is in [0, 1]:
        the format holds it there, and the client draws an answer outside it again.

        :param response: The response from the LLM containing the ranking.

        :return: The parsed ConfidenceRanking.
        """
        answer = _answer(response, ConfidenceRankingJSONFormat)
        ranked = [
            RankedCard(word=item.word.strip(), confidence=item.confidence)
            for item in answer.rankings
            if item.word.strip()
        ]
        logger.debug(
            "confidence ranking: [%s] reasoning=%r",
            ", ".join(f"{item.word} ({item.confidence})" for item in ranked),
            answer.reasoning,
        )
        return ConfidenceRanking(
            reasoning=answer.reasoning.strip(), rankings=ranked, raw_payload=response.raw_payload
        )

    def _format_own_key(self, game_state: GameState, player_id: int) -> str:
        """
        Renders the guesser's own side of the key card for the guess prompts. Every player sees
        their own side (§3) and may reason with it (§4.3), although only the partner's side decides
        what a guess is (§6.3). The section states facts and leaves the deductions to the model:

        - All 3 black words, with what this seat's own earlier guesses showed about them. Only this
          seat can have touched one: the partner touching it is resolved on this side, as an
          assassin, and ends the game. So an agent card on it means it was this seat's black agent,
          and this seat's own time token on it means it was the civilian.
        - The green and the beige words this seat can still guess, with the same filter as the
          words offered for guessing (``card.is_guessable_by(player_id)``).

        The measurement builders never call this: the measurement scores what the clue points to.

        :param game_state: The current state of the game.
        :param player_id: The seat of the guesser (0 or 1).

        :return: The rendered section, without its heading.
        """

        def own_role(card) -> CardRole:
            return card.llm_perspective_role if player_id == 0 else card.human_perspective_role

        def black_line(card) -> str:
            if card.revealed:
                return (
                    f"- {card.text} (covered by an agent card: you guessed it, and it was an "
                    "Agent for you)"
                )
            if player_id in card.time_marker_by:
                return (
                    f"- {card.text} (your time token is on it: you guessed it, and it was a "
                    "Civilian for you)"
                )
            return f"- {card.text}"

        def still_guessable(role: CardRole) -> list[str]:
            words = [
                f"- {card.text}"
                for card in game_state.board.cards
                if own_role(card) == role and card.is_guessable_by(player_id)
            ]
            return words or ["None."]

        black = [
            black_line(card)
            for card in game_state.board.cards
            if own_role(card) == CardRole.ASSASSIN
        ]
        return "\n".join(
            [
                "Black on your side (all 3):",
                *black,
                "",
                "Green on your side (only the words you can still guess):",
                *still_guessable(CardRole.AGENT),
                "",
                "Beige on your side (only the words you can still guess):",
                *still_guessable(CardRole.CIVILIAN),
            ]
        )

    def _clue_giver_board(self, game_state: GameState, player_id: int) -> dict[str, list[str]]:
        """
        Sorts the board words the way the clue giver at ``player_id`` sees them on the table: by its
        own side of the key card and by what sits on each word. It is derived from the board on
        every call, so a word drops out as soon as an agent card or a second time token covers it.

        - ``agents``: its green words not yet covered, the ones its partner still has to find.
        - ``assassins`` / ``civilians``: its black and beige words its partner can still touch.
        - ``other_visible``: words its partner can no longer touch, because the partner's own time
          token sits on them (they are beige on this side). They carry no risk, but they are still
          visible (§8.2), so the clue cannot be one of them or a form of one.
        - ``revealed``: its green words already covered by an agent card.

        A word covered by an agent card or by two time tokens is out of play and no longer visible,
        so it is in none of the lists, except a covered green word, which is listed in ``revealed``.

        :param game_state: The current state of the game.
        :param player_id: The seat of the clue giver (0 or 1).

        :return: The ``- WORD`` lines of each list, in board order.
        """
        partner = 1 - player_id
        board: dict[str, list[str]] = {
            "agents": [],
            "assassins": [],
            "civilians": [],
            "other_visible": [],
            "revealed": [],
        }
        for card in game_state.board.cards:
            role = card.llm_perspective_role if player_id == 0 else card.human_perspective_role
            line = f"- {card.text}"
            if card.is_covered:
                if card.revealed and role == CardRole.AGENT:
                    board["revealed"].append(line)
            elif not card.is_guessable_by(partner):
                board["other_visible"].append(line)
            elif role == CardRole.AGENT:
                board["agents"].append(line)
            elif role == CardRole.ASSASSIN:
                board["assassins"].append(line)
            else:
                board["civilians"].append(line)
        return board
