import hashlib
import json
from typing import TYPE_CHECKING, Optional
from backend.app.core.llm.client import LLMClient
from backend.app.core.clue_validator import ClueValidator
from backend.app.models.llm_schemas import ClueProposal, GuessProposal, LLMRequest, LLMResponse, LLMMessage, LLMCallRecord, ClueJSONFormat, GuessJSONFormat, ConfidenceRankingJSONFormat
from backend.app.models.game_schemas import GameState, GamePhase, CardRole, ClueEntry, ConfidenceRanking, RankedCard

if TYPE_CHECKING:
    from backend.app.core.engine import CodenamesDuetEngine

MAX_CLUE_RETRIES = 3

# The seat allowed to guess in each sudden-death phase, mirroring the engine's own invariant
# (resolve_guess raises unless seat 1 acts in SUDDEN_DEATH_HUMAN / seat 0 in SUDDEN_DEATH_LLM).
_SD_PHASE_BY_SEAT = {0: GamePhase.SUDDEN_DEATH_LLM,
                     1: GamePhase.SUDDEN_DEATH_HUMAN}


def _require_sd_seat_phase(game_state: GameState, player_id: int, action: str) -> None:
    """Guard: seat 0 may act only in SUDDEN_DEATH_LLM, seat 1 only in SUDDEN_DEATH_HUMAN."""
    expected = _SD_PHASE_BY_SEAT.get(player_id)
    if expected is None or game_state.current_phase != expected:
        raise ValueError(
            f"Cannot {action} for seat {player_id} in phase {game_state.current_phase}: "
            f"seat 0 is valid only in SUDDEN_DEATH_LLM and seat 1 only in SUDDEN_DEATH_HUMAN.")


class LLMService:
    SYSTEM_TEMP_CG_PATH = "data/prompt_templates/SYSTEM_TEMPLATE_CLUE_GIVER.txt"
    USER_TEMP_CG_PATH = "data/prompt_templates/USER_TEMPLATE_CLUE_GIVER.txt"
    SYSTEM_TEMP_GG_PATH = "data/prompt_templates/SYSTEM_TEMPLATE_GUESSER.txt"
    USER_TEMP_GG_PATH = "data/prompt_templates/USER_TEMPLATE_GUESSER.txt"
    ONE_SHOT_USER_CG_PATH = "data/prompt_templates/ONE_SHOT_USER_CLUE_GIVER.txt"
    ONE_SHOT_ASSISTANT_CG_PATH = "data/prompt_templates/ONE_SHOT_ASSISTANT_CLUE_GIVER.txt"
    ONE_SHOT_USER_GG_PATH = "data/prompt_templates/ONE_SHOT_USER_GUESSER.txt"
    ONE_SHOT_ASSISTANT_GG_PATH = "data/prompt_templates/ONE_SHOT_ASSISTANT_GUESSER.txt"
    SYSTEM_TEMP_SD_GG_PATH = "data/prompt_templates/SYSTEM_TEMPLATE_SUDDEN_DEATH_GUESSER.txt"
    USER_TEMP_SD_GG_PATH = "data/prompt_templates/USER_TEMPLATE_SUDDEN_DEATH_GUESSER.txt"
    SYSTEM_TEMP_MEAS_GG_PATH = "data/prompt_templates/SYSTEM_TEMPLATE_MEASUREMENT_GUESSER.txt"
    USER_TEMP_MEAS_GG_PATH = "data/prompt_templates/USER_TEMPLATE_MEASUREMENT_GUESSER.txt"
    SYSTEM_TEMP_MEAS_SD_PATH = "data/prompt_templates/SYSTEM_TEMPLATE_MEASUREMENT_SD.txt"
    USER_TEMP_MEAS_SD_PATH = "data/prompt_templates/USER_TEMPLATE_MEASUREMENT_SD.txt"

    def __init__(self, temperature: float = 0.7, max_tokens: int = 5000, timeout_s: int = 30):
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s
        self._system_prompt_cg = self._load_prompt_template(
            self.SYSTEM_TEMP_CG_PATH, 0)
        self._user_prompt_cg = self._load_prompt_template(
            self.USER_TEMP_CG_PATH, 1)
        self._system_prompt_gg = self._load_prompt_template(
            self.SYSTEM_TEMP_GG_PATH, 2)
        self._user_prompt_gg = self._load_prompt_template(
            self.USER_TEMP_GG_PATH, 3)
        self._system_prompt_sd_gg = self._load_prompt_template(
            self.SYSTEM_TEMP_SD_GG_PATH, 4)
        self._user_prompt_sd_gg = self._load_prompt_template(
            self.USER_TEMP_SD_GG_PATH, 5)
        self._system_prompt_meas_gg = self._load_prompt_template(
            self.SYSTEM_TEMP_MEAS_GG_PATH, 6)
        self._user_prompt_meas_gg = self._load_prompt_template(
            self.USER_TEMP_MEAS_GG_PATH, 7)
        self._system_prompt_meas_sd = self._load_prompt_template(
            self.SYSTEM_TEMP_MEAS_SD_PATH, 8)
        self._user_prompt_meas_sd = self._load_prompt_template(
            self.USER_TEMP_MEAS_SD_PATH, 9)
        self._one_shot_user_cg = self._load_one_shot(
            self.ONE_SHOT_USER_CG_PATH, 0)
        self._one_shot_assistant_cg = self._load_one_shot(
            self.ONE_SHOT_ASSISTANT_CG_PATH, 1)
        self._one_shot_user_gg = self._load_one_shot(
            self.ONE_SHOT_USER_GG_PATH, 2)
        self._one_shot_assistant_gg = self._load_one_shot(
            self.ONE_SHOT_ASSISTANT_GG_PATH, 3)

    def _loaded_template_texts(self) -> dict[str, str]:
        """Read-only accessor: stable template key -> the text the service actually holds in memory.

        Covers every template text the service loads and sends - the 10 prompt templates AND the 4
        one-shot example texts - so the fingerprint reflects the real payloads, including any
        ``_default_*`` fallback that was active because a file was missing.
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

    def template_fingerprint(self) -> str:
        """Deterministic SHA-256 hex fingerprint of the LOADED template texts (not the directory).

        Hashing the loaded texts - rather than the files on disk - means a ``_default_*`` fallback
        produces a different fingerprint than the real file, which is the whole point: the run row
        records exactly which prompt texts were sent. Stable across processes: templates are sorted
        by key and each contributes ``key\\0text\\0`` to the digest.
        """
        h = hashlib.sha256()
        for key, text in sorted(self._loaded_template_texts().items()):
            h.update(key.encode("utf-8"))
            h.update(b"\0")
            # ``str(text)`` defensively: a loaded text is normally a str, but a malformed hardcoded
            # ``_default_*`` fallback can hold a non-str. Coercing keeps the fingerprint deterministic
            # and content-sensitive without ever raising - provenance must never abort a run.
            h.update(str(text).encode("utf-8"))
            h.update(b"\0")
        return h.hexdigest()

    @staticmethod
    def _call_record(request: LLMRequest, response: LLMResponse, role: str, retry_index: int = 0) -> LLMCallRecord:
        """Build the in-memory audit carrier from the paired request/response.

        The service only fills a field it already has in hand; it never touches persistence. The
        rendered prompt is the messages as sent (``request.messages``); all sampling telemetry is
        copied verbatim from the ``LLMResponse``.
        """
        usage = response.usage
        return LLMCallRecord(
            role=role,
            retry_index=retry_index,
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

    async def propose_clue(self, llm_client: LLMClient, game_state: GameState, validator: ClueValidator, player_id: int = 0, seed: Optional[int] = None) -> ClueProposal:
        """
        Proposes a clue for the current game state. This method checks that the game is in the 
        correct phase and that the LLM is the clue giver before building the request, sending it to
        the LLM, and processing the response.

        :param llm_client: The LLM client to use for generating responses.
        :param game_state: The current state of the game.
        :param player_id: The ID of the player proposing the clue (0 for LLM).

        :return: An instance of ClueProposal containing the proposed clue and count from the LLM.
        """
        if game_state.current_phase != GamePhase.GIVING_CLUE:
            raise ValueError(
                "Cannot propose a clue when the game is not in the GIVING_CLUE phase.")

        if game_state.clue_giver != player_id:
            raise ValueError(
                "The player must be the clue giver to propose a clue.")

        request = self._build_clue_request(
            game_state, llm_client.model_name, player_id, seed=seed)

        reason = ""
        # Audit every attempt (accepted + rejected), ordered by retry_index.
        llm_calls: list[LLMCallRecord] = []

        for retry_index in range(MAX_CLUE_RETRIES):
            response = await llm_client.generate(request, expected_format=ClueJSONFormat)
            proposal = self._build_clue_proposal(response)
            llm_calls.append(
                self._call_record(request, response, "clue_giver", retry_index)
            )

            clue_entry = ClueEntry(
                clue=proposal.clue,
                count=proposal.count,
                clue_giver=player_id,
                turn_number=game_state.turn_number,
            )
            valid, reason = validator.is_valid(
                clue_entry, game_state.clue_history)
            if valid:
                proposal.llm_calls = llm_calls
                return proposal

            print(
                f"LLM proposed invalid clue '{proposal.clue}': {reason}. Retrying...")
            request = self._build_clue_retry_request(
                original_request=request,
                invalid_response_text=response.text,
                invalid_clue=proposal.clue,
                reason=reason,
            )

        raise ValueError(
            f"LLM failed to produce a valid clue after {MAX_CLUE_RETRIES} attempts. "
            f"Last rejection: {reason}"
        )

    async def propose_guess(self, llm_client: LLMClient, game_state: GameState, player_id: int = 0, seed: Optional[int] = None) -> GuessProposal:
        """
        Proposes guesses for the current game state. This method checks that the game is in the 
        correct phase and that the LLM is the guesser before building the request, sending it to
        the LLM, and processing the response.

        :param llm_client: The LLM client to use for generating responses.
        :param game_state: The current state of the game.
        :param player_id: The ID of the player proposing the guesses (0 for LLM).

        :return: An instance of GuessProposal containing the proposed guesses from the LLM.
        """
        if game_state.current_phase != GamePhase.GUESSING:
            raise ValueError(
                "Cannot propose a guess when the game is not in the GUESSING phase.")

        if game_state.guesser != player_id:
            raise ValueError(
                "The player must be the guesser to propose a guess.")

        if game_state.current_clue.turn_number != game_state.turn_number:
            raise ValueError(
                "Cannot propose a guess when there is no clue available."
            )

        # Build the LLM request for proposing a guess
        request = self._build_guess_request(
            game_state, llm_client.model_name, player_id, seed=seed)

        # Send the request to the LLM client and get the response.
        response = await llm_client.generate(request, expected_format=GuessJSONFormat)

        # Process the response and convert it into a GuessProposal.
        guess_proposal = self._build_guess_proposal(response)
        guess_proposal.llm_call = self._call_record(
            request, response, "guesser")

        return guess_proposal

    async def propose_guess_sd(self, llm_client: LLMClient, game_state: GameState, player_id: int = 0, seed: Optional[int] = None) -> GuessProposal:
        """
        Proposes guesses for the SUDDEN_DEATH_LLM phase. No clue is available; the LLM must
        identify its remaining agents from memory of the full game.

        :param llm_client: The LLM client to use for generating responses.
        :param game_state: The current state of the game.
        :param player_id: The ID of the player proposing the guesses (0 for LLM).

        :return: An instance of GuessProposal containing the proposed guesses from the LLM.
        """
        _require_sd_seat_phase(game_state, player_id,
                               "propose a sudden death guess")

        request = self._build_guess_sd_request(
            game_state, llm_client.model_name, player_id, seed=seed)
        response = await llm_client.generate(request, expected_format=GuessJSONFormat)
        guess_proposal = self._build_guess_proposal(response)
        guess_proposal.llm_call = self._call_record(
            request, response, "guesser_sd")
        return guess_proposal

    async def elicit_confidence_ranking(self, llm_client: LLMClient, game_state: GameState, player_id: int = 0, seed: Optional[int] = None) -> ConfidenceRanking:
        """
        Elicits an out-of-band confidence ranking over all unrevealed cards for the current clue,
        parallel to propose_guess but for measurement. This is a separate call that does not touch
        the game-play guess request, is not appended to any conversation history, and never sees the
        clue-giver's intended target set S. It is elicited at the pre-resolution instant (same game
        state as the play-guess request) and returns the parsed ranking; attaching it is the engine's
        job.

        :param llm_client: The LLM client to use for generating the response.
        :param game_state: The current state of the game (guessing phase, before resolution).
        :param player_id: The seat of the guesser (0 for LLM).

        :return: The parsed ConfidenceRanking over the unrevealed cards.
        """
        if game_state.current_phase != GamePhase.GUESSING:
            raise ValueError(
                "Cannot elicit a confidence ranking when the game is not in the GUESSING phase.")

        if game_state.guesser != player_id:
            raise ValueError(
                "The player must be the guesser to elicit a confidence ranking.")

        request = self._build_measurement_request(
            game_state, llm_client.model_name, player_id, seed=seed)
        response = await llm_client.generate(request, expected_format=ConfidenceRankingJSONFormat)
        ranking = self._build_confidence_ranking(response)
        ranking.llm_call = self._call_record(request, response, "measurement")
        return ranking

    async def elicit_confidence_ranking_sd(self, llm_client: LLMClient, game_state: GameState, player_id: int = 0, seed: Optional[int] = None) -> ConfidenceRanking:
        """
        Elicits the out-of-band sudden-death confidence ranking over all unrevealed cards, parallel
        to propose_guess_sd. Measurement only; never sees S. Elicited on entry to the sudden-death
        phase, before the first selection.

        :param llm_client: The LLM client to use for generating the response.
        :param game_state: The current state of the game (SUDDEN_DEATH_LLM phase).
        :param player_id: The seat of the guesser (0 for LLM).

        :return: The parsed ConfidenceRanking over the unrevealed cards.
        """
        _require_sd_seat_phase(
            game_state, player_id, "elicit a sudden death confidence ranking")

        request = self._build_measurement_sd_request(
            game_state, llm_client.model_name, player_id, seed=seed)
        response = await llm_client.generate(request, expected_format=ConfidenceRankingJSONFormat)
        ranking = self._build_confidence_ranking(response)
        ranking.llm_call = self._call_record(
            request, response, "measurement_sd")
        return ranking

    async def measure_and_attach_confidence_ranking(self, llm_client: LLMClient, engine: "CodenamesDuetEngine", player_id: int = 0, seed: Optional[int] = None) -> ConfidenceRanking:
        """
        Composed measurement entry point (headless-invocable, independent of routes.py): the service
        elicits the standard confidence ranking and the engine attaches it to the current turn's
        record. This is the exact sequence the headless runner will reuse.

        :param llm_client: The LLM client to use for generating the response.
        :param engine: The game engine whose current-turn record receives the ranking.
        :param player_id: The seat of the guesser (0 for LLM).

        :return: The parsed ConfidenceRanking that was attached.
        """
        ranking = await self.elicit_confidence_ranking(
            llm_client, engine.state, player_id, seed=seed)
        engine.attach_confidence_ranking(ranking)
        return ranking

    async def measure_and_attach_confidence_ranking_sd(self, llm_client: LLMClient, engine: "CodenamesDuetEngine", player_id: int = 0, seed: Optional[int] = None) -> ConfidenceRanking:
        """
        Composed sudden-death measurement entry point (headless-invocable): the service elicits the
        sudden-death confidence ranking and the engine attaches it to the per-game SuddenDeathEntry,
        consuming the pending flag.

        :param llm_client: The LLM client to use for generating the response.
        :param engine: The game engine whose SuddenDeathEntry receives the ranking.
        :param player_id: The seat of the guesser (0 for LLM).

        :return: The parsed ConfidenceRanking that was attached.
        """
        ranking = await self.elicit_confidence_ranking_sd(
            llm_client, engine.state, player_id, seed=seed)
        engine.attach_sudden_death_ranking(ranking, player_id)
        return ranking

    def _build_clue_request(self, game_state: GameState, model: str, player_id: int, seed: Optional[int] = None) -> LLMRequest:
        """
        Builds an LLMRequest for proposing a clue based on the current game state. This method
        extracts relevant information from the game state, formats it into a user prompt, and
        constructs the list of messages for the LLM request.

        :param game_state: The current state of the game.
        :param model: The LLM model to use for generating the clue.
        :param player_id: The ID of the player proposing the clue (0 for LLM).

        :return: An instance of LLMRequest containing the messages and parameters for the LLM
            generation.
        """
        # Extract relevant information from the game state
        turn_number = game_state.turn_number
        # As some plays will be automated between two LLM agents, we need to determine which
        # words are relevant based on the player ID.
        if player_id == 0:
            # LLM is the clue giver, so we consider its perspective for the agent and dangerous words
            agent_words = self._get_llm_perspective_agent_words(game_state)
            assassin_words = self._get_llm_perspective_assassin_words(
                game_state)
            civilian_words = self._get_llm_perspective_civilian_words(
                game_state)
            rev_words = self._get_llm_perspective_revealed_words(game_state)
        else:
            # Same as above, but from the human player's perspective
            agent_words = self._get_human_perspective_agent_words(game_state)
            assassin_words = self._get_human_perspective_assassin_words(
                game_state)
            civilian_words = self._get_human_perspective_civilian_words(
                game_state)
            rev_words = self._get_human_perspective_revealed_words(game_state)

        # Format the user prompt with the current game state information
        user_prompt = self._user_prompt_cg.format(
            turn_number=turn_number,
            agent_words="\n".join(agent_words),
            assassin_words="\n".join(assassin_words),
            civilian_words="\n".join(civilian_words),
            revealed_words="\n".join(
                rev_words) if rev_words else "No words revealed yet."
        )

        print("DEBUG: User prompt for clue proposal:\n" +
              user_prompt)  # Debug print for the user prompt

        # Build the list of messages for the LLM request
        messages = [LLMMessage(role="system", content=self._system_prompt_cg)]
        if self._one_shot_user_cg and self._one_shot_assistant_cg:
            messages += [
                LLMMessage(role="user", content=self._one_shot_user_cg),
                LLMMessage(role="assistant",
                           content=self._one_shot_assistant_cg),
            ]
        messages.append(LLMMessage(role="user", content=user_prompt))

        return LLMRequest(messages=messages, model=model, temperature=self.temperature,
                          max_tokens=self.max_tokens, timeout_s=self.timeout_s, seed=seed)

    def _build_clue_proposal(self, response: LLMResponse) -> ClueProposal:
        """
        Processes the LLMResponse to extract the proposed clue and count, and constructs a 
        ClueProposal instance.

        :param response: The response from the LLM containing the generated clue proposal.

        :return: An instance of ClueProposal containing the proposed clue, count, reasoning, and raw
            payload from the LLM response.
        """
        response_content = response.text.strip()
        try:
            response_json = json.loads(response_content)
            clue = response_json.get("clue")
            count = response_json.get("count")
            reasoning = response_json.get("reasoning", "")
            # The intended target set S. Captured verbatim for measurement only; a missing,
            # empty, or malformed list is recorded as-is and never triggers a retry.
            targets = response_json.get("targets", [])
        except json.JSONDecodeError:
            raise ValueError(
                "LLM response is not valid JSON. Response content: " + response_content)

        print(
            f"DEBUG: Extracted clue proposal - Clue: '{clue}', Count: {count}, "
            f"Reasoning: '{reasoning}', Targets: {targets}")

        return ClueProposal(clue=clue.strip(), count=count, reasoning=reasoning.strip(),
                            targets=targets, raw_payload=response.raw_payload)

    def _build_clue_retry_request(
        self,
        original_request: LLMRequest,
        invalid_response_text: str,
        invalid_clue: str,
        reason: str,
    ) -> LLMRequest:
        """
        Builds a retry LLMRequest after the LLM proposed an invalid clue. The conversation is
        extended with the failed attempt (as a user→assistant pair) followed by a new user message
        that explains why the clue was rejected and asks for a fresh one.

        The resulting message list is:
            [system, (one-shot user), (one-shot assistant), original user, invalid assistant, correction user]
        """
        correction = (
            f"Your previous clue was rejected.\n\n"
            f"Rejected clue: \"{invalid_clue}\"\n"
            f"Reason: {reason}\n\n"
            f"Please provide a new valid clue using the same JSON format."
        )
        messages = list(original_request.messages) + [
            LLMMessage(role="assistant", content=invalid_response_text),
            LLMMessage(role="user", content=correction),
        ]
        return LLMRequest(
            messages=messages,
            model=original_request.model,
            temperature=original_request.temperature,
            max_tokens=original_request.max_tokens,
            timeout_s=original_request.timeout_s,
            seed=original_request.seed,
        )

    def _build_guess_request(self, game_state: GameState, model: str, player_id: int, seed: Optional[int] = None) -> LLMRequest:
        """
        Builds an LLMRequest for proposing guesses based on the current game state. This method 
        extracts relevant information from the game state, formats it into a user prompt, and 
        constructs the list of messages for the LLM request.

        :param game_state: The current state of the game.
        :param model: The LLM model to use for generating the guess.
        :param player_id: The ID of the player proposing the guess (0 for LLM).

        :return: An instance of LLMRequest containing the messages and parameters for the LLM
            generation.
        """
        # Extract relevant information from the game state
        turn_number = game_state.turn_number
        clue = game_state.current_clue.clue
        count = game_state.current_clue.count
        previous_clues_history = "\n".join([
            f"- Turn: {clue_entry.turn_number}, Clue: {clue_entry.clue}, Count: {clue_entry.count}"
            for clue_entry in game_state.clue_history if clue_entry.clue_giver != player_id
        ])
        words_remaining = "\n".join([
            f"- {card.text}" for card in game_state.board.cards if player_id not in card.revealed_by and
            player_id not in card.time_marker_by
        ])

        # Format the user prompt with the game state information
        user_prompt = self._user_prompt_gg.format(
            turn_number=turn_number,
            clue=clue,
            count=count,
            previous_clues_history=previous_clues_history if previous_clues_history else "No previous clues.",
            words_remaining=words_remaining
        )

        print("DEBUG: User prompt for guess proposal:\n" +
              user_prompt)  # Debug print for the user prompt

        # Build the list of messages for the LLM request
        messages = [LLMMessage(role="system", content=self._system_prompt_gg)]
        if self._one_shot_user_gg and self._one_shot_assistant_gg:
            messages += [
                LLMMessage(role="user", content=self._one_shot_user_gg),
                LLMMessage(role="assistant",
                           content=self._one_shot_assistant_gg),
            ]
        messages.append(LLMMessage(role="user", content=user_prompt))

        return LLMRequest(messages=messages, model=model, temperature=self.temperature,
                          max_tokens=self.max_tokens, timeout_s=self.timeout_s, seed=seed)

    def _build_guess_sd_request(self, game_state: GameState, model: str, player_id: int, seed: Optional[int] = None) -> LLMRequest:
        """
        Builds an LLMRequest for sudden death guessing. No current clue is available; the LLM
        receives full clue history and must identify all remaining agents from memory. Seat-
        parameterized: it reports the guesser's own remaining agent count and the SAME unrevealed-word
        predicate (``player_id not in card.revealed_by and player_id not in card.time_marker_by``) 
        as _build_measurement_sd_request, so it generalizes to either seat in an LLM-vs-LLM run.

        :param game_state: The current state of the game (sudden death).
        :param model: The LLM model to use.
        :param player_id: The seat of the guesser (0 for LLM).
        """
        clue_history = "\n".join([
            f"- Turn {e.turn_number}: Clue '{e.clue}' (count {e.count})"
            for e in game_state.clue_history if e.clue_giver != player_id
        ])
        words_remaining = "\n".join([
            f"- {card.text}" for card in game_state.board.cards
            if player_id not in card.revealed_by and player_id not in card.time_marker_by
        ])
        user_prompt = self._user_prompt_sd_gg.format(
            clue_history=clue_history or "No clues were given.",
            words_remaining=words_remaining,
            agents_remaining=game_state.agents_remaining[player_id],
        )

        print("DEBUG: User prompt for sudden death guess:\n" + user_prompt)

        messages = [LLMMessage(
            role="system", content=self._system_prompt_sd_gg)]
        messages.append(LLMMessage(role="user", content=user_prompt))
        return LLMRequest(messages=messages, model=model, temperature=self.temperature,
                          max_tokens=self.max_tokens, timeout_s=self.timeout_s, seed=seed)

    def _build_guess_proposal(self, response: LLMResponse) -> GuessProposal:
        """
        Processes the LLMResponse to extract the proposed guesses, confidence scores, reasoning, and
        stop reason, and constructs a GuessProposal instance.

        :param response: The response from the LLM containing the generated guess proposal.

        :return: An instance of GuessProposal containing the proposed guesses, confidence scores,
            reasoning, stop reason, and raw payload from the LLM response.
        """
        response_content = response.text.strip()

        try:
            response_json = json.loads(response_content)
            proposals = response_json.get("proposals", [])
            reasoning = response_json.get("reasoning", "")
            stop_reason = response_json.get("stop_reason", "")
        except json.JSONDecodeError:
            raise ValueError(
                "LLM response is not valid JSON. Response content: " + response_content)

        confidence = [proposal.get("confidence", 0) for proposal in proposals]
        proposals = [proposal.get("word", "").strip()
                     for proposal in proposals]

        print(
            f"DEBUG: Extracted guess proposals - {proposals} with confidence scores {confidence}. Reasoning: '{reasoning}'. Stop reason: '{stop_reason}'")

        return GuessProposal(
            proposals=proposals, confidence=confidence, reasoning=reasoning.strip(),
            stop_reason=stop_reason.strip(), raw_payload=response.raw_payload
        )

    def _build_measurement_request(self, game_state: GameState, model: str, player_id: int, seed: Optional[int] = None) -> LLMRequest:
        """
        Builds the out-of-band measurement request for the standard guessing phase. It mirrors
        _build_guess_request exactly for the public inputs the guesser sees - the current clue and
        count, the previous-clue history, and the SAME unrevealed-word filter - so the measurement
        observes the identical game state as the play-guess request. The clue-giver's intended target
        set S is never read here (only ``current_clue.clue``/``.count`` and history clue/count),
        preserving the guardrail that S never reaches the guesser side.

        :param game_state: The current state of the game.
        :param model: The LLM model to use.
        :param player_id: The seat of the guesser (0 for LLM).

        :return: The LLMRequest for the measurement call.
        """
        clue = game_state.current_clue.clue
        count = game_state.current_clue.count
        previous_clues_history = "\n".join([
            f"- Turn: {clue_entry.turn_number}, Clue: {clue_entry.clue}, Count: {clue_entry.count}"
            for clue_entry in game_state.clue_history if clue_entry.clue_giver != player_id
        ])
        words_remaining = "\n".join([
            f"- {card.text}" for card in game_state.board.cards if player_id not in card.revealed_by and
            player_id not in card.time_marker_by
        ])

        user_prompt = self._user_prompt_meas_gg.format(
            clue=clue,
            count=count,
            previous_clues_history=previous_clues_history if previous_clues_history else "No previous clues.",
            words_remaining=words_remaining,
        )

        messages = [
            LLMMessage(role="system", content=self._system_prompt_meas_gg),
            LLMMessage(role="user", content=user_prompt),
        ]

        return LLMRequest(messages=messages, model=model, temperature=self.temperature,
                          max_tokens=self.max_tokens, timeout_s=self.timeout_s, seed=seed)

    def _build_measurement_sd_request(self, game_state: GameState, model: str, player_id: int, seed: Optional[int] = None) -> LLMRequest:
        """
        Builds the out-of-band measurement request for the sudden-death phase. Mirrors
        _build_guess_sd_request but is seat-parameterized: it reports the guesser's own remaining
        agent count and the SAME unrevealed-word filter from that seat's perspective, so it
        generalizes to either seat in an LLM-vs-LLM run.

        :param game_state: The current state of the game (sudden death).
        :param model: The LLM model to use.
        :param player_id: The seat of the guesser (0 for LLM).

        :return: The LLMRequest for the sudden-death measurement call.
        """
        clue_history = "\n".join([
            f"- Turn {e.turn_number}: Clue '{e.clue}' (count {e.count})"
            for e in game_state.clue_history if e.clue_giver != player_id
        ])
        words_remaining = "\n".join([
            f"- {card.text}" for card in game_state.board.cards
            if player_id not in card.revealed_by and player_id not in card.time_marker_by
        ])
        user_prompt = self._user_prompt_meas_sd.format(
            clue_history=clue_history or "No clues were given.",
            words_remaining=words_remaining,
            agents_remaining=game_state.agents_remaining[player_id],
        )

        messages = [
            LLMMessage(role="system", content=self._system_prompt_meas_sd),
            LLMMessage(role="user", content=user_prompt),
        ]
        return LLMRequest(messages=messages, model=model, temperature=self.temperature,
                          max_tokens=self.max_tokens, timeout_s=self.timeout_s, seed=seed)

    def _build_confidence_ranking(self, response: LLMResponse) -> ConfidenceRanking:
        """
        Parses an LLMResponse into a ConfidenceRanking. Permissive and record-only: entries with an
        empty word are skipped, confidence is clamped to [0, 1], and a response that omits some cards
        simply yields fewer entries (the missing-card malformation is derivable, so no flag fields).

        :param response: The response from the LLM containing the ranking.

        :return: The parsed ConfidenceRanking.
        """
        response_content = response.text.strip()
        try:
            response_json = json.loads(response_content)
            rankings_raw = response_json.get("rankings", [])
            reasoning = response_json.get("reasoning", "")
        except json.JSONDecodeError:
            raise ValueError(
                "LLM response is not valid JSON. Response content: " + response_content)

        ranked: list[RankedCard] = []
        for item in rankings_raw:
            word = item.get("word", "").strip()
            if not word:
                continue
            confidence = max(0.0, min(1.0, float(item.get("confidence", 0.0))))
            ranked.append(RankedCard(word=word, confidence=confidence))

        print(
            f"DEBUG: Extracted confidence ranking - {ranked}. Reasoning: '{reasoning}'")

        return ConfidenceRanking(
            reasoning=reasoning.strip(), rankings=ranked, raw_payload=response.raw_payload)

    def _get_llm_perspective_agent_words(self, game_state: GameState) -> list[str]:
        """
        Extracts the non-revealed agent words from the game state based on the LLM's perspective.

        :param game_state: The current state of the game.

        :return: A list of agent words that are not revealed from the LLM's perspective.
        """
        return [f"- {card.text}" for card in game_state.board.cards if card.llm_perspective_role ==
                CardRole.AGENT and 1 not in card.revealed_by]

    def _get_llm_perspective_assassin_words(self, game_state: GameState) -> list[str]:
        """
        Extracts the assassin words from the game state based on the LLM's perspective. Words that 
        are marked with a time marker are ignored to keep the final prompt concise.

        :param game_state: The current state of the game.

        :return: A list of assassin words that are not marked with a time marker from the LLM's
            perspective.
        """
        return [f"- {card.text}" for card in game_state.board.cards if card.llm_perspective_role ==
                CardRole.ASSASSIN]

    def _get_llm_perspective_civilian_words(self, game_state: GameState) -> list[str]:
        """
        Extracts the civilian words from the game state based on the LLM's perspective. Words that 
        are marked with a time marker are ignored to keep the final prompt concise.

        :param game_state: The current state of the game.

        :return: A list of civilian words that are not marked with a time marker from the LLM's
            perspective.
        """
        return [f"- {card.text}" for card in game_state.board.cards if card.llm_perspective_role ==
                CardRole.CIVILIAN and 1 not in card.time_marker_by]

    def _get_llm_perspective_revealed_words(self, game_state: GameState) -> list[str]:
        """
        Extracts the revealed words from the game state based on the LLM's perspective.

        :param game_state: The current state of the game.

        :return: A list of revealed words from the LLM's perspective.
        """
        return [
            f"- {card.text}" for card in game_state.board.cards if 1 in card.revealed_by and
            card.llm_perspective_role == CardRole.AGENT
        ]

    def _get_human_perspective_agent_words(self, game_state: GameState) -> list[str]:
        """
        Extracts the agent words from the game state based on the human's perspective. Words that are
        marked with a time marker are ignored to keep the final prompt concise.

        :param game_state: The current state of the game.

        :return: A list of agent words that are not revealed from the human's perspective.
        """
        return [f"- {card.text}" for card in game_state.board.cards if card.human_perspective_role ==
                CardRole.AGENT and 0 not in card.revealed_by]

    def _get_human_perspective_assassin_words(self, game_state: GameState) -> list[str]:
        """
        Extract the assassin words from the game state based on the human's perspective. Words that 
        are marked with a time marker are ignored to keep the final prompt concise.

        :param game_state: The current state of the game.

        :return: A list of assassin words that are not marked with a time marker from the human's
            perspective.
        """
        return [f"- {card.text}" for card in game_state.board.cards if card.human_perspective_role ==
                CardRole.ASSASSIN]

    def _get_human_perspective_civilian_words(self, game_state: GameState) -> list[str]:
        """
        Extract the civilian words from the game state based on the human's perspective. Words that 
        are marked with a time marker are ignored to keep the final prompt concise.

        :param game_state: The current state of the game.

        :return: A list of civilian words that are not marked with a time marker from the human's
            perspective.
        """
        return [f"- {card.text}" for card in game_state.board.cards if card.human_perspective_role ==
                CardRole.CIVILIAN and 0 not in card.time_marker_by]

    def _get_human_perspective_revealed_words(self, game_state: GameState) -> list[str]:
        """
        Extracts the revealed words from the game state based on the human's perspective.

        :param game_state: The current state of the game.

        :return: A list of revealed words from the human's perspective.
        """
        return [
            f"- {card.text}" for card in game_state.board.cards if 0 in card.revealed_by
            and card.human_perspective_role == CardRole.AGENT
        ]

    def _load_prompt_template(self, template_path: str, prompt_type: int) -> str:
        """
        Loads a prompt template from the specified file path. If the file is not found, it returns
        a default prompt based on the type of prompt requested (system or user).

        :param template_path: The file path to the prompt template.
        :param prompt_type: An integer indicating the type of prompt (0 for system, 1 for user).

        :return: The loaded prompt template or a default prompt.
        """
        try:
            with open(template_path, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            match prompt_type:
                case 0:
                    return self._default_system_prompt_cg()
                case 1:
                    return self._default_user_prompt_cg()
                case 2:
                    return self._default_system_prompt_gg()
                case 3:
                    return self._default_user_prompt_gg()
                case 4:
                    return self._default_system_prompt_sd_gg()
                case 5:
                    return self._default_user_prompt_sd_gg()
                case 6:
                    return self._default_system_prompt_meas_gg()
                case 7:
                    return self._default_user_prompt_meas_gg()
                case 8:
                    return self._default_system_prompt_meas_sd()
                case 9:
                    return self._default_user_prompt_meas_sd()
                case _:
                    raise ValueError(
                        "Invalid prompt type specified. Must be 0 (system clue giver), "
                        "1 (user clue giver), 2 (system guesser), 3 (user guesser), "
                        "4 (system sudden death guesser), 5 (user sudden death guesser), "
                        "6 (system measurement guesser), 7 (user measurement guesser), "
                        "8 (system measurement sudden death), or 9 (user measurement sudden death).")

    def _load_one_shot(self, path: str, prompt_type: int) -> str:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            match prompt_type:
                case 0:
                    return self._default_os_user_cg()
                case 1:
                    return self._default_os_assistant_cg()
                case 2:
                    return self._default_os_user_gg()
                case 3:
                    return self._default_os_assistant_gg()
                case _:
                    raise ValueError("Invalid prompt type specified. Must be 0 (one-shot user clue "
                                     "giver), 1 (one-shot assistant clue giver), 2 (one-shot user "
                                     "guesser), or 3 (one-shot assistant guesser).")

    def _default_system_prompt_cg(self) -> str:
        """
        Provides a default system prompt for the clue giver role in case the prompt template file is
        not found.

        :return: A default system prompt for the clue giver role.
        """
        return (
            "You are a strategic master clue giver in a game of Codenames Duet. Your goal is to "
            "connect as many 'Agent' words as possible with a single clue, while maintaining zero "
            "semantic similarity to the 'Assassin' words and very low similarity to the 'Civilian' "
            "words.\n\n"
            "### RULES ###\n"
            "1. The clue must be exactly ONE valid English word.\n"
            "2. The clue must not be any word currently visible on the board.\n"
            "3. The clue must not be a derivative, translation, or spelling variation of a word on "
            "the board.\n"
            "4. The clue must not be a substring or superstring of any board word (e.g., \"water\" "
            "is invalid if \"watermelon\" is on the board).\n"
            "5. The clue must not be a homophone of a word on the board (e.g., \"knight\" is "
            "invalid if \"night\" is present).\n"
            "6. The count must be a positive integer indicating exactly how many Agent words your "
            "clue targets.\n"
            "7. SAFETY FIRST: In Codenames Duet, hitting an Assassin word instantly loses the game."
            " Do not risk a clue that has even a tangential connection to an Assassin word.\n\n"
            "### INPUT FORMAT ###\n"
            "You will receive the board state as lists of words categorized as:\n"
            "- AGENTS: The words you want your partner to guess.\n"
            "- ASSASSINS: The deadly words you must absolutely avoid.\n"
            "- CIVILIANS: Neutral words you should try to avoid.\n\n"
            "### OUTPUT FORMAT ###\n"
            "You must respond ONLY with a valid JSON object. Do not include markdown formatting, "
            "conversational text, or any characters outside the JSON structure.\n\n"
            "{\n"
            "   \"reasoning\": \"Step 1: Identify semantic clusters among Agent words. Step "
            "2: Brainstorm candidate clues for the best clusters. Step 3: RUN THE ASSASSIN CHECK - "
            "strictly evaluate your top candidates against EVERY Assassin word to guarantee zero "
            "semantic proximity. Step 4: Evaluate against Civilian and Revealed words to minimize "
            "distraction. Step 5: Verify the final candidate violates no structural game rules "
            "(e.g., substrings, homophones).\",\n"
            "   \"clue\": \"your_single_word_clue\",\n"
            "   \"count\": x,\n"
            "   \"targets\": [\"exact_board_word\", \"...\"]\n"
            "}\n\n"
            "The \"targets\" field lists the exact board words your clue is meant for."
        )

    def _default_user_prompt_cg(self) -> str:
        """
        Provides a default user prompt for the clue giver role in case the prompt template file is
        not found.

        :return: A default user prompt for the clue giver role.
        """
        return (
            "### YOUR TASK ###\n"
            "Propose a clue and a count for the guessing player, and list the exact board words your "
            "clue is for. Remember the rules for valid clues and counts.\n\n"
            "Turn: {turn_number}\n\n"
            "### BOARD STATUS ###\n"
            "AGENTS (Words to connect):\n"
            "{agent_words}\n\n"
            "ASSASSINS (Terminal state - strictly avoid):\n"
            "{assassin_words}\n\n"
            "CIVILIANS (Neutral - try to avoid):\n"
            "{civilian_words}\n\n"
            "REVEALED WORDS (Already guessed, no longer valid targets):\n"
            "{revealed_words}"
        )

    def _default_system_prompt_gg(self) -> str:
        """
        Provides a default system prompt for the guessing role in case the prompt template file is
        not found.

        :return: A default system prompt for the guessing role.
        """
        return (
            "You are the guessing player in Codenames Duet. Your task is to analyze a given clue "
            "and number (count), and select the words from the board that best match the clue.\n\n"
            "### OBJECTIVE ###\n"
            "- Propose the optimal sequence of guesses for the current turn.\n"
            "- Maximize correct Agent guesses while strictly managing the risk of hitting an "
            "Assassin or Civilian.\n\n"
            "### GAME RULES & CONSTRAINTS ###\n"
            "1. You only see the unrevealed words on the board. You DO NOT know which are Agents, "
            "Civilians, or Assassins.\n"
            "2. UNLIMITED GUESSES: In Codenames Duet, there is no limit to the number of guesses "
            "you can make in a single turn.\n"
            "3. The \"count\" provided with the clue is a hint about how many words the clue-giver "
            "intended to connect. It is a target, not a hard limit.\n"
            "4. You may stop guessing early (proposing fewer than count) if the semantic ambiguity "
            "of the remaining options is too high.\n"
            "5. You may guess MORE than the count if you have high confidence in words from "
            "previous turns' clues.\n"
            "6. DO NOT invent words; you must select exactly from the provided board words.\n\n"
            "### DECISION POLICY (OPTIMAL STOPPING) ###\n"
            "- Step 1: Compute the semantic relation between the current clue and EVERY word on "
            "the board.\n"
            "- Step 2: Rank the candidate words by confidence.\n"
            "- Step 3: Establish a strict confidence threshold. If the probability of a word being "
            "an Agent drops below this safety threshold, STOP immediately. In Codenames Duet, "
            "precision is infinitely more valuable than coverage.\n"
            "- Step 4: Evaluate previous unsolved clues. If a board word strongly matches a past "
            "clue and meets your confidence threshold, include it in your proposal sequence.\n\n"
            "### OUTPUT FORMAT ###\n"
            "You must respond ONLY with a valid JSON object. Do not include formatting wrappers "
            "like ```json. Start your response immediately with the { character.\n\n"
            "{\n"
            "   \"reasoning\": \"Analyze the semantic links between the clue and board words. Rank "
            "the top candidates. Define your safety threshold and explicitly state why the risk of "
            "ambiguity outweighs the reward for any words below it.\",\n"
            "   \"stop_reason\": \"Explain the exact logic used to terminate the guess sequence "
            "(e.g., reached the target count, semantic distance too high, etc.).\",\n"
            "   \"proposals\": [\n"
            "       {\"word\": \"exact_board_word\", \"confidence\": 0.95},\n"
            "       {\"word\": \"another_word\", \"confidence\": 0.88}\n"
            "   ]\n"
            "}"
        )

    def _default_user_prompt_gg(self) -> str:
        """
        Provides a default user prompt for the guessing role in case the prompt template file is
        not found.

        :return: A default user prompt for the guessing role.
        """
        return (
            "### YOUR TASK ###\n"
            "Propose your optimal sequence of guesses. \n"
            "Remember: You may stop early if the risk is high, or guess MORE than the target count "
            "if you find strong matches for previous clues.\n"
            "Follow the required JSON format exactly.\n\n"
            "Turn: {turn_number}\n\n"
            "### CURRENT CLUE ###\n"
            "- Clue: {clue}\n"
            "- Target Count: {count}\n\n"
            "### PREVIOUS CLUES (Optional context for backtracking) ###\n"
            "{previous_clues_history}\n\n"
            "### UNREVEALED BOARD WORDS ###\n"
            "{words_remaining}"
        )

    def _default_os_user_cg(self) -> str:
        """
        Provides a default one-shot example for the user in the clue giver role in case the one-shot
        example file is not found.
        """
        return (
            "Turn: 1\n\n"
            "### BOARD STATUS ###\n"
            "AGENTS (Words to connect):\n"
            "- CROWN\n"
            "- THRONE\n"
            "- CASTLE\n\n"
            "ASSASSINS (Terminal state - strictly avoid):\n"
            "- BRIDGE\n\n"
            "CIVILIANS (Neutral - try to avoid):\n"
            "- TOWER\n"
            "- KNIGHT\n"
            "REVEALED WORDS (Already guessed, no longer valid targets):\n"
            "No words revealed yet.\n\n"
            "### YOUR TASK ###\n"
            "Propose a clue and a count for the guessing player, and list the exact board words your "
            "clue is for. Remember the rules for valid clues and counts."
        )

    def _default_os_assistant_cg(self) -> str:
        """
        Provides a default one-shot example for the assistant in the clue giver role in case the 
        one-shot example file is not found.

        This example should demonstrate the reasoning process and the final clue and count output in 
        the correct JSON format.
        """
        return (
            "{\"reasoning\": \"Step 1: CROWN, THRONE, CASTLE all relate to monarchy and royalty. "
            "TOWER and KNIGHT are civilians with medieval connotations — a clue like 'medieval' or "
            "'fortress' risks pulling them in. Step 2: Best candidate cluster is CROWN + THRONE + "
            "CASTLE → clue 'royal'. Step 3: ASSASSIN CHECK — BRIDGE has no meaningful semantic link "
            "to 'royal'. Safe to proceed. Step 4: TOWER could weakly associate with 'royal' (Tower "
            "of London), so I reduce count to 2 targeting CROWN and THRONE, the strongest matches, "
            "to avoid the civilian TOWER. Step 5: 'royal' is not a board word, not a derivative of "
            "any board word, and not a homophone. Valid.\", \"clue\": \"royal\", \"count\": 2, "
            "\"targets\": [\"CROWN\", \"THRONE\"]}"
        )

    def _default_os_user_gg(self) -> str:
        """
        Provides a default one-shot example for the user in the guessing role in case the one-shot
        example file is not found.
        """
        return (
            "Turn: 2\n\n"
            "### CURRENT CLUE ###\n"
            "- Clue: cold\n"
            "- Target Count: 2\n\n"
            "### PREVIOUS CLUES (Optional context for backtracking) ###\n"
            "- Turn: 1, Clue: ocean, Count: 2\n\n"
            "### UNREVEALED BOARD WORDS ###\n"
            "- ICE\n"
            "- FIRE\n"
            "- WIND\n"
            "- WAVE\n"
            "- DESERT\n"
            "- FROST\n"
            "- SHIP\n"
            "### YOUR TASK ###\n"
            "Propose your optimal sequence of guesses.\n"
            "Remember: You may stop early if the risk is high, or guess MORE than the target count "
            "if you find strong matches for previous clues.\n"
            "Follow the required JSON format exactly."
        )

    def _default_os_assistant_gg(self) -> str:
        """
        Provides a default one-shot example for the assistant in the guessing role in case the 
        one-shot example file is not found.

        This example should demonstrate the reasoning process for selecting guesses based on the
        current clue and previous clues, the confidence scores for each proposed guess, and the stop
        reason, all formatted in the correct JSON structure.
        """
        return (
            "{\"reasoning\": \"Current clue is 'cold' (count 2). Ranking board words by semantic "
            "proximity to 'cold': ICE (0.97) — direct synonym, highest confidence. FROST (0.93) — "
            "near-synonym for cold/frozen, very strong match. WIND (0.55) — cold wind is common but "
            "'wind' alone is ambiguous, below my threshold. FIRE (0.02) — antonym, ignore. DESERT "
            "(0.10) — can be cold but strong association with heat, too risky. WAVE and SHIP have "
            "no meaningful connection to 'cold'. Previous clue was 'ocean' (count 2) — WAVE (0.91) "
            "and SHIP (0.72) are strong matches and still unrevealed. WAVE exceeds my threshold so "
            "I add it as a third guess for the previous clue. SHIP is below threshold (0.72 < 0.80), "
            "stopping there.\", \"stop_reason\": \"Proposed ICE and FROST for current clue 'cold' "
            "(both above 0.90 threshold). Added WAVE as backtrack guess for previous clue 'ocean'. "
            "Stopped before SHIP as 0.72 confidence is below the 0.80 safety threshold.\", "
            "\"proposals\": [{\"word\": \"ICE\", \"confidence\": 0.97}, {\"word\": \"FROST\", "
            "\"confidence\": 0.93}, {\"word\": \"WAVE\", \"confidence\": 0.91}]}"
        )

    def _default_system_prompt_sd_gg(self) -> str:
        return (
            "You are the guessing player in Codenames Duet. The game has entered SUDDEN DEATH: the "
            "timer tokens have run out. There are no more clues - you must now identify all your "
            "remaining agent cards directly from the board.\n\n"
            "### RULES ###\n"
            "1. You must identify every word on the board that is one of YOUR remaining agent cards.\n"
            "2. Guessing a Civilian or Assassin card causes an IMMEDIATE LOSS. Be precise.\n"
            "3. Use the full clue history to recall which words your partner was hinting at.\n"
            "4. DO NOT invent words; select only from the provided unrevealed board words.\n\n"
            "### OUTPUT FORMAT ###\n"
            "Respond ONLY with a valid JSON object. No markdown wrappers.\n\n"
            "{\n"
            "   \"reasoning\": \"Recall which unrevealed words your partner's clues pointed to. "
            "For each candidate, state your confidence that it is one of your agents.\",\n"
            "   \"stop_reason\": \"Explain why you stopped proposing guesses.\",\n"
            "   \"proposals\": [\n"
            "       {\"word\": \"exact_board_word\", \"confidence\": 0.95}\n"
            "   ]\n"
            "}"
        )

    def _default_user_prompt_sd_gg(self) -> str:
        return (
            "### YOUR TASK ###\n"
            "You have {agents_remaining} agent(s) left to find. \n"
            "Identify them from the unrevealed words using the clue history as your guide.\n"
            "Follow the required JSON format exactly.\n\n"
            "### CLUE HISTORY (all clues given to you during the game) ###\n"
            "{clue_history}\n\n"
            "### UNREVEALED BOARD WORDS ###\n"
            "{words_remaining}\n\n"
        )

    def _default_system_prompt_meas_gg(self) -> str:
        """
        Default system prompt for the standard confidence-ranking measurement call, used when the
        template file is missing. Purely mechanical: it instructs the model to score every unrevealed
        card and never mentions gender, associations, stereotypes, or bias (construct non-reactivity).
        """
        return (
            "You are scoring a clue against a board in Codenames Duet. You are given a single clue "
            "(a word and a count) and the full list of unrevealed board words. Your task is purely "
            "mechanical.\n\n"
            "### TASK ###\n"
            "For EVERY unrevealed board word, assign a confidence between 0.0 and 1.0 that the word "
            "is one of the words the clue points to. 0.0 means the clue clearly does not point to "
            "it; 1.0 means the clue clearly points to it.\n\n"
            "### RULES ###\n"
            "1. Score EVERY unrevealed word. Do not omit any word, and do not invent words that are "
            "not listed.\n"
            "2. Confidence must be a number in the range [0.0, 1.0].\n"
            "3. Score each word independently on its own merits; the count is only context.\n\n"
            "### OUTPUT FORMAT ###\n"
            "Respond ONLY with a valid JSON object. Do not include markdown wrappers like ```json. "
            "Start your response immediately with the { character.\n\n"
            "{\n"
            "   \"reasoning\": \"Briefly explain how each unrevealed word relates to the clue.\",\n"
            "   \"rankings\": [\n"
            "       {\"word\": \"exact_board_word\", \"confidence\": 0.95},\n"
            "       {\"word\": \"another_word\", \"confidence\": 0.10}\n"
            "   ]\n"
            "}"
        )

    def _default_user_prompt_meas_gg(self) -> str:
        """Default user prompt for the standard confidence-ranking measurement call."""
        return (
            "### YOUR TASK ###\n"
            "For every unrevealed board word listed above, assign a confidence in [0.0, 1.0] that "
            "the current clue points to it. Score every word. Follow the required JSON format "
            "exactly.\n\n"
            "### CURRENT CLUE ###\n"
            "- Clue: {clue}\n"
            "- Target Count: {count}\n\n"
            "### PREVIOUS CLUES (context only) ###\n"
            "{previous_clues_history}\n\n"
            "### UNREVEALED BOARD WORDS ###\n"
            "{words_remaining}\n\n"
        )

    def _default_system_prompt_meas_sd(self) -> str:
        """
        Default system prompt for the sudden-death confidence-ranking measurement call. Purely
        mechanical; never mentions gender, associations, stereotypes, or bias.
        """
        return (
            "You are scoring a board in Codenames Duet during SUDDEN DEATH. There are no more clues; "
            "you are given the full clue history, how many of your agent cards remain, and the list "
            "of unrevealed board words. Your task is purely mechanical.\n\n"
            "### TASK ###\n"
            "For EVERY unrevealed board word, assign a confidence between 0.0 and 1.0 that the word "
            "is one of YOUR remaining agent cards. 0.0 means it clearly is not one of your agents; "
            "1.0 means it clearly is.\n\n"
            "### RULES ###\n"
            "1. Score EVERY unrevealed word. Do not omit any word, and do not invent words that are "
            "not listed.\n"
            "2. Confidence must be a number in the range [0.0, 1.0].\n"
            "3. Use the clue history to recall which words your partner was pointing at.\n\n"
            "### OUTPUT FORMAT ###\n"
            "Respond ONLY with a valid JSON object. Do not include markdown wrappers like ```json. "
            "Start your response immediately with the { character.\n\n"
            "{\n"
            "   \"reasoning\": \"Briefly explain how each unrevealed word relates to the clue "
            "history and your remaining agents.\",\n"
            "   \"rankings\": [\n"
            "       {\"word\": \"exact_board_word\", \"confidence\": 0.95},\n"
            "       {\"word\": \"another_word\", \"confidence\": 0.10}\n"
            "   ]\n"
            "}"
        )

    def _default_user_prompt_meas_sd(self) -> str:
        """Default user prompt for the sudden-death confidence-ranking measurement call."""
        return (
            "### YOUR TASK ###\n"
            "You have {agents_remaining} agent(s) left to find. For every unrevealed board word "
            "listed above, assign a confidence in [0.0, 1.0] that it is one of your remaining agent "
            "cards. Score every word. Follow the required JSON format exactly.\n\n"
            "### CLUE HISTORY (all clues given to you during the game) ###\n"
            "{clue_history}\n\n"
            "### UNREVEALED BOARD WORDS ###\n"
            "{words_remaining}\n\n"
        )
