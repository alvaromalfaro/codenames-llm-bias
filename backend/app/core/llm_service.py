import json
from backend.app.core.llm_client import LLMClient
from backend.app.models.llm_schemas import ClueProposal, GuessProposal, LLMRequest, LLMResponse, LLMMessage
from backend.app.models.game_schemas import GameState, GamePhase, CardRole


class CodenamesLLMService:
    SYSTEM_TEMP_CG_PATH = "data/prompt_templates/SYSTEM_TEMPLATE_CLUE_GIVER.txt"
    USER_TEMP_CG_PATH = "data/prompt_templates/USER_TEMPLATE_CLUE_GIVER.txt"
    SYSTEM_TEMP_GG_PATH = "data/prompt_templates/SYSTEM_TEMPLATE_GUESSER.txt"
    USER_TEMP_GG_PATH = "data/prompt_templates/USER_TEMPLATE_GUESSER.txt"

    def __init__(self, llm_client: LLMClient, default_model: str = "local", temperature: float = 0.7,
                 max_tokens: int = 1000, timeout_s: int = 30):
        self.llm_client = llm_client
        self.default_model = default_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s
        # Load prompt templates for both clue giver and guesser roles, with fallbacks to default
        # prompts if the template files are not found.
        self._system_prompt_cg = self._load_prompt_template(
            self.SYSTEM_TEMP_CG_PATH, 0)
        self._user_prompt_cg = self._load_prompt_template(
            self.USER_TEMP_CG_PATH, 1)
        self._system_prompt_gg = self._load_prompt_template(
            self.SYSTEM_TEMP_GG_PATH, 2)
        self._user_prompt_gg = self._load_prompt_template(
            self.USER_TEMP_GG_PATH, 3)

    async def propose_clue(self, game_state: GameState, player_id: int = 0) -> ClueProposal:
        """
        Proposes a clue for the current game state. This method checks that the game is in the 
        correct phase and that the LLM is the clue giver before building the request, sending it to
        the LLM, and processing the response.

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

        # Build the LLM request for proposing a clue
        request = self._build_clue_request(game_state, player_id)

        # Send the request to the LLM client and get the response
        response = await self.llm_client.generate(request)

        # Process the response and convert it into a ClueProposal
        clue_proposal = self._build_clue_proposal(response)

        return clue_proposal

    async def propose_guess(self, game_state: GameState, player_id: int = 0) -> GuessProposal:
        """
        Proposes guesses for the current game state. This method checks that the game is in the 
        correct phase and that the LLM is the guesser before building the request, sending it to
        the LLM, and processing the response.

        :param game_state: The current state of the game.
        :param player_id: The ID of the player proposing the guesses (0 for LLM).

        :return: An instance of GuessProposal containing the proposed guesses from the LLM.
        """
        if game_state.current_phase != GamePhase.GUESSING:
            raise ValueError(
                "Cannot propose a guess when the game is not in the GUESSING phase.")

        if game_state.guesser != player_id:
            raise ValueError(
                "The player must be the guessing player to propose a guess.")

        if game_state.clue_history[-1].turn_number != game_state.turn_number:
            raise ValueError(
                "Cannot propose a guess without an active clue. No clue has been proposed for this "
                "turn."
            )

        # Build the LLM request for proposing a guess
        request = self._build_guess_request(game_state, player_id)

        # Send the request to the LLM client and get the response.
        response = await self.llm_client.generate(request)

        # Process the response and convert it into a GuessProposal.
        guess_proposal = self._build_guess_proposal(response)

        return guess_proposal

    def _build_clue_request(self, game_state: GameState, player_id: int) -> LLMRequest:
        """
        Builds an LLMRequest for proposing a clue based on the current game state. This method
        extracts relevant information from the game state, formats it into a user prompt, and
        constructs the list of messages for the LLM request.

        :param game_state: The current state of the game.
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
            danger_words = self._get_llm_perspective_danger_words(game_state)
        else:
            # Same as above, but from the human player's perspective
            agent_words = self._get_human_perspective_agent_words(game_state)
            danger_words = self._get_human_perspective_danger_words(game_state)

        rev_words = [
            card.text for card in game_state.board.cards if card.revealed]

        # Format the user prompt with the current game state information
        user_prompt = self._user_prompt_cg.format(
            turn_number=turn_number,
            agent_words=", ".join(agent_words),
            danger_words=", ".join(danger_words),
            revealed_words=", ".join(rev_words)
        )

        # Build the list of messages for the LLM request
        messages = [
            LLMMessage(role="system", content=self._system_prompt_cg),
            LLMMessage(role="user", content=user_prompt)
        ]

        return LLMRequest(messages=messages, model=self.default_model, temperature=self.temperature,
                          max_tokens=self.max_tokens, timeout_s=self.timeout_s)

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
        except json.JSONDecodeError:
            raise ValueError(
                "LLM response is not valid JSON. Response content: " + response_content)

        # TODO: Implement validation logic for the clue and count based on the game rules.
        # For now, we will assume the clue and count are valid if they are present and of the
        # correct types
        if not isinstance(clue, str) or not clue.strip():
            raise ValueError("Clue must be a non-empty string.")
        if not isinstance(count, int) or count <= 0:
            raise ValueError("Count must be a positive integer.")

        return ClueProposal(clue=clue.strip(), count=count, reasoning=reasoning.strip(),
                            raw_payload=response.raw_payload)

    def _build_guess_request(self, game_state: GameState, player_id: int) -> LLMRequest:
        """
        Builds an LLMRequest for proposing guesses based on the current game state. This method 
        extracts relevant information from the game state, formats it into a user prompt, and 
        constructs the list of messages for the LLM request.

        :param game_state: The current state of the game.
        :param player_id: The ID of the player proposing the guess (0 for LLM).

        :return: An instance of LLMRequest containing the messages and parameters for the LLM
            generation.
        """
        # Extract relevant information from the game state
        turn_number = game_state.turn_number
        clue = game_state.current_clue.clue
        count = game_state.current_clue.count
        words_remaining = [card.text for card in game_state.board.cards if not card.revealed and
                           player_id not in card.time_marker_by]

        # Format the user prompt with the game state information
        user_prompt = self._user_prompt_gg.format(
            turn_number=turn_number,
            clue=clue,
            count=count,
            words_remaining=", ".join(words_remaining)
        )

        # Build the list of messages for the LLM request
        messages = [
            LLMMessage(role="system", content=self._system_prompt_gg),
            LLMMessage(role="user", content=user_prompt)
        ]

        return LLMRequest(messages=messages, model=self.default_model, temperature=self.temperature,
                          max_tokens=self.max_tokens, timeout_s=self.timeout_s)

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

        return GuessProposal(
            proposals=proposals, confidence=confidence, reasoning=reasoning.strip(),
            stop_reason=stop_reason.strip(), raw_payload=response.raw_payload
        )

    def _get_llm_perspective_agent_words(self, game_state: GameState) -> list[str]:
        """
        Extracts the non-revealed agent words from the game state based on the LLM's perspective.

        :param game_state: The current state of the game.

        :return: A list of agent words that are not revealed from the LLM's perspective.
        """
        return [card.text for card in game_state.board.cards if card.llm_perspective_role ==
                CardRole.AGENT and not card.revealed]

    def _get_llm_perspective_danger_words(self, game_state: GameState) -> list[str]:
        """
        Extracts the dangerous words (assassins and civilians) from the game state based on the LLM's
        perspective. Words that are marked with a time marker are ignored to keep the final prompt
        concise.

        :param game_state: The current state of the game.

        :return: A list of dangerous words that are not marked with a time marker from the LLM's
            perspective.
        """
        return [card.text for card in game_state.board.cards if card.llm_perspective_role !=
                CardRole.AGENT and 1 not in card.time_marker_by]

    def _get_human_perspective_agent_words(self, game_state: GameState) -> list[str]:
        """
        Extracts the agent words from the game state based on the human's perspective. Words that are
        marked with a time marker are ignored to keep the final prompt concise.

        :param game_state: The current state of the game.

        :return: A list of agent words that are not revealed from the human's perspective.
        """
        return [card.text for card in game_state.board.cards if card.human_perspective_role ==
                CardRole.AGENT and not card.revealed]

    def _get_human_perspective_danger_words(self, game_state: GameState) -> list[str]:
        """
        Extract the dangerous words (assassins and civilians) from the game state based on the 
        human's perspective. Words that are marked with a time marker are ignored to keep the final
        prompt concise.

        :param game_state: The current state of the game.

        :return: A list of dangerous words that are not marked with a time marker from the human's
            perspective.
        """
        return [card.text for card in game_state.board.cards if card.human_perspective_role !=
                CardRole.AGENT and 0 not in card.time_marker_by]

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
                case _:
                    raise ValueError(
                        "Invalid prompt type specified. Must be 0 (system clue giver),"
                        "1 (user clue giver), 2 (system guesser), or 3 (user guesser).")

    def _default_system_prompt_cg(self) -> str:
        """
        Provides a default system prompt for the clue giver role in case the prompt template file is
        not found.

        :return: A default system prompt for the clue giver role.
        """
        return (
            "You are a strategic master clue giver in a game of Codenames Duet. Your goal "
            "is to connect as many agent words as possible with a single clue while "
            "maintaining very low similarity to the assassin words and civilian words. "
            "You have to propose a clue and a count for the guessing player.\n\n"
            "### RULES ###\n"
            "- The clue must be a single word.\n"
            "- The clue must not be any word on the board.\n"
            "- The clue must not be a derivative of a word on the board.\n"
            "- The clue must not be contained in any of the words on the board.\n"
            "- The count must be a positive integer indicating how many words on the board are "
            "associated with the clue.\n"
            "- Try to connect as many agent words as possible.\n"
            "- Avoid clues strongly associated with dangerous words.\n\n"
            "### OUTPUT FORMAT ###\n"
            "Provide the clue and count in a JSON format like {\"clue\": \"example_clue\", "
            "\"count\": x, \"reasoning\": \"explanation of your reasoning for the clue and "
            "count\"}, where \"example_clue\" is the proposed clue, x is the number of agent "
            "words you are trying to connect with \"example_clue\", and the reasoning field "
            "contains your explanation for why you chose that clue and count."
        )

    def _default_user_prompt_cg(self) -> str:
        """
        Provides a default user prompt for the clue giver role in case the prompt template file is
        not found.

        :return: A default user prompt for the clue giver role.
        """
        return (
            "You are generating a clue for the current Codenames Duet turn. The game is "
            "in the turn {turn_number}.\n\n"
            "### BOARD STATUS ###\n"
            "The agent words are: {agent_words}\n\n"
            "The dangerous words to avoid (assassins and civilians) are: {danger_words}\n\n"
            "Already revealed words: {revealed_words}\n\n"
            "### YOUR TASK ###\n"
            "Propose a clue and a count for the guessing player. Remember the rules for valid "
            "clues and counts."
        )

    def _default_system_prompt_gg(self) -> str:
        """
        Provides a default system prompt for the guessing role in case the prompt template file is
        not found.

        :return: A default system prompt for the guessing role.
        """
        return (
            "You are the guessing player in Codenames Duet. Your task is to propose all guesses for "
            "the current turn in a single response, based on the received clue, count and words on "
            "the board.\n\n"
            "### OBJECTIVE ###\n"
            "- Propose the best guesses for the current turn.\n"
            "- Balance semantic relevance and ambiguity risk.\n\n"
            "### RULES ###\n"
            "- Do not invent words that are not on the board.\n"
            "- Return at least 1 proposal.\n"
            "- Return at most count proposals.\n"
            "- If ambiguity is high, return fewer than count proposals.\n"
            "- Do not output markdown or extra text outside the required JSON.\n"
            "\n"
            "### DECISION POLICY ###\n"
            "- Use semantic relation between clue and candidate words.\n"
            "- Prefer precision over coverage when uncertain.\n"
            "- If two options are similar, prefer the less ambiguous one.\n"
            "- Confidence must reflect relative certainty for each proposed word.\n\n"
            "### OUTPUT FORMAT ###\n"
            "Return ONLY valid JSON with this exact schema:\n"
            "{\n"
            "   \"proposals\": [\n"
            "       {\"word\": \"apple\", \"confidence\": 0.82},\n"
            "       {\"word\": \"tree\", \"confidence\": 0.67}\n"
            "   ],"
            "\"reasoning\": \"brief explanation of why these words fit the clue\",\n"
            "\"stop_reason\": \"why you propose this number of guesses\""
            "}\n\n"
            "Field requirements:\n"
            "- 'proposals': array with length 1..count\n"
            "- each proposals[i].word: unique board word.\n"
            "- each proposals[i].confidence: number between 0 and 1.\n"
            "- 'reasoning': concise turn-specific rationale.\n"
            "- 'stop_reason': concise reason for proposing exactly that many guesses."
        )

    def _default_user_prompt_gg(self) -> str:
        """
        Provides a default user prompt for the guessing role in case the prompt template file is
        not found.

        :return: A default user prompt for the guessing role.
        """
        return (
            "You are guessing in Codenames Duet turn {turn_number}.\n\n"
            "### CLUE RECEIVED ###\n"
            "- clue: {clue}\n"
            "- count: {count}\n\n"
            "### BOARD STATUS ###\n"
            "Choose from the following words:\n"
            "{words_remaining}\n\n"
            "### YOUR TASK ###\n"
            "Propose all guesses you want to make this turn in one response.\n"
            "Return a list of up to count proposals (you may return fewer if risk is high).\n"
            "Follow the required JSON format exactly."
        )
