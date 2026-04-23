import json
from backend.app.core.llm.client import LLMClient
from backend.app.models.llm_schemas import ClueProposal, GuessProposal, LLMRequest, LLMResponse, LLMMessage, ClueJSONFormat, GuessJSONFormat
from backend.app.models.game_schemas import GameState, GamePhase, CardRole


class LLMService:
    SYSTEM_TEMP_CG_PATH = "data/prompt_templates/SYSTEM_TEMPLATE_CLUE_GIVER.txt"
    USER_TEMP_CG_PATH = "data/prompt_templates/USER_TEMPLATE_CLUE_GIVER.txt"
    SYSTEM_TEMP_GG_PATH = "data/prompt_templates/SYSTEM_TEMPLATE_GUESSER.txt"
    USER_TEMP_GG_PATH = "data/prompt_templates/USER_TEMPLATE_GUESSER.txt"
    ONE_SHOT_USER_CG_PATH = "data/prompt_templates/ONE_SHOT_USER_CLUE_GIVER.txt"
    ONE_SHOT_ASSISTANT_CG_PATH = "data/prompt_templates/ONE_SHOT_ASSISTANT_CLUE_GIVER.txt"
    ONE_SHOT_USER_GG_PATH = "data/prompt_templates/ONE_SHOT_USER_GUESSER.txt"
    ONE_SHOT_ASSISTANT_GG_PATH = "data/prompt_templates/ONE_SHOT_ASSISTANT_GUESSER.txt"

    def __init__(self, temperature: float = 0.7, max_tokens: int = 1000, timeout_s: int = 30):
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
        self._one_shot_user_cg = self._load_one_shot(
            self.ONE_SHOT_USER_CG_PATH, 0)
        self._one_shot_assistant_cg = self._load_one_shot(
            self.ONE_SHOT_ASSISTANT_CG_PATH, 1)
        self._one_shot_user_gg = self._load_one_shot(
            self.ONE_SHOT_USER_GG_PATH, 2)
        self._one_shot_assistant_gg = self._load_one_shot(
            self.ONE_SHOT_ASSISTANT_GG_PATH, 3)

    async def propose_clue(self, llm_client: LLMClient, game_state: GameState, player_id: int = 0) -> ClueProposal:
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

        # Build the LLM request for proposing a clue
        request = self._build_clue_request(
            game_state, llm_client.local_model, player_id)

        # Send the request to the LLM client and get the response
        response = await llm_client.generate(request, expected_format=ClueJSONFormat)

        # Process the response and convert it into a ClueProposal
        clue_proposal = self._build_clue_proposal(response)

        return clue_proposal

    async def propose_guess(self, llm_client: LLMClient, game_state: GameState, player_id: int = 0) -> GuessProposal:
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
            game_state, llm_client.local_model, player_id)

        # Send the request to the LLM client and get the response.
        response = await llm_client.generate(request, expected_format=GuessJSONFormat)

        # Process the response and convert it into a GuessProposal.
        guess_proposal = self._build_guess_proposal(response)

        return guess_proposal

    def _build_clue_request(self, game_state: GameState, model: str, player_id: int) -> LLMRequest:
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

    def _build_guess_request(self, game_state: GameState, model: str, player_id: int) -> LLMRequest:
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
            f"- {card.text}" for card in game_state.board.cards if 0 not in card.revealed_by and
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
                case _:
                    raise ValueError(
                        "Invalid prompt type specified. Must be 0 (system clue giver),"
                        "1 (user clue giver), 2 (system guesser), or 3 (user guesser).")

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
            "You must respond ONLY with a valid JSON object. Do not include markdown formatting "
            "(like ```json), conversational text, or any characters outside the JSON structure.\n\n"
            "{\n"
            "   \"reasoning\": \"Step 1: Identify semantic clusters among Agent words. Step "
            "2: Brainstorm candidate clues for the best clusters. Step 3: RUN THE ASSASSIN CHECK - "
            "strictly evaluate your top candidates against EVERY Assassin word to guarantee zero "
            "semantic proximity. Step 4: Evaluate against Civilian and Revealed words to minimize "
            "distraction. Step 5: Verify the final candidate violates no structural game rules "
            "(e.g., substrings, homophones).\n",
            "   \"clue\": \"your_single_word_clue\",\n"
            "   \"count\": x\n"
            "}"
        )

    def _default_user_prompt_cg(self) -> str:
        """
        Provides a default user prompt for the clue giver role in case the prompt template file is
        not found.

        :return: A default user prompt for the clue giver role.
        """
        return (
            "Turn: {turn_number}\n\n"
            "### BOARD STATUS ###\n"
            "AGENTS (Words to connect):\n"
            "{agent_words}\n\n"
            "ASSASSINS (Terminal state - strictly avoid):\n"
            "{assassin_words}\n\n"
            "CIVILIANS (Neutral - try to avoid):\n"
            "{civilian_words}\n\n"
            "REVEALED WORDS (Already guessed, no longer valid targets):\n"
            "{revealed_words}\n\n"
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
            "Turn: {turn_number}\n\n"
            "### CURRENT CLUE ###\n"
            "- Clue: {clue}\n"
            "- Target Count: {count}\n\n"
            "### PREVIOUS CLUES (Optional context for backtracking) ###\n"
            "{previous_clues_history}\n\n"
            "### UNREVEALED BOARD WORDS ###\n"
            "{words_remaining}\n\n"
            "### YOUR TASK ###\n"
            "Propose your optimal sequence of guesses.\n"
            "Remember: You may stop early if the risk is high, or guess MORE than the target count "
            "if you find strong matches for previous clues.\n"
            "Follow the required JSON format exactly.\n"
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
            "Propose a clue and a count for the guessing player. Remember the rules for valid clues "
            "and counts."
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
            "any board word, and not a homophone. Valid.\", \"clue\": \"royal\", \"count\": 2}"
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
