from llm_client import LLMClient
from backend.app.models.llm_schemas import ClueProposal, LLMRequest
from backend.app.models.game_schemas import GameState, GamePhase


class CodenamesLLMService:
    SYSTEM_TEMP_CG = ("You are a strategic master clue giver in a game of Codenames Duet. "
                      "Your goal is to connect as many agent words as possible with a single "
                      "clue while maintaining very low similarity to the assassin words and "
                      "civilian words. You have to propose a clue and a count for the guessing "
                      "player. The clue must be a single word that is not on the board, "
                      "a derivative of a word on the board, or be contained in any of the words "
                      "on the board. The count must be a positive integer indicating how many "
                      "words on the board are associated with the clue. Provide the clue and count "
                      "in a JSON format like {\"clue\": \"example_clue\", \"count\": 2, "
                      "\"reasoning\": \"explanation of your reasoning for the clue and count\"}. "
                      "Do not include any additional text or explanation in your response.")

    def __init__(self, llm_client: LLMClient, default_model: str = "local", temperature: float = 0.7,
                 max_tokens: int = 1000, timeout_s: int = 30):
        self.llm_client = llm_client
        self.default_model = default_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s

    async def propose_clue(self, game_state: GameState, player_id: int = 0) -> ClueProposal:
        """
        Proposes a clue for the current game state. This method checks that the game is in the 
        correct phase and that the LLM is the clue giver before building the request, sending it to
        the LLM, and processing the response.

        :param game_state: The current state of the game, which includes information about the 
            board, current phase, clue giver, and other relevant details needed to generate a clue.
        :param player_id: The ID of the player proposing the clue (0 for LLM).

        :return: An instance of ClueProposal containing the proposed clue and count from the LLM.
        """
        if game_state.current_phase != GamePhase.GIVING_CLUE:
            raise ValueError(
                "Cannot propose a clue when the game is not in the GIVING_CLUE phase.")

        if game_state.clue_giver != player_id:
            raise ValueError(
                "The player must be the clue giver to propose a clue.")

        # Build the LLM request
        request = self._build_clue_request(game_state, player_id)

        # Send the request to the LLM client and get the response
        response = await self.llm_client.generate(request)

        # Process the response and convert it into a ClueProposal
        clue_proposal = self._build_clue_response(response)

        return clue_proposal

    def _build_clue_request(self, game_state: GameState, player_id: int) -> LLMRequest:
        pass

    def _build_clue_response(self, response) -> ClueProposal:
        pass
