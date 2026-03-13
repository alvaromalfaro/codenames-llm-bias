from llm_client import LLMClient
from backend.app.models.llm_schemas import ClueProposal, LLMRequest
from backend.app.models.game_schemas import GameState, GamePhase


class CodenamesLLMService:
    def __init__(self, llm_client: LLMClient, default_model: str = "local", temperature: float = 0.7,
                 max_tokens: int = 1000, timeout_s: int = 30):
        self.llm_client = llm_client
        self.default_model = default_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s

    async def propose_clue(self, game_state: GameState, player_id: int) -> ClueProposal:
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
