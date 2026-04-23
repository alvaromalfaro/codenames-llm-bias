from abc import ABC, abstractmethod
from backend.app.models.llm_schemas import LLMRequest, LLMResponse


class LLMClient(ABC):
    @abstractmethod
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """
        Abstract method to generate a response from the LLM based on the given request. This method
        must be implemented by any concrete subclass of LLMClient.

        :param request: An instance of LLMRequest containing the messages and parameters for the LLM
            generation.

        :return: An instance of LLMResponse containing the generated response from the LLM.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    async def health_check(self) -> bool:
        """
        Optional method to perform a health check on the LLM provider. This can be used to verify
        that the provider is available and functioning properly before attempting to generate a
        response.

        :return: A boolean indicating whether the LLM provider is healthy (True) or not (False).
        """
        # Default implementation assumes the provider is healthy.
        return True

    async def close(self) -> None:
        """
        Optional method to close any resources or connections used by the LLM client. This can be
        overridden by subclasses if they need to perform any cleanup when the client is no longer
        needed.
        """
        pass
