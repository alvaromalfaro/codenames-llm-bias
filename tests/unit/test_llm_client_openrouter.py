import json
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from backend.app.core.llm.client_openrouter import LLMClientOpenRouter
from backend.app.models.llm_schemas import LLMResponse, ClueJSONFormat


def _mock_openai_response() -> MagicMock:
    """Builds a MagicMock mimicking a successful OpenAI/OpenRouter chat completion response."""
    content = json.dumps({"reasoning": "test reasoning", "clue": "battle", "count": 3})
    message = MagicMock()
    message.content = content
    choice = MagicMock()
    choice.message = message
    choice.finish_reason = "stop"

    response = MagicMock()
    response.choices = [choice]
    response.model = "openrouter/some-model"
    response.id = "req-123"
    response.system_fingerprint = "fp_abc123"
    response.usage.prompt_tokens = 10
    response.usage.completion_tokens = 20
    response.usage.total_tokens = 30
    response.model_dump.return_value = {"id": "req-123"}
    return response


@pytest.mark.asyncio
async def test_llm_client_openrouter_forwards_seed(llm_request_cg):
    """
    Tests that the OpenRouter client forwards `seed` to chat.completions.create().
    """
    request = llm_request_cg.model_copy(update={"seed": 321})

    with patch("backend.app.core.llm.client_openrouter.AsyncOpenAI") as MockOpenAI:
        mock_create = AsyncMock(return_value=_mock_openai_response())
        MockOpenAI.return_value.chat.completions.create = mock_create
        client = LLMClientOpenRouter("openrouter/some-model")

        await client.generate(request, expected_format=ClueJSONFormat)

        _, kwargs = mock_create.call_args
        assert kwargs["seed"] == 321
        assert kwargs["temperature"] == request.temperature


@pytest.mark.asyncio
async def test_llm_client_openrouter_populates_telemetry(llm_request_cg):
    """
    Tests that the OpenRouter client populates the sampling telemetry on LLMResponse:
    requested_temperature/requested_seed from the request, and system_fingerprint/resolved_model
    from the provider response.
    """
    request = llm_request_cg.model_copy(update={"seed": 7, "temperature": 0.5})

    with patch("backend.app.core.llm.client_openrouter.AsyncOpenAI") as MockOpenAI:
        mock_create = AsyncMock(return_value=_mock_openai_response())
        MockOpenAI.return_value.chat.completions.create = mock_create
        client = LLMClientOpenRouter("openrouter/some-model")

        result = await client.generate(request, expected_format=ClueJSONFormat)

        assert isinstance(result, LLMResponse)
        assert result.requested_temperature == 0.5
        assert result.requested_seed == 7
        assert result.system_fingerprint == "fp_abc123"
        assert result.resolved_model == "openrouter/some-model"


@pytest.mark.asyncio
async def test_llm_client_openrouter_missing_system_fingerprint(llm_request_cg):
    """
    Tests that a missing system_fingerprint on the provider response is guarded and stored as None.
    """
    response = _mock_openai_response()
    del response.system_fingerprint  # provider did not return a fingerprint

    with patch("backend.app.core.llm.client_openrouter.AsyncOpenAI") as MockOpenAI:
        mock_create = AsyncMock(return_value=response)
        MockOpenAI.return_value.chat.completions.create = mock_create
        client = LLMClientOpenRouter("openrouter/some-model")

        result = await client.generate(llm_request_cg, expected_format=ClueJSONFormat)

        assert result.system_fingerprint is None
