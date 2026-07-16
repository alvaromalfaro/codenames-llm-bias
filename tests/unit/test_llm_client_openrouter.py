import json
import httpx
import pytest
from openai import APITimeoutError
from unittest.mock import patch, AsyncMock, MagicMock
from backend.app.core.llm import client as client_module
from backend.app.core.llm.client_openrouter import LLMClientOpenRouter, _OPENROUTER_BASE_URL
from backend.app.models.llm_errors import LLMParseError, LLMTimeoutError
from backend.app.models.llm_schemas import LLMResponse, ClueJSONFormat


def _timeout_error() -> APITimeoutError:
    """A raw provider timeout the client maps to the retriable LLMTimeoutError."""
    return APITimeoutError(request=httpx.Request("POST", _OPENROUTER_BASE_URL))


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


# transient retry (5c.2)
@pytest.mark.asyncio
async def test_openrouter_retries_retriable_then_succeeds(llm_request_cg, monkeypatch):
    """A retriable provider error (timeout) twice, then success: generate() returns the response
    after exactly 3 attempts, and the request sent each time is IDENTICAL (same seed) - proving no
    reseed on retry."""
    monkeypatch.setattr(client_module, "_RETRY_BACKOFF_BASE_S", 0.0)  # no real sleeping
    request = llm_request_cg.model_copy(update={"seed": 321})

    with patch("backend.app.core.llm.client_openrouter.AsyncOpenAI") as MockOpenAI:
        mock_create = AsyncMock(
            side_effect=[_timeout_error(), _timeout_error(), _mock_openai_response()])
        MockOpenAI.return_value.chat.completions.create = mock_create
        client = LLMClientOpenRouter("openrouter/some-model", max_retries=2)

        result = await client.generate(request, expected_format=ClueJSONFormat)

        assert isinstance(result, LLMResponse)
        assert mock_create.call_count == 3
        seeds = [call.kwargs["seed"] for call in mock_create.call_args_list]
        assert seeds == [321, 321, 321]  # same request re-sent, never reseeded


@pytest.mark.asyncio
async def test_openrouter_non_retriable_raises_immediately(llm_request_cg, monkeypatch):
    """A non-retriable error (parse failure) raises on the first attempt even with a retry budget."""
    monkeypatch.setattr(client_module, "_RETRY_BACKOFF_BASE_S", 0.0)
    bad = _mock_openai_response()
    bad.choices[0].message.content = "not valid json for the schema"

    with patch("backend.app.core.llm.client_openrouter.AsyncOpenAI") as MockOpenAI:
        mock_create = AsyncMock(return_value=bad)
        MockOpenAI.return_value.chat.completions.create = mock_create
        client = LLMClientOpenRouter("openrouter/some-model", max_retries=3)

        with pytest.raises(LLMParseError):
            await client.generate(llm_request_cg, expected_format=ClueJSONFormat)
        assert mock_create.call_count == 1


@pytest.mark.asyncio
async def test_openrouter_default_no_retry(llm_request_cg, monkeypatch):
    """max_retries defaults to 0 (the interactive path): a retriable error raises on the first
    attempt, exactly one provider call - pre-5c.2 behavior."""
    monkeypatch.setattr(client_module, "_RETRY_BACKOFF_BASE_S", 0.0)

    with patch("backend.app.core.llm.client_openrouter.AsyncOpenAI") as MockOpenAI:
        mock_create = AsyncMock(side_effect=_timeout_error())
        MockOpenAI.return_value.chat.completions.create = mock_create
        client = LLMClientOpenRouter("openrouter/some-model")  # no max_retries

        with pytest.raises(LLMTimeoutError):
            await client.generate(llm_request_cg, expected_format=ClueJSONFormat)
        assert mock_create.call_count == 1


@pytest.mark.asyncio
async def test_openrouter_exhausts_retry_budget(llm_request_cg, monkeypatch):
    """A retriable error failing k+1 times raises the last error after exactly k+1 attempts."""
    monkeypatch.setattr(client_module, "_RETRY_BACKOFF_BASE_S", 0.0)
    k = 2

    with patch("backend.app.core.llm.client_openrouter.AsyncOpenAI") as MockOpenAI:
        mock_create = AsyncMock(side_effect=[_timeout_error() for _ in range(k + 1)])
        MockOpenAI.return_value.chat.completions.create = mock_create
        client = LLMClientOpenRouter("openrouter/some-model", max_retries=k)

        with pytest.raises(LLMTimeoutError):
            await client.generate(llm_request_cg, expected_format=ClueJSONFormat)
        assert mock_create.call_count == k + 1
