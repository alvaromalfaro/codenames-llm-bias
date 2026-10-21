import json
import logging
import httpx
import pytest
from openai import APITimeoutError, AuthenticationError
from unittest.mock import patch, AsyncMock, MagicMock
from backend.app.core.llm import client as client_module
from backend.app.core.llm.client_openrouter import LLMClientOpenRouter, _OPENROUTER_BASE_URL
from backend.app.models.llm_errors import (
    LLMAuthError,
    LLMMalformedResponseError,
    LLMTimeoutError,
)
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
async def test_llm_client_openrouter_without_usage_records_unknown_counts(llm_request_cg):
    """A response without a usage block records the token counts as unknown (None), not as 0."""
    response = _mock_openai_response()
    response.usage = None

    with patch("backend.app.core.llm.client_openrouter.AsyncOpenAI") as MockOpenAI:
        MockOpenAI.return_value.chat.completions.create = AsyncMock(return_value=response)
        client = LLMClientOpenRouter("openrouter/some-model")

        result = await client.generate(llm_request_cg, expected_format=ClueJSONFormat)

    assert result.usage.prompt_tokens is None
    assert result.usage.completion_tokens is None
    assert result.usage.total_tokens is None


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
            side_effect=[_timeout_error(), _timeout_error(), _mock_openai_response()]
        )
        MockOpenAI.return_value.chat.completions.create = mock_create
        client = LLMClientOpenRouter("openrouter/some-model", max_retries=2)

        result = await client.generate(request, expected_format=ClueJSONFormat)

        assert isinstance(result, LLMResponse)
        assert mock_create.call_count == 3
        seeds = [call.kwargs["seed"] for call in mock_create.call_args_list]
        assert seeds == [321, 321, 321]  # same request re-sent, never reseeded


@pytest.mark.asyncio
async def test_openrouter_non_retriable_raises_immediately(llm_request_cg, monkeypatch):
    """A non-retriable error (a rejected key) raises on the first attempt even with a retry
    budget."""
    monkeypatch.setattr(client_module, "_RETRY_BACKOFF_BASE_S", 0.0)
    rejected = AuthenticationError(
        "invalid key",
        body=None,
        response=httpx.Response(401, request=httpx.Request("POST", _OPENROUTER_BASE_URL)),
    )

    with patch("backend.app.core.llm.client_openrouter.AsyncOpenAI") as MockOpenAI:
        mock_create = AsyncMock(side_effect=rejected)
        MockOpenAI.return_value.chat.completions.create = mock_create
        client = LLMClientOpenRouter("openrouter/some-model", max_retries=3)

        with pytest.raises(LLMAuthError):
            await client.generate(llm_request_cg, expected_format=ClueJSONFormat)
        assert mock_create.call_count == 1


@pytest.mark.asyncio
async def test_openrouter_draws_a_malformed_answer_again_and_counts_it(llm_request_cg):
    """An answer that does not match the schema is a bad draw: drawn again with the next derived
    seed, and counted on the answer kept."""
    request = llm_request_cg.model_copy(update={"seed": 321})
    bad = _mock_openai_response()
    bad.choices[0].message.content = "not valid json for the schema"

    with patch("backend.app.core.llm.client_openrouter.AsyncOpenAI") as MockOpenAI:
        mock_create = AsyncMock(side_effect=[bad, _mock_openai_response()])
        MockOpenAI.return_value.chat.completions.create = mock_create
        client = LLMClientOpenRouter("openrouter/some-model", max_retries=2)

        result = await client.generate(request, expected_format=ClueJSONFormat)

    assert result.raw_payload[client_module.MALFORMED_RESAMPLE_KEY] == 1
    seeds = [call.kwargs["seed"] for call in mock_create.call_args_list]
    assert seeds == [321, client_module.redraw_seed(321, 1) & 0x7FFFFFFF]


@pytest.mark.asyncio
async def test_openrouter_malformed_answer_keeps_the_answer(llm_request_cg, caplog):
    """A malformed answer carries the provider's payload and logs the whole answer at WARNING: only
    its count reaches the database, and the batch logs at INFO."""
    bad = _mock_openai_response()
    bad.choices[0].message.content = "not valid json for the schema"

    with patch("backend.app.core.llm.client_openrouter.AsyncOpenAI") as MockOpenAI:
        MockOpenAI.return_value.chat.completions.create = AsyncMock(return_value=bad)
        client = LLMClientOpenRouter("openrouter/some-model")

        with caplog.at_level(logging.WARNING, logger="backend.app.core.llm.client_openrouter"):
            with pytest.raises(LLMMalformedResponseError) as excinfo:
                await client.generate(llm_request_cg, expected_format=ClueJSONFormat)

    assert excinfo.value.raw_payload == {"id": "req-123"}
    assert "not valid json for the schema" in caplog.text


@pytest.mark.asyncio
async def test_openrouter_default_no_retry(llm_request_cg, monkeypatch):
    """max_retries defaults to 0: a retriable error raises on the first
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


@pytest.mark.asyncio
async def test_openrouter_draws_a_cut_answer_again_with_a_new_seed(llm_request_cg):
    """A cut answer is a bad draw: unlike a transient, it is drawn again with the next derived seed
    (masked to the 31 bits OpenRouter takes)."""
    request = llm_request_cg.model_copy(update={"seed": 321})
    cut = _mock_openai_response()
    cut.choices[0].finish_reason = "length"

    with patch("backend.app.core.llm.client_openrouter.AsyncOpenAI") as MockOpenAI:
        mock_create = AsyncMock(side_effect=[cut, _mock_openai_response()])
        MockOpenAI.return_value.chat.completions.create = mock_create
        client = LLMClientOpenRouter("openrouter/some-model", max_retries=2)

        await client.generate(request, expected_format=ClueJSONFormat)

    seeds = [call.kwargs["seed"] for call in mock_create.call_args_list]
    assert seeds == [321, client_module.redraw_seed(321, 1) & 0x7FFFFFFF]
