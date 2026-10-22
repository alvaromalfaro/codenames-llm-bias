import asyncio
import json
import logging
import httpx
import pytest
from unittest.mock import patch, MagicMock
from ollama import RequestError, ResponseError
from backend.app import config
from backend.app.core.llm import client as client_module
from backend.app.core.llm.client_local import LLMClientLocal
from backend.app.models.llm_errors import (
    LLMModelNotProvidedError,
    LLMRefusalError,
    LLMParseError,
    LLMTimeoutError,
    LLMEmptyResponseError,
    LLMDegenerateResponseError,
    LLMTruncatedResponseError,
    LLMModelNotFoundError,
    LLMInvalidRequestError,
    LLMAuthError,
    LLMRateLimitError,
    LLMProviderUnavailableError,
    LLMMalformedResponseError,
)
from backend.app.models.llm_schemas import (
    LLMResponse,
    ClueJSONFormat,
    GuessJSONFormat,
    ConfidenceRankingJSONFormat,
)


def _patch_ollama():
    """Patch the ollama client the local client builds. Specced on the real AsyncClient, so its
    awaited methods (chat, show, close) are AsyncMocks and a call with a wrong argument fails."""
    return patch("backend.app.core.llm.client_local.AsyncClient", autospec=True)


@pytest.mark.asyncio
async def test_llm_client_local_generate_success(llm_request_cg):
    """
    Tests that the LLMClientLocal correctly processes a valid response from the Ollama client and
    returns the expected LLMResponse.
    """
    mock_content = json.dumps(
        {
            "reasoning": "The word 'battle' captures a military commander ('NAPOLEON'), "
            "the hardware ('RIFLE') and a primary theater of conflict ('RUSSIA')",
            "clue": "battle",
            "count": 3,
        }
    )
    mock_response = MagicMock()
    mock_response.message.content = mock_content
    mock_response.total_duration = 3200
    mock_response.prompt_eval_count = 10
    mock_response.eval_count = 20
    mock_response.done_reason = "stop"
    mock_response.model_dump_json.return_value = json.dumps({"message": {"content": mock_content}})

    with _patch_ollama() as MockClient:
        MockClient.return_value.chat.return_value = mock_response
        client = LLMClientLocal("ollama3.2:latest")

        result = await client.generate(llm_request_cg, expected_format=ClueJSONFormat)

        assert isinstance(result, LLMResponse)
        assert result.text == mock_response.message.content
        assert result.model_used == client.model_name
        # Ollama reports nanoseconds; latency_ms is milliseconds (ns / 1e6).
        assert result.latency_ms == round(mock_response.total_duration / 1_000_000)
        assert result.usage.prompt_tokens == mock_response.prompt_eval_count
        assert result.usage.completion_tokens == mock_response.eval_count
        assert (
            result.usage.total_tokens == mock_response.prompt_eval_count + mock_response.eval_count
        )
        assert result.finish_reason == mock_response.done_reason
        assert result.provider == "ollama"
        assert result.execution_mode == "local"


@pytest.mark.asyncio
async def test_llm_client_local_raises_model_not_provided_error(llm_request_cg):
    """
    Tests that the LLMClientLocal raises an LLMModelNotProvidedError when the Ollama client raises a
    RequestError indicating that the model was not provided.
    """
    with _patch_ollama() as MockClient:
        MockClient.return_value.chat.side_effect = RequestError("Model not provided")
        client = LLMClientLocal(None)

        with pytest.raises(LLMModelNotProvidedError):
            await client.generate(llm_request_cg)


# An error status is the daemon's or the request's, never a refusal by the model.
@pytest.mark.parametrize(
    "status, text, expected, retriable",
    [
        (404, "model 'ollama3.2:latest' not found", LLMModelNotFoundError, False),
        (400, '"ollama3.2:latest" does not support thinking', LLMInvalidRequestError, False),
        (401, "unauthorized", LLMAuthError, False),
        (429, "too many requests", LLMRateLimitError, True),
        (
            500,
            "llama runner process has terminated: CUDA error: out of memory",
            LLMProviderUnavailableError,
            True,
        ),
        (
            503,
            "server busy, please try again.  maximum pending requests exceeded",
            LLMProviderUnavailableError,
            True,
        ),
    ],
)
@pytest.mark.asyncio
async def test_an_error_status_is_mapped_by_its_code(
    llm_request_cg, status, text, expected, retriable
):
    with _patch_ollama() as MockClient:
        MockClient.return_value.chat.side_effect = ResponseError(
            json.dumps({"error": text}), status
        )
        client = LLMClientLocal("ollama3.2:latest")

        with pytest.raises(expected) as excinfo:
            await client.generate(llm_request_cg)

    assert not isinstance(excinfo.value, LLMRefusalError)
    assert excinfo.value.retriable is retriable
    assert excinfo.value.http_status == status
    assert excinfo.value.provider == "ollama"


@pytest.mark.parametrize(
    "raw",
    [
        # What ollama raises when nothing listens at OLLAMA_HOST.
        ConnectionError("Failed to connect to Ollama."),
        # What httpx raises when the daemon drops the connection mid-answer (stopped, or crashed).
        httpx.RemoteProtocolError("Server disconnected without sending a response."),
        httpx.ReadError("connection reset"),
    ],
)
@pytest.mark.asyncio
async def test_a_daemon_that_cannot_be_reached_is_unavailable(llm_request_cg, raw):
    """ollama wraps neither: unmapped, they escaped as plain exceptions, never retried and with no
    error code for the batch."""
    with _patch_ollama() as MockClient:
        MockClient.return_value.chat.side_effect = raw
        client = LLMClientLocal("ollama3.2:latest")

        with pytest.raises(LLMProviderUnavailableError) as excinfo:
            await client.generate(llm_request_cg)

    assert excinfo.value.retriable is True
    assert excinfo.value.cause is raw


@pytest.mark.asyncio
async def test_a_daemon_that_fails_is_sent_the_same_request_again(llm_request_cg, monkeypatch):
    """A runner that crashed (500) is the daemon's failure, not the model's: the same request is
    sent again after the backoff, its seed unchanged, and the daemon's next answer is kept."""
    monkeypatch.setattr(client_module, "_RETRY_BACKOFF_BASE_S", 0.0)
    request = llm_request_cg.model_copy(update={"seed": 123})
    with _patch_ollama() as MockClient:
        mock_chat = MockClient.return_value.chat
        mock_chat.side_effect = [
            ResponseError('{"error": "llama runner process has terminated"}', 500),
            ConnectionError("Failed to connect to Ollama."),
            _mock_ollama_response(),
        ]
        client = LLMClientLocal("ollama3.2:latest", max_retries=3)

        result = await client.generate(request, expected_format=ClueJSONFormat)

    assert isinstance(result, LLMResponse)
    assert _sent_seeds(mock_chat) == [123, 123, 123]


@pytest.mark.asyncio
async def test_llm_client_local_raises_llm_parse_error(llm_request_cg):
    """
    Tests that the LLMClientLocal raises an LLMParseError when the Ollama client returns a response
    that cannot be parsed as valid JSON.
    """
    mock_response = MagicMock()
    mock_response.message.content = "This is not a valid JSON response"
    mock_response.model_dump_json.return_value = mock_response.message.content

    with _patch_ollama() as MockClient:
        MockClient.return_value.chat.return_value = mock_response
        client = LLMClientLocal("ollama3.2:latest")

        with pytest.raises(LLMParseError):
            await client.generate(llm_request_cg)


@pytest.mark.asyncio
@pytest.mark.parametrize(("prompt_count", "answer_count"), [(None, 20), (10, None), (None, None)])
async def test_llm_client_local_missing_token_count(llm_request_cg, prompt_count, answer_count):
    """
    Tests that a token count the server leaves out is recorded as None, with the total unknown,
    instead of failing the call with a TypeError when the two are added.
    """
    mock_content = json.dumps({"reasoning": "test reasoning", "clue": "battle", "count": 3})
    mock_response = MagicMock()
    mock_response.message.content = mock_content
    mock_response.total_duration = 3200
    mock_response.prompt_eval_count = prompt_count
    mock_response.eval_count = answer_count
    mock_response.done_reason = "stop"
    mock_response.model_dump_json.return_value = json.dumps({"message": {"content": mock_content}})

    with _patch_ollama() as MockClient:
        MockClient.return_value.chat.return_value = mock_response
        client = LLMClientLocal("ollama3.2:latest")

        result = await client.generate(llm_request_cg, expected_format=ClueJSONFormat)

    assert result.usage.prompt_tokens == prompt_count
    assert result.usage.completion_tokens == answer_count
    assert result.usage.total_tokens is None


@pytest.mark.asyncio
async def test_llm_client_local_generate_think_false(llm_request_cg):
    """
    Tests that when think=False, the chat() call is made without the think parameter.
    """
    mock_content = json.dumps({"reasoning": "test reasoning", "clue": "battle", "count": 3})
    mock_response = MagicMock()
    mock_response.message.content = mock_content
    mock_response.total_duration = 3200
    mock_response.prompt_eval_count = 10
    mock_response.eval_count = 20
    mock_response.done_reason = "stop"
    mock_response.model_dump_json.return_value = json.dumps({"message": {"content": mock_content}})

    with _patch_ollama() as MockClient:
        mock_chat = MockClient.return_value.chat
        mock_chat.return_value = mock_response
        client = LLMClientLocal("ollama3.2:latest", think=False)

        result = await client.generate(llm_request_cg, expected_format=ClueJSONFormat)

        assert isinstance(result, LLMResponse)
        _, kwargs = mock_chat.call_args
        assert "think" not in kwargs


@pytest.mark.asyncio
async def test_llm_client_local_raises_malformed_on_schema_mismatch(llm_request_cg):
    """
    Tests that the LLMClientLocal raises an LLMMalformedResponseError, a retriable LLMParseError,
    when the response is valid JSON but does not match the expected schema (missing required
    fields).
    """
    mock_content = json.dumps({"reasoning": "some reasoning", "clue": "battle"})  # missing "count"
    mock_response = MagicMock()
    mock_response.message.content = mock_content
    mock_response.model_dump_json.return_value = json.dumps({"message": {"content": mock_content}})

    with _patch_ollama() as MockClient:
        MockClient.return_value.chat.return_value = mock_response
        client = LLMClientLocal("ollama3.2:latest")

        with pytest.raises(LLMMalformedResponseError) as excinfo:
            await client.generate(llm_request_cg, expected_format=ClueJSONFormat)
    assert isinstance(excinfo.value, LLMParseError)
    assert excinfo.value.code == "malformed_response" and excinfo.value.retriable is True


@pytest.mark.asyncio
async def test_llm_client_local_malformed_answer_keeps_the_answer(llm_request_cg, caplog):
    """
    Tests that a malformed answer carries the provider's payload and logs the whole answer at
    WARNING: only its count reaches the database, and the batch logs at INFO.
    """
    mock_content = json.dumps({"reasoning": "some reasoning", "clue": "battle"})  # missing "count"
    mock_response = MagicMock()
    mock_response.message.content = mock_content
    mock_response.model_dump_json.return_value = json.dumps({"message": {"content": mock_content}})

    with _patch_ollama() as MockClient:
        MockClient.return_value.chat.return_value = mock_response
        client = LLMClientLocal("ollama3.2:latest")

        with caplog.at_level(logging.WARNING, logger="backend.app.core.llm.client_local"):
            with pytest.raises(LLMMalformedResponseError) as excinfo:
                await client.generate(llm_request_cg, expected_format=ClueJSONFormat)

    assert excinfo.value.raw_payload == {"message": {"content": mock_content}}
    assert mock_content in caplog.text


@pytest.mark.asyncio
async def test_llm_client_local_generate_without_expected_format(llm_request_cg):
    """
    Tests that generate() succeeds without an expected_format, skipping schema validation
    and using format="json" as the fallback.
    """
    mock_content = json.dumps({"reasoning": "test reasoning", "clue": "battle", "count": 3})
    mock_response = MagicMock()
    mock_response.message.content = mock_content
    mock_response.total_duration = 3200
    mock_response.prompt_eval_count = 10
    mock_response.eval_count = 20
    mock_response.done_reason = "stop"
    mock_response.model_dump_json.return_value = json.dumps({"message": {"content": mock_content}})

    with _patch_ollama() as MockClient:
        mock_chat = MockClient.return_value.chat
        mock_chat.return_value = mock_response
        client = LLMClientLocal("ollama3.2:latest")

        result = await client.generate(llm_request_cg)

        assert isinstance(result, LLMResponse)
        assert result.text == mock_content
        _, kwargs = mock_chat.call_args
        assert kwargs.get("format") == "json"


def _mock_ollama_response(model: str | None = None) -> MagicMock:
    """Builds a MagicMock mimicking a successful Ollama chat response for the tests below."""
    mock_content = json.dumps({"reasoning": "test reasoning", "clue": "battle", "count": 3})
    mock_response = MagicMock()
    mock_response.message.content = mock_content
    mock_response.total_duration = 3200
    mock_response.prompt_eval_count = 10
    mock_response.eval_count = 20
    mock_response.done_reason = "stop"
    dumped = {"message": {"content": mock_content}}
    if model is not None:
        dumped["model"] = model
    mock_response.model_dump_json.return_value = json.dumps(dumped)
    return mock_response


@pytest.mark.asyncio
async def test_llm_client_local_forwards_temperature_and_seed_when_seeded(llm_request_cg):
    """
    When request.seed is not None, the mocked chat() must be called with an `options` dict that
    carries BOTH the temperature and the seed.
    """
    request = llm_request_cg.model_copy(update={"seed": 123})

    with _patch_ollama() as MockClient:
        mock_chat = MockClient.return_value.chat
        mock_chat.return_value = _mock_ollama_response()
        client = LLMClientLocal("ollama3.2:latest")

        await client.generate(request, expected_format=ClueJSONFormat)

        _, kwargs = mock_chat.call_args
        assert "options" in kwargs
        assert kwargs["options"]["temperature"] == request.temperature
        assert kwargs["options"]["seed"] == 123


@pytest.mark.asyncio
async def test_llm_client_local_omits_seed_when_unset(llm_request_cg):
    """
    When request.seed is None, `options` must carry the temperature but NOT a `seed` key (guarding
    against passing `"seed": None`, which some ollama-client versions serialize).
    """
    assert llm_request_cg.seed is None

    with _patch_ollama() as MockClient:
        mock_chat = MockClient.return_value.chat
        mock_chat.return_value = _mock_ollama_response()
        client = LLMClientLocal("ollama3.2:latest")

        await client.generate(llm_request_cg, expected_format=ClueJSONFormat)

        _, kwargs = mock_chat.call_args
        assert "options" in kwargs
        assert kwargs["options"]["temperature"] == llm_request_cg.temperature
        assert "seed" not in kwargs["options"]


@pytest.mark.asyncio
async def test_llm_client_local_always_passes_options_with_temperature(llm_request_cg):
    """
    Regression guard against the temperature leak returning: chat() must ALWAYS be called with an
    `options` dict carrying `temperature`.
    """
    with _patch_ollama() as MockClient:
        mock_chat = MockClient.return_value.chat
        mock_chat.return_value = _mock_ollama_response()
        client = LLMClientLocal("ollama3.2:latest")

        await client.generate(llm_request_cg, expected_format=ClueJSONFormat)

        _, kwargs = mock_chat.call_args
        assert "options" in kwargs
        assert "temperature" in kwargs["options"]


@pytest.mark.asyncio
async def test_llm_client_local_populates_sampling_telemetry(llm_request_cg):
    """
    The returned LLMResponse must carry the requested sampling telemetry (temperature + seed), a
    None system_fingerprint (Ollama has none), and a resolved_model.
    """
    request = llm_request_cg.model_copy(update={"seed": 99, "temperature": 0.3})

    with _patch_ollama() as MockClient:
        mock_chat = MockClient.return_value.chat
        mock_chat.return_value = _mock_ollama_response(model="llama3.2:latest")
        client = LLMClientLocal("ollama3.2:latest")

        result = await client.generate(request, expected_format=ClueJSONFormat)

        assert result.requested_temperature == 0.3
        assert result.requested_seed == 99
        assert result.system_fingerprint is None
        assert result.resolved_model == "llama3.2:latest"


@pytest.mark.asyncio
async def test_llm_client_local_resolved_model_falls_back_to_model_name(llm_request_cg):
    """
    When the Ollama response carries no `model` field, resolved_model falls back to the configured
    model name.
    """
    with _patch_ollama() as MockClient:
        mock_chat = MockClient.return_value.chat
        mock_chat.return_value = _mock_ollama_response(model=None)
        client = LLMClientLocal("ollama3.2:latest")

        result = await client.generate(llm_request_cg, expected_format=ClueJSONFormat)

        assert result.resolved_model == "ollama3.2:latest"


# latency_ms unit conversion (nanoseconds -> milliseconds)
_MAX_INT4 = 2_147_483_647  # llm_call.latency_ms is a 32-bit int column


@pytest.mark.asyncio
async def test_llm_client_local_latency_ms_converts_nanoseconds(llm_request_cg):
    """Ollama reports total_duration in NANOSECONDS; latency_ms must be milliseconds.

    A ~42.7s call reports total_duration ~4.27e10 ns, which as a raw value overflows the 32-bit
    llm_call.latency_ms column. After ns->ms it is ~42707 ms and fits comfortably.
    """
    mock_response = _mock_ollama_response()
    mock_response.total_duration = 42_706_620_491  # ~42.7s in nanoseconds

    with _patch_ollama() as MockClient:
        MockClient.return_value.chat.return_value = mock_response
        client = LLMClientLocal("ollama3.2:latest")

        result = await client.generate(llm_request_cg, expected_format=ClueJSONFormat)

        assert result.latency_ms == 42707  # round(42_706_620_491 / 1e6)
        assert result.latency_ms <= _MAX_INT4  # fits the int4 column, no overflow


@pytest.mark.asyncio
async def test_llm_client_local_latency_ms_none_duration_is_zero(llm_request_cg):
    """total_duration is absent on some responses; latency_ms falls back to 0 (int, ge=0)."""
    mock_response = _mock_ollama_response()
    mock_response.total_duration = None

    with _patch_ollama() as MockClient:
        MockClient.return_value.chat.return_value = mock_response
        client = LLMClientLocal("ollama3.2:latest")

        result = await client.generate(llm_request_cg, expected_format=ClueJSONFormat)

        assert result.latency_ms == 0


# transient retry
@pytest.mark.asyncio
async def test_llm_client_local_non_retriable_raises_immediately(llm_request_cg):
    """A non-retriable error (a model that is not pulled) raises on the first attempt even with a
    retry budget: exactly one chat() call."""
    with _patch_ollama() as MockClient:
        mock_chat = MockClient.return_value.chat
        mock_chat.side_effect = ResponseError("model 'ollama3.2:latest' not found", 404)
        client = LLMClientLocal("ollama3.2:latest", max_retries=3)

        with pytest.raises(LLMModelNotFoundError):
            await client.generate(llm_request_cg, expected_format=ClueJSONFormat)
        assert mock_chat.call_count == 1


@pytest.mark.asyncio
async def test_llm_client_local_default_no_retry(llm_request_cg):
    """max_retries defaults to 0: a single attempt, even for a retriable error."""
    with _patch_ollama() as MockClient:
        mock_chat = MockClient.return_value.chat
        mock_chat.side_effect = ResponseError("llama runner process has terminated", 500)
        client = LLMClientLocal("ollama3.2:latest")  # no max_retries

        with pytest.raises(LLMProviderUnavailableError):
            await client.generate(llm_request_cg, expected_format=ClueJSONFormat)
        assert mock_chat.call_count == 1


@pytest.mark.asyncio
async def test_llm_client_local_retries_retriable_then_succeeds(llm_request_cg, monkeypatch):
    """A transient is retried by generate_with_retries: two timeouts, then an answer. generate()
    returns after exactly 3 attempts, each re-sending the identical request (same seed) - no
    reseed."""
    monkeypatch.setattr(client_module, "_RETRY_BACKOFF_BASE_S", 0.0)  # no real sleeping
    request = llm_request_cg.model_copy(update={"seed": 123})

    with _patch_ollama() as MockClient:
        mock_chat = MockClient.return_value.chat
        mock_chat.side_effect = [
            httpx.ReadTimeout("timed out"),
            httpx.ReadTimeout("timed out"),
            _mock_ollama_response(),
        ]
        client = LLMClientLocal("ollama3.2:latest", max_retries=2)

        result = await client.generate(request, expected_format=ClueJSONFormat)

        assert isinstance(result, LLMResponse)
        assert mock_chat.call_count == 3
        seeds = [call.kwargs["options"]["seed"] for call in mock_chat.call_args_list]
        assert seeds == [123, 123, 123]  # same request re-sent, never reseeded


# the wait on the daemon is bounded
def test_the_ollama_client_waits_the_configured_time():
    """ollama's own default is no limit at all: a hung daemon would stop the batch for good."""
    with _patch_ollama() as MockClient:
        LLMClientLocal("ollama3.2:latest")

    timeout = MockClient.call_args.kwargs["timeout"]
    assert timeout.read == timeout.write == timeout.pool == config.OLLAMA_TIMEOUT_S
    assert timeout.connect == config.OLLAMA_CONNECT_TIMEOUT_S


@pytest.mark.parametrize("raw", [httpx.ReadTimeout("timed out"), httpx.ConnectTimeout("timed out")])
@pytest.mark.asyncio
async def test_a_call_that_times_out_is_a_retriable_timeout(llm_request_cg, raw):
    """The ollama client does not wrap httpx's timeouts; the local client maps them onto the
    taxonomy, as the retriable LLMTimeoutError (a provider error for the batch)."""
    with _patch_ollama() as MockClient:
        MockClient.return_value.chat.side_effect = raw
        client = LLMClientLocal("ollama3.2:latest")

        with pytest.raises(LLMTimeoutError) as excinfo:
            await client.generate(llm_request_cg, expected_format=ClueJSONFormat)

    assert excinfo.value.retriable is True
    assert excinfo.value.cause is raw
    assert excinfo.value.provider == "ollama"


@pytest.mark.asyncio
async def test_a_daemon_that_never_answers_fails_after_the_budget(llm_request_cg, monkeypatch):
    """Bounded: every attempt times out, the budget runs out (max_retries=3 -> 4 attempts) and the
    timeout ends the call instead of waiting forever."""
    monkeypatch.setattr(client_module, "_RETRY_BACKOFF_BASE_S", 0.0)
    with _patch_ollama() as MockClient:
        mock_chat = MockClient.return_value.chat
        mock_chat.side_effect = httpx.ReadTimeout("timed out")
        client = LLMClientLocal("ollama3.2:latest", max_retries=3)

        with pytest.raises(LLMTimeoutError):
            await client.generate(llm_request_cg, expected_format=ClueJSONFormat)

    assert mock_chat.call_count == 4


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raw",
    [
        httpx.ReadTimeout("timed out"),
        ConnectionError("Failed to connect to Ollama."),
        ResponseError('{"error": "llama runner process has terminated"}', 500),
    ],
)
async def test_a_probe_the_daemon_fails_is_made_again_not_cached(llm_request_cg, monkeypatch, raw):
    """A capability probe the daemon fails fails the attempt as the chat would, and caches nothing.
    Cached as "not capable", it would leave `think` out for the whole game, and a reasoning model
    would play with thinking on once the daemon came back."""
    monkeypatch.setattr(client_module, "_RETRY_BACKOFF_BASE_S", 0.0)
    with _patch_ollama() as MockClient:
        instance = MockClient.return_value
        instance.show.side_effect = [raw, MagicMock(capabilities=["completion", "thinking"])]
        instance.chat.return_value = _mock_ollama_response()
        client = LLMClientLocal("gemma4:12b", think=False, max_retries=3)

        await client.generate(llm_request_cg, expected_format=ClueJSONFormat)

    assert instance.show.call_count == 2
    assert instance.chat.call_count == 1
    assert instance.chat.call_args.kwargs["think"] is False


@pytest.mark.asyncio
async def test_a_probe_for_a_model_that_is_not_pulled_fails_the_call(llm_request_cg):
    """The probe maps a status as the chat would: a model that is not pulled ends the call at once,
    before the chat is sent."""
    with _patch_ollama() as MockClient:
        instance = MockClient.return_value
        instance.show.side_effect = ResponseError("model 'gemma4:12b' not found", 404)
        client = LLMClientLocal("gemma4:12b", think=False, max_retries=3)

        with pytest.raises(LLMModelNotFoundError):
            await client.generate(llm_request_cg, expected_format=ClueJSONFormat)

    assert instance.show.call_count == 1
    instance.chat.assert_not_called()


# the health check the batch asks before playing a game again
@pytest.mark.asyncio
async def test_a_daemon_that_has_the_model_is_healthy():
    with _patch_ollama() as MockClient:
        client = LLMClientLocal("gemma4:12b")

        assert await client.health_check() is True

    MockClient.return_value.show.assert_awaited_once_with("gemma4:12b")


@pytest.mark.parametrize(
    "raw",
    [
        ConnectionError("Failed to connect to Ollama."),
        httpx.ConnectError("connection refused"),
        httpx.ReadTimeout("timed out"),
        ResponseError('{"error": "llama runner process has terminated"}', 500),
        ResponseError("model 'gemma4:12b' not found", 404),
    ],
)
@pytest.mark.asyncio
async def test_a_daemon_that_fails_the_health_check_is_not_healthy(raw):
    with _patch_ollama() as MockClient:
        MockClient.return_value.show.side_effect = raw
        client = LLMClientLocal("gemma4:12b")

        assert await client.health_check() is False


@pytest.mark.asyncio
async def test_a_wedged_daemon_is_not_healthy(monkeypatch):
    """The check waits OLLAMA_HEALTH_TIMEOUT_S, not the 300 s a game's call may take."""
    monkeypatch.setattr(config, "OLLAMA_HEALTH_TIMEOUT_S", 0.01)

    async def never_answers(model):
        await asyncio.sleep(10)

    with _patch_ollama() as MockClient:
        MockClient.return_value.show.side_effect = never_answers
        client = LLMClientLocal("gemma4:12b")

        assert await client.health_check() is False


@pytest.mark.asyncio
async def test_close_closes_the_ollama_client():
    with _patch_ollama() as MockClient:
        client = LLMClientLocal("ollama3.2:latest")

        await client.close()

    MockClient.return_value.close.assert_awaited_once()


# empty-response re-sample
def _mock_empty_ollama_response(content: str = "") -> MagicMock:
    """Builds a MagicMock mimicking the intermittent empty ollama chat response."""
    mock_response = MagicMock()
    mock_response.message.content = content
    mock_response.total_duration = 3200
    mock_response.prompt_eval_count = 10
    mock_response.eval_count = 0
    mock_response.done_reason = "stop"
    mock_response.model_dump_json.return_value = json.dumps({"message": {"content": content}})
    return mock_response


@pytest.mark.parametrize("empty_content", ["", "   \n\t "])
@pytest.mark.asyncio
async def test_llm_client_local_empty_response_resampled_then_succeeds(
    llm_request_cg, empty_content
):
    """An empty (or whitespace-only) draw is re-sampled: attempt 1 returns '', attempt 2 returns
    valid JSON, and generate() succeeds after exactly 2 attempts. The number of empty draws absorbed
    is stamped on raw_payload for downstream audit."""
    request = llm_request_cg.model_copy(update={"seed": 123})

    with _patch_ollama() as MockClient:
        mock_chat = MockClient.return_value.chat
        mock_chat.side_effect = [
            _mock_empty_ollama_response(empty_content),
            _mock_ollama_response(),
        ]
        client = LLMClientLocal("ollama3.2:latest", max_retries=3)

        result = await client.generate(request, expected_format=ClueJSONFormat)

        assert isinstance(result, LLMResponse)
        assert mock_chat.call_count == 2
        assert result.raw_payload["empty_response_resamples"] == 1
        # The new draw has a new seed, derived from the request's own.
        seeds = [call.kwargs["options"]["seed"] for call in mock_chat.call_args_list]
        assert seeds == [123, client_module.redraw_seed(123, 1)]


@pytest.mark.asyncio
async def test_llm_client_local_empty_response_raises_after_cap(llm_request_cg):
    """A persistently empty response is NOT retried forever: it exhausts the budget (max_retries=3
    -> 4 attempts) and then raises."""
    with _patch_ollama() as MockClient:
        mock_chat = MockClient.return_value.chat
        mock_chat.return_value = _mock_empty_ollama_response()
        client = LLMClientLocal("ollama3.2:latest", max_retries=3)

        with pytest.raises(LLMEmptyResponseError):
            await client.generate(request=llm_request_cg, expected_format=ClueJSONFormat)
        assert mock_chat.call_count == 4


@pytest.mark.asyncio
async def test_llm_client_local_empty_response_not_retried_by_default(llm_request_cg):
    """With the default max_retries=0, an empty response raises on attempt 1."""
    with _patch_ollama() as MockClient:
        mock_chat = MockClient.return_value.chat
        mock_chat.return_value = _mock_empty_ollama_response()
        client = LLMClientLocal("ollama3.2:latest")  # no max_retries

        with pytest.raises(LLMEmptyResponseError):
            await client.generate(llm_request_cg, expected_format=ClueJSONFormat)
        assert mock_chat.call_count == 1


@pytest.mark.asyncio
async def test_llm_client_local_malformed_answer_is_drawn_again_and_counted(llm_request_cg):
    """An answer that does not match the schema is a bad draw: drawn again with a new seed, and
    counted apart on the answer kept."""
    request = llm_request_cg.model_copy(update={"seed": 123})

    with _patch_ollama() as MockClient:
        mock_chat = MockClient.return_value.chat
        # Well-formed JSON, but missing the fields ClueJSONFormat requires.
        mock_chat.side_effect = [
            _mock_empty_ollama_response('{"unexpected": "shape"}'),
            _mock_ollama_response(),
        ]
        client = LLMClientLocal("ollama3.2:latest", max_retries=3)

        result = await client.generate(request, expected_format=ClueJSONFormat)

    assert result.raw_payload["malformed_response_resamples"] == 1
    assert result.raw_payload["empty_response_resamples"] == 0
    seeds = [call.kwargs["options"]["seed"] for call in mock_chat.call_args_list]
    assert seeds == [123, client_module.redraw_seed(123, 1)]


@pytest.mark.asyncio
async def test_llm_client_local_unreadable_answer_stays_non_retriable(llm_request_cg):
    """An answer the client cannot read at all (no expected format to hold it to, and not JSON) is
    the plain LLMParseError, which is not retried even with a full budget."""
    mock_response = MagicMock()
    mock_response.message.content = "This is not a valid JSON response"
    mock_response.model_dump_json.return_value = mock_response.message.content

    with _patch_ollama() as MockClient:
        mock_chat = MockClient.return_value.chat
        mock_chat.return_value = mock_response
        client = LLMClientLocal("ollama3.2:latest", max_retries=3)

        with pytest.raises(LLMParseError) as excinfo:
            await client.generate(llm_request_cg)
    assert type(excinfo.value) is LLMParseError and excinfo.value.retriable is False
    assert mock_chat.call_count == 1


@pytest.mark.asyncio
async def test_llm_client_local_no_empty_draws_records_zero(llm_request_cg):
    """The common case is still auditable: zero empty draws is recorded as 0, not omitted."""
    with _patch_ollama() as MockClient:
        MockClient.return_value.chat.return_value = _mock_ollama_response()
        client = LLMClientLocal("ollama3.2:latest", max_retries=3)

        result = await client.generate(llm_request_cg, expected_format=ClueJSONFormat)

        assert result.raw_payload["empty_response_resamples"] == 0


# thinking-capability gate
def _patch_client_with_capabilities(MockClient, capabilities):
    """Wire the mocked ollama Client so /api/show reports ``capabilities`` and chat() succeeds.

    ``capabilities=None`` makes the probe fail for a reason other than the daemon: an answer the
    client cannot read.
    """
    instance = MockClient.return_value
    if capabilities is None:
        instance.show.side_effect = ValueError("unreadable /api/show answer")
    else:
        instance.show.return_value.capabilities = capabilities
    instance.chat.return_value = _mock_ollama_response()
    return instance


@pytest.mark.asyncio
async def test_thinking_capable_model_is_sent_think_false(llm_request_cg):
    """A model advertising `thinking` must be sent an explicit think=False.

    Omitting the flag is NOT equivalent: ollama defaults a reasoning model to thinking ON, which
    leaves it in a different inference regime than the non-reasoning roster models.
    """
    with _patch_ollama() as MockClient:
        instance = _patch_client_with_capabilities(
            MockClient, ["completion", "vision", "tools", "thinking"]
        )
        client = LLMClientLocal("gemma4:12b", think=False, max_retries=3)

        await client.generate(llm_request_cg, expected_format=ClueJSONFormat)

        _, kwargs = instance.chat.call_args
        assert "think" in kwargs, "a thinking-capable model must receive an explicit `think`"
        assert kwargs["think"] is False


@pytest.mark.asyncio
async def test_non_thinking_model_is_not_sent_think(llm_request_cg):
    """A model that does not advertise `thinking` must NOT receive the flag at all: ollama rejects
    think=True for such models (HTTP 400 'does not support thinking')."""
    with _patch_ollama() as MockClient:
        instance = _patch_client_with_capabilities(MockClient, ["completion", "tools"])
        client = LLMClientLocal("llama3.1:8b", think=False, max_retries=3)

        await client.generate(llm_request_cg, expected_format=ClueJSONFormat)

        _, kwargs = instance.chat.call_args
        assert "think" not in kwargs


@pytest.mark.asyncio
async def test_thinking_gate_applies_to_guess_calls_too(llm_request_cg):
    """The gate lives in _generate_once, so it covers guess/ranking calls, not just clues."""
    with _patch_ollama() as MockClient:
        instance = _patch_client_with_capabilities(MockClient, ["completion", "thinking"])
        instance.chat.return_value = _mock_empty_ollama_response(
            json.dumps(
                {
                    "reasoning": "r",
                    "stop_reason": "done",
                    "proposals": [{"word": "OCEAN", "confidence": 0.9}],
                }
            )
        )
        client = LLMClientLocal("gemma4:12b", think=False, max_retries=3)

        await client.generate(llm_request_cg, expected_format=GuessJSONFormat)

        _, kwargs = instance.chat.call_args
        assert kwargs["think"] is False


@pytest.mark.asyncio
async def test_think_true_is_still_honoured_for_capable_model(llm_request_cg):
    """The gate decides whether the flag is SENT, not what it says: an explicit think=True on a
    capable model still enables reasoning."""
    with _patch_ollama() as MockClient:
        instance = _patch_client_with_capabilities(MockClient, ["completion", "thinking"])
        client = LLMClientLocal("gemma4:12b", think=True, max_retries=3)

        await client.generate(llm_request_cg, expected_format=ClueJSONFormat)

        _, kwargs = instance.chat.call_args
        assert kwargs["think"] is True


@pytest.mark.asyncio
async def test_capability_probe_failure_degrades_to_omitting_think(llm_request_cg):
    """A probe that fails for a reason other than the daemon must not break generation: it degrades
    to omitting `think`."""
    with _patch_ollama() as MockClient:
        instance = _patch_client_with_capabilities(MockClient, None)
        client = LLMClientLocal("gemma4:12b", think=False, max_retries=3)

        result = await client.generate(llm_request_cg, expected_format=ClueJSONFormat)

        assert isinstance(result, LLMResponse)
        _, kwargs = instance.chat.call_args
        assert "think" not in kwargs


@pytest.mark.asyncio
async def test_capability_probe_is_cached_across_calls(llm_request_cg):
    """The probe costs one /api/show per client, not one per generate()."""
    with _patch_ollama() as MockClient:
        instance = _patch_client_with_capabilities(MockClient, ["completion", "thinking"])
        client = LLMClientLocal("gemma4:12b", think=False, max_retries=3)

        await client.generate(llm_request_cg, expected_format=ClueJSONFormat)
        await client.generate(llm_request_cg, expected_format=ClueJSONFormat)

        assert instance.show.call_count == 1
        assert instance.chat.call_count == 2


# degenerate (well-formed but unusable) re-sample
_VALID_GUESS_JSON = json.dumps(
    {
        "reasoning": "marine words",
        "stop_reason": "done",
        "proposals": [{"word": "OCEAN", "confidence": 0.9}],
    }
)
# violates GuessProposal.proposals (min_length=1).
_DEGENERATE_GUESS_JSON = json.dumps(
    {
        "reasoning": "nothing clears my confidence threshold",
        "stop_reason": "risk avoidance",
        "proposals": [],
    }
)


@pytest.mark.asyncio
async def test_degenerate_empty_proposals_is_resampled_then_succeeds(llm_request_cg):
    """An empty proposals list parses fine but cannot build a GuessProposal. It must be re-sampled
    by the SAME bounded machinery, and counted separately from empty-content draws."""
    with _patch_ollama() as MockClient:
        mock_chat = MockClient.return_value.chat
        mock_chat.side_effect = [
            _mock_empty_ollama_response(_DEGENERATE_GUESS_JSON),
            _mock_empty_ollama_response(_VALID_GUESS_JSON),
        ]
        client = LLMClientLocal("ollama3.2:latest", max_retries=3)

        result = await client.generate(llm_request_cg, expected_format=GuessJSONFormat)

        assert isinstance(result, LLMResponse)
        assert mock_chat.call_count == 2
        assert result.raw_payload["degenerate_response_resamples"] == 1
        # Counted apart: this was not an empty-content draw.
        assert result.raw_payload["empty_response_resamples"] == 0


@pytest.mark.asyncio
async def test_degenerate_empty_proposals_raises_after_cap(llm_request_cg):
    """Bounded: a persistently degenerate model exhausts the budget and still errors out. The domain
    rule (an empty guess list is illegal) is never relaxed."""
    with _patch_ollama() as MockClient:
        mock_chat = MockClient.return_value.chat
        mock_chat.return_value = _mock_empty_ollama_response(_DEGENERATE_GUESS_JSON)
        client = LLMClientLocal("ollama3.2:latest", max_retries=3)

        with pytest.raises(LLMDegenerateResponseError):
            await client.generate(llm_request_cg, expected_format=GuessJSONFormat)
        assert mock_chat.call_count == 4


@pytest.mark.asyncio
async def test_degenerate_not_retried_by_default(llm_request_cg):
    """With the default max_retries=0: one attempt, then raise."""
    with _patch_ollama() as MockClient:
        mock_chat = MockClient.return_value.chat
        mock_chat.return_value = _mock_empty_ollama_response(_DEGENERATE_GUESS_JSON)
        client = LLMClientLocal("ollama3.2:latest")  # no max_retries

        with pytest.raises(LLMDegenerateResponseError):
            await client.generate(llm_request_cg, expected_format=GuessJSONFormat)
        assert mock_chat.call_count == 1


@pytest.mark.asyncio
async def test_a_blank_proposed_word_is_degenerate_and_drawn_again(llm_request_cg):
    """A word of blanks passes the schema's minLength (it is how a model constrained to it writes an
    empty word) but cannot build a GuessProposal: drawn again, as a degenerate draw."""
    blank = json.dumps(
        {
            "reasoning": "r",
            "stop_reason": "done",
            "proposals": [{"word": "OCEAN", "confidence": 0.9}, {"word": " ", "confidence": 0.5}],
        }
    )
    with _patch_ollama() as MockClient:
        mock_chat = MockClient.return_value.chat
        mock_chat.side_effect = [
            _mock_empty_ollama_response(blank),
            _mock_empty_ollama_response(_VALID_GUESS_JSON),
        ]
        client = LLMClientLocal("ollama3.2:latest", max_retries=3)

        result = await client.generate(llm_request_cg, expected_format=GuessJSONFormat)

    assert mock_chat.call_count == 2
    assert result.raw_payload["degenerate_response_resamples"] == 1


@pytest.mark.parametrize(
    "fmt, bad",
    [
        (
            GuessJSONFormat,
            {
                "reasoning": "r",
                "stop_reason": "done",
                "proposals": [{"word": "OCEAN", "confidence": 95}],
            },
        ),
        (
            ConfidenceRankingJSONFormat,
            {"reasoning": "r", "rankings": [{"word": "OCEAN", "confidence": 90}]},
        ),
    ],
)
@pytest.mark.asyncio
async def test_a_confidence_out_of_range_is_malformed_and_drawn_again(llm_request_cg, fmt, bad):
    """A confidence written as a percentage fails the format (Ollama's decoding does not hold the
    model to a number's limits): drawn again with a new seed, in play and in measurement alike,
    instead of ending the game or being clamped into another score."""
    good = {
        GuessJSONFormat: _VALID_GUESS_JSON,
        ConfidenceRankingJSONFormat: json.dumps(
            {"reasoning": "r", "rankings": [{"word": "OCEAN", "confidence": 0.9}]}
        ),
    }[fmt]
    with _patch_ollama() as MockClient:
        mock_chat = MockClient.return_value.chat
        mock_chat.side_effect = [
            _mock_empty_ollama_response(json.dumps(bad)),
            _mock_empty_ollama_response(good),
        ]
        client = LLMClientLocal("ollama3.2:latest", max_retries=3)

        result = await client.generate(llm_request_cg, expected_format=fmt)

    assert mock_chat.call_count == 2
    assert result.raw_payload["malformed_response_resamples"] == 1
    assert json.loads(result.text) == json.loads(good)


@pytest.mark.parametrize(
    "bad_clue,label",
    [
        ({"reasoning": "r", "clue": "   ", "count": 2, "targets": []}, "blank clue"),
        ({"reasoning": "r", "clue": "battle", "count": 0, "targets": []}, "count below 1"),
    ],
)
@pytest.mark.asyncio
async def test_degenerate_clue_is_resampled_then_succeeds(llm_request_cg, bad_clue, label):
    """The clue path has the same gap: ClueJSONFormat accepts a blank clue / count=0, ClueProposal
    rejects both. Covered by the same re-sample."""
    with _patch_ollama() as MockClient:
        mock_chat = MockClient.return_value.chat
        mock_chat.side_effect = [
            _mock_empty_ollama_response(json.dumps(bad_clue)),
            _mock_ollama_response(),
        ]
        client = LLMClientLocal("ollama3.2:latest", max_retries=3)

        result = await client.generate(llm_request_cg, expected_format=ClueJSONFormat)

        assert mock_chat.call_count == 2, label
        assert result.raw_payload["degenerate_response_resamples"] == 1


@pytest.mark.asyncio
async def test_healthy_response_records_zero_degenerate_resamples(llm_request_cg):
    """The common case stays auditable: zero is recorded, not omitted."""
    with _patch_ollama() as MockClient:
        MockClient.return_value.chat.return_value = _mock_ollama_response()
        client = LLMClientLocal("ollama3.2:latest", max_retries=3)

        result = await client.generate(llm_request_cg, expected_format=ClueJSONFormat)

        assert result.raw_payload["degenerate_response_resamples"] == 0


@pytest.mark.asyncio
async def test_empty_rankings_are_not_treated_as_degenerate(llm_request_cg):
    """Measurement rankings are permissive BY DESIGN: an empty ranking must NOT trigger a re-sample,
    or the measurement path would start burning budget on responses it is happy to accept."""
    with _patch_ollama() as MockClient:
        mock_chat = MockClient.return_value.chat
        mock_chat.return_value = _mock_empty_ollama_response(
            json.dumps({"reasoning": "r", "rankings": []})
        )
        client = LLMClientLocal("ollama3.2:latest", max_retries=3)

        result = await client.generate(llm_request_cg, expected_format=ConfidenceRankingJSONFormat)

        assert mock_chat.call_count == 1
        assert result.raw_payload["degenerate_response_resamples"] == 0


# The context window and the answer cap, and an answer cut at that cap (E1 of the test-game report)


@pytest.mark.asyncio
async def test_every_request_carries_the_fixed_context_and_the_answer_cap(llm_request_cg):
    """The context window is set by config, never left to ollama (which picks it from the GPU's
    memory), and the answer is capped at the request's max_tokens."""
    with _patch_ollama() as MockClient:
        MockClient.return_value.chat.return_value = _mock_ollama_response()
        client = LLMClientLocal("ollama3.2:latest")

        await client.generate(llm_request_cg, expected_format=ClueJSONFormat)

        options = MockClient.return_value.chat.call_args.kwargs["options"]
        assert options["num_ctx"] == config.OLLAMA_NUM_CTX
        assert options["num_predict"] == llm_request_cg.max_tokens


def _mock_cut_ollama_response() -> MagicMock:
    """An answer cut at the output limit: a model that looped, its JSON left unfinished."""
    content = "{\"reasoning\": \"Let's try ART. No. Let's try ART. No. Let's try"
    response = _mock_empty_ollama_response(content)
    response.eval_count = 2048
    response.done_reason = "length"
    return response


@pytest.mark.asyncio
async def test_an_answer_cut_at_the_limit_is_drawn_again(llm_request_cg):
    """A cut answer is a bad draw, not malformed output: it is re-sampled within the budget, and
    counted apart from the empty and the degenerate ones."""
    with _patch_ollama() as MockClient:
        mock_chat = MockClient.return_value.chat
        mock_chat.side_effect = [_mock_cut_ollama_response(), _mock_ollama_response()]
        client = LLMClientLocal("ollama3.2:latest", max_retries=3)

        result = await client.generate(llm_request_cg, expected_format=ClueJSONFormat)

        assert mock_chat.call_count == 2
        assert result.raw_payload["truncated_response_resamples"] == 1
        assert result.raw_payload["empty_response_resamples"] == 0
        assert result.raw_payload["degenerate_response_resamples"] == 0


@pytest.mark.asyncio
async def test_an_answer_cut_every_time_still_fails_after_the_budget(llm_request_cg):
    """Bounded: a model that always loops exhausts the budget and the error propagates."""
    with _patch_ollama() as MockClient:
        mock_chat = MockClient.return_value.chat
        mock_chat.return_value = _mock_cut_ollama_response()
        client = LLMClientLocal("ollama3.2:latest", max_retries=3)

        with pytest.raises(LLMTruncatedResponseError):
            await client.generate(llm_request_cg, expected_format=ClueJSONFormat)
        assert mock_chat.call_count == 4


# a bad draw is drawn again with a new seed; a transient re-sends the same request
def _sent_seeds(mock_chat) -> list:
    return [call.kwargs["options"].get("seed") for call in mock_chat.call_args_list]


@pytest.mark.asyncio
async def test_each_new_draw_of_a_bad_answer_has_its_own_seed(llm_request_cg):
    """With the same seed a local model gives the same answer again (gemma4 proposed nothing three
    times in a row, word for word), so each new draw gets the next derived seed, and the answer kept
    carries the seed it was drawn with."""
    request = llm_request_cg.model_copy(update={"seed": 123})
    with _patch_ollama() as MockClient:
        mock_chat = MockClient.return_value.chat
        mock_chat.side_effect = [
            _mock_empty_ollama_response(_DEGENERATE_GUESS_JSON),
            _mock_cut_ollama_response(),
            _mock_empty_ollama_response(_DEGENERATE_GUESS_JSON),
            _mock_empty_ollama_response(_VALID_GUESS_JSON),
        ]
        client = LLMClientLocal("ollama3.2:latest", max_retries=3)

        result = await client.generate(request, expected_format=GuessJSONFormat)

    redraws = [client_module.redraw_seed(123, k) for k in (1, 2, 3)]
    assert _sent_seeds(mock_chat) == [123, *redraws]
    assert len(set(_sent_seeds(mock_chat))) == 4
    assert result.requested_seed == redraws[-1]
    assert result.raw_payload["degenerate_response_resamples"] == 2
    assert result.raw_payload["truncated_response_resamples"] == 1


@pytest.mark.asyncio
async def test_a_transient_failure_after_a_bad_draw_resends_that_draw(llm_request_cg, monkeypatch):
    """A transient failure is the network's, not the model's: the request that failed is sent again
    as it was, the redrawn seed included."""
    monkeypatch.setattr(client_module, "_RETRY_BACKOFF_BASE_S", 0.0)
    request = llm_request_cg.model_copy(update={"seed": 123})
    with _patch_ollama() as MockClient:
        mock_chat = MockClient.return_value.chat
        mock_chat.side_effect = [
            _mock_cut_ollama_response(),
            httpx.ReadTimeout("timed out"),
            _mock_ollama_response(),
        ]
        client = LLMClientLocal("ollama3.2:latest", max_retries=3)

        await client.generate(request, expected_format=ClueJSONFormat)

    redraw = client_module.redraw_seed(123, 1)
    assert _sent_seeds(mock_chat) == [123, redraw, redraw]


@pytest.mark.asyncio
async def test_an_unseeded_request_is_drawn_again_unseeded(llm_request_cg):
    request = llm_request_cg.model_copy(update={"seed": None})
    with _patch_ollama() as MockClient:
        mock_chat = MockClient.return_value.chat
        mock_chat.side_effect = [_mock_cut_ollama_response(), _mock_ollama_response()]
        client = LLMClientLocal("ollama3.2:latest", max_retries=3)

        await client.generate(request, expected_format=ClueJSONFormat)

    assert _sent_seeds(mock_chat) == [None, None]


def test_redraw_seeds_are_deterministic_distinct_and_64_bit():
    seeds = [client_module.redraw_seed(2**64 - 1, k) for k in range(1, 6)]
    assert seeds == [client_module.redraw_seed(2**64 - 1, k) for k in range(1, 6)]
    assert len(set(seeds)) == 5 and 2**64 - 1 not in seeds
    assert all(0 <= s < 2**64 for s in seeds)
    assert client_module.redraw_seed(7, 1) != client_module.redraw_seed(8, 1)
    assert client_module.redraw_seed(None, 1) is None


def test_every_bad_draw_is_a_retriable_parse_error():
    """A bad draw that is not retriable is never drawn again: the game ends on it."""
    for kind in client_module._BAD_DRAWS:
        assert issubclass(kind, LLMParseError) and kind.retriable, kind.__name__


# the JSON of an answer, out of markdown or prose

_CLUE = '{"reasoning": "r", "clue": "battle", "count": 2}'


@pytest.mark.parametrize(
    "answer, expected",
    [
        (_CLUE, _CLUE),
        (f"  {_CLUE}\n", _CLUE),
        (f"```json\n{_CLUE}\n```", _CLUE),
        (f"```JSON\n{_CLUE}\n```", _CLUE),
        (f"```\n{_CLUE}\n```", _CLUE),
        (f"```json {_CLUE}```", _CLUE),
        (f"Here is my clue:\n```json\n{_CLUE}\n```\nGood luck!", _CLUE),
        (f"Here is my clue: {_CLUE} Good luck!", _CLUE),
        ('Note {not json}, then {"clue": "a {b}", "count": 1}', '{"clue": "a {b}", "count": 1}'),
        ("```json\n```", ""),
        ('{"reasoning": "r", "clue": "batt', '{"reasoning": "r", "clue": "batt'),
        ("no JSON here", "no JSON here"),
    ],
    ids=[
        "json",
        "padded",
        "fenced-json",
        "fenced-JSON",
        "fenced-untagged",
        "fenced-one-line",
        "fenced-in-prose",
        "in-prose",
        "first-object",
        "empty-block",
        "cut",
        "no-json",
    ],
)
def test_extract_json(answer, expected):
    assert client_module.extract_json(answer) == expected


@pytest.mark.asyncio
async def test_an_answer_in_an_untagged_code_block_is_read(llm_request_cg):
    """Only ```json blocks used to be unwrapped: an untagged one was a malformed draw."""
    with _patch_ollama() as MockClient:
        MockClient.return_value.chat.side_effect = [
            _mock_empty_ollama_response(f"```\n{_CLUE}\n```")
        ]
        client = LLMClientLocal("ollama3.2:latest")

        result = await client.generate(llm_request_cg, expected_format=ClueJSONFormat)

    assert result.parsed.clue == "battle" and result.text == _CLUE


@pytest.mark.asyncio
async def test_an_answer_without_content_is_an_empty_draw(llm_request_cg):
    """The ollama client types message.content as optional: an answer without one is an empty draw,
    not a TypeError that no retry or error code would know."""
    with _patch_ollama() as MockClient:
        MockClient.return_value.chat.side_effect = [_mock_empty_ollama_response(None)]
        client = LLMClientLocal("ollama3.2:latest")

        with pytest.raises(LLMEmptyResponseError):
            await client.generate(llm_request_cg, expected_format=ClueJSONFormat)
