import json
import pytest
from unittest.mock import patch, MagicMock
from ollama import RequestError, ResponseError
from backend.app.core.llm import client as client_module
from backend.app.core.llm.client_local import LLMClientLocal
from backend.app.models.llm_errors import (
    LLMModelNotProvidedError, LLMRefusalError, LLMParseError, LLMTimeoutError,
    LLMEmptyResponseError, LLMDegenerateResponseError,
)
from backend.app.models.llm_schemas import (
    LLMResponse, ClueJSONFormat, GuessJSONFormat, ConfidenceRankingJSONFormat,
)


@pytest.mark.asyncio
async def test_llm_client_local_generate_success(llm_request_cg):
    """
    Tests that the LLMClientLocal correctly processes a valid response from the Ollama client and
    returns the expected LLMResponse.
    """
    mock_content = json.dumps({
        "reasoning": "The word 'battle' captures a military commander ('NAPOLEON'), "
                     "the hardware ('RIFLE') and a primary theater of conflict ('RUSSIA')",
        "clue": "battle",
        "count": 3
    })
    mock_response = MagicMock()
    mock_response.message.content = mock_content
    mock_response.total_duration = 3200
    mock_response.prompt_eval_count = 10
    mock_response.eval_count = 20
    mock_response.done_reason = "stop"
    mock_response.model_dump_json.return_value = json.dumps({
        "message": {"content": mock_content}
    })

    with patch("backend.app.core.llm.client_local.Client") as MockClient:
        MockClient.return_value.chat.return_value = mock_response
        client = LLMClientLocal('ollama3.2:latest')

        result = await client.generate(llm_request_cg, expected_format=ClueJSONFormat)

        assert isinstance(result, LLMResponse)
        assert result.text == mock_response.message.content
        assert result.model_used == client.model_name
        # Ollama reports nanoseconds; latency_ms is milliseconds (ns / 1e6).
        assert result.latency_ms == round(
            mock_response.total_duration / 1_000_000)
        assert result.usage.prompt_tokens == mock_response.prompt_eval_count
        assert result.usage.completion_tokens == mock_response.eval_count
        assert result.usage.total_tokens == mock_response.prompt_eval_count + \
            mock_response.eval_count
        assert result.finish_reason == mock_response.done_reason
        assert result.provider == "ollama"
        assert result.execution_mode == "local"


@pytest.mark.asyncio
async def test_llm_client_local_raises_model_not_provided_error(llm_request_cg):
    """
    Tests that the LLMClientLocal raises an LLMModelNotProvidedError when the Ollama client raises a
    RequestError indicating that the model was not provided.
    """
    with patch("backend.app.core.llm.client_local.Client") as MockClient:
        MockClient.return_value.chat.side_effect = RequestError(
            "Model not provided")
        client = LLMClientLocal(None)

        with pytest.raises(LLMModelNotProvidedError):
            await client.generate(llm_request_cg)


@pytest.mark.asyncio
async def test_llm_client_local_raises_llm_refusal_error(llm_request_cg):
    """
    Tests that the LLMClientLocal raises an LLMRefusalError when the Ollama client raises a
    ResponseError indicating that the LLM refused to generate a response.
    """
    with patch("backend.app.core.llm.client_local.Client") as MockClient:
        MockClient.return_value.chat.side_effect = ResponseError(
            "LLM refused to generate a response")
        client = LLMClientLocal('ollama3.2:latest')

        with pytest.raises(LLMRefusalError):
            await client.generate(llm_request_cg)


@pytest.mark.asyncio
async def test_llm_client_local_raises_llm_parse_error(llm_request_cg):
    """
    Tests that the LLMClientLocal raises an LLMParseError when the Ollama client returns a response
    that cannot be parsed as valid JSON.
    """
    mock_response = MagicMock()
    mock_response.message.content = "This is not a valid JSON response"
    mock_response.model_dump_json.return_value = mock_response.message.content

    with patch("backend.app.core.llm.client_local.Client") as MockClient:
        MockClient.return_value.chat.return_value = mock_response
        client = LLMClientLocal('ollama3.2:latest')

        with pytest.raises(LLMParseError):
            await client.generate(llm_request_cg)


@pytest.mark.asyncio
async def test_llm_client_local_generate_think_false(llm_request_cg):
    """
    Tests that when think=False, the chat() call is made without the think parameter.
    """
    mock_content = json.dumps(
        {"reasoning": "test reasoning", "clue": "battle", "count": 3})
    mock_response = MagicMock()
    mock_response.message.content = mock_content
    mock_response.total_duration = 3200
    mock_response.prompt_eval_count = 10
    mock_response.eval_count = 20
    mock_response.done_reason = "stop"
    mock_response.model_dump_json.return_value = json.dumps(
        {"message": {"content": mock_content}})

    with patch("backend.app.core.llm.client_local.Client") as MockClient:
        mock_chat = MockClient.return_value.chat
        mock_chat.return_value = mock_response
        client = LLMClientLocal('ollama3.2:latest', think=False)

        result = await client.generate(llm_request_cg, expected_format=ClueJSONFormat)

        assert isinstance(result, LLMResponse)
        _, kwargs = mock_chat.call_args
        assert "think" not in kwargs


@pytest.mark.asyncio
async def test_llm_client_local_raises_parse_error_on_schema_mismatch(llm_request_cg):
    """
    Tests that the LLMClientLocal raises an LLMParseError when the response is valid JSON
    but does not match the expected schema (missing required fields).
    """
    mock_content = json.dumps(
        {"reasoning": "some reasoning", "clue": "battle"})  # missing "count"
    mock_response = MagicMock()
    mock_response.message.content = mock_content
    mock_response.model_dump_json.return_value = json.dumps(
        {"message": {"content": mock_content}})

    with patch("backend.app.core.llm.client_local.Client") as MockClient:
        MockClient.return_value.chat.return_value = mock_response
        client = LLMClientLocal('ollama3.2:latest')

        with pytest.raises(LLMParseError):
            await client.generate(llm_request_cg, expected_format=ClueJSONFormat)


@pytest.mark.asyncio
async def test_llm_client_local_generate_without_expected_format(llm_request_cg):
    """
    Tests that generate() succeeds without an expected_format, skipping schema validation
    and using format="json" as the fallback.
    """
    mock_content = json.dumps(
        {"reasoning": "test reasoning", "clue": "battle", "count": 3})
    mock_response = MagicMock()
    mock_response.message.content = mock_content
    mock_response.total_duration = 3200
    mock_response.prompt_eval_count = 10
    mock_response.eval_count = 20
    mock_response.done_reason = "stop"
    mock_response.model_dump_json.return_value = json.dumps(
        {"message": {"content": mock_content}})

    with patch("backend.app.core.llm.client_local.Client") as MockClient:
        mock_chat = MockClient.return_value.chat
        mock_chat.return_value = mock_response
        client = LLMClientLocal('ollama3.2:latest')

        result = await client.generate(llm_request_cg)

        assert isinstance(result, LLMResponse)
        assert result.text == mock_content
        _, kwargs = mock_chat.call_args
        assert kwargs.get("format") == "json"


def _mock_ollama_response(model: str | None = None) -> MagicMock:
    """Builds a MagicMock mimicking a successful Ollama chat response for the tests below."""
    mock_content = json.dumps(
        {"reasoning": "test reasoning", "clue": "battle", "count": 3})
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

    with patch("backend.app.core.llm.client_local.Client") as MockClient:
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

    with patch("backend.app.core.llm.client_local.Client") as MockClient:
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
    with patch("backend.app.core.llm.client_local.Client") as MockClient:
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
    request = llm_request_cg.model_copy(
        update={"seed": 99, "temperature": 0.3})

    with patch("backend.app.core.llm.client_local.Client") as MockClient:
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
    with patch("backend.app.core.llm.client_local.Client") as MockClient:
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

    with patch("backend.app.core.llm.client_local.Client") as MockClient:
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

    with patch("backend.app.core.llm.client_local.Client") as MockClient:
        MockClient.return_value.chat.return_value = mock_response
        client = LLMClientLocal("ollama3.2:latest")

        result = await client.generate(llm_request_cg, expected_format=ClueJSONFormat)

        assert result.latency_ms == 0


# transient retry
@pytest.mark.asyncio
async def test_llm_client_local_non_retriable_raises_immediately(llm_request_cg):
    """A non-retriable error (refusal) raises on the first attempt even with a retry budget:
    exactly one chat() call."""
    with patch("backend.app.core.llm.client_local.Client") as MockClient:
        mock_chat = MockClient.return_value.chat
        mock_chat.side_effect = ResponseError("refused")
        client = LLMClientLocal("ollama3.2:latest", max_retries=3)

        with pytest.raises(LLMRefusalError):
            await client.generate(llm_request_cg, expected_format=ClueJSONFormat)
        assert mock_chat.call_count == 1


@pytest.mark.asyncio
async def test_llm_client_local_default_no_retry(llm_request_cg):
    """max_retries defaults to 0 (the interactive path): a single attempt."""
    with patch("backend.app.core.llm.client_local.Client") as MockClient:
        mock_chat = MockClient.return_value.chat
        mock_chat.side_effect = ResponseError("refused")
        client = LLMClientLocal("ollama3.2:latest")  # no max_retries

        with pytest.raises(LLMRefusalError):
            await client.generate(llm_request_cg, expected_format=ClueJSONFormat)
        assert mock_chat.call_count == 1


@pytest.mark.asyncio
async def test_llm_client_local_retries_retriable_then_succeeds(llm_request_cg, monkeypatch):
    """Apart from the empty-response case (covered below), the local client's own mapping yields no
    retriable error, so we inject a retriable mapped
    error (LLMTimeoutError) at the provider boundary to exercise the retry wiring: it propagates
    uncaught through _generate_once and is retried by generate_with_retries. generate() returns after
    exactly 3 attempts, each re-sending the identical request (same seed) - no reseed."""
    monkeypatch.setattr(
        client_module, "_RETRY_BACKOFF_BASE_S", 0.0)  # no real sleeping
    request = llm_request_cg.model_copy(update={"seed": 123})

    with patch("backend.app.core.llm.client_local.Client") as MockClient:
        mock_chat = MockClient.return_value.chat
        mock_chat.side_effect = [
            LLMTimeoutError(), LLMTimeoutError(), _mock_ollama_response()]
        client = LLMClientLocal("ollama3.2:latest", max_retries=2)

        result = await client.generate(request, expected_format=ClueJSONFormat)

        assert isinstance(result, LLMResponse)
        assert mock_chat.call_count == 3
        seeds = [call.kwargs["options"]["seed"]
                 for call in mock_chat.call_args_list]
        assert seeds == [123, 123, 123]  # same request re-sent, never reseeded


# empty-response re-sample
def _mock_empty_ollama_response(content: str = "") -> MagicMock:
    """Builds a MagicMock mimicking the intermittent empty ollama chat response."""
    mock_response = MagicMock()
    mock_response.message.content = content
    mock_response.total_duration = 3200
    mock_response.prompt_eval_count = 10
    mock_response.eval_count = 0
    mock_response.done_reason = "stop"
    mock_response.model_dump_json.return_value = json.dumps(
        {"message": {"content": content}})
    return mock_response


@pytest.mark.parametrize("empty_content", ["", "   \n\t "])
@pytest.mark.asyncio
async def test_llm_client_local_empty_response_resampled_then_succeeds(
        llm_request_cg, empty_content):
    """An empty (or whitespace-only) draw is re-sampled: attempt 1 returns '', attempt 2 returns
    valid JSON, and generate() succeeds after exactly 2 attempts. The number of empty draws absorbed
    is stamped on raw_payload for downstream audit."""
    request = llm_request_cg.model_copy(update={"seed": 123})

    with patch("backend.app.core.llm.client_local.Client") as MockClient:
        mock_chat = MockClient.return_value.chat
        mock_chat.side_effect = [
            _mock_empty_ollama_response(empty_content), _mock_ollama_response()]
        client = LLMClientLocal("ollama3.2:latest", max_retries=3)

        result = await client.generate(request, expected_format=ClueJSONFormat)

        assert isinstance(result, LLMResponse)
        assert mock_chat.call_count == 2
        assert result.raw_payload[client_module.EMPTY_RESAMPLE_KEY] == 1
        # The identical request is re-sent - the re-sample must never reseed.
        seeds = [call.kwargs["options"]["seed"]
                 for call in mock_chat.call_args_list]
        assert seeds == [123, 123]


@pytest.mark.asyncio
async def test_llm_client_local_empty_response_raises_after_cap(llm_request_cg):
    """A persistently empty response is NOT retried forever: it exhausts the budget (max_retries=3
    -> 4 attempts) and then raises."""
    with patch("backend.app.core.llm.client_local.Client") as MockClient:
        mock_chat = MockClient.return_value.chat
        mock_chat.return_value = _mock_empty_ollama_response()
        client = LLMClientLocal("ollama3.2:latest", max_retries=3)

        with pytest.raises(LLMEmptyResponseError):
            await client.generate(request=llm_request_cg, expected_format=ClueJSONFormat)
        assert mock_chat.call_count == 4


@pytest.mark.asyncio
async def test_llm_client_local_empty_response_not_retried_interactively(llm_request_cg):
    """The interactive path (max_retries=0) is unchanged: an empty response raises on attempt 1."""
    with patch("backend.app.core.llm.client_local.Client") as MockClient:
        mock_chat = MockClient.return_value.chat
        mock_chat.return_value = _mock_empty_ollama_response()
        client = LLMClientLocal("ollama3.2:latest")  # no max_retries

        with pytest.raises(LLMEmptyResponseError):
            await client.generate(llm_request_cg, expected_format=ClueJSONFormat)
        assert mock_chat.call_count == 1


@pytest.mark.asyncio
async def test_llm_client_local_malformed_nonempty_stays_non_retriable(llm_request_cg):
    """The retry is scoped to the empty case only. Genuinely malformed non-empty output remains a
    non-retriable LLMParseError and raises on the first attempt even with a full retry budget -
    re-sending it would only reproduce the same bad parse."""
    with patch("backend.app.core.llm.client_local.Client") as MockClient:
        mock_chat = MockClient.return_value.chat
        # Well-formed JSON, but missing the fields ClueJSONFormat requires.
        mock_chat.return_value = _mock_empty_ollama_response(
            '{"unexpected": "shape"}')
        client = LLMClientLocal("ollama3.2:latest", max_retries=3)

        with pytest.raises(LLMParseError) as excinfo:
            await client.generate(llm_request_cg, expected_format=ClueJSONFormat)
        assert not isinstance(excinfo.value, LLMEmptyResponseError)
        assert excinfo.value.retriable is False
        assert mock_chat.call_count == 1


@pytest.mark.asyncio
async def test_llm_client_local_no_empty_draws_records_zero(llm_request_cg):
    """The common case is still auditable: zero empty draws is recorded as 0, not omitted."""
    with patch("backend.app.core.llm.client_local.Client") as MockClient:
        MockClient.return_value.chat.return_value = _mock_ollama_response()
        client = LLMClientLocal("ollama3.2:latest", max_retries=3)

        result = await client.generate(llm_request_cg, expected_format=ClueJSONFormat)

        assert result.raw_payload[client_module.EMPTY_RESAMPLE_KEY] == 0


# thinking-capability gate
def _patch_client_with_capabilities(MockClient, capabilities):
    """Wire the mocked ollama Client so /api/show reports ``capabilities`` and chat() succeeds.

    ``capabilities=None`` makes the probe raise, standing in for an unreachable daemon.
    """
    instance = MockClient.return_value
    if capabilities is None:
        instance.show.side_effect = ConnectionError("daemon unreachable")
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
    with patch("backend.app.core.llm.client_local.Client") as MockClient:
        instance = _patch_client_with_capabilities(
            MockClient, ["completion", "vision", "tools", "thinking"])
        client = LLMClientLocal("gemma4:12b", think=False, max_retries=3)

        await client.generate(llm_request_cg, expected_format=ClueJSONFormat)

        _, kwargs = instance.chat.call_args
        assert "think" in kwargs, "a thinking-capable model must receive an explicit `think`"
        assert kwargs["think"] is False


@pytest.mark.asyncio
async def test_non_thinking_model_is_not_sent_think(llm_request_cg):
    """A model that does not advertise `thinking` must NOT receive the flag at all: ollama rejects
    think=True for such models (HTTP 400 'does not support thinking')."""
    with patch("backend.app.core.llm.client_local.Client") as MockClient:
        instance = _patch_client_with_capabilities(
            MockClient, ["completion", "tools"])
        client = LLMClientLocal("llama3.1:8b", think=False, max_retries=3)

        await client.generate(llm_request_cg, expected_format=ClueJSONFormat)

        _, kwargs = instance.chat.call_args
        assert "think" not in kwargs


@pytest.mark.asyncio
async def test_thinking_gate_applies_to_guess_calls_too(llm_request_cg):
    """The gate lives in _generate_once, so it covers guess/ranking calls, not just clues."""
    with patch("backend.app.core.llm.client_local.Client") as MockClient:
        instance = _patch_client_with_capabilities(
            MockClient, ["completion", "thinking"])
        instance.chat.return_value = _mock_empty_ollama_response(json.dumps({
            "reasoning": "r", "stop_reason": "done",
            "proposals": [{"word": "OCEAN", "confidence": 0.9}],
        }))
        client = LLMClientLocal("gemma4:12b", think=False, max_retries=3)

        await client.generate(llm_request_cg, expected_format=GuessJSONFormat)

        _, kwargs = instance.chat.call_args
        assert kwargs["think"] is False


@pytest.mark.asyncio
async def test_think_true_is_still_honoured_for_capable_model(llm_request_cg):
    """The gate decides whether the flag is SENT, not what it says: an explicit think=True on a
    capable model still enables reasoning."""
    with patch("backend.app.core.llm.client_local.Client") as MockClient:
        instance = _patch_client_with_capabilities(
            MockClient, ["completion", "thinking"])
        client = LLMClientLocal("gemma4:12b", think=True, max_retries=3)

        await client.generate(llm_request_cg, expected_format=ClueJSONFormat)

        _, kwargs = instance.chat.call_args
        assert kwargs["think"] is True


@pytest.mark.asyncio
async def test_capability_probe_failure_degrades_to_omitting_think(llm_request_cg):
    """An unreachable daemon must not break generation: the probe degrades to omitting `think`."""
    with patch("backend.app.core.llm.client_local.Client") as MockClient:
        instance = _patch_client_with_capabilities(MockClient, None)
        client = LLMClientLocal("gemma4:12b", think=False, max_retries=3)

        result = await client.generate(llm_request_cg, expected_format=ClueJSONFormat)

        assert isinstance(result, LLMResponse)
        _, kwargs = instance.chat.call_args
        assert "think" not in kwargs


@pytest.mark.asyncio
async def test_capability_probe_is_cached_across_calls(llm_request_cg):
    """The probe costs one /api/show per client, not one per generate()."""
    with patch("backend.app.core.llm.client_local.Client") as MockClient:
        instance = _patch_client_with_capabilities(
            MockClient, ["completion", "thinking"])
        client = LLMClientLocal("gemma4:12b", think=False, max_retries=3)

        await client.generate(llm_request_cg, expected_format=ClueJSONFormat)
        await client.generate(llm_request_cg, expected_format=ClueJSONFormat)

        assert instance.show.call_count == 1
        assert instance.chat.call_count == 2


# degenerate (well-formed but unusable) re-sample
_VALID_GUESS_JSON = json.dumps({
    "reasoning": "marine words", "stop_reason": "done",
    "proposals": [{"word": "OCEAN", "confidence": 0.9}],
})
# violates GuessProposal.proposals (min_length=1).
_DEGENERATE_GUESS_JSON = json.dumps({
    "reasoning": "nothing clears my confidence threshold", "stop_reason": "risk avoidance",
    "proposals": [],
})


@pytest.mark.asyncio
async def test_degenerate_empty_proposals_is_resampled_then_succeeds(llm_request_cg):
    """An empty proposals list parses fine but cannot build a GuessProposal. It must be re-sampled
    by the SAME bounded machinery, and counted separately from empty-content draws."""
    with patch("backend.app.core.llm.client_local.Client") as MockClient:
        mock_chat = MockClient.return_value.chat
        mock_chat.side_effect = [
            _mock_empty_ollama_response(_DEGENERATE_GUESS_JSON),
            _mock_empty_ollama_response(_VALID_GUESS_JSON),
        ]
        client = LLMClientLocal("ollama3.2:latest", max_retries=3)

        result = await client.generate(llm_request_cg, expected_format=GuessJSONFormat)

        assert isinstance(result, LLMResponse)
        assert mock_chat.call_count == 2
        assert result.raw_payload[client_module.DEGENERATE_RESAMPLE_KEY] == 1
        # Counted apart: this was not an empty-content draw.
        assert result.raw_payload[client_module.EMPTY_RESAMPLE_KEY] == 0


@pytest.mark.asyncio
async def test_degenerate_empty_proposals_raises_after_cap(llm_request_cg):
    """Bounded: a persistently degenerate model exhausts the budget and still errors out. The domain
    rule (an empty guess list is illegal) is never relaxed."""
    with patch("backend.app.core.llm.client_local.Client") as MockClient:
        mock_chat = MockClient.return_value.chat
        mock_chat.return_value = _mock_empty_ollama_response(
            _DEGENERATE_GUESS_JSON)
        client = LLMClientLocal("ollama3.2:latest", max_retries=3)

        with pytest.raises(LLMDegenerateResponseError):
            await client.generate(llm_request_cg, expected_format=GuessJSONFormat)
        assert mock_chat.call_count == 4


@pytest.mark.asyncio
async def test_degenerate_not_retried_interactively(llm_request_cg):
    """The interactive path (max_retries=0) is unchanged: one attempt, then raise."""
    with patch("backend.app.core.llm.client_local.Client") as MockClient:
        mock_chat = MockClient.return_value.chat
        mock_chat.return_value = _mock_empty_ollama_response(
            _DEGENERATE_GUESS_JSON)
        client = LLMClientLocal("ollama3.2:latest")  # no max_retries

        with pytest.raises(LLMDegenerateResponseError):
            await client.generate(llm_request_cg, expected_format=GuessJSONFormat)
        assert mock_chat.call_count == 1


@pytest.mark.parametrize("bad_clue,label", [
    ({"reasoning": "r", "clue": "   ", "count": 2, "targets": []}, "blank clue"),
    ({"reasoning": "r", "clue": "battle", "count": 0, "targets": []}, "count below 1"),
])
@pytest.mark.asyncio
async def test_degenerate_clue_is_resampled_then_succeeds(llm_request_cg, bad_clue, label):
    """The clue path has the same gap: ClueJSONFormat accepts a blank clue / count=0, ClueProposal
    rejects both. Covered by the same re-sample."""
    with patch("backend.app.core.llm.client_local.Client") as MockClient:
        mock_chat = MockClient.return_value.chat
        mock_chat.side_effect = [
            _mock_empty_ollama_response(json.dumps(bad_clue)),
            _mock_ollama_response(),
        ]
        client = LLMClientLocal("ollama3.2:latest", max_retries=3)

        result = await client.generate(llm_request_cg, expected_format=ClueJSONFormat)

        assert mock_chat.call_count == 2, label
        assert result.raw_payload[client_module.DEGENERATE_RESAMPLE_KEY] == 1


@pytest.mark.asyncio
async def test_healthy_response_records_zero_degenerate_resamples(llm_request_cg):
    """The common case stays auditable: zero is recorded, not omitted."""
    with patch("backend.app.core.llm.client_local.Client") as MockClient:
        MockClient.return_value.chat.return_value = _mock_ollama_response()
        client = LLMClientLocal("ollama3.2:latest", max_retries=3)

        result = await client.generate(llm_request_cg, expected_format=ClueJSONFormat)

        assert result.raw_payload[client_module.DEGENERATE_RESAMPLE_KEY] == 0


@pytest.mark.asyncio
async def test_empty_rankings_are_not_treated_as_degenerate(llm_request_cg):
    """Measurement rankings are permissive BY DESIGN: an empty ranking must NOT trigger a re-sample,
    or the measurement path would start burning budget on responses it is happy to accept."""
    with patch("backend.app.core.llm.client_local.Client") as MockClient:
        mock_chat = MockClient.return_value.chat
        mock_chat.return_value = _mock_empty_ollama_response(
            json.dumps({"reasoning": "r", "rankings": []}))
        client = LLMClientLocal("ollama3.2:latest", max_retries=3)

        result = await client.generate(
            llm_request_cg, expected_format=ConfidenceRankingJSONFormat)

        assert mock_chat.call_count == 1
        assert result.raw_payload[client_module.DEGENERATE_RESAMPLE_KEY] == 0
