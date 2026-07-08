import json
import pytest
from unittest.mock import patch, MagicMock
from ollama import RequestError, ResponseError
from backend.app.core.llm.client_local import LLMClientLocal
from backend.app.models.llm_errors import LLMModelNotProvidedError, LLMRefusalError, LLMParseError
from backend.app.models.llm_schemas import LLMResponse, ClueJSONFormat


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
        assert result.model_used == client.local_model
        assert result.latency_ms == mock_response.total_duration
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
        MockClient.return_value.chat.side_effect = RequestError("Model not provided")
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
        MockClient.return_value.chat.side_effect = ResponseError("LLM refused to generate a response")
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
    mock_content = json.dumps({"reasoning": "test reasoning", "clue": "battle", "count": 3})
    mock_response = MagicMock()
    mock_response.message.content = mock_content
    mock_response.total_duration = 3200
    mock_response.prompt_eval_count = 10
    mock_response.eval_count = 20
    mock_response.done_reason = "stop"
    mock_response.model_dump_json.return_value = json.dumps({"message": {"content": mock_content}})

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
    mock_content = json.dumps({"reasoning": "some reasoning", "clue": "battle"})  # missing "count"
    mock_response = MagicMock()
    mock_response.message.content = mock_content
    mock_response.model_dump_json.return_value = json.dumps({"message": {"content": mock_content}})

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
    mock_content = json.dumps({"reasoning": "test reasoning", "clue": "battle", "count": 3})
    mock_response = MagicMock()
    mock_response.message.content = mock_content
    mock_response.total_duration = 3200
    mock_response.prompt_eval_count = 10
    mock_response.eval_count = 20
    mock_response.done_reason = "stop"
    mock_response.model_dump_json.return_value = json.dumps({"message": {"content": mock_content}})

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
    request = llm_request_cg.model_copy(update={"seed": 99, "temperature": 0.3})

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
