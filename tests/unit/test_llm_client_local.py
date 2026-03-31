import pytest
from unittest.mock import patch, MagicMock
from ollama import RequestError, ResponseError
from backend.app.core.lm.llm_client_local import LLMClientLocal
from backend.app.models.llm_errors import LLMModelNotProvidedError, LLMRefusalError, LLMParseError


@pytest.mark.asyncio
async def test_llm_client_local_generate_success(llm_request_cg):
    mock_response = MagicMock()
    mock_response.message.content = "{\"clue\": \"battle\", " \
        "\"count\": 3, " \
        "\"reasoning\": \"The word 'battle' captures a military commander ('NAPOLEON')," \
        "the hardware ('RIFLE') and a primary theater of conflict ('RUSSIA')\"" \
        "}"
    mock_response.total_duration = 3200
    mock_response.prompt_eval_count = 10
    mock_response.eval_count = 20
    mock_response.done_reason = "stop"
    mock_response.model_dump_json.return_value = mock_response.message.content

    with patch("backend.app.core.lm.llm_client_local.chat", return_value=mock_response):
        client = LLMClientLocal()

        result = await client.generate(llm_request_cg)

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
    with patch("backend.app.core.lm.llm_client_local.chat",
               side_effect=RequestError("Model not provided")):
        client = LLMClientLocal()

        with pytest.raises(LLMModelNotProvidedError):
            await client.generate(llm_request_cg)


@pytest.mark.asyncio
async def test_llm_client_local_raises_llm_refusal_error(llm_request_cg):
    with patch("backend.app.core.lm.llm_client_local.chat",
               side_effect=ResponseError("LLM refused to generate a response")):
        client = LLMClientLocal()

        with pytest.raises(LLMRefusalError):
            await client.generate(llm_request_cg)


@pytest.mark.asyncio
async def test_llm_client_local_raises_llm_parse_error(llm_request_cg):
    mock_response = MagicMock()
    mock_response.message.content = "This is not a valid JSON response"
    mock_response.model_dump_json.return_value = mock_response.message.content

    with patch("backend.app.core.lm.llm_client_local.chat", return_value=mock_response):
        client = LLMClientLocal()

        with pytest.raises(LLMParseError):
            await client.generate(llm_request_cg)
