"""The OpenRouter client against OpenRouter's answers (tests/fixtures/openrouter_responses.json),
served through the real openai SDK over a mock HTTP transport: the SDK parses each body, and turns
each error status into its exception, as it would against OpenRouter."""

import contextlib
import copy
import json
import logging
from typing import Any

import httpx
import pytest
from openai import AsyncOpenAI
from unittest.mock import patch
from backend.app import config
from backend.app.core.llm import client as client_module
from backend.app.core.llm.client_openrouter import LLMClientOpenRouter, provider_routing
from backend.app.models.llm_errors import (
    LLMAuthError,
    LLMEmptyResponseError,
    LLMInsufficientCreditsError,
    LLMInvalidRequestError,
    LLMMalformedResponseError,
    LLMProviderUnavailableError,
    LLMRateLimitError,
    LLMRefusalError,
    LLMTimeoutError,
)
from backend.app.models.llm_schemas import LLMResponse, ClueJSONFormat, GuessJSONFormat
from backend.app.paths import REPO_ROOT

_FIXTURES = json.loads(
    (REPO_ROOT / "tests" / "fixtures" / "openrouter_responses.json").read_text(encoding="utf-8")
)

_GUESS = {
    "reasoning": "r",
    "stop_reason": "done",
    "proposals": [{"word": "apple", "confidence": 1}],
}

# An HTTP answer (status, JSON body), or an exception the transport raises instead of answering.
Answer = tuple[int, dict[str, Any]] | Exception


def _fixture(name: str) -> tuple[int, dict[str, Any]]:
    case = copy.deepcopy(_FIXTURES[name])
    return case["status"], case["body"]


def _completion(content: Any = ..., **choice: Any) -> tuple[int, dict[str, Any]]:
    """The clue completion, with its first choice's content and other fields replaced as given."""
    status, body = _fixture("clue")
    if content is not ...:
        body["choices"][0]["message"]["content"] = content
    body["choices"][0].update(choice)
    return status, body


@contextlib.contextmanager
def _answering(*answers: Answer):
    """The client's AsyncOpenAI answers its calls with ``answers``, in order. Yields the JSON bodies
    of the requests sent."""
    sent: list[dict[str, Any]] = []
    queue = iter(answers)

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        answer = next(queue)
        if isinstance(answer, Exception):
            raise answer
        status, body = answer
        return httpx.Response(status, json=body)

    def build(**kwargs: Any) -> AsyncOpenAI:
        transport = httpx.MockTransport(handler)
        return AsyncOpenAI(**kwargs, http_client=httpx.AsyncClient(transport=transport))

    with patch("backend.app.core.llm.client_openrouter.AsyncOpenAI", side_effect=build):
        yield sent


def _timeout() -> httpx.ReadTimeout:
    """The transport times out: the SDK raises APITimeoutError, mapped to LLMTimeoutError."""
    return httpx.ReadTimeout("timed out")


@pytest.fixture(autouse=True)
def api_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")


@pytest.fixture(autouse=True)
def route(monkeypatch):
    """The test model, "m", pinned to a provider as config must pin every OpenRouter model."""
    monkeypatch.setitem(config.llm_models, "OpenRouter", {"m": {"provider": "DeepInfra"}})


@pytest.fixture
def no_backoff(monkeypatch):
    monkeypatch.setattr(client_module, "_RETRY_BACKOFF_BASE_S", 0.0)


def test_a_model_pinned_to_no_provider_is_refused(monkeypatch):
    """OpenRouter could serve each of its calls from another provider: the client is not built."""
    monkeypatch.setitem(config.llm_models, "OpenRouter", {"m": {}})

    assert provider_routing("m") is None
    with pytest.raises(LLMInvalidRequestError, match="no provider pinned"):
        LLMClientOpenRouter("m")


def test_a_client_without_a_key_is_an_auth_error(monkeypatch):
    """The SDK would refuse to build it with an error of its own, outside the taxonomy."""
    monkeypatch.delenv("OPENROUTER_API_KEY")

    with pytest.raises(LLMAuthError):
        LLMClientOpenRouter("m")


# the request


async def test_the_request_carries_the_seed_the_temperature_and_the_schema(llm_request_cg):
    """The schema is sent as ollama gets it, not strict: OpenAI's strict mode needs every field
    required and no other allowed, which the formats do not say."""
    request = llm_request_cg.model_copy(update={"seed": 321})

    with _answering(_completion()) as sent:
        await LLMClientOpenRouter("m").generate(request, expected_format=ClueJSONFormat)

    assert sent[0]["seed"] == 321
    assert sent[0]["temperature"] == request.temperature
    assert sent[0]["provider"] == {
        "order": ["DeepInfra"],
        "allow_fallbacks": False,
        "require_parameters": True,
    }
    assert sent[0]["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "ClueJSONFormat",
            "schema": ClueJSONFormat.model_json_schema(),
            "strict": False,
        },
    }


async def test_the_request_pins_the_quantization_config_names(llm_request_cg, monkeypatch):
    route = {"m": {"provider": "DeepInfra", "quantization": "fp8"}}
    monkeypatch.setitem(config.llm_models, "OpenRouter", route)

    with _answering(_completion()) as sent:
        await LLMClientOpenRouter("m").generate(llm_request_cg, expected_format=ClueJSONFormat)

    assert sent[0]["provider"]["order"] == ["DeepInfra"]
    assert sent[0]["provider"]["quantizations"] == ["fp8"]


async def test_the_seed_recorded_is_the_seed_sent(llm_request_cg):
    """OpenRouter takes a 32-bit signed seed: the request's is masked, and the call records the seed
    the provider was sent, not the one it was not."""
    request = llm_request_cg.model_copy(update={"seed": 2**40 + 5})

    with _answering(_completion()) as sent:
        result = await LLMClientOpenRouter("m").generate(request, expected_format=ClueJSONFormat)

    assert sent[0]["seed"] == 5 and result.requested_seed == 5


# the answer


async def test_an_answer_populates_the_telemetry(llm_request_cg):
    request = llm_request_cg.model_copy(update={"seed": 7, "temperature": 0.5})

    with _answering(_completion()):
        result = await LLMClientOpenRouter("m").generate(request, expected_format=ClueJSONFormat)

    assert isinstance(result, LLMResponse)
    assert result.parsed.clue == "battle" and result.parsed.targets == ["war", "army"]
    assert (result.requested_temperature, result.requested_seed) == (0.5, 7)
    assert result.system_fingerprint == "fp_6f2b9a4c1d"
    assert result.resolved_model == "mistralai/mistral-small-3.2-24b-instruct"
    assert result.request_id == "gen-1761130000-k2mQ7vXwYpR4sT8uLz3n"
    assert (result.usage.prompt_tokens, result.usage.completion_tokens) == (812, 41)
    # Who served the call is in the payload kept (see LLM-06 f).
    assert result.raw_payload["provider"] == "DeepInfra"


async def test_an_answer_without_usage_records_unknown_counts(llm_request_cg):
    """A response without a usage block records the token counts as unknown (None), not as 0."""
    status, body = _completion()
    del body["usage"]

    with _answering((status, body)):
        result = await LLMClientOpenRouter("m").generate(
            llm_request_cg, expected_format=ClueJSONFormat
        )

    assert result.usage.prompt_tokens is None
    assert result.usage.completion_tokens is None
    assert result.usage.total_tokens is None


async def test_an_answer_without_a_system_fingerprint_records_none(llm_request_cg):
    status, body = _completion()
    del body["system_fingerprint"]

    with _answering((status, body)):
        result = await LLMClientOpenRouter("m").generate(
            llm_request_cg, expected_format=ClueJSONFormat
        )

    assert result.system_fingerprint is None


async def test_the_json_is_read_out_of_a_code_block(llm_request_cg):
    """A provider that does not honour response_format may wrap the answer in markdown and prose:
    its JSON is read, not drawn again as malformed."""
    clue = _completion()[1]["choices"][0]["message"]["content"]

    with _answering(_completion(f"Here it is:\n```json\n{clue}\n```")):
        result = await LLMClientOpenRouter("m").generate(
            llm_request_cg, expected_format=ClueJSONFormat
        )

    assert result.parsed.clue == "battle" and result.text == clue


# bad draws


@pytest.mark.parametrize("content", ["", "   ", None])
async def test_an_empty_answer_is_an_empty_draw(llm_request_cg, content):
    """It used to fail validation as a malformed one: the same bad draw, counted as another."""
    with _answering(_completion(content)):
        with pytest.raises(LLMEmptyResponseError) as excinfo:
            await LLMClientOpenRouter("m").generate(llm_request_cg, expected_format=ClueJSONFormat)

    assert excinfo.value.retriable and excinfo.value.raw_payload["provider"] == "DeepInfra"


async def test_a_malformed_answer_is_drawn_again_and_counted(llm_request_cg):
    """An answer that does not match the schema is a bad draw: drawn again with the next derived
    seed, and counted on the answer kept."""
    request = llm_request_cg.model_copy(update={"seed": 321})

    with _answering(_completion("not valid json for the schema"), _completion()) as sent:
        client = LLMClientOpenRouter("m", max_retries=2)
        result = await client.generate(request, expected_format=ClueJSONFormat)

    assert result.raw_payload["malformed_response_resamples"] == 1
    assert [body["seed"] for body in sent] == [321, client_module.redraw_seed(321, 1) & 0x7FFFFFFF]


async def test_a_malformed_answer_keeps_the_answer(llm_request_cg, caplog):
    """A malformed answer carries the provider's payload and logs the whole answer at WARNING: only
    its count reaches the database, and the batch logs at INFO."""
    with _answering(_completion("not valid json for the schema")):
        with caplog.at_level(logging.WARNING, logger="backend.app.core.llm.client"):
            with pytest.raises(LLMMalformedResponseError) as excinfo:
                await LLMClientOpenRouter("m").generate(
                    llm_request_cg, expected_format=ClueJSONFormat
                )

    assert excinfo.value.raw_payload["id"] == "gen-1761130000-k2mQ7vXwYpR4sT8uLz3n"
    assert "not valid json for the schema" in caplog.text


async def test_a_degenerate_answer_is_drawn_again_and_counted(llm_request_cg):
    """A well-formed answer that cannot build a domain object (an empty proposals list) is a bad
    draw, as from the local client: drawn again with the next derived seed and counted on the
    answer kept, instead of failing in the service, outside the retry loop."""
    request = llm_request_cg.model_copy(update={"seed": 321})
    degenerate = json.dumps({**_GUESS, "proposals": []})

    with _answering(_completion(degenerate), _completion(json.dumps(_GUESS))) as sent:
        client = LLMClientOpenRouter("m", max_retries=2)
        result = await client.generate(request, expected_format=GuessJSONFormat)

    assert [p.word for p in result.parsed.proposals] == ["apple"]
    assert result.raw_payload["degenerate_response_resamples"] == 1
    assert [body["seed"] for body in sent] == [321, client_module.redraw_seed(321, 1) & 0x7FFFFFFF]


async def test_a_cut_answer_is_drawn_again_with_a_new_seed(llm_request_cg):
    """A cut answer is a bad draw: unlike a transient, it is drawn again with the next derived seed
    (masked to the 31 bits OpenRouter takes)."""
    request = llm_request_cg.model_copy(update={"seed": 321})

    with _answering(
        _completion('{"reasoning": "r", "cl', finish_reason="length"), _completion()
    ) as sent:
        client = LLMClientOpenRouter("m", max_retries=2)
        result = await client.generate(request, expected_format=ClueJSONFormat)

    assert result.raw_payload["truncated_response_resamples"] == 1
    assert [body["seed"] for body in sent] == [321, client_module.redraw_seed(321, 1) & 0x7FFFFFFF]


# refusals and errors


async def test_a_refusal_is_the_models_and_is_not_drawn_again(llm_request_cg):
    """A model that refuses under structured output says so in message.refusal, with no content: a
    refusal (counted as the model's behaviour), not an empty draw."""
    with _answering(_fixture("refusal")) as sent:
        with pytest.raises(LLMRefusalError) as excinfo:
            client = LLMClientOpenRouter("m", max_retries=3)
            await client.generate(llm_request_cg, expected_format=ClueJSONFormat)

    assert len(sent) == 1
    assert "I can't help with that request" in excinfo.value.message


@pytest.mark.parametrize("name", ["error_in_body", "error_in_choice"])
async def test_an_error_in_a_200_is_mapped_by_its_code(llm_request_cg, name):
    """Once the response has started, OpenRouter reports a failure in a 200: in place of the
    choices, or in the choice. Its code says what failed (here the upstream provider: retriable),
    not a missing choice or an empty answer."""
    with _answering(_fixture(name)):
        with pytest.raises(LLMProviderUnavailableError) as excinfo:
            await LLMClientOpenRouter("m").generate(llm_request_cg, expected_format=ClueJSONFormat)

    assert excinfo.value.http_status == 502 and excinfo.value.retriable
    assert "OpenRouter reported an error" in excinfo.value.message


@pytest.mark.parametrize(
    "name, error",
    [
        ("bad_request", LLMInvalidRequestError),
        ("invalid_key", LLMAuthError),
        ("insufficient_credits", LLMInsufficientCreditsError),
        ("moderation", LLMRefusalError),
        ("timed_out", LLMTimeoutError),
        ("rate_limited", LLMRateLimitError),
        ("model_down", LLMProviderUnavailableError),
        ("no_provider", LLMProviderUnavailableError),
    ],
)
async def test_an_error_status_is_mapped_by_its_code(llm_request_cg, name, error):
    """Every status OpenRouter documents, as its taxonomy error; and one request for each: the SDK,
    which by default retries a 408, a 429 and a 5xx twice, retries nothing (the client's retry is
    the only one, so every attempt is counted)."""
    status, body = _fixture(name)

    with _answering((status, body)) as sent:
        with pytest.raises(error) as excinfo:
            await LLMClientOpenRouter("m").generate(llm_request_cg, expected_format=ClueJSONFormat)

    assert type(excinfo.value) is error
    assert excinfo.value.http_status == status
    assert len(sent) == 1


# the retry of transients


async def test_a_transient_is_retried_with_the_same_request(llm_request_cg, no_backoff):
    """A retriable provider error (timeout) twice, then success: the response comes after exactly 3
    attempts, and the request sent each time is IDENTICAL (same seed) - no reseed on retry."""
    request = llm_request_cg.model_copy(update={"seed": 321})

    with _answering(_timeout(), _timeout(), _completion()) as sent:
        client = LLMClientOpenRouter("m", max_retries=2)
        result = await client.generate(request, expected_format=ClueJSONFormat)

    assert isinstance(result, LLMResponse)
    assert [body["seed"] for body in sent] == [321, 321, 321]


async def test_a_non_retriable_error_raises_at_once(llm_request_cg, no_backoff):
    """A non-retriable error (a rejected key) raises on the first attempt even with a budget."""
    with _answering(_fixture("invalid_key")) as sent:
        with pytest.raises(LLMAuthError):
            client = LLMClientOpenRouter("m", max_retries=3)
            await client.generate(llm_request_cg, expected_format=ClueJSONFormat)

    assert len(sent) == 1


async def test_there_is_no_retry_by_default(llm_request_cg, no_backoff):
    """max_retries defaults to 0: a retriable error raises at the first attempt, after one call."""
    with _answering(_timeout()) as sent:
        with pytest.raises(LLMTimeoutError):
            await LLMClientOpenRouter("m").generate(llm_request_cg, expected_format=ClueJSONFormat)

    assert len(sent) == 1


async def test_the_retry_budget_is_exhausted(llm_request_cg, no_backoff):
    """A retriable error failing k+1 times raises the last error after exactly k+1 attempts."""
    k = 2

    with _answering(*[_timeout() for _ in range(k + 1)]) as sent:
        with pytest.raises(LLMTimeoutError):
            client = LLMClientOpenRouter("m", max_retries=k)
            await client.generate(llm_request_cg, expected_format=ClueJSONFormat)

    assert len(sent) == k + 1
