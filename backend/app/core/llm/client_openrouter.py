import logging
import os
import time
from typing import Any, Optional, cast

from openai import (
    AsyncOpenAI,
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
)
from openai.types.chat import ChatCompletionMessageParam
from openai.types.chat.completion_create_params import ResponseFormat
from backend.app import config
from backend.app.core.llm.client import (
    LLMClient,
    error_for_status,
    generate_with_retries,
    provider_seed,
    read_answer,
)
from backend.app.models.llm_schemas import LLMFormat, LLMRequest, LLMResponse, TokenUsage
from backend.app.models.llm_errors import (
    LLMAuthError,
    LLMError,
    LLMInvalidRequestError,
    LLMRefusalError,
    LLMTimeoutError,
    LLMProviderUnavailableError,
)

logger = logging.getLogger(__name__)

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def provider_routing(model_name: str) -> Optional[dict[str, Any]]:
    """The ``provider`` object OpenRouter is sent for ``model_name``, from the model's entry in
    ``config.llm_models["OpenRouter"]``; None when the entry pins no provider.

    Unless told otherwise, OpenRouter may serve each call from another provider, or another
    quantization. The object pins the provider the entry names (and its quantization, if it names
    one), with no fallback to another, and only if that provider honours every parameter sent
    (``seed``, ``response_format``): a call it cannot serve fails, instead of being served
    otherwise."""
    models = cast("dict[str, dict[str, Any]]", config.llm_models).get("OpenRouter", {})
    entry = models.get(model_name) or {}
    if not entry.get("provider"):
        return None
    routing: dict[str, Any] = {
        "order": [entry["provider"]],
        "allow_fallbacks": False,
        "require_parameters": True,
    }
    if entry.get("quantization"):
        routing["quantizations"] = [entry["quantization"]]
    return routing


def _status_error(status: Optional[int], message: Optional[str] = None, **context: Any) -> LLMError:
    """The taxonomy error for an error status from OpenRouter, the response's or one in its body.

    As ``error_for_status``, but for a 403: OpenRouter's says that the moderation the chosen model
    requires flagged the input. That is a refusal of the content (``LLMRefusalError``), which the
    same input would meet again, not a rejected key."""
    if status == 403:
        context["http_status"] = status
        return LLMRefusalError(message, **context)
    return error_for_status(status, message, **context)


def _body_error(error: Any, **context: Any) -> LLMError:
    """The taxonomy error for an error OpenRouter reports in the body of a 200, by its code.

    OpenRouter answers with the error's status only when it rejects the request; an error once the
    response has started (the upstream model failed, or its provider) comes in a 200, as an
    ``{"code", "message"}`` in place of the choices, or in the choice."""
    error = error if isinstance(error, dict) else {}
    code, message = error.get("code"), error.get("message")
    return _status_error(
        code if isinstance(code, int) else None,
        f"OpenRouter reported an error: {message}" if message else None,
        **context,
    )


class LLMClientOpenRouter(LLMClient):
    def __init__(self, model_name: str, max_retries: int = 0):
        self.model_name = model_name
        # Bounded retry of retriable LLM errors (0 = one attempt, the default). The game runner
        # passes a non-zero budget; see game_runner._CLIENT_MAX_RETRIES.
        self.max_retries = max_retries
        # Every call of the model is served by the provider its config pins (see provider_routing).
        routing = provider_routing(model_name)
        if routing is None:
            raise LLMInvalidRequestError(
                f"{model_name} has no provider pinned in config.llm_models['OpenRouter'].",
                provider="openrouter",
                execution_mode="api",
            )
        self._routing = routing
        # The SDK refuses to build a client without a key, with an error of its own.
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise LLMAuthError(
                "OPENROUTER_API_KEY is not set.", provider="openrouter", execution_mode="api"
            )
        # No retries of the SDK's own (it retries twice by default): generate_with_retries is the
        # only retry, so every attempt is one call and is counted.
        self._client = AsyncOpenAI(base_url=_OPENROUTER_BASE_URL, api_key=api_key, max_retries=0)

    async def generate(
        self, request: LLMRequest, expected_format: type[LLMFormat] | None = None
    ) -> LLMResponse:
        """Generate a response, retrying retriable errors per ``self.max_retries``.

        A bad draw (empty, cut, malformed or degenerate) is drawn again with a new seed, and a
        transient failure re-sends the same request; see ``generate_with_retries``."""
        return await generate_with_retries(
            lambda req: self._generate_once(req, expected_format),
            request,
            max_retries=self.max_retries,
            provider="openrouter",
            model=self.model_name,
        )

    async def _generate_once(
        self, request: LLMRequest, expected_format: type[LLMFormat] | None = None
    ) -> LLMResponse:
        # LLMMessage.role is one of the roles the API takes (system, user, assistant).
        messages = [
            cast(ChatCompletionMessageParam, {"role": m.role, "content": m.content})
            for m in request.messages
        ]

        response_format: ResponseFormat
        if expected_format:
            # Not strict: OpenAI's strict mode needs every field required and no other field
            # allowed, which the formats do not say (``targets`` has a default), and changing them
            # would change the schema ollama is sent and the run's fingerprint. The schema is sent
            # as ollama gets it, and the validation of the answer holds the model to it.
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": expected_format.__name__,
                    "schema": expected_format.model_json_schema(),
                    "strict": False,
                },
            }
        else:
            response_format = {"type": "json_object"}

        # OpenRouter takes a 32-bit signed seed: the request's is masked (see provider_seed), and
        # the seed recorded is the one sent.
        seed = provider_seed(request.seed)

        context: dict[str, Any] = {"provider": "openrouter", "execution_mode": "api"}
        try:
            start = time.monotonic()
            response = await self._client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=request.temperature,
                seed=seed,
                max_tokens=request.max_tokens,
                timeout=request.timeout_s,
                response_format=response_format,
                extra_body={"provider": self._routing},
            )
            # Wall-clock time of the call, network included (see LLMResponse.latency_ms).
            latency_ms = int((time.monotonic() - start) * 1000)
        except APITimeoutError as e:
            raise LLMTimeoutError(cause=e, **context) from e
        except APIConnectionError as e:
            raise LLMProviderUnavailableError(cause=e, **context) from e
        except APIStatusError as e:
            raise _status_error(e.status_code, cause=e, request_id=e.request_id, **context) from e

        raw_payload = response.model_dump()
        context.update(raw_payload=raw_payload, request_id=response.id)

        # An error once the response had started: in place of the choices, or in the choice.
        if not response.choices:
            raise _body_error(getattr(response, "error", None), **context)
        choice = response.choices[0]
        if choice.finish_reason == "error" or getattr(choice, "error", None):
            raise _body_error(getattr(choice, "error", None), **context)

        # A model that refuses under structured output says so in a field of its own, not in the
        # content: the model's behaviour, which the same input would meet again.
        if choice.message.refusal:
            raise LLMRefusalError(f"The LLM refused: {choice.message.refusal}", **context)

        content, parsed = read_answer(
            choice.message.content or "",
            cut=choice.finish_reason == "length",
            expected_format=expected_format,
            model=self.model_name,
            **context,
        )

        usage = response.usage
        return LLMResponse(
            text=content,
            model_used=response.model,
            latency_ms=latency_ms,
            # Without a usage block the counts are unknown: None, not 0.
            usage=TokenUsage(
                prompt_tokens=usage.prompt_tokens if usage else None,
                completion_tokens=usage.completion_tokens if usage else None,
                total_tokens=usage.total_tokens if usage else None,
            ),
            finish_reason=choice.finish_reason,
            raw_payload=raw_payload,
            request_id=response.id,
            execution_mode="api",
            provider="openrouter",
            requested_temperature=request.temperature,
            requested_seed=seed,
            system_fingerprint=getattr(response, "system_fingerprint", None),
            resolved_model=response.model,
            parsed=parsed,
        )

    async def close(self) -> None:
        await self._client.close()
