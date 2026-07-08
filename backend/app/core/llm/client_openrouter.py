import os
import time
from pydantic import BaseModel
from openai import AsyncOpenAI, AuthenticationError, RateLimitError, APITimeoutError, APIConnectionError
from backend.app.core.llm.client import LLMClient
from backend.app.models.llm_schemas import LLMRequest, LLMResponse, TokenUsage
from backend.app.models.llm_errors import (
    LLMAuthError, LLMRateLimitError, LLMTimeoutError, LLMProviderUnavailableError, LLMParseError
)

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class LLMClientOpenRouter(LLMClient):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._client = AsyncOpenAI(
            base_url=_OPENROUTER_BASE_URL,
            api_key=os.environ.get("OPENROUTER_API_KEY", ""),
        )

    async def generate(self, request: LLMRequest, expected_format: type[BaseModel] = None) -> LLMResponse:
        messages = [{"role": m.role, "content": m.content} for m in request.messages]

        if expected_format:
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": expected_format.__name__,
                    "schema": expected_format.model_json_schema(),
                    "strict": True,
                },
            }
        else:
            response_format = {"type": "json_object"}

        try:
            start = time.monotonic()
            response = await self._client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=request.temperature,
                seed=request.seed,
                max_tokens=request.max_tokens,
                timeout=request.timeout_s,
                response_format=response_format,
            )
            latency_ms = int((time.monotonic() - start) * 1000)

            content = response.choices[0].message.content or ""

            if expected_format:
                try:
                    expected_format.model_validate_json(content)
                except Exception as e:
                    raise LLMParseError(provider="openrouter", cause=e, execution_mode="api")

            usage = response.usage
            raw_payload = response.model_dump()

            return LLMResponse(
                text=content,
                model_used=response.model,
                latency_ms=latency_ms,
                usage=TokenUsage(
                    prompt_tokens=usage.prompt_tokens if usage else 0,
                    completion_tokens=usage.completion_tokens if usage else 0,
                    total_tokens=usage.total_tokens if usage else 0,
                ),
                finish_reason=response.choices[0].finish_reason,
                raw_payload=raw_payload,
                request_id=response.id,
                execution_mode="api",
                provider="openrouter",
                requested_temperature=request.temperature,
                requested_seed=request.seed,
                system_fingerprint=getattr(response, "system_fingerprint", None),
                resolved_model=response.model,
            )

        except AuthenticationError as e:
            raise LLMAuthError(provider="openrouter", cause=e, execution_mode="api")
        except RateLimitError as e:
            raise LLMRateLimitError(provider="openrouter", cause=e, execution_mode="api")
        except APITimeoutError as e:
            raise LLMTimeoutError(provider="openrouter", cause=e, execution_mode="api")
        except APIConnectionError as e:
            raise LLMProviderUnavailableError(provider="openrouter", cause=e, execution_mode="api")

    async def close(self) -> None:
        await self._client.close()
