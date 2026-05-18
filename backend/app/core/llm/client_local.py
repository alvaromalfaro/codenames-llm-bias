import json
import os
from json import JSONDecodeError
import re
from pydantic import BaseModel
from backend.app.core.llm.client import LLMClient
from backend.app.models.llm_schemas import LLMRequest, LLMResponse, TokenUsage
from backend.app.models.llm_errors import LLMModelNotProvidedError, LLMRefusalError, LLMParseError
from ollama import Client, RequestError, ResponseError


class LLMClientLocal(LLMClient):
    def __init__(self, local_model: str, think: bool = True):
        self.local_model = local_model
        self.think = think
        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self._client = Client(host=host)

    async def generate(self, request: LLMRequest, expected_format: type[BaseModel] = None) -> LLMResponse:
        messages = [
            {"role": message.role, "content": message.content} for message in request.messages
        ]

        format = expected_format.model_json_schema() if expected_format else "json"

        try:
            if self.think:
                ollama_response = self._client.chat(
                    model=self.local_model,
                    messages=messages,
                    think=self.think,
                    format=format
                )
            else:
                ollama_response = self._client.chat(
                    model=self.local_model,
                    messages=messages,
                    format=format
                )

            # Convert the Ollama response to JSON
            response_json = json.loads(ollama_response.model_dump_json())
            content = response_json.get("message", {}).get("content", "")
            content = self._remove_markdown_code_blocks(content)

            # Validate the response against the expected format if provided
            if expected_format:
                try:
                    expected_format.model_validate_json(content)
                except Exception as e:
                    raise LLMParseError(
                        provider="ollama", cause=e, execution_mode="local"
                    )

            return LLMResponse(
                text=content,
                model_used=self.local_model,
                latency_ms=ollama_response.total_duration,
                usage=TokenUsage(
                    prompt_tokens=ollama_response.prompt_eval_count,
                    completion_tokens=ollama_response.eval_count,
                    total_tokens=ollama_response.prompt_eval_count + ollama_response.eval_count,
                ),
                finish_reason=ollama_response.done_reason,
                raw_payload=response_json,
                request_id=None,
                execution_mode="local",
                provider="ollama"
            )

        except RequestError as re:
            raise LLMModelNotProvidedError(
                provider="ollama", cause=re, execution_mode="local"
            )
        except ResponseError as re:
            raise LLMRefusalError(
                provider="ollama", cause=re, execution_mode="local"
            )
        except JSONDecodeError as je:
            raise LLMParseError(
                provider="ollama", cause=je, execution_mode="local"
            )

    def _remove_markdown_code_blocks(self, text: str) -> str:
        """
        Utility method to extract JSON content from a string, especially if it's wrapped in markdown
        code blocks.
        """
        return re.sub(r"```json(.*?)```", r"\1", text, flags=re.DOTALL).strip()
