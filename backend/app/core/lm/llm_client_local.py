import json
from json import JSONDecodeError
from backend.app.core.lm.llm_client import LLMClient
from backend.app.models.llm_schemas import LLMRequest, LLMResponse, TokenUsage
from backend.app.models.llm_errors import LLMModelNotProvidedError, LLMRefusalError, LLMParseError
from ollama import chat, RequestError, ResponseError


class LLMClientLocal(LLMClient):
    LLM_MODEL = "llama3.2:latest"

    def __init__(self, local_model: str = LLM_MODEL, think: bool = True):
        self.local_model = local_model
        self.think = think

    async def generate(self, request: LLMRequest) -> LLMResponse:
        # Convert LLMRequest to the format expected by the Ollama client
        messages = [
            {"role": message.role, "content": message.content} for message in request.messages
        ]

        try:
            # Call the Ollama client to get the response from the local LLM
            ollama_response = chat(
                model=self.local_model,
                messages=messages,
                think=self.think,
            )

            # Convert the Ollama response to LLMResponse format
            response_json = json.loads(ollama_response.model_dump_json())

            return LLMResponse(
                text=ollama_response.message.content,
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
