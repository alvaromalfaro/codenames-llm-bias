import json
import logging
import os
from json import JSONDecodeError
import re
from typing import Any
from pydantic import BaseModel
from backend.app.core.llm.client import LLMClient, generate_with_retries
from backend.app.models.llm_schemas import LLMRequest, LLMResponse, TokenUsage
from backend.app.models.llm_errors import (
    LLMModelNotProvidedError, LLMRefusalError, LLMParseError, LLMEmptyResponseError,
    LLMDegenerateResponseError,
)
from ollama import Client, RequestError, ResponseError

logger = logging.getLogger(__name__)

# The capability string ollama's /api/show reports for reasoning-capable models.
_THINKING_CAPABILITY = "thinking"


def degenerate_reason(parsed: BaseModel) -> str | None:
    """Why this well-formed response cannot yield a legal domain object, or None if it can.

    A model can return JSON that parses cleanly and still be a non-answer. Because the domain object 
    is built in ``llm_service`` outside the client's retry loop, that mismatch would otherwise 
    escape as a fatal ``ValidationError`` mid-game.

    Detecting it here - against the same permissive parse - lets the existing bounded re-sample
    absorb it. The checks intentionally mirror the domain constraints and nothing more; they never
    relax a rule, they only decide what is worth re-sampling.

    Measurement rankings are deliberately not checked: an incomplete or empty ranking is parsed
    permissively downstream by design and must not trigger a re-sample.
    """
    data = parsed.model_dump()

    # Guess path: GuessProposal.proposals / .confidence require at least one item.
    proposals = data.get("proposals")
    if isinstance(proposals, list) and not proposals:
        return "empty proposals list"

    # Clue path: ClueProposal rejects a blank clue and requires count >= 1.
    if "clue" in data and not str(data.get("clue") or "").strip():
        return "blank clue"
    count = data.get("count")
    if isinstance(count, int) and not isinstance(count, bool) and count < 1:
        return f"non-positive count ({count})"

    return None


class LLMClientLocal(LLMClient):
    def __init__(self, model_name: str, think: bool = False, max_retries: int = 0):
        self.model_name = model_name
        # Whether this seat should reason. Defaults to False to match the roster config
        # (config.llm_models sets "think": False for every local model) and SeatSpec.think. Note the
        # flag is only ever SENT for models that advertise the capability; see _supports_thinking.
        self.think = think
        # Bounded same-request retry of retriable LLM errors (0 = one attempt, the interactive
        # default). The headless driver passes a non-zero budget; see game_runner._CLIENT_MAX_RETRIES.
        self.max_retries = max_retries
        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self._client = Client(host=host)
        # Lazily resolved, then cached for the client's lifetime: the daemon's answer for a given
        # model cannot change without a re-pull, and this must not cost an /api/show per call.
        self._thinking_capable: bool | None = None

    def _supports_thinking(self) -> bool:
        """Whether this model advertises the ``thinking`` capability, per the daemon's /api/show.

        For a reasoning-capable model ollama defaults thinking ON, which under structured output can 
        leave ``message.content`` empty and always puts that model in a different inference regime 
        (and a far slower one) than the non-reasoning models it is compared against. So thinking-capable
        models must be sent an explicit ``think`` value.

        The flag is withheld from models that do not advertise the capability: ollama rejects
        ``think=True`` for them outright (HTTP 400 "does not support thinking"), and older daemons
        may reject the parameter in any form.

        Never raises: an unreachable daemon or a client without ``show`` degrades to False, i.e.
        omit ``think`` and preserve the previous behaviour rather than break generation on a probe.
        """
        if self._thinking_capable is None:
            try:
                capabilities = self._client.show(self.model_name).capabilities
                self._thinking_capable = any(
                    str(c) == _THINKING_CAPABILITY for c in (capabilities or []))
            except Exception as e:
                logger.warning(
                    "capability probe failed for %r (%s); omitting `think` from the request.",
                    self.model_name, e)
                self._thinking_capable = False
        return self._thinking_capable

    async def generate(self, request: LLMRequest, expected_format: type[BaseModel] = None) -> LLMResponse:
        """Generate a response, retrying transient (retriable) errors per ``self.max_retries``.

        The retry re-sends the identical request (same seed/prompt) around the taxonomy mapping in
        ``_generate_once``; see ``generate_with_retries``."""
        return await generate_with_retries(
            lambda: self._generate_once(request, expected_format),
            max_retries=self.max_retries, provider="ollama", model=self.model_name)

    async def _generate_once(self, request: LLMRequest, expected_format: type[BaseModel] = None) -> LLMResponse:
        messages = [
            {"role": message.role, "content": message.content} for message in request.messages
        ]

        format = expected_format.model_json_schema() if expected_format else "json"

        # Ollama ignores sampling settings unless they are passed explicitly via `options`. Always
        # forward the temperature; forward the seed ONLY when set (some ollama-client versions
        # serialize a `"seed": None`, which would change behaviour).
        options: dict[str, Any] = {"temperature": request.temperature}
        if request.seed is not None:
            options["seed"] = request.seed

        # Send `think` only to models that advertise the capability, and then send it - omitting it
        # lets ollama default a reasoning model to thinking ON. Probed outside the try so a probe
        # failure can never be mis-mapped onto the provider-error taxonomy below.
        chat_kwargs: dict[str, Any] = {}
        if self._supports_thinking():
            chat_kwargs["think"] = self.think

        try:
            ollama_response = self._client.chat(
                model=self.model_name,
                messages=messages,
                format=format,
                options=options,
                **chat_kwargs
            )

            # Convert the Ollama response to JSON
            response_json = json.loads(ollama_response.model_dump_json())
            content = response_json.get("message", {}).get("content", "")
            content = self._remove_markdown_code_blocks(content)

            # An empty / whitespace-only draw is a stochastic non-answer, not malformed output:
            # raised as the retriable parse error so generate_with_retries re-samples it within the
            # caller's budget instead of killing the game on a single bad draw.
            if not content.strip():
                raise LLMEmptyResponseError(
                    provider="ollama", raw_payload=response_json, execution_mode="local"
                )

            # Validate the response against the expected format if provided
            if expected_format:
                try:
                    parsed = expected_format.model_validate_json(content)
                except Exception as e:
                    raise LLMParseError(
                        provider="ollama", cause=e, execution_mode="local"
                    )

                # Well-formed but degenerate (e.g. an empty proposals list): the permissive
                # LLM-facing schema accepts it, the domain model would not. Raised as the RETRIABLE
                # parse error so the SAME bounded re-sample absorbs it, instead of a fatal
                # ValidationError escaping from llm_service outside the retry loop.
                reason = degenerate_reason(parsed)
                if reason is not None:
                    raise LLMDegenerateResponseError(
                        message=f"The LLM returned a well-formed but degenerate response: {reason}.",
                        provider="ollama", raw_payload=response_json, execution_mode="local",
                    )

            return LLMResponse(
                text=content,
                model_used=self.model_name,
                # Convert ns -> ms
                latency_ms=(round(ollama_response.total_duration / 1_000_000)
                            if ollama_response.total_duration is not None else 0),
                usage=TokenUsage(
                    prompt_tokens=ollama_response.prompt_eval_count,
                    completion_tokens=ollama_response.eval_count,
                    total_tokens=ollama_response.prompt_eval_count + ollama_response.eval_count,
                ),
                finish_reason=ollama_response.done_reason,
                raw_payload=response_json,
                request_id=None,
                execution_mode="local",
                provider="ollama",
                requested_temperature=request.temperature,
                requested_seed=request.seed,
                system_fingerprint=None,
                resolved_model=response_json.get("model") or self.model_name,
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
