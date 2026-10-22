import asyncio
import json
import logging
import os
from json import JSONDecodeError
from typing import Any, Literal, Optional
import httpx
from backend.app import config
from backend.app.core.llm.client import (
    LLMClient,
    error_for_status,
    generate_with_retries,
    read_answer,
)
from backend.app.models.llm_schemas import LLMFormat, LLMRequest, LLMResponse, TokenUsage
from backend.app.models.llm_errors import (
    LLMError,
    LLMModelNotProvidedError,
    LLMParseError,
    LLMTimeoutError,
    LLMProviderUnavailableError,
)
from ollama import AsyncClient, RequestError, ResponseError

logger = logging.getLogger(__name__)

# The capability string ollama's /api/show reports for reasoning-capable models.
_THINKING_CAPABILITY = "thinking"

# What a call to the daemon raises when it fails instead of answering: an error status
# (ResponseError), the builtin ConnectionError ollama raises when it cannot connect, and the httpx
# errors it lets through (a timeout, a connection dropped mid-answer).
_DAEMON_ERRORS = (ResponseError, ConnectionError, httpx.TransportError)


def _daemon_error(e: Exception) -> LLMError:
    """The taxonomy error for a call to the daemon that failed with ``e`` (a ``_DAEMON_ERRORS``):
    by its status when the daemon answered with one (see ``error_for_status``)."""
    context: dict[str, Any] = {"provider": "ollama", "cause": e, "execution_mode": "local"}
    # No answer within config.OLLAMA_TIMEOUT_S, or no connection within the connect limit.
    if isinstance(e, httpx.TimeoutException):
        return LLMTimeoutError(**context)
    # The daemon is not running, or dropped the connection mid-answer (stopped, or crashed).
    if not isinstance(e, ResponseError):
        return LLMProviderUnavailableError(**context)
    # An error reported mid-answer has no status (-1).
    return error_for_status(e.status_code if e.status_code > 0 else None, **context)


def _ms(ns: Optional[int]) -> Optional[int]:
    """A duration ollama reports in nanoseconds, in milliseconds; None when it reports none."""
    return None if ns is None else round(ns / 1_000_000)


class LLMClientLocal(LLMClient):
    def __init__(self, model_name: str, think: bool = False, max_retries: int = 0):
        self.model_name = model_name
        # Whether this seat should reason. Defaults to False to match the roster config
        # (config.llm_models sets "think": False for every local model) and SeatSpec.think. Note the
        # flag is only ever SENT for models that advertise the capability; see _supports_thinking.
        self.think = think
        # Bounded retry of retriable LLM errors (0 = one attempt, the default). The game runner
        # passes a non-zero budget; see game_runner._CLIENT_MAX_RETRIES.
        self.max_retries = max_retries
        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        # The async client, so a call in flight leaves the event loop free, and a bounded wait, so a
        # hung daemon fails the call as a timeout instead of stopping the batch (ollama's own
        # default is no limit at all). See config.OLLAMA_TIMEOUT_S for the value.
        self._client = AsyncClient(
            host=host,
            timeout=httpx.Timeout(config.OLLAMA_TIMEOUT_S, connect=config.OLLAMA_CONNECT_TIMEOUT_S),
        )
        # Lazily resolved, then cached for the client's lifetime: the daemon's answer for a given
        # model cannot change without a re-pull, and this must not cost an /api/show per call.
        self._thinking_capable: bool | None = None

    async def _supports_thinking(self) -> bool:
        """Whether this model advertises the ``thinking`` capability, per the daemon's /api/show.

        For a reasoning-capable model ollama defaults thinking ON, which under structured output can
        leave ``message.content`` empty and always puts that model in a different inference regime
        (and a far slower one) than the non-reasoning models it is compared against. So
        thinking-capable models must be sent an explicit ``think`` value.

        The flag is withheld from models that do not advertise the capability: ollama rejects
        ``think=True`` for them outright (HTTP 400 "does not support thinking"), and older daemons
        may reject the parameter in any form.

        A failed call to the daemon (no answer in time, no connection, an error status) raises what
        a failed chat would (see ``_daemon_error``), retried the same way, and nothing is cached, so
        the next attempt probes again. Degrading would cache "not capable" for the whole game, and
        once the daemon came back a reasoning model would play the rest of it thinking. A probe
        that fails for any other reason (a client without ``show``, an answer it cannot read)
        degrades to False, i.e. omit ``think`` and preserve the previous behaviour rather than break
        generation on a probe.
        """
        if self._thinking_capable is None:
            try:
                capabilities = (await self._client.show(self.model_name)).capabilities
                self._thinking_capable = any(
                    str(c) == _THINKING_CAPABILITY for c in (capabilities or [])
                )
            except _DAEMON_ERRORS as e:
                raise _daemon_error(e) from e
            except Exception as e:
                logger.warning(
                    "capability probe failed for %r (%s); omitting `think` from the request.",
                    self.model_name,
                    e,
                )
                self._thinking_capable = False
        return self._thinking_capable

    async def generate(
        self, request: LLMRequest, expected_format: type[LLMFormat] | None = None
    ) -> LLMResponse:
        """Generate a response, retrying retriable errors per ``self.max_retries``.

        A bad draw (empty, degenerate, cut or malformed) is drawn again with a new seed, and a
        transient failure re-sends the same request; see ``generate_with_retries``."""
        return await generate_with_retries(
            lambda req: self._generate_once(req, expected_format),
            request,
            max_retries=self.max_retries,
            provider="ollama",
            model=self.model_name,
        )

    async def _generate_once(
        self, request: LLMRequest, expected_format: type[LLMFormat] | None = None
    ) -> LLMResponse:
        messages = [
            {"role": message.role, "content": message.content} for message in request.messages
        ]

        format: dict[str, Any] | Literal["json"] = (
            expected_format.model_json_schema() if expected_format else "json"
        )

        # Ollama ignores sampling settings unless they are passed explicitly via `options`. Always
        # forward the temperature; forward the seed ONLY when set (some ollama-client versions
        # serialize a `"seed": None`, which would change behaviour).
        options: dict[str, Any] = {"temperature": request.temperature}
        if request.seed is not None:
            options["seed"] = request.seed
        # A fixed context window, so the game does not depend on the machine (ollama would pick one
        # from the GPU's memory), and a cap on the answer, so a model that loops is cut instead of
        # filling the context. A cut answer is drawn again (LLMTruncatedResponseError, below).
        options["num_ctx"] = config.OLLAMA_NUM_CTX
        options["num_predict"] = request.max_tokens

        # Send `think` only to models that advertise the capability, and then send it - omitting it
        # lets ollama default a reasoning model to thinking ON. Probed outside the try: the probe
        # handles its own failures (see _supports_thinking).
        chat_kwargs: dict[str, Any] = {}
        if await self._supports_thinking():
            chat_kwargs["think"] = self.think

        try:
            ollama_response = await self._client.chat(
                model=self.model_name,
                messages=messages,
                format=format,
                options=options,
                **chat_kwargs,
            )

            # Convert the Ollama response to JSON. The client types the content as optional: an
            # answer without one is an empty answer.
            response_json = json.loads(ollama_response.model_dump_json())
            # The answer is cut when it reaches the output limit (num_predict) or fills the context.
            content, parsed = read_answer(
                response_json.get("message", {}).get("content") or "",
                cut=ollama_response.done_reason == "length",
                expected_format=expected_format,
                model=self.model_name,
                provider="ollama",
                raw_payload=response_json,
                execution_mode="local",
            )

            return LLMResponse(
                text=content,
                model_used=self.model_name,
                latency_ms=_ms(ollama_response.total_duration) or 0,
                load_ms=_ms(ollama_response.load_duration),
                prompt_eval_ms=_ms(ollama_response.prompt_eval_duration),
                eval_ms=_ms(ollama_response.eval_duration),
                # The ollama client types both counts as optional: one the server leaves out is
                # recorded as None (and the total with it), never summed as if it were 0.
                usage=TokenUsage(
                    prompt_tokens=ollama_response.prompt_eval_count,
                    completion_tokens=ollama_response.eval_count,
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
                parsed=parsed,
            )

        except _DAEMON_ERRORS as de:
            raise _daemon_error(de) from de
        except RequestError as re:
            raise LLMModelNotProvidedError(
                provider="ollama", cause=re, execution_mode="local"
            ) from re
        except JSONDecodeError as je:
            raise LLMParseError(provider="ollama", cause=je, execution_mode="local") from je

    async def health_check(self) -> bool:
        """Whether the daemon answers and still has this model: its /api/show for the model, within
        ``config.OLLAMA_HEALTH_TIMEOUT_S``. The batch asks before it plays again a game the daemon
        failed (see ``batch_runner``). False when the daemon is down, wedged or failing (5xx), or
        no longer has the model (404)."""
        try:
            await asyncio.wait_for(
                self._client.show(self.model_name), timeout=config.OLLAMA_HEALTH_TIMEOUT_S
            )
        except (*_DAEMON_ERRORS, TimeoutError) as e:
            logger.info("ollama health check for %r failed: %r", self.model_name, e)
            return False
        return True

    async def close(self) -> None:
        await self._client.close()
