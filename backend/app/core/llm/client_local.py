import asyncio
import json
import logging
import os
from json import JSONDecodeError
import re
from typing import Any, Literal
import httpx
from pydantic import BaseModel
from backend.app import config
from backend.app.core.llm.client import LLMClient, generate_with_retries
from backend.app.models.llm_schemas import LLMRequest, LLMResponse, TokenUsage
from backend.app.models.llm_errors import (
    LLMError,
    LLMModelNotProvidedError,
    LLMParseError,
    LLMEmptyResponseError,
    LLMDegenerateResponseError,
    LLMTruncatedResponseError,
    LLMMalformedResponseError,
    LLMTimeoutError,
    LLMAuthError,
    LLMRateLimitError,
    LLMProviderUnavailableError,
    LLMModelNotFoundError,
    LLMInvalidRequestError,
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
    """The taxonomy error for a call to the daemon that failed with ``e`` (a ``_DAEMON_ERRORS``).

    None of them is a refusal by the model: a model that will not play still answers, in text. An
    error status is the daemon's or the request's, so nothing here maps to ``LLMRefusalError``,
    which the batch counts as the model's behaviour."""
    context: dict[str, Any] = {"provider": "ollama", "cause": e, "execution_mode": "local"}
    # No answer within config.OLLAMA_TIMEOUT_S, or no connection within the connect limit.
    if isinstance(e, httpx.TimeoutException):
        return LLMTimeoutError(**context)
    # The daemon is not running, or dropped the connection mid-answer (stopped, or crashed).
    if not isinstance(e, ResponseError):
        return LLMProviderUnavailableError(**context)
    status = e.status_code
    context["http_status"] = status if status > 0 else None
    if status == 404:
        # The model is not pulled.
        return LLMModelNotFoundError(**context)
    if status in (401, 403):
        return LLMAuthError(**context)
    if status == 429:
        return LLMRateLimitError(**context)
    if 400 <= status < 500:
        # The daemon rejects what was sent (an option or a parameter the model does not support):
        # the same request would be rejected again.
        return LLMInvalidRequestError(**context)
    # A 5xx: the daemon failed (its runner crashed or ran out of memory, or its queue is full), and
    # may recover. Also an error reported mid-answer, which has no status.
    return LLMProviderUnavailableError(**context)


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

    # Guess path: GuessProposal.proposals / .confidence require at least one item, and no blank
    # word (the schema's minLength lets a word of blanks through).
    proposals = data.get("proposals")
    if isinstance(proposals, list) and not proposals:
        return "empty proposals list"
    if isinstance(proposals, list) and any(
        isinstance(p, dict) and not str(p.get("word") or "").strip() for p in proposals
    ):
        return "blank proposed word"

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
        self, request: LLMRequest, expected_format: type[BaseModel] | None = None
    ) -> LLMResponse:
        """Generate a response, retrying retriable errors per ``self.max_retries``.

        A bad draw (empty, degenerate or cut) is drawn again with a new seed, and a transient
        failure re-sends the same request; see ``generate_with_retries``."""
        return await generate_with_retries(
            lambda req: self._generate_once(req, expected_format),
            request,
            max_retries=self.max_retries,
            provider="ollama",
            model=self.model_name,
        )

    async def _generate_once(
        self, request: LLMRequest, expected_format: type[BaseModel] | None = None
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

            # The answer reached the output limit (num_predict) or filled the context, so its JSON
            # is unfinished: a bad draw, re-sampled within the caller's budget like the empty one.
            if ollama_response.done_reason == "length":
                logger.debug(
                    "ollama %s answer cut at the output limit (%s prompt + %s answer "
                    "tokens); discarded draw:\n%s",
                    self.model_name,
                    ollama_response.prompt_eval_count,
                    ollama_response.eval_count,
                    content,
                )
                raise LLMTruncatedResponseError(
                    provider="ollama", raw_payload=response_json, execution_mode="local"
                )

            # Validate the response against the expected format if provided
            parsed = None
            if expected_format:
                try:
                    parsed = expected_format.model_validate_json(content)
                except Exception as e:
                    # A bad draw, drawn again within the caller's budget. Only its count reaches the
                    # database, and the schema the answer was constrained by should have kept it
                    # from being written, so it is logged whole (num_predict bounds it), at a level
                    # the batch's log keeps.
                    logger.warning(
                        "ollama %s response does not match %s (%s); discarded draw:\n%s",
                        self.model_name,
                        expected_format.__name__,
                        e,
                        content,
                    )
                    raise LLMMalformedResponseError(
                        provider="ollama",
                        raw_payload=response_json,
                        cause=e,
                        execution_mode="local",
                    ) from e

                # Well-formed but degenerate (e.g. an empty proposals list): the permissive
                # LLM-facing schema accepts it, the domain model would not. Raised as the RETRIABLE
                # parse error so the SAME bounded re-sample absorbs it, instead of a fatal
                # ValidationError escaping from llm_service outside the retry loop.
                reason = degenerate_reason(parsed)
                if reason is not None:
                    logger.debug(
                        "ollama %s response is degenerate (%s); discarded draw:\n%s",
                        self.model_name,
                        reason,
                        content,
                    )
                    raise LLMDegenerateResponseError(
                        message=(
                            f"The LLM returned a well-formed but degenerate response: {reason}."
                        ),
                        provider="ollama",
                        raw_payload=response_json,
                        execution_mode="local",
                    )

            return LLMResponse(
                text=content,
                model_used=self.model_name,
                # Convert ns -> ms
                latency_ms=(
                    round(ollama_response.total_duration / 1_000_000)
                    if ollama_response.total_duration is not None
                    else 0
                ),
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

    def _remove_markdown_code_blocks(self, text: str) -> str:
        """
        Utility method to extract JSON content from a string, especially if it's wrapped in markdown
        code blocks.
        """
        return re.sub(r"```json(.*?)```", r"\1", text, flags=re.DOTALL).strip()
