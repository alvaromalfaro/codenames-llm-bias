import asyncio
import logging
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from pydantic import BaseModel
from backend.app.models.llm_errors import (
    LLMError, LLMEmptyResponseError, LLMDegenerateResponseError,
)
from backend.app.models.llm_schemas import LLMRequest, LLMResponse

logger = logging.getLogger(__name__)

# Exponential backoff base for the client-side transient retry: 15s, 30s, 60s, ...
_RETRY_BACKOFF_BASE_S = 15

# Key under which the empty-response re-sample count is stamped into the returned response's
# raw_payload. That payload is copied verbatim onto LLMCallRecord.raw_payload and written to the
# JSONB llm_call.raw_payload column, so the count is auditable downstream with no schema change.
EMPTY_RESAMPLE_KEY = "empty_response_resamples"

# Same, for WELL-FORMED but degenerate draws (e.g. an empty proposals list). Counted apart from
# EMPTY_RESAMPLE_KEY because "returned nothing" and "returned a well-formed refusal to guess" are
# different model behaviours, and both are data about the model, not noise.
DEGENERATE_RESAMPLE_KEY = "degenerate_response_resamples"


async def generate_with_retries(
    attempt: Callable[[], Awaitable[LLMResponse]], *,
    max_retries: int, provider: str, model: str,
) -> LLMResponse:
    """Run a single provider attempt with a bounded, same-REQUEST retry of retriable LLM errors.

    ``attempt`` is a zero-arg coroutine factory that performs one provider call and either returns an
    ``LLMResponse`` or raises an ``LLMError`` from the taxonomy (the mapping already applied). This
    helper classifies on the MAPPED error's ``retriable`` flag: a retriable error is retried after an
    exponential backoff (re-invoking ``attempt``, which re-sends the identical request - never
    reseed); a non-retriable ``LLMError`` or an exhausted budget re-raises; all other exceptions 
    propagate untouched.

    ``max_retries=0`` (the default for the interactive path) means exactly one attempt: the first
    retriable error raises. ``max_retries=k`` allows up to ``k+1`` attempts. Transients produce no
    ``LLMResponse`` and hence no telemetry - they are logged, never persisted (a network failure is
    not model behavior, unlike clue-legality retries).

    The one exception is the empty-response re-sample (``LLMEmptyResponseError``): an empty draw is
    model behavior, so the number absorbed by this call is stamped onto the successful response's
    ``raw_payload`` under ``EMPTY_RESAMPLE_KEY`` and rides the existing audit path to
    ``llm_call.raw_payload``. It is also skipped for backoff, being a local re-roll not a transient.
    """
    n = 0
    empty_resamples = 0
    degenerate_resamples = 0
    while True:
        try:
            response = await attempt()
        except LLMError as exc:
            if not exc.retriable or n >= max_retries:
                raise
            # An empty or degenerate draw is a local re-sample, not a network transient: back off
            # only for the latter, so a stochastic non-answer costs a re-roll rather than minutes of
            # sleeping.
            is_empty = isinstance(exc, LLMEmptyResponseError)
            is_degenerate = isinstance(exc, LLMDegenerateResponseError)
            is_resample = is_empty or is_degenerate
            backoff = 0.0 if is_resample else _RETRY_BACKOFF_BASE_S * (2 ** n)
            logger.warning(
                "retriable LLM error provider=%s model=%s attempt=%s/%s: %s; retrying after %.2fs",
                provider, model, n + 1, max_retries + 1, exc, backoff)
            n += 1
            if is_empty:
                empty_resamples += 1
            if is_degenerate:
                degenerate_resamples += 1
            if backoff:
                await asyncio.sleep(backoff)
        else:
            # Stamp the re-sample counts onto the response that finally succeeded, so the audit trail
            # records how many bad draws this single call absorbed (0 for the common case).
            response.raw_payload[EMPTY_RESAMPLE_KEY] = empty_resamples
            response.raw_payload[DEGENERATE_RESAMPLE_KEY] = degenerate_resamples
            return response


class LLMClient(ABC):
    @abstractmethod
    async def generate(self, request: LLMRequest, expected_format: type[BaseModel] = None) -> LLMResponse:
        """
        Abstract method to generate a response from the LLM based on the given request. This method
        must be implemented by any concrete subclass of LLMClient.

        :param request: An instance of LLMRequest containing the messages and parameters for the LLM
            generation.

        :return: An instance of LLMResponse containing the generated response from the LLM.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    async def health_check(self) -> bool:
        """
        Optional method to perform a health check on the LLM provider. This can be used to verify
        that the provider is available and functioning properly before attempting to generate a
        response.

        :return: A boolean indicating whether the LLM provider is healthy (True) or not (False).
        """
        # Default implementation assumes the provider is healthy.
        return True

    async def close(self) -> None:
        """
        Optional method to close any resources or connections used by the LLM client. This can be
        overridden by subclasses if they need to perform any cleanup when the client is no longer
        needed.
        """
        pass
