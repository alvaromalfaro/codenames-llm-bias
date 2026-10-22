import asyncio
import hashlib
import logging
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Awaitable, Callable
from typing import Optional
from pydantic import BaseModel
from backend.app.models.llm_errors import (
    LLMError,
    LLMEmptyResponseError,
    LLMDegenerateResponseError,
    LLMTruncatedResponseError,
    LLMMalformedResponseError,
)
from backend.app.models.llm_schemas import LLMRequest, LLMResponse

logger = logging.getLogger(__name__)

# Exponential backoff base for the client-side transient retry: 15s, 30s, 60s, ...
_RETRY_BACKOFF_BASE_S = 15

# The bad draws: a model's answer that cannot be used, drawn again with a new seed. Each kind is
# counted apart, because each is a different model behaviour (data about the model, not noise), and
# its count is stamped into the kept response's raw_payload under "<code>_resamples" (e.g.
# "empty_response_resamples"), 0 included. That payload is copied verbatim onto
# LLMCallRecord.raw_payload and written to the JSONB llm_call.raw_payload column, so the counts are
# auditable downstream with no schema change.
_BAD_DRAWS: tuple[type[LLMError], ...] = (
    LLMEmptyResponseError,
    LLMDegenerateResponseError,
    LLMTruncatedResponseError,
    LLMMalformedResponseError,
)

_MASK64 = (1 << 64) - 1


def redraw_seed(seed: Optional[int], draw: int) -> Optional[int]:
    """The seed of the ``draw``-th new draw (1, 2, ...) of a request whose own seed is ``seed``:
    ``SHA-256(seed_as_8B_be || b"redraw" || draw_as_8B_be) mod 2**64``, the scheme of the game
    runner's own seeds. Deterministic, so a run with redraws is still reproducible from its master
    seed. An unseeded request (``None``) stays unseeded: each of its draws is already a new one."""
    if seed is None:
        return None
    h = hashlib.sha256((seed & _MASK64).to_bytes(8, "big"))
    h.update(b"redraw")
    h.update(draw.to_bytes(8, "big"))
    return int.from_bytes(h.digest(), "big") & _MASK64


async def generate_with_retries(
    attempt: Callable[[LLMRequest], Awaitable[LLMResponse]],
    request: LLMRequest,
    *,
    max_retries: int,
    provider: str,
    model: str,
) -> LLMResponse:
    """Send ``request`` with ``attempt``, with a bounded retry of retriable LLM errors.

    ``attempt`` performs one provider call for the request it is given, and either returns an
    ``LLMResponse`` or raises an ``LLMError`` from the taxonomy (the mapping already applied). This
    helper classifies on the MAPPED error's ``retriable`` flag; a non-retriable ``LLMError`` or an
    exhausted budget re-raises, and all other exceptions propagate untouched. A retriable error is
    retried in one of two ways:

    - **A bad draw** - an empty (``LLMEmptyResponseError``), degenerate
      (``LLMDegenerateResponseError``), cut (``LLMTruncatedResponseError``) or malformed
      (``LLMMalformedResponseError``) answer - is drawn again at once, with a new seed:
      ``redraw_seed(request.seed, k)`` for the k-th new draw. With the same seed a local model gives
      the same answer again (a model that loops loops again), so re-sending the request would only
      spend the budget. The answer that is finally kept carries the seed it was drawn with
      (``LLMResponse.requested_seed``, hence ``llm_call.requested_seed``).
    - **A transient failure** (timeout, rate limit, provider unavailable) re-sends the same request
      after an exponential backoff: the network failed, not the model.

    ``max_retries=0`` (the clients' default) means exactly one attempt: the first
    retriable error raises. ``max_retries=k`` allows up to ``k+1`` attempts. Transients produce no
    ``LLMResponse`` and hence no telemetry - they are logged, never persisted (a network failure is
    not model behavior, unlike clue-legality retries).

    Bad draws are model behavior, so the number of each kind absorbed by this call is stamped onto
    the successful response's ``raw_payload`` (``"<code>_resamples"``, one key per kind in
    ``_BAD_DRAWS``) and rides the existing audit path to ``llm_call.raw_payload``. Their sum is the
    ``k`` of the kept answer's seed.
    """
    n = 0
    bad_draws: Counter[str] = Counter()
    current = request
    while True:
        try:
            response = await attempt(current)
        except LLMError as exc:
            if not exc.retriable or n >= max_retries:
                raise
            # A bad draw is not a network transient: it is drawn again at once with a new seed. Only
            # a transient backs off, and re-sends the same request.
            n += 1
            if isinstance(exc, _BAD_DRAWS):
                bad_draws[exc.code] += 1
                current = request.model_copy(
                    update={"seed": redraw_seed(request.seed, bad_draws.total())}
                )
                logger.warning(
                    "retriable LLM error provider=%s model=%s attempt=%s/%s: %s; drawing again "
                    "with seed %s",
                    provider,
                    model,
                    n,
                    max_retries + 1,
                    exc,
                    current.seed,
                )
            else:
                backoff = _RETRY_BACKOFF_BASE_S * (2 ** (n - 1))
                logger.warning(
                    "retriable LLM error provider=%s model=%s attempt=%s/%s: %s; retrying after "
                    "%.2fs",
                    provider,
                    model,
                    n,
                    max_retries + 1,
                    exc,
                    backoff,
                )
                await asyncio.sleep(backoff)
        else:
            # Stamp the re-sample counts onto the response that finally succeeded, so the audit
            # trail records how many bad draws this single call absorbed (0 for the common case).
            response.raw_payload.update(
                {f"{kind.code}_resamples": bad_draws[kind.code] for kind in _BAD_DRAWS}
            )
            return response


class LLMClient(ABC):
    # The model this client serves; the service puts it in every request.
    model_name: str

    @abstractmethod
    async def generate(
        self, request: LLMRequest, expected_format: type[BaseModel] | None = None
    ) -> LLMResponse:
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

    async def close(self) -> None:  # noqa: B027 - a no-op by default, not an abstract method
        """
        Optional method to close any resources or connections used by the LLM client. This can be
        overridden by subclasses if they need to perform any cleanup when the client is no longer
        needed.
        """
        pass
