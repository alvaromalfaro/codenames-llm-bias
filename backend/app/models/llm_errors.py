from typing import Any


class LLMError(Exception):
    """
    A base class for errors related to interactions with the LLM. It includes a code to categorize
    the error, a message describing the error, a flag indicating whether the error is retriable, and
    optional metadata such as the provider, HTTP status code, request ID, raw payload from the LLM,
    the original exception that caused this error (if any), and the execution mode.
    """

    def __init__(self, code: str, message: str, retriable: bool = False, provider: str | None = None,
                 http_status: int | None = None, request_id: str | None = None,
                 raw_payload: dict[str, Any] | None = None, cause: Exception | None = None,
                 execution_mode: str | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.retriable = retriable
        self.provider = provider
        self.http_status = http_status
        self.request_id = request_id
        self.raw_payload = raw_payload
        self.cause = cause
        self.execution_mode = execution_mode

    def __str__(self) -> str:
        return f"LLMError(code={self.code}, message={self.message}, retriable={self.retriable}, " \
            f"provider={self.provider}, http_status={self.http_status}, request_id={self.request_id}, "\
            f"execution_mode={self.execution_mode}, cause={repr(self.cause)})"


class LLMModelNotProvidedError(LLMError):
    """
    A specific error class for when the LLM model is not provided in the request. This error is not
    retriable, as the model must be specified in order for the request to succeed on a subsequent
    attempt.
    """

    def __init__(self, message: str = "LLM model not provided in the request.",
                 provider: str | None = None, http_status: int | None = None,
                 request_id: str | None = None, raw_payload: dict[str, Any] | None = None,
                 cause: Exception | None = None, execution_mode: str | None = None):
        super().__init__(code="model_not_provided", message=message, retriable=False,
                         provider=provider, http_status=http_status, request_id=request_id,
                         raw_payload=raw_payload, cause=cause, execution_mode=execution_mode)


class LLMModelNotFoundError(LLMError):
    """
    A specific error class for when the provider does not have the requested model (e.g., a local
    model that was never pulled, or was removed). This error is not retriable, as the model must be
    made available before the request can succeed on a subsequent attempt.
    """

    def __init__(self, message: str = "The LLM provider does not have the requested model.",
                 provider: str | None = None, http_status: int | None = None,
                 request_id: str | None = None, raw_payload: dict[str, Any] | None = None,
                 cause: Exception | None = None, execution_mode: str | None = None):
        super().__init__(code="model_not_found", message=message, retriable=False,
                         provider=provider, http_status=http_status, request_id=request_id,
                         raw_payload=raw_payload, cause=cause, execution_mode=execution_mode)


class LLMInvalidRequestError(LLMError):
    """
    A specific error class for when the provider rejects the request as invalid (e.g., an option or
    a parameter the model does not support). This error is not retriable, as the same request would
    be rejected again: the code or the configuration that built it must be fixed.
    """

    def __init__(self, message: str = "The LLM provider rejected the request as invalid.",
                 provider: str | None = None, http_status: int | None = None,
                 request_id: str | None = None, raw_payload: dict[str, Any] | None = None,
                 cause: Exception | None = None, execution_mode: str | None = None):
        super().__init__(code="invalid_request", message=message, retriable=False,
                         provider=provider, http_status=http_status, request_id=request_id,
                         raw_payload=raw_payload, cause=cause, execution_mode=execution_mode)


class LLMRefusalError(LLMError):
    """
    A specific error class for when the LLM refuses to generate a response (e.g., due to content
    moderation, policy violations, etc.). This error is not retriable, as the refusal would likely
    occur again on a subsequent attempt unless the input is modified to comply with the LLM's
    policies.
    """

    def __init__(self, message: str = "The LLM refused to generate a response for the given input.",
                 provider: str | None = None, http_status: int | None = None,
                 request_id: str | None = None, raw_payload: dict[str, Any] | None = None,
                 cause: Exception | None = None, execution_mode: str | None = None):
        super().__init__(code="refused", message=message, retriable=False,
                         provider=provider, http_status=http_status, request_id=request_id,
                         raw_payload=raw_payload, cause=cause, execution_mode=execution_mode)


class LLMTimeoutError(LLMError):
    """
    A specific error class for LLM timeouts, indicating that the request to the LLM exceeded the
    allowed time limit. This error is retriable, as the timeout may have been a transient issue that
    could succeed on a subsequent attempt.
    """

    def __init__(self, message: str = "The LLM request timed out.", provider: str | None = None,
                 http_status: int | None = None, request_id: str | None = None,
                 raw_payload: dict[str, Any] | None = None, cause: Exception | None = None,
                 execution_mode: str | None = None):
        super().__init__(code="timeout", message=message, retriable=True, provider=provider,
                         http_status=http_status, request_id=request_id, raw_payload=raw_payload,
                         cause=cause, execution_mode=execution_mode)


class LLMRateLimitError(LLMError):
    """
    A specific error class for LLM rate limits, indicating that the request to the LLM was rejected
    due to exceeding the allowed number of requests in a given time period. This error is retriable,
    as the rate limit may reset after a certain amount of time, allowing the request to succeed on a
    subsequent attempt.
    """

    def __init__(self, message: str = "The LLM request was rate limited.",
                 provider: str | None = None, http_status: int | None = None,
                 request_id: str | None = None, raw_payload: dict[str, Any] | None = None,
                 cause: Exception | None = None, execution_mode: str | None = None):
        super().__init__(code="rate_limit", message=message, retriable=True, provider=provider,
                         http_status=http_status, request_id=request_id, raw_payload=raw_payload,
                         cause=cause, execution_mode=execution_mode)


class LLMAuthError(LLMError):
    """
    A specific error class for LLM authentication errors, indicating that the request to the LLM
    failed due to authentication issues (e.g., invalid API key, expired token, etc.). This error is
    not retriable, as the authentication issue must be resolved before the request can succeed on a
    subsequent attempt.
    """

    def __init__(self, message: str = "Authentication with the LLM provider failed.",
                 provider: str | None = None, http_status: int | None = None,
                 request_id: str | None = None, raw_payload: dict[str, Any] | None = None,
                 cause: Exception | None = None, execution_mode: str | None = None):
        super().__init__(code="auth", message=message, retriable=False, provider=provider,
                         http_status=http_status, request_id=request_id, raw_payload=raw_payload,
                         cause=cause, execution_mode=execution_mode)


class LLMParseError(LLMError):
    """
    A specific error class for LLM parse errors, indicating that the response from the LLM could not
    be parsed or understood. Raised as such only when the provider's own answer cannot be read; this
    error is not retriable, as the parsing logic would need to be updated before the request could
    succeed on a subsequent attempt. A model's answer that cannot be used is one of its retriable
    subclasses: empty, degenerate, cut, or malformed.
    """

    def __init__(self, message: str = "Failed to parse the LLM response.",
                 provider: str | None = None, http_status: int | None = None,
                 request_id: str | None = None, raw_payload: dict[str, Any] | None = None,
                 cause: Exception | None = None, execution_mode: str | None = None):
        super().__init__(code="parse_error", message=message, retriable=False, provider=provider,
                         http_status=http_status, request_id=request_id, raw_payload=raw_payload,
                         cause=cause, execution_mode=execution_mode)


class LLMEmptyResponseError(LLMParseError):
    """
    A specific parse error for when the provider returns empty or whitespace-only content. Unlike its
    parent, this error is retriable: an empty draw is a stochastic non-answer (observed
    intermittently from ollama at temperature > 0), and a new draw, with a new seed, usually
    succeeds. It stays a subclass of
    LLMParseError so existing ``except LLMParseError`` handlers keep catching it.
    """

    def __init__(self, message: str = "The LLM returned an empty response.",
                 provider: str | None = None, http_status: int | None = None,
                 request_id: str | None = None, raw_payload: dict[str, Any] | None = None,
                 cause: Exception | None = None, execution_mode: str | None = None):
        # Bypass LLMParseError.__init__ (which hardcodes retriable=False) to set the code and the
        # retriable flag that distinguish this case.
        LLMError.__init__(self, code="empty_response", message=message, retriable=True,
                          provider=provider, http_status=http_status, request_id=request_id,
                          raw_payload=raw_payload, cause=cause, execution_mode=execution_mode)


class LLMDegenerateResponseError(LLMParseError):
    """
    A specific parse error for a well-formed response whose content is degenerate: it satisfies the
    (deliberately permissive) LLM-facing schema but cannot produce a legal domain object - e.g. an
    empty ``proposals`` list, which parses fine against ``GuessJSONFormat`` but violates
    ``GuessProposal.proposals`` (``min_length=1``).

    Retriable for the same reason as an empty response: the model produced a non-answer, not
    malformed output, and a new draw, with a new seed, usually yields a usable one. This is
    tracked separately from ``LLMEmptyResponseError`` because the two describe different model
    behaviours - "returned nothing" versus "returned a well-formed refusal to guess" - and the
    experiment wants to count them apart.

    It stays a subclass of ``LLMParseError`` so existing handlers keep catching it. Note the domain
    constraint it protects (``min_length=1``) is not loosened: an empty guess list is illegal by game
    rules, so after the retry budget is exhausted this still errors out.
    """

    def __init__(self, message: str = "The LLM returned a well-formed but degenerate response.",
                 provider: str | None = None, http_status: int | None = None,
                 request_id: str | None = None, raw_payload: dict[str, Any] | None = None,
                 cause: Exception | None = None, execution_mode: str | None = None):
        # Bypass LLMParseError.__init__ (which hardcodes retriable=False) to set the code and the
        # retriable flag that distinguish this case.
        LLMError.__init__(self, code="degenerate_response", message=message, retriable=True,
                          provider=provider, http_status=http_status, request_id=request_id,
                          raw_payload=raw_payload, cause=cause, execution_mode=execution_mode)


class LLMTruncatedResponseError(LLMParseError):
    """
    A specific parse error for an answer the provider cut because it reached the output limit
    (``done_reason``/``finish_reason`` "length"): the model kept writing, usually because it fell
    into a loop, and the JSON is left unfinished.

    Retriable, like an empty or a degenerate response: it is a bad draw, and a new draw, with a new
    seed, usually ends normally. Counted apart from those two because it is a different model
    behaviour. It stays a subclass of ``LLMParseError`` so existing handlers keep catching it; after
    the retry budget is exhausted the game still errors out.
    """

    def __init__(self, message: str = "The LLM response was cut at the output limit.",
                 provider: str | None = None, http_status: int | None = None,
                 request_id: str | None = None, raw_payload: dict[str, Any] | None = None,
                 cause: Exception | None = None, execution_mode: str | None = None):
        # Bypass LLMParseError.__init__ (which hardcodes retriable=False) to set the code and the
        # retriable flag that distinguish this case.
        LLMError.__init__(self, code="truncated_response", message=message, retriable=True,
                          provider=provider, http_status=http_status, request_id=request_id,
                          raw_payload=raw_payload, cause=cause, execution_mode=execution_mode)


class LLMMalformedResponseError(LLMParseError):
    """
    A specific parse error for an answer that does not match the response schema it was asked for
    (the request's ``expected_format``): a field missing or of the wrong type, a value out of range,
    or no JSON at all.

    Retriable, like an empty, degenerate or cut response: it is a bad draw. With the same seed a
    local model would write the same answer again, but a bad draw is drawn again with a new seed,
    which usually yields a well-formed one. Counted apart from the other bad draws because it is a
    different model behaviour. It stays a subclass of ``LLMParseError`` so existing handlers keep
    catching it; after the retry budget is exhausted the game still errors out.
    """

    def __init__(self, message: str = "The LLM response does not match the expected format.",
                 provider: str | None = None, http_status: int | None = None,
                 request_id: str | None = None, raw_payload: dict[str, Any] | None = None,
                 cause: Exception | None = None, execution_mode: str | None = None):
        # Bypass LLMParseError.__init__ (which hardcodes retriable=False) to set the code and the
        # retriable flag that distinguish this case.
        LLMError.__init__(self, code="malformed_response", message=message, retriable=True,
                          provider=provider, http_status=http_status, request_id=request_id,
                          raw_payload=raw_payload, cause=cause, execution_mode=execution_mode)


class LLMProviderUnavailableError(LLMError):
    """
    A specific error class for LLM provider unavailability, indicating that the LLM provider is
    currently unavailable (e.g., due to maintenance, outages, etc.). This error is retriable, as the
    provider may become available again after some time, allowing the request to succeed on a
    subsequent attempt.
    """

    def __init__(self, message: str = "The LLM provider is currently unavailable.",
                 provider: str | None = None, http_status: int | None = None,
                 request_id: str | None = None, raw_payload: dict[str, Any] | None = None,
                 cause: Exception | None = None, execution_mode: str | None = None):
        super().__init__(code="provider_unavailable", message=message, retriable=True,
                         provider=provider, http_status=http_status, request_id=request_id,
                         raw_payload=raw_payload, cause=cause, execution_mode=execution_mode)
