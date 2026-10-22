from typing import Any, ClassVar


class LLMError(Exception):
    """
    A base class for errors related to interactions with the LLM. Each subclass is one kind of
    error, described by its class attributes: the ``code`` that categorizes it (what the batch
    classifies a failed game by, and what ``game.error_code`` records), whether it is
    ``retriable``, and the ``default_message`` it is raised with when no message is given. An
    instance carries optional metadata on the call that failed: the provider, HTTP status code,
    request ID, raw payload from the LLM, the original exception that caused this error (if any),
    and the execution mode.
    """

    code: ClassVar[str]
    default_message: ClassVar[str]
    retriable: ClassVar[bool] = False

    def __init__(
        self,
        message: str | None = None,
        *,
        provider: str | None = None,
        http_status: int | None = None,
        request_id: str | None = None,
        raw_payload: dict[str, Any] | None = None,
        cause: Exception | None = None,
        execution_mode: str | None = None,
    ):
        self.message = self.default_message if message is None else message
        super().__init__(self.message)
        self.provider = provider
        self.http_status = http_status
        self.request_id = request_id
        self.raw_payload = raw_payload
        self.cause = cause
        self.execution_mode = execution_mode

    def __str__(self) -> str:
        return (
            f"LLMError(code={self.code}, message={self.message}, retriable={self.retriable}, "
            f"provider={self.provider}, http_status={self.http_status}, "
            f"request_id={self.request_id}, "
            f"execution_mode={self.execution_mode}, cause={repr(self.cause)})"
        )


class LLMModelNotProvidedError(LLMError):
    """
    The LLM model is not provided in the request. Not retriable, as the model must be specified in
    order for the request to succeed on a subsequent attempt.
    """

    code = "model_not_provided"
    default_message = "LLM model not provided in the request."


class LLMModelNotFoundError(LLMError):
    """
    The provider does not have the requested model (e.g., a local model that was never pulled, or
    was removed). Not retriable, as the model must be made available before the request can succeed
    on a subsequent attempt.
    """

    code = "model_not_found"
    default_message = "The LLM provider does not have the requested model."


class LLMInvalidRequestError(LLMError):
    """
    The provider rejects the request as invalid (e.g., an option or a parameter the model does not
    support). Not retriable, as the same request would be rejected again: the code or the
    configuration that built it must be fixed.
    """

    code = "invalid_request"
    default_message = "The LLM provider rejected the request as invalid."


class LLMRefusalError(LLMError):
    """
    The LLM refuses to generate a response (e.g., due to content moderation, policy violations,
    etc.). Not retriable, as the refusal would likely occur again on a subsequent attempt unless the
    input is modified to comply with the LLM's policies.
    """

    code = "refused"
    default_message = "The LLM refused to generate a response for the given input."


class LLMTimeoutError(LLMError):
    """
    The request to the LLM exceeded the allowed time limit. Retriable, as the timeout may have been
    a transient issue that could succeed on a subsequent attempt.
    """

    code = "timeout"
    default_message = "The LLM request timed out."
    retriable = True


class LLMRateLimitError(LLMError):
    """
    The request to the LLM was rejected for exceeding the allowed number of requests in a given time
    period. Retriable, as the rate limit may reset after a certain amount of time, allowing the
    request to succeed on a subsequent attempt.
    """

    code = "rate_limit"
    default_message = "The LLM request was rate limited."
    retriable = True


class LLMAuthError(LLMError):
    """
    The request to the LLM failed due to authentication issues (e.g., invalid API key, expired
    token, etc.). Not retriable, as the authentication issue must be resolved before the request can
    succeed on a subsequent attempt.
    """

    code = "auth"
    default_message = "Authentication with the LLM provider failed."


class LLMParseError(LLMError):
    """
    The response from the LLM could not be parsed or understood. Raised as such only when the
    provider's own answer cannot be read; not retriable, as the parsing logic would need to be
    updated before the request could succeed on a subsequent attempt.

    A model's answer that cannot be used is one of its subclasses, the bad draws: empty, degenerate,
    cut, or malformed. They are retriable: a bad draw is drawn again with a new seed (see
    ``generate_with_retries``), which usually yields a usable answer, and each kind is counted apart
    because each is a different model behaviour. They stay subclasses of ``LLMParseError`` so
    ``except LLMParseError`` handlers keep catching them; after the retry budget is exhausted the
    game still errors out.
    """

    code = "parse_error"
    default_message = "Failed to parse the LLM response."


class LLMEmptyResponseError(LLMParseError):
    """
    A bad draw: the provider returns empty or whitespace-only content. A stochastic non-answer
    (observed intermittently from ollama at temperature > 0).
    """

    code = "empty_response"
    default_message = "The LLM returned an empty response."
    retriable = True


class LLMDegenerateResponseError(LLMParseError):
    """
    A bad draw: a well-formed response whose content is degenerate. It satisfies the (deliberately
    permissive) LLM-facing schema but cannot produce a legal domain object - e.g. an empty
    ``proposals`` list, which parses fine against ``GuessJSONFormat`` but violates
    ``GuessProposal.proposals`` (``min_length=1``).

    Counted apart from an empty response because "returned nothing" and "returned a well-formed
    refusal to guess" are different model behaviours. The domain constraint it protects is not
    loosened: an empty guess list is illegal by game rules.
    """

    code = "degenerate_response"
    default_message = "The LLM returned a well-formed but degenerate response."
    retriable = True


class LLMTruncatedResponseError(LLMParseError):
    """
    A bad draw: an answer the provider cut because it reached the output limit
    (``done_reason``/``finish_reason`` "length"). The model kept writing, usually because it fell
    into a loop, and the JSON is left unfinished; a new draw usually ends normally.
    """

    code = "truncated_response"
    default_message = "The LLM response was cut at the output limit."
    retriable = True


class LLMMalformedResponseError(LLMParseError):
    """
    A bad draw: an answer that does not match the response schema it was asked for (the request's
    ``expected_format``): a field missing or of the wrong type, a value out of range, or no JSON at
    all. With the same seed a local model would write the same answer again; with a new one it
    usually writes a well-formed one.
    """

    code = "malformed_response"
    default_message = "The LLM response does not match the expected format."
    retriable = True


class LLMProviderUnavailableError(LLMError):
    """
    The LLM provider is currently unavailable (e.g., due to maintenance, outages, etc.). Retriable,
    as the provider may become available again after some time, allowing the request to succeed on
    a subsequent attempt.
    """

    code = "provider_unavailable"
    default_message = "The LLM provider is currently unavailable."
    retriable = True
