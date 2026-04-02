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
    be parsed or understood (e.g., invalid JSON, unexpected format, etc.). This error is not 
    retriable, as the response from the LLM would need to be corrected or the parsing logic would 
    need to be updated before the request could succeed on a subsequent attempt.
    """

    def __init__(self, message: str = "Failed to parse the LLM response.",
                 provider: str | None = None, http_status: int | None = None,
                 request_id: str | None = None, raw_payload: dict[str, Any] | None = None,
                 cause: Exception | None = None, execution_mode: str | None = None):
        super().__init__(code="parse_error", message=message, retriable=False, provider=provider,
                         http_status=http_status, request_id=request_id, raw_payload=raw_payload,
                         cause=cause, execution_mode=execution_mode)


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
