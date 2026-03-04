from typing import Any


class LLMError(Exception):
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


class LLMTimeoutError(LLMError):
    def __init__(self, message: str = "The LLM request timed out.", provider: str | None = None,
                 http_status: int | None = None, request_id: str | None = None, raw_payload: dict[str, Any] | None = None,
                 cause: Exception | None = None, execution_mode: str | None = None):
        super().__init__(code="timeout", message=message, retriable=True, provider=provider,
                         http_status=http_status, request_id=request_id, raw_payload=raw_payload,
                         cause=cause, execution_mode=execution_mode)


class LLMRateLimitError(LLMError):
    def __init__(self, message: str = "The LLM request was rate limited.", provider: str | None = None,
                 http_status: int | None = None, request_id: str | None = None, raw_payload: dict[str, Any] | None = None,
                 cause: Exception | None = None, execution_mode: str | None = None):
        super().__init__(code="rate_limit", message=message, retriable=True, provider=provider,
                         http_status=http_status, request_id=request_id, raw_payload=raw_payload,
                         cause=cause, execution_mode=execution_mode)


class LLMAuthError(LLMError):
    def __init__(self, message: str = "Authentication with the LLM provider failed.", provider: str | None = None,
                 http_status: int | None = None, request_id: str | None = None,
                 raw_payload: dict[str, Any] | None = None, cause: Exception | None = None,
                 execution_mode: str | None = None):
        super().__init__(code="auth", message=message, retriable=False, provider=provider, http_status=http_status,
                         request_id=request_id, raw_payload=raw_payload, cause=cause, execution_mode=execution_mode)


class LLMParseError(LLMError):
    def __init__(self, message: str = "Failed to parse the LLM response.", provider: str | None = None,
                 http_status: int | None = None, request_id: str | None = None, raw_payload: dict[str, Any] | None = None,
                 cause: Exception | None = None, execution_mode: str | None = None):
        super().__init__(code="parse_error", message=message, retriable=False, provider=provider,
                         http_status=http_status, request_id=request_id, raw_payload=raw_payload,
                         cause=cause, execution_mode=execution_mode)


class LLMProviderUnavailableError(LLMError):
    def __init__(self, message: str = "The LLM provider is currently unavailable.", provider: str | None = None,
                 http_status: int | None = None, request_id: str | None = None, raw_payload: dict[str, Any] | None = None,
                 cause: Exception | None = None, execution_mode: str | None = None):
        super().__init__(code="provider_unavailable", message=message, retriable=True, provider=provider,
                         http_status=http_status, request_id=request_id, raw_payload=raw_payload,
                         cause=cause, execution_mode=execution_mode)
