import pytest
from pydantic import ValidationError
from backend.app.models.llm_schemas import LLMMessage, TokenUsage, LLMRequest, LLMResponse, \
    ClueProposal, GuessProposal


def test_llm_message_valid():
    """
    Validates that a correctly structured LLMMessage instance passes validation without errors.
    """
    message = LLMMessage(
        role="user", content="Hope it passes validation without errors...")
    assert isinstance(message, LLMMessage)
    assert message.role == "user"
    assert message.content == "Hope it passes validation without errors..."


@pytest.mark.parametrize("content, expected_error", [
    ("", "Content of the message cannot be empty."),
    ("   ", "Content of the message cannot be empty.")
])
def test_llm_message_invalid_content(content, expected_error):
    """
    Validates that an LLMMessage instance with empty or whitespace-only content fails validation.

    :param content: A string representing the content of the message to be tested.
    :param expected_error: A string indicating the expected error message to be raised during 
        validation.
    """
    with pytest.raises(ValidationError) as exc_info:
        LLMMessage(role="user", content=content)

    assert expected_error in str(exc_info.value)


def test_llm_message_invalid_role():
    """
    Validates that an LLMMessage instance with an invalid role fails validation.
    """
    with pytest.raises(ValidationError) as exc_info:
        LLMMessage(role="invalid_role", content="This is a test message.")

    # The error message should indicate that the role value is not one of the allowed literals
    # The exact error message may vary, but it should mention the role field
    assert "role" in str(exc_info.value)


def test_llm_message_missing_fields():
    """
    Validates that an LLMMessage instance missing required fields fails validation.
    """
    with pytest.raises(ValidationError) as exc_info:
        LLMMessage(content="This message is missing the role field.")

    assert "role" in str(exc_info.value)

    with pytest.raises(ValidationError) as exc_info:
        LLMMessage(role="user")

    assert "content" in str(exc_info.value)


def test_token_usage_valid():
    """
    Validates that a correctly structured TokenUsage instance passes validation without errors.
    """
    token_usage = TokenUsage(prompt_tokens=10, completion_tokens=5)

    assert isinstance(token_usage, TokenUsage)
    assert token_usage.prompt_tokens == 10
    assert token_usage.completion_tokens == 5
    assert token_usage.total_tokens == 15


def test_token_usage_invalid_total_tokens():
    """
    Validates that a TokenUsage instance with an incorrect total_tokens value fails validation.
    """
    with pytest.raises(ValidationError) as exc_info:
        TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=20)

    assert "Total tokens must be equal to the sum of prompt tokens and completion tokens." in str(
        exc_info.value)


def test_token_usage_negative_tokens():
    """
    Validates that a TokenUsage instance with negative token counts fails validation.
    """
    with pytest.raises(ValidationError) as exc_info:
        TokenUsage(prompt_tokens=-1, completion_tokens=5)

    assert "prompt_tokens" in str(exc_info.value)

    with pytest.raises(ValidationError) as exc_info:
        TokenUsage(prompt_tokens=10, completion_tokens=-5)

    assert "completion_tokens" in str(exc_info.value)


def test_llm_request_valid():
    """
    Validates that a correctly structured LLMRequest instance passes validation without errors.
    """
    request = LLMRequest(
        messages=[
            LLMMessage(role="system", content="You are a helpful assistant."),
            LLMMessage(role="user", content="Hello, how are you?"),
            LLMMessage(role="assistant", content="I'm doing well, thank you!")
        ],
        model="test-model",
        temperature=0.5,
        max_tokens=100,
        timeout_s=10,
        metadata={"request_id": "12345"},
        response_format="text"
    )

    assert isinstance(request, LLMRequest)
    assert len(request.messages) == 3
    assert request.model == "test-model"
    assert request.temperature == 0.5
    assert request.max_tokens == 100
    assert request.timeout_s == 10
    assert request.metadata == {"request_id": "12345"}
    assert request.response_format == "text"


def test_llm_request_empty_messages():
    """
    Validates that an LLMRequest instance with an empty messages list fails validation.
    """
    with pytest.raises(ValidationError) as exc_info:
        LLMRequest(
            messages=[],
            model="test-model",
            temperature=0.5,
            max_tokens=100,
            timeout_s=10,
            metadata={"request_id": "12345"},
            response_format="text"
        )

    assert "At least one message must be provided in the request." in str(
        exc_info.value)


def test_llm_request_no_user_role_in_message():
    """
    Validates that an LLMRequest instance with messages that do not include a user role fails 
    validation.
    """
    with pytest.raises(ValidationError) as exc_info:
        LLMRequest(
            messages=[
                LLMMessage(role="system",
                           content="This message is missing the role field.")
            ],
            model="test-model",
            temperature=0.5,
            max_tokens=100,
            timeout_s=10,
            metadata={"request_id": "12345"},
            response_format="text"
        )

    assert "At least one message must have the role of 'user'." in str(
        exc_info.value)


def test_llm_request_no_model():
    """
    Validates that an LLMRequest instance without a model specified fails validation.
    """
    with pytest.raises(ValidationError) as exc_info:
        LLMRequest(
            messages=[
                LLMMessage(
                    role="user", content="This request is missing the model field.")
            ],
            temperature=0.5,
            max_tokens=100,
            timeout_s=10,
            metadata={"request_id": "12345"},
            response_format="text"
        )

    assert "model" in str(exc_info.value)


def test_llm_request_empty_model():
    """
    Validates that an LLMRequest instance with an empty model string fails validation.
    """
    with pytest.raises(ValidationError) as exc_info:
        LLMRequest(
            messages=[
                LLMMessage(
                    role="user", content="This request has an empty model field.")
            ],
            model="",
            temperature=0.5,
            max_tokens=100,
            timeout_s=10,
            metadata={"request_id": "12345"},
            response_format="text"
        )

    assert "Model name cannot be empty." in str(exc_info.value)


@pytest.mark.parametrize("temperature, expected_error_substring", [
    (-0.1, "temperature"),
    (2.1, "temperature")
])
def test_llm_request_temperature_out_of_bounds(temperature, expected_error_substring):
    """
    Validates that an LLMRequest instance with a temperature value outside the valid range fails 
    validation.
    """
    with pytest.raises(ValidationError) as exc_info:
        LLMRequest(
            messages=[
                LLMMessage(
                    role="user",
                    content="This request has a temperature value outside the valid range."
                )
            ],
            model="test-model",
            temperature=temperature,
            max_tokens=100,
            timeout_s=10,
            metadata={"request_id": "12345"},
            response_format="text"
        )

    assert expected_error_substring in str(exc_info.value)


def test_llm_request_max_tokens_out_of_bounds():
    """
    Validates that an LLMRequest instance with a max_tokens value less than 1 fails validation.
    """
    with pytest.raises(ValidationError) as exc_info:
        LLMRequest(
            messages=[
                LLMMessage(
                    role="user", content="This request has a max_tokens value less than 1.")
            ],
            model="test-model",
            temperature=0.5,
            max_tokens=0,
            timeout_s=10,
            metadata={"request_id": "12345"},
            response_format="text"
        )

    assert "max_tokens" in str(exc_info.value)


def test_llm_request_timeout_out_of_bounds():
    """
    Validates that an LLMRequest instance with a timeout_s value less than or equal to 0 fails 
    validation.
    """
    with pytest.raises(ValidationError) as exc_info:
        LLMRequest(
            messages=[
                LLMMessage(
                    role="user",
                    content="This request has a timeout_s value less than or equal to 0."
                )
            ],
            model="test-model",
            temperature=0.5,
            max_tokens=100,
            timeout_s=0,
            metadata={"request_id": "12345"},
            response_format="text"
        )

    assert "timeout_s" in str(exc_info.value)


def test_llm_request_invalid_metadata_type():
    """
    Validates that an LLMRequest instance with metadata that is not a dictionary fails validation.
    """
    with pytest.raises(ValidationError) as exc_info:
        LLMRequest(
            messages=[
                LLMMessage(
                    role="user",
                    content="This request has metadata that is not a dictionary."
                )
            ],
            model="test-model",
            temperature=0.5,
            max_tokens=100,
            timeout_s=10,
            metadata="This should be a dictionary, not a string.",
            response_format="text"
        )

    assert "metadata" in str(exc_info.value)


def test_llm_request_invalid_response_format():
    """
    Validates that an LLMRequest instance with an invalid response_format value fails validation.
    """
    with pytest.raises(ValidationError) as exc_info:
        LLMRequest(
            messages=[
                LLMMessage(
                    role="user",
                    content="This request has an invalid response_format value."
                )
            ],
            model="test-model",
            temperature=0.5,
            max_tokens=100,
            timeout_s=10,
            metadata={"request_id": "12345"},
            response_format="invalid_format"
        )

    assert "response_format" in str(exc_info.value)


def test_llm_response_valid():
    """
    Validates that a correctly structured LLMResponse instance passes validation without errors.
    """
    response = LLMResponse(
        text="This is a test response.",
        model_used="test-model",
        latency_ms=100,
        usage=TokenUsage(prompt_tokens=10, completion_tokens=5),
        finish_reason="stop",
        raw_payload={"response": "This is a test response."},
        request_id="12345",
        execution_mode="local",
        provider="ollama"
    )

    assert isinstance(response, LLMResponse)
    assert response.text == "This is a test response."
    assert response.model_used == "test-model"
    assert response.latency_ms == 100
    assert isinstance(response.usage, TokenUsage)
    assert response.usage.prompt_tokens == 10
    assert response.usage.completion_tokens == 5
    assert response.usage.total_tokens == 15
    assert response.finish_reason == "stop"
    assert response.raw_payload == {"response": "This is a test response."}
    assert response.request_id == "12345"
    assert response.execution_mode == "local"
    assert response.provider == "ollama"


def test_llm_response_missing_test_field():
    """
    Validates that an LLMResponse instance missing required fields fails validation.
    """
    with pytest.raises(ValidationError) as exc_info:
        LLMResponse(
            model_used="test-model",
            latency_ms=100,
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5),
            finish_reason="stop",
            raw_payload={"response": "This is a test response."},
            request_id="12345",
            execution_mode="local",
            provider="ollama"
        )

    assert "text" in str(exc_info.value)


def test_llm_response_empty_text():
    """
    Validates that an LLMResponse instance with empty or whitespace-only text fails validation.
    """
    with pytest.raises(ValidationError) as exc_info:
        LLMResponse(
            text="   ",
            model_used="test-model",
            latency_ms=100,
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5),
            finish_reason="stop",
            raw_payload={"response": "This is a test response."},
            request_id="12345",
            execution_mode="local",
            provider="ollama"
        )

    assert "text" in str(exc_info.value)


def test_llm_response_model_not_provided():
    """
    Validates that an LLMResponse instance with an empty or whitespace-only model_used field fails 
    validation.
    """
    with pytest.raises(ValidationError) as exc_info:
        LLMResponse(
            text="This is a test response.",
            latency_ms=100,
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5),
            finish_reason="stop",
            raw_payload={"response": "This is a test response."},
            request_id="12345",
            execution_mode="local",
            provider="ollama"
        )

    assert "model_used" in str(exc_info.value)


def test_llm_response_model_used_empty():
    """
    Validates that an LLMResponse instance with an empty model_used field fails validation.
    """
    with pytest.raises(ValidationError) as exc_info:
        LLMResponse(
            text="This is a test response.",
            model_used=" ",
            latency_ms=100,
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5),
            finish_reason="stop",
            raw_payload={"response": "This is a test response."},
            request_id="12345",
            execution_mode="local",
            provider="ollama"
        )

    assert "Model used cannot be empty." in str(exc_info.value)


def test_llm_response_negative_latency():
    """
    Validates that an LLMResponse instance with a negative latency_ms value fails validation.
    """
    with pytest.raises(ValidationError) as exc_info:
        LLMResponse(
            text="This is a test response.",
            model_used="test-model",
            latency_ms=-10,
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5),
            finish_reason="stop",
            raw_payload={"response": "This is a test response."},
            request_id="12345",
            execution_mode="local",
            provider="ollama"
        )

    assert "latency_ms" in str(exc_info.value)


def test_llm_response_invalid_usage():
    """
    Validates that an LLMResponse instance with an invalid usage field fails validation.
    """
    with pytest.raises(ValidationError) as exc_info:
        LLMResponse(
            text="This is a test response.",
            model_used="test-model",
            latency_ms=100,
            usage={"invalid": "usage"},
            finish_reason="stop",
            raw_payload={"response": "This is a test response."},
            request_id="12345",
            execution_mode="local",
            provider="ollama"
        )

    assert "usage" in str(exc_info.value)


def test_llm_response_invalid_finish_reason():
    """
    Validates that an LLMResponse instance with an invalid finish_reason field fails validation.
    """
    with pytest.raises(ValidationError) as exc_info:
        LLMResponse(
            text="This is a test response.",
            model_used="test-model",
            latency_ms=100,
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5),
            finish_reason=123,  # Invalid type for finish_reason
            raw_payload={"response": "This is a test response."},
            request_id="12345",
            execution_mode="local",
            provider="ollama"
        )

    assert "finish_reason" in str(exc_info.value)


def test_llm_response_invalid_raw_payload():
    """
    Validates that an LLMResponse instance with an invalid raw_payload field fails validation.
    """
    with pytest.raises(ValidationError) as exc_info:
        LLMResponse(
            text="This is a test response.",
            model_used="test-model",
            latency_ms=100,
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5),
            finish_reason="stop",
            # Invalid type for raw_payload
            raw_payload="This should be a dictionary, not a string.",
            request_id="12345",
            execution_mode="local",
            provider="ollama"
        )

    assert "raw_payload" in str(exc_info.value)


def test_llm_response_invalid_request_id():
    """
    Validates that an LLMResponse instance with an invalid request_id field fails validation.
    """
    with pytest.raises(ValidationError) as exc_info:
        LLMResponse(
            text="This is a test response.",
            model_used="test-model",
            latency_ms=100,
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5),
            finish_reason="stop",
            raw_payload={"response": "This is a test response."},
            request_id=12345,  # Invalid type for request_id
            execution_mode="local",
            provider="ollama"
        )

    assert "request_id" in str(exc_info.value)


def test_llm_response_invalid_execution_mode():
    """
    Validates that an LLMResponse instance with an invalid execution_mode field fails validation.
    """
    with pytest.raises(ValidationError) as exc_info:
        LLMResponse(
            text="This is a test response.",
            model_used="test-model",
            latency_ms=100,
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5),
            finish_reason="stop",
            raw_payload={"response": "This is a test response."},
            request_id="12345",
            execution_mode="invalid_mode",  # Invalid value for execution_mode
            provider="ollama"
        )

    assert "execution_mode" in str(exc_info.value)


def test_llm_response_invalid_provider():
    """
    Validates that an LLMResponse instance with an invalid provider field fails validation.
    """
    with pytest.raises(ValidationError) as exc_info:
        LLMResponse(
            text="This is a test response.",
            model_used="test-model",
            latency_ms=100,
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5),
            finish_reason="stop",
            raw_payload={"response": "This is a test response."},
            request_id="12345",
            execution_mode="local",
            provider=123  # Invalid type for provider
        )

    assert "provider" in str(exc_info.value)


def test_llm_clue_proposal_valid():
    """
    Validates that a correctly structured ClueProposal instance passes validation without errors.
    """
    proposal = ClueProposal(
        clue="Test clue",
        count=3,
        reasoning="This is the reasoning for the clue proposal.",
        raw_payload={"proposal": "Test clue proposal."}
    )

    assert isinstance(proposal, ClueProposal)
    assert proposal.clue == "Test clue"
    assert proposal.count == 3
    assert proposal.reasoning == "This is the reasoning for the clue proposal."
    assert proposal.raw_payload == {"proposal": "Test clue proposal."}


def test_llm_clue_proposal_not_clue():
    """
    Validates that a ClueProposal instance missing the clue field fails validation.
    """
    with pytest.raises(ValidationError) as exc_info:
        ClueProposal(
            count=3,
            reasoning="This is the reasoning for the clue proposal.",
            raw_payload={"proposal": "Test clue proposal."}
        )

    assert "clue" in str(exc_info.value)


def test_llm_clue_proposal_empty_clue():
    """
    Validates that a ClueProposal instance with an empty clue field fails validation.
    """
    with pytest.raises(ValidationError) as exc_info:
        ClueProposal(
            clue="   ",
            count=3,
            reasoning="This is the reasoning for the clue proposal.",
            raw_payload={"proposal": "Test clue proposal."}
        )

    assert "clue" in str(exc_info.value)


def test_llm_clue_proposal_count_out_of_bounds():
    """
    Validates that a ClueProposal instance with a count value less than 1 fails validation.
    """
    with pytest.raises(ValidationError) as exc_info:
        ClueProposal(
            clue="Test clue",
            count=0,
            reasoning="This is the reasoning for the clue proposal.",
            raw_payload={"proposal": "Test clue proposal."}
        )

    assert "count" in str(exc_info.value)


def test_llm_clue_proposal_invalid_reasoning():
    """
    Validates that a ClueProposal instance with an invalid reasoning field fails validation.
    """
    with pytest.raises(ValidationError) as exc_info:
        ClueProposal(
            clue="Test clue",
            count=3,
            reasoning=123,  # Invalid type for reasoning
            raw_payload={"proposal": "Test clue proposal."}
        )

    assert "reasoning" in str(exc_info.value)


def test_llm_clue_proposal_invalid_raw_payload():
    """
    Validates that a ClueProposal instance with an invalid raw_payload field fails validation.
    """
    with pytest.raises(ValidationError) as exc_info:
        ClueProposal(
            clue="Test clue",
            count=3,
            reasoning="This is the reasoning for the clue proposal.",
            # Invalid type for raw_payload
            raw_payload="This should be a dictionary, not a string."
        )

    assert "raw_payload" in str(exc_info.value)


def test_llm_guess_proposal_valid():
    """
    Validates that a correctly structured GuessProposal instance passes validation without errors.
    """
    proposal = GuessProposal(
        proposals=["Card1", "Card2", "Card3"],
        confidence=[0.9, 0.8, 0.7],
        reasoning="This is the reasoning for the guess proposal.",
        stop_reason="max_tokens",
        raw_payload={"proposal": "Test guess proposal."}
    )

    assert isinstance(proposal, GuessProposal)
    assert proposal.proposals == ["Card1", "Card2", "Card3"]
    assert proposal.confidence == [0.9, 0.8, 0.7]
    assert proposal.reasoning == "This is the reasoning for the guess proposal."
    assert proposal.stop_reason == "max_tokens"
    assert proposal.raw_payload == {"proposal": "Test guess proposal."}


def test_llm_guess_proposal_empty_proposals():
    """
    Validates that a GuessProposal instance with an empty proposals list fails validation.
    """
    with pytest.raises(ValidationError) as exc_info:
        GuessProposal(
            proposals=[],
            confidence=[0.9, 0.8, 0.7],
            reasoning="This is the reasoning for the guess proposal.",
            stop_reason="max_tokens",
            raw_payload={"proposal": "Test guess proposal."}
        )

    assert "proposals" in str(exc_info.value)


def test_llm_guess_proposal_invalid_proposal():
    """
    Validates that a GuessProposal instance with an invalid proposal in the proposals list fails 
    validation.
    """
    with pytest.raises(ValidationError) as exc_info:
        GuessProposal(
            # Invalid proposal (whitespace-only)
            proposals=["Card1", "   ", "Card3"],
            confidence=[0.9, 0.8, 0.7],
            reasoning="This is the reasoning for the guess proposal.",
            stop_reason="max_tokens",
            raw_payload={"proposal": "Test guess proposal."}
        )

    assert "Proposals cannot contain empty strings." in str(exc_info.value)


def test_llm_guess_proposal_empty_confidence():
    """
    Validates that a GuessProposal instance with an empty confidence list fails validation.
    """
    with pytest.raises(ValidationError) as exc_info:
        GuessProposal(
            proposals=["Card1", "Card2", "Card3"],
            confidence=[],
            reasoning="This is the reasoning for the guess proposal.",
            stop_reason="max_tokens",
            raw_payload={"proposal": "Test guess proposal."}
        )

    assert "confidence" in str(exc_info.value)


def test_llm_guess_proposal_proposals_confidence_length_mismatch():
    """
    Validates that a GuessProposal instance where the length of the proposals list does not match 
    the length of the confidence list fails validation.
    """
    with pytest.raises(ValidationError) as exc_info:
        GuessProposal(
            proposals=["Card1", "Card2"],
            confidence=[0.9, 0.8, 0.7],  # Length mismatch with proposals
            reasoning="This is the reasoning for the guess proposal.",
            stop_reason="max_tokens",
            raw_payload={"proposal": "Test guess proposal."}
        )

    assert "The number of proposals must match the number of confidence scores." in str(
        exc_info.value)


def test_llm_guess_proposal_confidence_out_of_bounds():
    """
    Validates that a GuessProposal instance with confidence scores outside the range of 0 to 1 fails 
    validation.
    """
    with pytest.raises(ValidationError) as exc_info:
        GuessProposal(
            proposals=["Card1", "Card2", "Card3"],
            confidence=[0.9, -0.1, 0.7],  # Invalid confidence score
            reasoning="This is the reasoning for the guess proposal.",
            stop_reason="max_tokens",
            raw_payload={"proposal": "Test guess proposal."}
        )

    assert "Confidence scores must be between 0 and 1." in str(exc_info.value)


def test_llm_guess_proposal_invalid_reasoning():
    """
    Validates that a GuessProposal instance with an invalid reasoning field fails validation.
    """
    with pytest.raises(ValidationError) as exc_info:
        GuessProposal(
            proposals=["Card1", "Card2", "Card3"],
            confidence=[0.9, 0.8, 0.7],
            reasoning=123,  # Invalid type for reasoning
            stop_reason="max_tokens",
            raw_payload={"proposal": "Test guess proposal."}
        )

    assert "reasoning" in str(exc_info.value)


def test_llm_guess_proposal_invalid_stop_reason():
    """
    Validates that a GuessProposal instance with an invalid stop_reason field fails validation.
    """
    with pytest.raises(ValidationError) as exc_info:
        GuessProposal(
            proposals=["Card1", "Card2", "Card3"],
            confidence=[0.9, 0.8, 0.7],
            reasoning="This is the reasoning for the guess proposal.",
            stop_reason=123,  # Invalid type for stop_reason
            raw_payload={"proposal": "Test guess proposal."}
        )

    assert "stop_reason" in str(exc_info.value)


def test_llm_guess_proposal_invalid_raw_payload():
    """
    Validates that a GuessProposal instance with an invalid raw_payload field fails validation.
    """
    with pytest.raises(ValidationError) as exc_info:
        GuessProposal(
            proposals=["Card1", "Card2", "Card3"],
            confidence=[0.9, 0.8, 0.7],
            reasoning="This is the reasoning for the guess proposal.",
            stop_reason="max_tokens",
            # Invalid type for raw_payload
            raw_payload="This should be a dictionary, not a string."
        )

    assert "raw_payload" in str(exc_info.value)
