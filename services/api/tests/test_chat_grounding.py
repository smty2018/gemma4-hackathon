from datetime import UTC, datetime
from typing import Any

import pytest

from app.chat.pipeline import (
    MAX_CHAT_HISTORY_TURNS,
    ChatGroundingError,
    ChatGroundingPipeline,
    GroundedChatRequest,
)
from app.inference.gemma import (
    DEFAULT_MODEL_ID,
    GemmaInferenceError,
    GemmaRequest,
    GemmaResponse,
)
from app.schemas.chat import ChatRole, ChatTurn
from app.schemas.explanation import (
    DocumentExplanation,
    EvidenceKind,
    ExplanationSummary,
    SourceEvidence,
)
from app.tools.base import ToolResult
from app.tools.executor import ToolExecutionReceipt


class StubGemma:
    def __init__(self, responses: list[dict[str, Any] | Exception]) -> None:
        self.responses = responses
        self.requests: list[GemmaRequest] = []

    def generate(self, request: GemmaRequest) -> GemmaResponse:
        self.requests.append(request)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return GemmaResponse(
            model_id=DEFAULT_MODEL_ID,
            structured=response,
        )


def document_explanation(*, confidence: float = 0.82) -> DocumentExplanation:
    source = SourceEvidence(
        evidence_id="E1",
        page=1,
        kind=EvidenceKind.AMOUNT,
        label="Amount",
        value="₹1,250",
        normalized_value="1250",
        currency="INR",
        quote="Amount due: ₹1,250",
        confidence=confidence,
    )
    return DocumentExplanation(
        language="English",
        audience="general public",
        simple_summary=ExplanationSummary(
            text="A payment of ₹1,250 is due.",
            evidence_ids=["E1"],
        ),
        source_evidence=[source],
        confidence=confidence,
    )


def tool_receipt(*, ok: bool = True, data: dict[str, Any] | None = None) -> ToolExecutionReceipt:
    return ToolExecutionReceipt(
        proposal_id="proposal-123456789",
        tool_name="add_amounts",
        arguments={"amounts": ["1250", "250"]},
        result=ToolResult(
            ok=ok,
            summary="Total: 1500",
            data=data or {"total": "1500"},
        ),
        executed_at=datetime(2026, 7, 5, 10, 30, tzinfo=UTC),
    )


def test_follow_up_answer_is_grounded_in_document_and_tool_results() -> None:
    gemma = StubGemma(
        [
            {
                "answer": "The document amount is ₹1,250; the calculated total is ₹1,500.",
                "answerable": True,
                "document_evidence_ids": ["E1", "E1"],
                "tool_result_ids": ["T1"],
                "confidence": 0.97,
            }
        ]
    )
    history = [
        ChatTurn(role=ChatRole.USER, content="What amount is due?"),
        ChatTurn(role=ChatRole.ASSISTANT, content="The amount due is ₹1,250."),
    ]

    answer = ChatGroundingPipeline(gemma).answer(
        GroundedChatRequest(
            question="What is that amount after adding ₹250?",
            document=document_explanation(),
            tool_results=[tool_receipt()],
            history=history,
            language="English",
            explanation_style="Simple",
        )
    )

    assert answer.answerable is True
    assert answer.document_evidence_ids == ["E1"]
    assert answer.tool_result_ids == ["T1"]
    assert answer.document_sources[0].page == 1
    assert answer.tool_sources[0].data == {"total": "1500"}
    assert answer.confidence == 0.82

    request = gemma.requests[0]
    assert request.response_schema is not None
    assert request.temperature == 0
    assert "What amount is due?" in request.prompt
    assert "The amount due is ₹1,250." in request.prompt
    assert '"id":"E1"' in request.prompt
    assert '"id":"T1"' in request.prompt


@pytest.mark.parametrize(
    ("language", "expected"),
    [
        ("English", "I cannot answer"),
        ("Hindi", "मैं अपलोड"),
        ("Bengali", "আপলোড করা"),
    ],
)
def test_unanswerable_questions_use_localized_deterministic_refusal(
    language: str,
    expected: str,
) -> None:
    gemma = StubGemma(
        [
            {
                "answer": "Invented answer that must be ignored.",
                "answerable": False,
                "document_evidence_ids": [],
                "tool_result_ids": [],
                "confidence": 0.9,
            }
        ]
    )

    answer = ChatGroundingPipeline(gemma).answer(
        GroundedChatRequest(
            question="What is tomorrow's weather?",
            document=document_explanation(),
            language=language,
        )
    )

    assert answer.answerable is False
    assert expected in answer.answer
    assert "Invented" not in answer.answer
    assert answer.confidence == 0
    assert answer.document_sources == []


@pytest.mark.parametrize(
    "payload",
    [
        {
            "answer": "Unsupported.",
            "answerable": True,
            "document_evidence_ids": [],
            "tool_result_ids": [],
            "confidence": 0.8,
        },
        {
            "answer": "Unknown document citation.",
            "answerable": True,
            "document_evidence_ids": ["E99"],
            "tool_result_ids": [],
            "confidence": 0.8,
        },
        {
            "answer": "Unknown tool citation.",
            "answerable": True,
            "document_evidence_ids": [],
            "tool_result_ids": ["T99"],
            "confidence": 0.8,
        },
    ],
)
def test_unsupported_or_unknown_citations_are_rejected(payload: dict[str, Any]) -> None:
    with pytest.raises(ChatGroundingError) as error:
        ChatGroundingPipeline(StubGemma([payload])).answer(
            GroundedChatRequest(
                question="Answer this.",
                document=document_explanation(),
                tool_results=[tool_receipt()],
            )
        )

    assert error.value.code == "chat_grounding_response_invalid"
    assert error.value.retryable is True


def test_failed_tool_result_caps_answer_confidence() -> None:
    payload = {
        "answer": "The tool reported a failure result.",
        "answerable": True,
        "document_evidence_ids": [],
        "tool_result_ids": ["T1"],
        "confidence": 0.9,
    }
    answer = ChatGroundingPipeline(StubGemma([payload])).answer(
        GroundedChatRequest(
            question="What did the tool report?",
            document=document_explanation(),
            tool_results=[tool_receipt(ok=False)],
        )
    )

    assert answer.confidence == 0.25
    assert answer.tool_sources[0].succeeded is False


def test_model_failure_is_sanitized() -> None:
    gemma = StubGemma(
        [
            GemmaInferenceError(
                "gemma_inference_failed",
                "private model details",
                retryable=True,
            )
        ]
    )

    with pytest.raises(ChatGroundingError) as error:
        ChatGroundingPipeline(gemma).answer(
            GroundedChatRequest(
                question="What is due?",
                document=document_explanation(),
            )
        )

    assert error.value.code == "chat_grounding_inference_failed"
    assert error.value.retryable is True
    assert "private model details" not in str(error.value)


@pytest.mark.parametrize(
    ("request_factory", "code"),
    [
        (
            lambda: GroundedChatRequest(question="", document=document_explanation()),
            "invalid_chat_question",
        ),
        (
            lambda: GroundedChatRequest(
                question="Question",
                document=document_explanation(),
                history=[
                    ChatTurn(role=ChatRole.USER, content="turn")
                    for _ in range(MAX_CHAT_HISTORY_TURNS + 1)
                ],
            ),
            "chat_history_too_long",
        ),
        (
            lambda: GroundedChatRequest(
                question="Question",
                document=document_explanation(),
                tool_results=[tool_receipt() for _ in range(21)],
            ),
            "chat_tool_results_limit",
        ),
        (
            lambda: GroundedChatRequest(
                question="Question",
                document=document_explanation(),
                tool_results=[tool_receipt(data={"large": "x" * 90_000})],
            ),
            "chat_grounding_context_too_large",
        ),
    ],
)
def test_invalid_chat_context_stops_before_inference(request_factory, code: str) -> None:
    gemma = StubGemma([])

    with pytest.raises(ChatGroundingError) as error:
        ChatGroundingPipeline(gemma).answer(request_factory())

    assert error.value.code == code
    assert gemma.requests == []
