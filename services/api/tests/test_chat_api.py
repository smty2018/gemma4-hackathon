from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.api.routes.chat import get_chat_grounding_pipeline
from app.chat.pipeline import ChatGroundingPipeline
from app.inference.gemma import DEFAULT_MODEL_ID, GemmaInferenceError, GemmaRequest, GemmaResponse
from app.main import app


class StubGemma:
    def __init__(self, responses: list[dict[str, Any] | Exception]) -> None:
        self.responses = responses
        self.requests: list[GemmaRequest] = []

    def generate(self, request: GemmaRequest) -> GemmaResponse:
        self.requests.append(request)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return GemmaResponse(model_id=DEFAULT_MODEL_ID, structured=response)


def document_payload() -> dict[str, Any]:
    return {
        "language": "English",
        "audience": "general public",
        "simple_summary": {
            "text": "This is a payment notice for ₹1,250.",
            "evidence_ids": ["E1"],
        },
        "key_facts": [],
        "required_actions": [],
        "warnings": [],
        "source_evidence": [
            {
                "evidence_id": "E1",
                "page": 1,
                "kind": "amount",
                "label": "Amount",
                "value": "₹1,250",
                "normalized_value": "1250",
                "currency": "INR",
                "quote": "Amount due: ₹1,250",
                "confidence": 0.9,
            }
        ],
        "confidence": 0.9,
    }


def override_pipeline(responses: list[dict[str, Any] | Exception]) -> StubGemma:
    gemma = StubGemma(responses)
    app.dependency_overrides[get_chat_grounding_pipeline] = lambda: ChatGroundingPipeline(gemma)
    return gemma


def test_ask_endpoint_returns_grounded_answer() -> None:
    override_pipeline(
        [
            {
                "answer": "Pay ₹1,250 by the due date.",
                "answerable": True,
                "document_evidence_ids": ["E1"],
                "tool_result_ids": [],
                "confidence": 0.9,
            }
        ]
    )
    try:
        response = TestClient(app).post(
            "/api/v1/chat/ask",
            json={
                "question": "How much do I owe?",
                "document": document_payload(),
                "language": "English",
                "explanation_style": "Simple",
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "Pay ₹1,250 by the due date."
    assert payload["answerable"] is True
    assert payload["document_sources"][0]["evidence_id"] == "E1"


def test_ask_endpoint_forwards_history() -> None:
    gemma = override_pipeline(
        [
            {
                "answer": "Yes, that is correct.",
                "answerable": True,
                "document_evidence_ids": ["E1"],
                "tool_result_ids": [],
                "confidence": 0.85,
            }
        ]
    )
    try:
        response = TestClient(app).post(
            "/api/v1/chat/ask",
            json={
                "question": "Is that due date final?",
                "document": document_payload(),
                "history": [
                    {"role": "user", "content": "How much do I owe?"},
                    {"role": "assistant", "content": "Pay ₹1,250 by the due date."},
                ],
                "language": "English",
                "explanation_style": "Simple",
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert "History question" not in response.text
    assert len(gemma.requests) == 1
    assert "How much do I owe?" in gemma.requests[0].prompt


@pytest.mark.parametrize(
    ("response", "expected_status"),
    [
        (GemmaInferenceError("gemma_inference_failed", "boom", retryable=True), 503),
        ({"answer": "", "answerable": False, "confidence": 0}, 422),
    ],
)
def test_ask_endpoint_maps_grounding_errors_to_http_status(
    response: dict[str, Any] | Exception,
    expected_status: int,
) -> None:
    override_pipeline([response])
    try:
        api_response = TestClient(app).post(
            "/api/v1/chat/ask",
            json={
                "question": "How much do I owe?",
                "document": document_payload(),
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert api_response.status_code == expected_status
    assert api_response.json()["detail"]["code"]
