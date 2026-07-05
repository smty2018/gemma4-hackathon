import json
from pathlib import Path
from typing import Any

import pytest

from app.chat.pipeline import ChatGroundingError
from app.demo import DemoFlow, DemoFlowRequest
from app.inference.gemma import DEFAULT_MODEL_ID, GemmaRequest, GemmaResponse
from app.ocr.pipeline import OcrPipelineError

REPOSITORY_ROOT = Path(__file__).parents[3]
DEMO_ROOT = REPOSITORY_ROOT / "samples" / "demo"


class DemoGemmaStub:
    def __init__(
        self,
        *,
        expected: dict[str, str],
        document_text: str,
        invalid_chat_citation: bool = False,
    ) -> None:
        self.expected = expected
        self.document_text = document_text
        self.invalid_chat_citation = invalid_chat_citation
        self.requests: list[GemmaRequest] = []

    def generate(self, request: GemmaRequest) -> GemmaResponse:
        self.requests.append(request)
        schema_name = getattr(request.response_schema, "__name__", "")
        if schema_name == "_PageExtractionPayload":
            return GemmaResponse(
                model_id=DEFAULT_MODEL_ID,
                structured={
                    "text": self.document_text,
                    "evidence": [
                        {
                            "label": "Document type",
                            "value": self.expected["document_label"],
                            "evidence_text": self.expected["document_label"],
                            "confidence": 0.98,
                        }
                    ],
                    "dates": [
                        {
                            "value": self.expected["date_value"],
                            "normalized_value": "2026-07-10",
                            "evidence_text": self.expected["date_evidence"],
                            "confidence": 0.96,
                        }
                    ],
                    "amounts": [
                        {
                            "value": "₹1,250",
                            "normalized_value": "1250",
                            "currency": "INR",
                            "evidence_text": self.expected["amount_evidence"],
                            "confidence": 0.97,
                        }
                    ],
                    "confidence": 0.96,
                },
            )
        if schema_name == "_ExplanationPayload":
            return GemmaResponse(
                model_id=DEFAULT_MODEL_ID,
                structured={
                    "simple_summary": {
                        "text": self.expected["summary"],
                        "evidence_ids": ["P1"],
                    },
                    "key_facts": [
                        {
                            "label": "Amount",
                            "value": "₹1,250",
                            "evidence_ids": ["E3"],
                            "confidence": 0.97,
                        }
                    ],
                    "required_actions": [
                        {
                            "action": self.expected["action"],
                            "deadline": self.expected["date_value"],
                            "urgency": "soon",
                            "evidence_ids": ["E2", "E3"],
                            "confidence": 0.95,
                        }
                    ],
                    "warnings": [],
                    "confidence": 0.95,
                },
            )
        if schema_name == "_GroundedAnswerPayload":
            evidence_ids = ["E404"] if self.invalid_chat_citation else ["E2", "E3"]
            return GemmaResponse(
                model_id=DEFAULT_MODEL_ID,
                structured={
                    "answer": self.expected["follow_up"],
                    "answerable": True,
                    "document_evidence_ids": evidence_ids,
                    "tool_result_ids": [],
                    "confidence": 0.94,
                },
            )
        raise AssertionError(f"Unexpected demo schema: {schema_name}")


def demo_cases() -> list[dict[str, Any]]:
    return json.loads(
        (DEMO_ROOT / "expected-extractions.json").read_text(encoding="utf-8")
    )


@pytest.mark.parametrize("expected", demo_cases(), ids=lambda item: item["language"])
def test_sample_documents_complete_polished_demo_flow(expected: dict[str, str]) -> None:
    document_text = (DEMO_ROOT / expected["file"]).read_text(encoding="utf-8")
    gemma = DemoGemmaStub(expected=expected, document_text=document_text)

    result = DemoFlow(gemma).run(
        DemoFlowRequest(
            document_text=document_text,
            language=expected["language"],
            follow_up_question="What should I do and by when?",
        )
    )

    assert result.ocr.dates[0].normalized_value == "2026-07-10"
    assert result.ocr.amounts[0].normalized_value == "1250"
    assert result.ocr.amounts[0].currency == "INR"
    assert result.ocr.dates[0].grounded is True
    assert result.ocr.amounts[0].grounded is True
    assert result.explanation.simple_summary.text == expected["summary"]
    assert result.explanation.required_actions[0].evidence_ids == ["E2", "E3"]
    assert result.follow_up.answer == expected["follow_up"]
    assert result.follow_up.document_evidence_ids == ["E2", "E3"]
    assert result.follow_up.answerable is True
    assert len(gemma.requests) == 3
    assert expected["summary"] in gemma.requests[2].prompt
    assert "DEMO-1024" in document_text


def test_empty_demo_document_fails_before_model_inference() -> None:
    expected = demo_cases()[0]
    gemma = DemoGemmaStub(expected=expected, document_text="")

    with pytest.raises(OcrPipelineError) as error:
        DemoFlow(gemma).run(DemoFlowRequest(document_text=""))

    assert error.value.code == "empty_ocr_page"
    assert gemma.requests == []


def test_demo_rejects_hallucinated_follow_up_citation() -> None:
    expected = demo_cases()[0]
    document_text = (DEMO_ROOT / expected["file"]).read_text(encoding="utf-8")
    gemma = DemoGemmaStub(
        expected=expected,
        document_text=document_text,
        invalid_chat_citation=True,
    )

    with pytest.raises(ChatGroundingError) as error:
        DemoFlow(gemma).run(DemoFlowRequest(document_text=document_text))

    assert error.value.code == "chat_grounding_response_invalid"
