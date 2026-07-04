from collections.abc import Sequence
from typing import Any

import pytest

from app.explanation.pipeline import (
    MAX_EVIDENCE_PROMPT_CHARACTERS,
    ExplanationPipeline,
    ExplanationPipelineError,
    ExplanationRequest,
)
from app.inference.gemma import (
    DEFAULT_MODEL_ID,
    GemmaInferenceError,
    GemmaRequest,
    GemmaResponse,
)
from app.schemas.explanation import ActionUrgency, WarningSeverity
from app.schemas.ocr import (
    OcrAmount,
    OcrDate,
    OcrDocumentResult,
    OcrEvidence,
    OcrPageResult,
    OcrSource,
)


class StubGemma:
    def __init__(self, responses: Sequence[dict[str, Any] | Exception]) -> None:
        self.responses = list(responses)
        self.requests: list[GemmaRequest] = []

    def generate(self, request: GemmaRequest) -> GemmaResponse:
        self.requests.append(request)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return GemmaResponse(
            text="structured explanation",
            model_id=DEFAULT_MODEL_ID,
            structured=response,
        )


def ocr_result(
    *,
    confidence: float = 0.9,
    page_warnings: list[str] | None = None,
) -> OcrDocumentResult:
    page = OcrPageResult(
        page=1,
        source=OcrSource.IMAGE,
        text="Payment notice\nDue date: 10 July 2026\nAmount due: ₹1,250",
        evidence=[
            OcrEvidence(
                page=1,
                label="Document type",
                value="Payment notice",
                evidence_text="Payment notice",
                confidence=0.92,
                grounded=True,
            )
        ],
        dates=[
            OcrDate(
                page=1,
                value="10 July 2026",
                normalized_value="2026-07-10",
                evidence_text="Due date: 10 July 2026",
                confidence=0.9,
                grounded=True,
            )
        ],
        amounts=[
            OcrAmount(
                page=1,
                value="₹1,250",
                normalized_value="1250",
                currency="INR",
                evidence_text="Amount due: ₹1,250",
                confidence=0.94,
                grounded=True,
            )
        ],
        confidence=confidence,
        warnings=page_warnings or [],
    )
    return OcrDocumentResult(
        page_count=1,
        text="--- Page 1 ---\n" + page.text,
        pages=[page],
        evidence=page.evidence,
        dates=page.dates,
        amounts=page.amounts,
        confidence=confidence,
    )


def explanation_payload() -> dict[str, Any]:
    return {
        "simple_summary": {
            "text": "This is a payment notice with a stated due date and amount.",
            "evidence_ids": ["P1"],
        },
        "key_facts": [
            {
                "label": "Amount due",
                "value": "₹1,250",
                "evidence_ids": ["E3", "E3"],
                "confidence": 0.99,
            }
        ],
        "required_actions": [
            {
                "action": "Pay the stated amount by the due date.",
                "deadline": "10 July 2026",
                "urgency": "soon",
                "evidence_ids": ["E2", "E3"],
                "confidence": 0.98,
            }
        ],
        "warnings": [
            {
                "message": "Verify the account details before paying.",
                "severity": "caution",
                "evidence_ids": ["E1"],
            }
        ],
        "confidence": 0.95,
    }


def test_pipeline_returns_grounded_document_explanation() -> None:
    gemma = StubGemma([explanation_payload()])

    result = ExplanationPipeline(gemma).explain(
        ExplanationRequest(
            ocr=ocr_result(),
            language="Hindi",
            audience="first-time reader",
        )
    )

    assert result.language == "Hindi"
    assert result.audience == "first-time reader"
    assert result.simple_summary.evidence_ids == ["P1"]
    assert result.key_facts[0].evidence_ids == ["E3"]
    assert result.key_facts[0].confidence == 0.94
    assert result.required_actions[0].urgency is ActionUrgency.SOON
    assert result.required_actions[0].confidence == 0.9
    assert result.warnings[0].severity is WarningSeverity.CAUTION
    assert [item.evidence_id for item in result.source_evidence] == [
        "P1",
        "E1",
        "E2",
        "E3",
    ]
    assert result.confidence == 0.9

    request = gemma.requests[0]
    assert request.temperature == 0
    assert request.max_new_tokens == 2_048
    assert request.response_schema is not None
    assert request.system_instruction is not None
    assert "in Hindi for first-time reader" in request.prompt
    assert '"id":"E3"' in request.prompt
    assert "<evidence_catalog>" in request.prompt


def test_no_explicit_actions_can_return_an_empty_action_list() -> None:
    payload = explanation_payload()
    payload["required_actions"] = []
    result = ExplanationPipeline(StubGemma([payload])).explain(
        ExplanationRequest(ocr=ocr_result())
    )

    assert result.required_actions == []


def test_ocr_warnings_and_low_confidence_are_preserved() -> None:
    payload = explanation_payload()
    payload["warnings"] = []
    result = ExplanationPipeline(StubGemma([payload])).explain(
        ExplanationRequest(
            ocr=ocr_result(
                confidence=0.5,
                page_warnings=["One extracted item was not grounded."],
            )
        )
    )

    assert len(result.warnings) == 2
    assert result.warnings[0].message.startswith("OCR warning for page 1")
    assert result.warnings[0].evidence_ids == ["P1"]
    assert "confidence is low" in result.warnings[1].message
    assert result.warnings[1].evidence_ids == []
    assert result.confidence == 0.5


def test_unknown_model_citations_reject_the_explanation() -> None:
    payload = explanation_payload()
    payload["key_facts"][0]["evidence_ids"] = ["E999"]

    with pytest.raises(ExplanationPipelineError) as error:
        ExplanationPipeline(StubGemma([payload])).explain(
            ExplanationRequest(ocr=ocr_result())
        )

    assert error.value.code == "explanation_response_invalid"
    assert error.value.retryable is True


def test_empty_ocr_evidence_stops_before_model_inference() -> None:
    page = OcrPageResult(
        page=1,
        source=OcrSource.IMAGE,
        text="",
        confidence=0,
    )
    ocr = OcrDocumentResult(
        page_count=1,
        text="--- Page 1 ---\n",
        pages=[page],
        confidence=0,
    )
    gemma = StubGemma([])

    with pytest.raises(ExplanationPipelineError) as error:
        ExplanationPipeline(gemma).explain(ExplanationRequest(ocr=ocr))

    assert error.value.code == "explanation_evidence_required"
    assert gemma.requests == []


@pytest.mark.parametrize(
    ("language", "audience", "code"),
    [
        ("", "reader", "invalid_explanation_language"),
        ("x" * 33, "reader", "invalid_explanation_language"),
        ("English", "", "invalid_explanation_audience"),
        ("English", "x" * 101, "invalid_explanation_audience"),
    ],
)
def test_invalid_preferences_have_stable_errors(
    language: str,
    audience: str,
    code: str,
) -> None:
    with pytest.raises(ExplanationPipelineError) as error:
        ExplanationPipeline(StubGemma([])).explain(
            ExplanationRequest(
                ocr=ocr_result(),
                language=language,
                audience=audience,
            )
        )

    assert error.value.code == code
    assert error.value.retryable is False


def test_gemma_failure_is_normalized_without_leaking_details() -> None:
    gemma = StubGemma(
        [
            GemmaInferenceError(
                "gemma_inference_failed",
                "private GPU failure",
                retryable=True,
            )
        ]
    )

    with pytest.raises(ExplanationPipelineError) as error:
        ExplanationPipeline(gemma).explain(ExplanationRequest(ocr=ocr_result()))

    assert error.value.code == "explanation_inference_failed"
    assert error.value.retryable is True
    assert "private GPU failure" not in str(error.value)


def test_invalid_structured_output_is_normalized() -> None:
    payload = explanation_payload()
    payload["confidence"] = 2

    with pytest.raises(ExplanationPipelineError) as error:
        ExplanationPipeline(StubGemma([payload])).explain(
            ExplanationRequest(ocr=ocr_result())
        )

    assert error.value.code == "explanation_response_invalid"


def test_evidence_catalog_in_prompt_stays_bounded() -> None:
    pages = []
    for page_number in range(1, 101):
        pages.append(
            OcrPageResult(
                page=page_number,
                source=OcrSource.EMBEDDED_TEXT,
                text="x" * 10_000,
                confidence=0.8,
            )
        )
    ocr = OcrDocumentResult(
        page_count=len(pages),
        text="large document",
        pages=pages,
        confidence=0.8,
    )
    payload = {
        "simple_summary": {"text": "Large document.", "evidence_ids": ["P1"]},
        "key_facts": [],
        "required_actions": [],
        "warnings": [],
        "confidence": 0.8,
    }
    gemma = StubGemma([payload])

    ExplanationPipeline(gemma).explain(ExplanationRequest(ocr=ocr))

    catalog = gemma.requests[0].prompt.split("<evidence_catalog>\n", 1)[1].split(
        "\n</evidence_catalog>", 1
    )[0]
    assert len(catalog) <= MAX_EVIDENCE_PROMPT_CHARACTERS
