from collections.abc import Sequence
from typing import Any

import pytest

from app.inference.gemma import (
    DEFAULT_MODEL_ID,
    GemmaInferenceError,
    GemmaRequest,
    GemmaResponse,
)
from app.ocr.pipeline import (
    MAX_EMBEDDED_TEXT_CHARACTERS,
    MAX_OCR_PAGES,
    OcrPageInput,
    OcrPipeline,
    OcrPipelineError,
)
from app.schemas.ocr import OcrSource


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
            text="structured response",
            model_id=DEFAULT_MODEL_ID,
            structured=response,
        )


def page_payload(
    text: str,
    *,
    confidence: float = 0.9,
    evidence: list[dict[str, Any]] | None = None,
    dates: list[dict[str, Any]] | None = None,
    amounts: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "text": text,
        "evidence": evidence or [],
        "dates": dates or [],
        "amounts": amounts or [],
        "confidence": confidence,
    }


def test_pipeline_extracts_and_flattens_page_level_results() -> None:
    first_page = page_payload(
        "Invoice date: 5 July 2026\nTotal due: ₹1,250.00",
        confidence=0.95,
        evidence=[
            {
                "label": "Document type",
                "value": "Invoice",
                "evidence_text": "Invoice date",
                "confidence": 0.9,
            }
        ],
        dates=[
            {
                "value": "5 July 2026",
                "normalized_value": "2026-07-05",
                "evidence_text": "Invoice date: 5 July 2026",
                "confidence": 0.88,
            }
        ],
        amounts=[
            {
                "value": "₹1,250.00",
                "normalized_value": "1250.00",
                "currency": "INR",
                "evidence_text": "Total due: ₹1,250.00",
                "confidence": 0.92,
            }
        ],
    )
    second_page = page_payload("Payment instructions", confidence=0.8)
    gemma = StubGemma([first_page, second_page])
    image = object()

    result = OcrPipeline(gemma).extract(
        [
            OcrPageInput(page=2, embedded_text="Payment instructions"),
            OcrPageInput(page=1, image=image),
        ]
    )

    assert [page.page for page in result.pages] == [1, 2]
    assert result.page_count == 2
    assert result.pages[0].source is OcrSource.IMAGE
    assert result.pages[1].source is OcrSource.EMBEDDED_TEXT
    assert result.text.startswith("--- Page 1 ---\nInvoice date")
    assert "--- Page 2 ---\nPayment instructions" in result.text

    assert result.evidence[0].page == 1
    assert result.evidence[0].grounded is True
    assert result.dates[0].normalized_value == "2026-07-05"
    assert result.dates[0].page == 1
    assert result.amounts[0].normalized_value == "1250.00"
    assert result.amounts[0].currency == "INR"
    assert result.amounts[0].page == 1
    assert result.pages[0].confidence == pytest.approx(0.9)
    assert result.confidence == pytest.approx(0.85)

    assert gemma.requests[0].images == (image,)
    assert gemma.requests[0].temperature == 0
    assert gemma.requests[0].response_schema is not None
    assert "page 1" in gemma.requests[0].prompt
    assert gemma.requests[1].images == ()
    assert "<document_text>\nPayment instructions" in gemma.requests[1].prompt


def test_image_and_embedded_text_are_used_together() -> None:
    gemma = StubGemma([page_payload("Visible text")])

    result = OcrPipeline(gemma).extract(
        [OcrPageInput(page=3, image=object(), embedded_text="Embedded hint")]
    )

    assert result.pages[0].source is OcrSource.IMAGE_AND_EMBEDDED_TEXT
    assert len(gemma.requests[0].images) == 1
    assert "Embedded hint" in gemma.requests[0].prompt


def test_ungrounded_items_are_flagged_and_confidence_is_capped() -> None:
    gemma = StubGemma(
        [
            page_payload(
                "Visible total: ₹500",
                confidence=0.96,
                amounts=[
                    {
                        "value": "₹9,999",
                        "normalized_value": "9999",
                        "currency": "INR",
                        "evidence_text": "Secret total: ₹9,999",
                        "confidence": 0.99,
                    }
                ],
            )
        ]
    )

    result = OcrPipeline(gemma).extract([OcrPageInput(page=1, image=object())])

    amount = result.amounts[0]
    assert amount.grounded is False
    assert amount.confidence == 0.25
    assert result.pages[0].confidence == 0.25
    assert result.pages[0].warnings == [
        "1 extracted item(s) could not be matched to the page text."
    ]


def test_matching_ignores_case_and_whitespace_differences() -> None:
    gemma = StubGemma(
        [
            page_payload(
                "DUE DATE:\n 10 JULY 2026",
                dates=[
                    {
                        "value": "10 July 2026",
                        "normalized_value": "2026-07-10",
                        "evidence_text": "due date: 10 july 2026",
                        "confidence": 0.9,
                    }
                ],
            )
        ]
    )

    result = OcrPipeline(gemma).extract(
        [OcrPageInput(page=1, embedded_text="DUE DATE: 10 JULY 2026")]
    )

    assert result.dates[0].grounded is True
    assert result.dates[0].confidence == 0.9
    assert result.pages[0].warnings == []


@pytest.mark.parametrize(
    ("pages", "code", "page"),
    [
        ([], "ocr_pages_required", None),
        ([OcrPageInput(page=0, image=object())], "invalid_ocr_page_number", 0),
        (
            [
                OcrPageInput(page=1, image=object()),
                OcrPageInput(page=1, embedded_text="duplicate"),
            ],
            "duplicate_ocr_page",
            1,
        ),
        ([OcrPageInput(page=4)], "empty_ocr_page", 4),
        (
            [
                OcrPageInput(
                    page=2,
                    embedded_text="x" * (MAX_EMBEDDED_TEXT_CHARACTERS + 1),
                )
            ],
            "ocr_page_text_too_large",
            2,
        ),
        (
            [OcrPageInput(page=index + 1, image=object()) for index in range(MAX_OCR_PAGES + 1)],
            "ocr_page_limit_exceeded",
            None,
        ),
    ],
)
def test_invalid_page_batches_have_stable_errors(
    pages: list[OcrPageInput],
    code: str,
    page: int | None,
) -> None:
    with pytest.raises(OcrPipelineError) as error:
        OcrPipeline(StubGemma([])).extract(pages)

    assert error.value.code == code
    assert error.value.page == page
    assert error.value.retryable is False


def test_gemma_failure_is_scoped_to_the_page_without_leaking_details() -> None:
    gemma = StubGemma(
        [
            GemmaInferenceError(
                "gemma_inference_failed",
                "private GPU details",
                retryable=True,
            )
        ]
    )

    with pytest.raises(OcrPipelineError) as error:
        OcrPipeline(gemma).extract([OcrPageInput(page=7, image=object())])

    assert error.value.code == "ocr_page_inference_failed"
    assert error.value.page == 7
    assert error.value.retryable is True
    assert "private GPU details" not in str(error.value)


def test_invalid_structured_result_is_normalized() -> None:
    gemma = StubGemma(
        [
            {
                "text": "Total: ₹500",
                "evidence": [],
                "dates": [],
                "amounts": [],
                "confidence": 3,
            }
        ]
    )

    with pytest.raises(OcrPipelineError) as error:
        OcrPipeline(gemma).extract([OcrPageInput(page=2, image=object())])

    assert error.value.code == "ocr_page_response_invalid"
    assert error.value.page == 2
    assert error.value.retryable is True


@pytest.mark.parametrize(
    ("dates", "amounts"),
    [
        (
            [
                {
                    "value": "next Friday",
                    "normalized_value": "next-Friday",
                    "evidence_text": "next Friday",
                    "confidence": 0.8,
                }
            ],
            [],
        ),
        (
            [],
            [
                {
                    "value": "₹1,250",
                    "normalized_value": "1,250",
                    "currency": "INR",
                    "evidence_text": "₹1,250",
                    "confidence": 0.8,
                }
            ],
        ),
    ],
)
def test_invalid_normalized_dates_and_amounts_are_rejected(
    dates: list[dict[str, Any]],
    amounts: list[dict[str, Any]],
) -> None:
    gemma = StubGemma(
        [
            page_payload(
                "Pay ₹1,250 next Friday",
                dates=dates,
                amounts=amounts,
            )
        ]
    )

    with pytest.raises(OcrPipelineError) as error:
        OcrPipeline(gemma).extract([OcrPageInput(page=1, image=object())])

    assert error.value.code == "ocr_page_response_invalid"
