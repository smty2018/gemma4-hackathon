from io import BytesIO
from typing import Any

import pymupdf
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.api.routes.documents import get_document_analysis_service
from app.documents import DocumentAnalysisService, DocumentIngestionError
from app.documents.analysis import prepare_document
from app.inference.gemma import DEFAULT_MODEL_ID, GemmaRequest, GemmaResponse
from app.main import app


class PipelineGemmaStub:
    model_id = DEFAULT_MODEL_ID

    def __init__(self) -> None:
        self.requests: list[GemmaRequest] = []

    def generate(self, request: GemmaRequest) -> GemmaResponse:
        self.requests.append(request)
        schema_name = getattr(request.response_schema, "__name__", "")
        if schema_name == "_PageExtractionPayload":
            text = _embedded_text(request.prompt) or "Scanned payment notice"
            evidence_text = "Payment notice" if "Payment notice" in text else text
            return GemmaResponse(
                model_id=self.model_id,
                structured={
                    "text": text,
                    "evidence": [
                        {
                            "label": "Document type",
                            "value": "Payment notice",
                            "evidence_text": evidence_text,
                            "confidence": 0.95,
                        }
                    ],
                    "dates": [],
                    "amounts": [],
                    "confidence": 0.94,
                },
            )
        if schema_name == "_ExplanationPayload":
            return GemmaResponse(
                model_id=self.model_id,
                structured={
                    "simple_summary": {
                        "text": "This document is a payment notice.",
                        "evidence_ids": ["P1"],
                    },
                    "key_facts": [],
                    "required_actions": [],
                    "warnings": [],
                    "confidence": 0.92,
                },
            )
        raise AssertionError(f"Unexpected schema: {schema_name}")


def _embedded_text(prompt: str) -> str:
    marker = "<document_text>\n"
    if marker not in prompt:
        return ""
    return prompt.split(marker, 1)[1].split("\n</document_text>", 1)[0]


def text_pdf(text: str = "Payment notice\nReference: DEMO-1024") -> bytes:
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), text)
    content = document.tobytes()
    document.close()
    return content


def png_image() -> bytes:
    output = BytesIO()
    Image.new("RGB", (64, 64), "white").save(output, format="PNG")
    return output.getvalue()


def service() -> tuple[DocumentAnalysisService, PipelineGemmaStub]:
    gemma = PipelineGemmaStub()
    return DocumentAnalysisService(gemma), gemma


def test_pdf_analysis_runs_ocr_then_explanation() -> None:
    analysis_service, gemma = service()

    result = analysis_service.analyze(
        filename="notice.pdf",
        content_type="application/pdf",
        content=text_pdf(),
        language="English",
        audience="general public",
    )

    assert result.document_name == "notice.pdf"
    assert result.model_id == DEFAULT_MODEL_ID
    assert result.ocr.page_count == 1
    assert "DEMO-1024" in result.ocr.text
    assert result.explanation.simple_summary.text == "This document is a payment notice."
    assert [getattr(item.response_schema, "__name__", "") for item in gemma.requests] == [
        "_PageExtractionPayload",
        "_ExplanationPayload",
    ]


def test_image_is_decoded_and_sent_to_ocr_model() -> None:
    analysis_service, gemma = service()

    result = analysis_service.analyze(
        filename="scan.png",
        content_type="image/png",
        content=png_image(),
        language="Hindi",
        audience="general public",
    )

    assert result.ocr.pages[0].source.value == "image"
    assert len(gemma.requests[0].images) == 1
    assert gemma.requests[0].images[0].mode == "RGB"
    assert result.explanation.language == "Hindi"


@pytest.mark.parametrize(
    ("filename", "content_type", "content", "code"),
    [
        ("notes.txt", "text/plain", b"hello", "unsupported_document_type"),
        ("broken.pdf", "application/pdf", b"not a pdf", "invalid_pdf_signature"),
        ("empty.png", "image/png", b"", "empty_document"),
    ],
)
def test_invalid_documents_have_stable_error_codes(
    filename: str,
    content_type: str,
    content: bytes,
    code: str,
) -> None:
    with pytest.raises(DocumentIngestionError) as error:
        prepare_document(filename=filename, content_type=content_type, content=content)

    assert error.value.code == code


def test_live_route_returns_typed_pipeline_result() -> None:
    analysis_service, _gemma = service()
    app.dependency_overrides[get_document_analysis_service] = lambda: analysis_service
    try:
        response = TestClient(app).post(
            "/api/v1/documents/analyze",
            files={"document": ("notice.pdf", text_pdf(), "application/pdf")},
            data={"language": "Bengali", "audience": "general public"},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    payload: dict[str, Any] = response.json()
    assert payload["model_id"] == DEFAULT_MODEL_ID
    assert payload["ocr"]["page_count"] == 1
    assert payload["explanation"]["language"] == "Bengali"
    assert payload["explanation"]["simple_summary"]["text"]


def test_plan_route_remains_available_without_model_inference() -> None:
    response = TestClient(app).post(
        "/api/v1/documents/plan",
        files={"document": ("notice.pdf", text_pdf(), "application/pdf")},
        data={"language": "English", "audience": "general public"},
    )

    assert response.status_code == 202
    assert response.json()["status"] == "accepted"
