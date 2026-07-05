from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from io import BytesIO
from pathlib import PurePath
from typing import Protocol
from warnings import catch_warnings, simplefilter

import pymupdf
from PIL import Image, ImageOps, UnidentifiedImageError

from app.explanation.pipeline import ExplanationPipeline, ExplanationRequest
from app.inference.gemma import GemmaRequest, GemmaResponse
from app.ocr.pipeline import OcrPageInput, OcrPipeline
from app.schemas.document import DocumentAnalysisResult

MAX_DOCUMENT_BYTES = 20 * 1024 * 1024
MAX_DOCUMENT_PAGES = 100
MAX_IMAGE_PIXELS = 40_000_000
MAX_TEXT_CHARACTERS_PER_PAGE = 50_000
MAX_TEXT_CHARACTERS_TOTAL = 500_000
SCANNED_PAGE_DPI = 120

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
_IMAGE_CONTENT_TYPES = {"image/jpeg", "image/jpg", "image/png"}
_GENERIC_CONTENT_TYPES = {"", "application/octet-stream"}


class DocumentIngestionError(ValueError):
    def __init__(self, code: str, message: str, *, status_code: int = 422) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code


class GemmaGenerator(Protocol):
    model_id: str

    def generate(self, request: GemmaRequest) -> GemmaResponse: ...


@dataclass(frozen=True)
class PreparedDocument:
    filename: str
    content_type: str
    pages: Sequence[OcrPageInput]


class DocumentAnalysisService:
    def __init__(self, gemma: GemmaGenerator) -> None:
        self._gemma = gemma

    def analyze(
        self,
        *,
        filename: str,
        content_type: str | None,
        content: bytes,
        language: str,
        audience: str,
    ) -> DocumentAnalysisResult:
        prepared = prepare_document(
            filename=filename,
            content_type=content_type,
            content=content,
        )
        ocr = OcrPipeline(self._gemma).extract(prepared.pages)
        explanation = ExplanationPipeline(self._gemma).explain(
            ExplanationRequest(
                ocr=ocr,
                language=language,
                audience=audience,
            )
        )
        return DocumentAnalysisResult(
            document_name=prepared.filename,
            content_type=prepared.content_type,
            model_id=self._gemma.model_id,
            ocr=ocr,
            explanation=explanation,
        )


def prepare_document(
    *,
    filename: str,
    content_type: str | None,
    content: bytes,
) -> PreparedDocument:
    safe_filename = _safe_filename(filename)
    extension = PurePath(safe_filename).suffix.lower()
    normalized_content_type = (content_type or "").strip().lower()

    if not content:
        raise DocumentIngestionError("empty_document", "The uploaded document is empty.")
    if len(content) > MAX_DOCUMENT_BYTES:
        raise DocumentIngestionError(
            "document_too_large",
            "The uploaded document exceeds the 20 MB limit.",
            status_code=413,
        )

    if extension == ".pdf":
        if normalized_content_type not in {*_GENERIC_CONTENT_TYPES, "application/pdf"}:
            raise DocumentIngestionError(
                "document_content_type_mismatch",
                "The upload content type does not match a PDF.",
            )
        if not content.startswith(b"%PDF-"):
            raise DocumentIngestionError(
                "invalid_pdf_signature",
                "The uploaded file is not a valid PDF.",
            )
        return PreparedDocument(
            filename=safe_filename,
            content_type="application/pdf",
            pages=_prepare_pdf_pages(content),
        )

    if extension in _IMAGE_EXTENSIONS:
        if normalized_content_type not in {*_GENERIC_CONTENT_TYPES, *_IMAGE_CONTENT_TYPES}:
            raise DocumentIngestionError(
                "document_content_type_mismatch",
                "The upload content type does not match an image.",
            )
        return PreparedDocument(
            filename=safe_filename,
            content_type="image/png" if extension == ".png" else "image/jpeg",
            pages=(OcrPageInput(page=1, image=_decode_image(content)),),
        )

    raise DocumentIngestionError(
        "unsupported_document_type",
        "Upload a PDF, PNG, JPG, or JPEG document.",
        status_code=415,
    )


def _prepare_pdf_pages(content: bytes) -> tuple[OcrPageInput, ...]:
    try:
        with pymupdf.open(stream=content, filetype="pdf") as document:
            if document.needs_pass:
                raise DocumentIngestionError(
                    "pdf_password_protected",
                    "Password-protected PDFs are not supported.",
                )
            if document.page_count < 1:
                raise DocumentIngestionError(
                    "pdf_has_no_pages",
                    "The PDF does not contain any pages.",
                )
            if document.page_count > MAX_DOCUMENT_PAGES:
                raise DocumentIngestionError(
                    "pdf_page_limit_exceeded",
                    f"The PDF exceeds the {MAX_DOCUMENT_PAGES}-page limit.",
                    status_code=413,
                )

            remaining_text = MAX_TEXT_CHARACTERS_TOTAL
            pages: list[OcrPageInput] = []
            for index in range(document.page_count):
                page = document.load_page(index)
                embedded_text = _normalize_text(page.get_text("text", sort=True))
                if embedded_text:
                    text_limit = min(MAX_TEXT_CHARACTERS_PER_PAGE, remaining_text)
                    if text_limit < 1:
                        raise DocumentIngestionError(
                            "pdf_text_limit_exceeded",
                            "The PDF contains too much embedded text to process safely.",
                            status_code=413,
                        )
                    embedded_text = embedded_text[:text_limit]
                    remaining_text -= len(embedded_text)
                    pages.append(OcrPageInput(page=index + 1, embedded_text=embedded_text))
                    continue

                pixmap = page.get_pixmap(
                    dpi=SCANNED_PAGE_DPI,
                    colorspace=pymupdf.csRGB,
                    alpha=False,
                )
                pages.append(
                    OcrPageInput(
                        page=index + 1,
                        image=_decode_image(pixmap.tobytes("png")),
                    )
                )
            return tuple(pages)
    except DocumentIngestionError:
        raise
    except (pymupdf.FileDataError, RuntimeError, ValueError) as error:
        raise DocumentIngestionError(
            "invalid_pdf",
            "The PDF could not be opened or processed.",
        ) from error


def _decode_image(content: bytes) -> Image.Image:
    try:
        with catch_warnings():
            simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(content)) as opened:
                width, height = opened.size
                if width * height > MAX_IMAGE_PIXELS:
                    raise DocumentIngestionError(
                        "image_pixel_limit_exceeded",
                        "The image dimensions are too large to process safely.",
                        status_code=413,
                    )
                opened.load()
                return ImageOps.exif_transpose(opened).convert("RGB").copy()
    except DocumentIngestionError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
        raise DocumentIngestionError(
            "image_pixel_limit_exceeded",
            "The image dimensions are too large to process safely.",
            status_code=413,
        ) from error
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise DocumentIngestionError(
            "invalid_image",
            "The uploaded image could not be decoded.",
        ) from error


def _safe_filename(filename: str) -> str:
    safe_name = PurePath(filename.replace("\\", "/")).name.strip()
    if not safe_name or safe_name in {".", ".."}:
        raise DocumentIngestionError(
            "invalid_document_filename",
            "The uploaded document needs a valid filename.",
        )
    return safe_name


def _normalize_text(text: str) -> str:
    lines = []
    for line in text.replace("\x00", "").splitlines():
        normalized = re.sub(r"[\t ]+", " ", line).strip()
        if normalized:
            lines.append(normalized)
    return "\n".join(lines)
