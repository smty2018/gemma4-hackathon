import re

import pymupdf

from ingestion.models import (
    PdfContent,
    PdfPagePreview,
    PdfPageText,
    PreviewGenerationError,
)
from ingestion.pdf import MAX_PDF_PAGES


DEFAULT_SCANNED_PAGE_PREVIEWS = 8
SCANNED_PAGE_DPI = 120
MAX_TEXT_CHARACTERS_PER_PAGE = 50_000
MAX_TEXT_CHARACTERS_TOTAL = 500_000


def _normalize_text(text: str) -> str:
    lines = []
    for line in text.replace("\x00", "").splitlines():
        normalized_line = re.sub(r"[\t ]+", " ", line).strip()
        if normalized_line:
            lines.append(normalized_line)
    return "\n".join(lines)


def _meaningful_character_count(text: str) -> int:
    return sum(not character.isspace() for character in text)


def process_pdf_content(
    content: bytes,
    *,
    max_scanned_page_previews: int = DEFAULT_SCANNED_PAGE_PREVIEWS,
    dpi: int = SCANNED_PAGE_DPI,
    max_characters_per_page: int = MAX_TEXT_CHARACTERS_PER_PAGE,
    max_characters_total: int = MAX_TEXT_CHARACTERS_TOTAL,
) -> PdfContent:
    if max_scanned_page_previews < 0:
        raise ValueError("max_scanned_page_previews cannot be negative")
    if not 72 <= dpi <= 200:
        raise ValueError("dpi must be between 72 and 200")
    if max_characters_per_page < 1:
        raise ValueError("max_characters_per_page must be positive")
    if max_characters_total < 1:
        raise ValueError("max_characters_total must be positive")

    try:
        with pymupdf.open(stream=content, filetype="pdf") as document:
            if document.needs_pass:
                raise PreviewGenerationError(
                    "pdf_password_protected",
                    "Password-protected PDFs are not supported yet.",
                )

            page_count = document.page_count
            if page_count < 1:
                raise PreviewGenerationError(
                    "pdf_has_no_pages",
                    "The PDF does not contain any pages.",
                )
            if page_count > MAX_PDF_PAGES:
                raise PreviewGenerationError(
                    "pdf_too_many_pages",
                    f"The PDF has {page_count} pages; the limit is {MAX_PDF_PAGES}.",
                )

            pages: list[PdfPageText] = []
            scanned_page_numbers: list[int] = []
            scanned_pages: list[PdfPagePreview] = []
            remaining_text_characters = max_characters_total
            text_truncated = False

            for index in range(page_count):
                page = document.load_page(index)
                normalized_text = _normalize_text(page.get_text("text", sort=True))
                character_count = len(normalized_text)
                has_embedded_text = _meaningful_character_count(normalized_text) > 0
                page_truncated = False
                stored_text = normalized_text

                if has_embedded_text:
                    page_limit = min(
                        max_characters_per_page,
                        remaining_text_characters,
                    )
                    if character_count > page_limit:
                        stored_text = normalized_text[:page_limit]
                        page_truncated = True
                        text_truncated = True
                    remaining_text_characters -= len(stored_text)
                else:
                    stored_text = ""
                    scanned_page_numbers.append(index + 1)
                    if len(scanned_pages) < max_scanned_page_previews:
                        pixmap = page.get_pixmap(
                            dpi=dpi,
                            colorspace=pymupdf.csRGB,
                            alpha=False,
                        )
                        scanned_pages.append(
                            PdfPagePreview(
                                page_number=index + 1,
                                content=pixmap.tobytes("png"),
                                width=pixmap.width,
                                height=pixmap.height,
                            )
                        )

                pages.append(
                    PdfPageText(
                        page_number=index + 1,
                        text=stored_text,
                        character_count=character_count,
                        has_embedded_text=has_embedded_text,
                        truncated=page_truncated,
                    )
                )

            return PdfContent(
                page_count=page_count,
                pages=tuple(pages),
                scanned_page_numbers=tuple(scanned_page_numbers),
                scanned_pages=tuple(scanned_pages),
                scanned_previews_truncated=(
                    len(scanned_pages) < len(scanned_page_numbers)
                ),
                text_truncated=text_truncated,
            )
    except PreviewGenerationError:
        raise
    except (pymupdf.FileDataError, RuntimeError, ValueError) as error:
        raise PreviewGenerationError(
            "invalid_pdf",
            "The PDF could not be opened or processed.",
        ) from error
