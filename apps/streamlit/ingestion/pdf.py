import pymupdf

from ingestion.models import PdfPagePreview, PdfPreview, PreviewGenerationError


DEFAULT_PREVIEW_PAGES = 4
MAX_PDF_PAGES = 100
PREVIEW_DPI = 120


def generate_pdf_previews(
    content: bytes,
    *,
    max_preview_pages: int = DEFAULT_PREVIEW_PAGES,
    dpi: int = PREVIEW_DPI,
) -> PdfPreview:
    if max_preview_pages < 1:
        raise ValueError("max_preview_pages must be positive")
    if not 72 <= dpi <= 200:
        raise ValueError("dpi must be between 72 and 200")

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

            preview_count = min(page_count, max_preview_pages)
            pages: list[PdfPagePreview] = []
            for index in range(preview_count):
                page = document.load_page(index)
                pixmap = page.get_pixmap(
                    dpi=dpi,
                    colorspace=pymupdf.csRGB,
                    alpha=False,
                )
                pages.append(
                    PdfPagePreview(
                        page_number=index + 1,
                        content=pixmap.tobytes("png"),
                        width=pixmap.width,
                        height=pixmap.height,
                    )
                )

            return PdfPreview(
                page_count=page_count,
                pages=tuple(pages),
                truncated=preview_count < page_count,
            )
    except PreviewGenerationError:
        raise
    except (pymupdf.FileDataError, RuntimeError, ValueError) as error:
        raise PreviewGenerationError(
            "invalid_pdf",
            "The PDF could not be opened or rendered.",
        ) from error
