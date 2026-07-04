import pymupdf
import pytest

from ingestion.models import PreviewGenerationError
from ingestion.pdf import MAX_PDF_PAGES, generate_pdf_previews


def pdf_bytes(page_count: int = 1) -> bytes:
    document = pymupdf.open()
    for page_number in range(1, page_count + 1):
        page = document.new_page(width=300, height=400)
        page.insert_text((40, 60), f"Test page {page_number}")
    content = document.tobytes()
    document.close()
    return content


def encrypted_pdf_bytes() -> bytes:
    document = pymupdf.open()
    document.new_page()
    content = document.tobytes(
        encryption=pymupdf.PDF_ENCRYPT_AES_256,
        owner_pw="owner-password",
        user_pw="user-password",
    )
    document.close()
    return content


def test_pdf_pages_are_rendered_as_numbered_png_previews() -> None:
    preview = generate_pdf_previews(pdf_bytes(3), max_preview_pages=2)

    assert preview.page_count == 3
    assert preview.truncated is True
    assert [page.page_number for page in preview.pages] == [1, 2]
    assert all(page.content.startswith(b"\x89PNG\r\n\x1a\n") for page in preview.pages)
    assert all(page.width > 0 and page.height > 0 for page in preview.pages)


def test_short_pdf_is_not_marked_as_truncated() -> None:
    preview = generate_pdf_previews(pdf_bytes(2), max_preview_pages=4)

    assert preview.page_count == 2
    assert len(preview.pages) == 2
    assert preview.truncated is False


def test_corrupt_pdf_has_stable_error_code() -> None:
    with pytest.raises(PreviewGenerationError) as error:
        generate_pdf_previews(b"%PDF-1.7\nnot a usable pdf")

    assert error.value.code == "invalid_pdf"


def test_password_protected_pdf_is_rejected() -> None:
    with pytest.raises(PreviewGenerationError) as error:
        generate_pdf_previews(encrypted_pdf_bytes())

    assert error.value.code == "pdf_password_protected"


def test_pdf_page_limit_prevents_unbounded_processing() -> None:
    with pytest.raises(PreviewGenerationError) as error:
        generate_pdf_previews(pdf_bytes(MAX_PDF_PAGES + 1))

    assert error.value.code == "pdf_too_many_pages"


@pytest.mark.parametrize(
    ("max_preview_pages", "dpi", "message"),
    [(0, 120, "positive"), (1, 50, "between 72 and 200"), (1, 250, "between 72 and 200")],
)
def test_render_bounds_are_validated(
    max_preview_pages: int, dpi: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        generate_pdf_previews(
            pdf_bytes(),
            max_preview_pages=max_preview_pages,
            dpi=dpi,
        )
