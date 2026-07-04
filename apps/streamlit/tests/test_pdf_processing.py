from io import BytesIO

import pymupdf
import pytest
from PIL import Image

from ingestion.models import PreviewGenerationError
from ingestion.pdf_processing import process_pdf_content


def text_pdf_bytes(*page_texts: str) -> bytes:
    document = pymupdf.open()
    for text in page_texts:
        page = document.new_page(width=300, height=400)
        page.insert_text((40, 60), text)
    content = document.tobytes()
    document.close()
    return content


def image_bytes() -> bytes:
    image = Image.new("RGB", (80, 60), color=(30, 90, 150))
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def mixed_pdf_bytes(scanned_page_count: int = 1) -> bytes:
    document = pymupdf.open()
    text_page = document.new_page(width=300, height=400)
    text_page.insert_text((40, 60), "Embedded application details")
    for _ in range(scanned_page_count):
        scanned_page = document.new_page(width=300, height=400)
        scanned_page.insert_image(scanned_page.rect, stream=image_bytes())
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


def test_embedded_text_is_extracted_per_page() -> None:
    result = process_pdf_content(text_pdf_bytes("First page", "Second page"))

    assert result.page_count == 2
    assert result.text_page_count == 2
    assert [page.text for page in result.pages] == ["First page", "Second page"]
    assert result.embedded_text == "Page 1\nFirst page\n\nPage 2\nSecond page"
    assert result.scanned_page_numbers == ()
    assert result.scanned_pages == ()
    assert result.text_truncated is False


def test_short_embedded_text_is_not_misclassified_as_a_scan() -> None:
    result = process_pdf_content(text_pdf_bytes("1"))

    assert result.pages[0].has_embedded_text is True
    assert result.scanned_page_numbers == ()


def test_image_only_pages_are_detected_and_rendered() -> None:
    result = process_pdf_content(mixed_pdf_bytes())

    assert result.text_page_count == 1
    assert result.scanned_page_numbers == (2,)
    assert len(result.scanned_pages) == 1
    scanned_page = result.scanned_pages[0]
    assert scanned_page.page_number == 2
    assert scanned_page.content.startswith(b"\x89PNG\r\n\x1a\n")
    assert scanned_page.width > 0 and scanned_page.height > 0
    assert result.scanned_previews_truncated is False


def test_scanned_page_rendering_is_bounded() -> None:
    result = process_pdf_content(
        mixed_pdf_bytes(scanned_page_count=3),
        max_scanned_page_previews=2,
    )

    assert result.scanned_page_numbers == (2, 3, 4)
    assert [page.page_number for page in result.scanned_pages] == [2, 3]
    assert result.scanned_previews_truncated is True


def test_text_output_respects_page_and_document_limits() -> None:
    result = process_pdf_content(
        text_pdf_bytes("abcdefghij", "klmnopqrst"),
        max_characters_per_page=6,
        max_characters_total=9,
    )

    assert [page.text for page in result.pages] == ["abcdef", "klm"]
    assert [page.character_count for page in result.pages] == [10, 10]
    assert all(page.truncated for page in result.pages)
    assert result.text_truncated is True


def test_corrupt_and_password_protected_pdfs_have_stable_errors() -> None:
    with pytest.raises(PreviewGenerationError) as corrupt_error:
        process_pdf_content(b"%PDF-1.7\nnot usable")
    assert corrupt_error.value.code == "invalid_pdf"

    with pytest.raises(PreviewGenerationError) as password_error:
        process_pdf_content(encrypted_pdf_bytes())
    assert password_error.value.code == "pdf_password_protected"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_scanned_page_previews": -1}, "cannot be negative"),
        ({"dpi": 50}, "between 72 and 200"),
        ({"dpi": 250}, "between 72 and 200"),
        ({"max_characters_per_page": 0}, "must be positive"),
        ({"max_characters_total": 0}, "must be positive"),
    ],
)
def test_processing_bounds_are_validated(
    kwargs: dict[str, int], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        process_pdf_content(text_pdf_bytes("text"), **kwargs)
