import pytest

from ingestion.models import MEBIBYTE, FileKind, UploadValidationError
from ingestion.validation import validate_upload


VALID_FILES = (
    ("scan.png", "image/png", b"\x89PNG\r\n\x1a\ncontent", FileKind.IMAGE),
    ("scan.jpg", "image/jpeg", b"\xff\xd8\xffcontent", FileKind.IMAGE),
    ("notice.pdf", "application/pdf", b"%PDF-1.7\ncontent", FileKind.PDF),
    ("question.wav", "audio/wav", b"RIFF\x00\x00\x00\x00WAVEcontent", FileKind.AUDIO),
    ("question.mp3", "audio/mpeg", b"ID3content", FileKind.AUDIO),
    ("question.m4a", "audio/mp4", b"\x00\x00\x00\x18ftypM4A content", FileKind.AUDIO),
)


@pytest.mark.parametrize(("filename", "mime", "content", "kind"), VALID_FILES)
def test_valid_uploads_are_canonicalized(
    filename: str, mime: str, content: bytes, kind: FileKind
) -> None:
    descriptor = validate_upload(
        filename=filename,
        content_type=mime,
        content=content,
    )

    assert descriptor.filename == filename
    assert descriptor.kind is kind
    assert descriptor.size_bytes == len(content)
    assert len(descriptor.content_hash) == 64


def test_filename_is_reduced_to_basename() -> None:
    descriptor = validate_upload(
        filename="../../private/notice.PDF",
        content_type="application/pdf",
        content=b"%PDF-1.7\ncontent",
    )

    assert descriptor.filename == "notice.PDF"
    assert descriptor.extension == ".pdf"


@pytest.mark.parametrize("mime", [None, "", "application/octet-stream"])
def test_generic_browser_mime_is_accepted(mime: str | None) -> None:
    descriptor = validate_upload(
        filename="notice.pdf",
        content_type=mime,
        content=b"%PDF-1.7\ncontent",
    )

    assert descriptor.content_type == "application/pdf"


@pytest.mark.parametrize(
    ("filename", "mime", "content", "code"),
    (
        ("notes.txt", "text/plain", b"hello", "unsupported_extension"),
        ("notice.pdf", "application/pdf", b"", "empty_file"),
        ("notice.pdf", "image/png", b"%PDF-1.7\ncontent", "mime_mismatch"),
        ("notice.pdf", "application/pdf", b"not a pdf", "content_mismatch"),
        ("scan.png", "image/png", b"not an image", "content_mismatch"),
    ),
)
def test_invalid_uploads_have_stable_error_codes(
    filename: str, mime: str, content: bytes, code: str
) -> None:
    with pytest.raises(UploadValidationError) as error:
        validate_upload(filename=filename, content_type=mime, content=content)

    assert error.value.code == code


def test_size_limit_is_enforced_before_signature_check() -> None:
    content = b"%PDF-" + b"0" * (20 * MEBIBYTE)

    with pytest.raises(UploadValidationError) as error:
        validate_upload(
            filename="large.pdf",
            content_type="application/pdf",
            content=content,
        )

    assert error.value.code == "file_too_large"
    assert "20 MB" in str(error.value)
