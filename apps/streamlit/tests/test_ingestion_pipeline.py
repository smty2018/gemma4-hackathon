from io import BytesIO
import wave

from PIL import Image
import pymupdf
import pytest

from ingestion.models import FileKind, UploadValidationError
from ingestion.pipeline import ingest_upload


def png_bytes() -> bytes:
    output = BytesIO()
    Image.new("RGB", (120, 80), color=(20, 108, 67)).save(output, format="PNG")
    return output.getvalue()


def pdf_bytes() -> bytes:
    document = pymupdf.open()
    document.new_page(width=300, height=400).insert_text((30, 40), "Notice")
    content = document.tobytes()
    document.close()
    return content


def wav_bytes() -> bytes:
    output = BytesIO()
    with wave.open(output, "wb") as recording:
        recording.setnchannels(1)
        recording.setsampwidth(2)
        recording.setframerate(8_000)
        recording.writeframes(b"\x00\x00" * 4_000)
    return output.getvalue()


@pytest.mark.parametrize(
    ("filename", "content_type", "content_factory", "kind", "preview_field"),
    (
        ("scan.png", "image/png", png_bytes, FileKind.IMAGE, "image_preview"),
        ("notice.pdf", "application/pdf", pdf_bytes, FileKind.PDF, "pdf_preview"),
        ("question.wav", "audio/wav", wav_bytes, FileKind.AUDIO, "audio_preview"),
    ),
)
def test_pipeline_dispatches_to_expected_preview(
    filename: str,
    content_type: str,
    content_factory,
    kind: FileKind,
    preview_field: str,
) -> None:
    result = ingest_upload(
        filename=filename,
        content_type=content_type,
        content=content_factory(),
    )

    assert result.descriptor.kind is kind
    assert getattr(result, preview_field) is not None
    populated_previews = [
        result.image_preview is not None,
        result.pdf_preview is not None,
        result.audio_preview is not None,
    ]
    assert sum(populated_previews) == 1


def test_pipeline_stops_before_preview_for_invalid_upload() -> None:
    with pytest.raises(UploadValidationError) as error:
        ingest_upload(
            filename="fake.pdf",
            content_type="application/pdf",
            content=b"not a pdf",
        )

    assert error.value.code == "content_mismatch"
