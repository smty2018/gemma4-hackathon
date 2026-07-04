from ingestion.audio import inspect_audio
from ingestion.image import generate_image_preview
from ingestion.models import FileKind, IngestionResult
from ingestion.pdf import generate_pdf_previews
from ingestion.pdf_processing import process_pdf_content
from ingestion.validation import validate_upload


def ingest_upload(
    *, filename: str, content_type: str | None, content: bytes
) -> IngestionResult:
    descriptor = validate_upload(
        filename=filename,
        content_type=content_type,
        content=content,
    )

    if descriptor.kind is FileKind.IMAGE:
        return IngestionResult(
            descriptor=descriptor,
            image_preview=generate_image_preview(content),
        )
    if descriptor.kind is FileKind.PDF:
        return IngestionResult(
            descriptor=descriptor,
            pdf_preview=generate_pdf_previews(content),
            pdf_content=process_pdf_content(content),
        )
    if descriptor.kind is FileKind.AUDIO:
        return IngestionResult(
            descriptor=descriptor,
            audio_preview=inspect_audio(content, extension=descriptor.extension),
        )

    raise AssertionError(f"Unhandled file kind: {descriptor.kind}")
