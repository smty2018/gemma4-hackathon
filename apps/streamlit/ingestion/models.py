from dataclasses import dataclass
from enum import StrEnum


class FileKind(StrEnum):
    IMAGE = "image"
    PDF = "pdf"
    AUDIO = "audio"


@dataclass(frozen=True)
class FileRule:
    kind: FileKind
    canonical_mime: str
    accepted_mimes: frozenset[str]
    max_size_bytes: int


@dataclass(frozen=True)
class FileDescriptor:
    filename: str
    extension: str
    kind: FileKind
    content_type: str
    size_bytes: int
    content_hash: str


@dataclass(frozen=True)
class PreviewImage:
    content: bytes
    width: int
    height: int
    source_width: int
    source_height: int
    format: str = "PNG"


@dataclass(frozen=True)
class PdfPagePreview:
    page_number: int
    content: bytes
    width: int
    height: int


@dataclass(frozen=True)
class PdfPreview:
    page_count: int
    pages: tuple[PdfPagePreview, ...]
    truncated: bool


@dataclass(frozen=True)
class PdfPageText:
    page_number: int
    text: str
    character_count: int
    has_embedded_text: bool
    truncated: bool


@dataclass(frozen=True)
class PdfContent:
    page_count: int
    pages: tuple[PdfPageText, ...]
    scanned_page_numbers: tuple[int, ...]
    scanned_pages: tuple[PdfPagePreview, ...]
    scanned_previews_truncated: bool
    text_truncated: bool

    @property
    def embedded_text(self) -> str:
        sections = (
            f"Page {page.page_number}\n{page.text}"
            for page in self.pages
            if page.has_embedded_text and page.text
        )
        return "\n\n".join(sections)

    @property
    def text_page_count(self) -> int:
        return sum(page.has_embedded_text for page in self.pages)


@dataclass(frozen=True)
class AudioPreview:
    content: bytes
    content_type: str
    duration_seconds: float
    sample_rate: int | None
    channels: int | None


@dataclass(frozen=True)
class IngestionResult:
    descriptor: FileDescriptor
    image_preview: PreviewImage | None = None
    pdf_preview: PdfPreview | None = None
    audio_preview: AudioPreview | None = None


class UploadValidationError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class PreviewGenerationError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


MEBIBYTE = 1024 * 1024

FILE_RULES: dict[str, FileRule] = {
    ".png": FileRule(
        FileKind.IMAGE,
        "image/png",
        frozenset({"image/png"}),
        10 * MEBIBYTE,
    ),
    ".jpg": FileRule(
        FileKind.IMAGE,
        "image/jpeg",
        frozenset({"image/jpeg", "image/jpg"}),
        10 * MEBIBYTE,
    ),
    ".jpeg": FileRule(
        FileKind.IMAGE,
        "image/jpeg",
        frozenset({"image/jpeg", "image/jpg"}),
        10 * MEBIBYTE,
    ),
    ".pdf": FileRule(
        FileKind.PDF,
        "application/pdf",
        frozenset({"application/pdf"}),
        20 * MEBIBYTE,
    ),
    ".wav": FileRule(
        FileKind.AUDIO,
        "audio/wav",
        frozenset({"audio/wav", "audio/x-wav", "audio/wave"}),
        15 * MEBIBYTE,
    ),
    ".mp3": FileRule(
        FileKind.AUDIO,
        "audio/mpeg",
        frozenset({"audio/mpeg", "audio/mp3"}),
        15 * MEBIBYTE,
    ),
    ".m4a": FileRule(
        FileKind.AUDIO,
        "audio/mp4",
        frozenset({"audio/mp4", "audio/x-m4a", "video/mp4"}),
        15 * MEBIBYTE,
    ),
}
