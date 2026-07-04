from hashlib import sha256
from pathlib import PurePath

from ingestion.models import FILE_RULES, FileDescriptor, UploadValidationError


GENERIC_MIMES = {"", "application/octet-stream"}


def _safe_filename(filename: str) -> str:
    normalized = filename.replace("\\", "/")
    safe_name = PurePath(normalized).name.strip()
    if not safe_name or safe_name in {".", ".."}:
        raise UploadValidationError("invalid_filename", "The uploaded file needs a valid name.")
    return safe_name


def _has_valid_signature(extension: str, content: bytes) -> bool:
    if extension == ".png":
        return content.startswith(b"\x89PNG\r\n\x1a\n")
    if extension in {".jpg", ".jpeg"}:
        return content.startswith(b"\xff\xd8\xff")
    if extension == ".pdf":
        return content.startswith(b"%PDF-")
    if extension == ".wav":
        return len(content) >= 12 and content[:4] == b"RIFF" and content[8:12] == b"WAVE"
    if extension == ".mp3":
        return content.startswith(b"ID3") or (
            len(content) >= 2 and content[0] == 0xFF and content[1] & 0xE0 == 0xE0
        )
    if extension == ".m4a":
        return len(content) >= 12 and content[4:8] == b"ftyp"
    return False


def validate_upload(
    *, filename: str, content_type: str | None, content: bytes
) -> FileDescriptor:
    safe_name = _safe_filename(filename)
    extension = PurePath(safe_name).suffix.lower()
    rule = FILE_RULES.get(extension)

    if rule is None:
        supported = ", ".join(sorted(FILE_RULES))
        raise UploadValidationError(
            "unsupported_extension",
            f"Unsupported file type. Choose one of: {supported}.",
        )
    if not content:
        raise UploadValidationError("empty_file", "The uploaded file is empty.")
    if len(content) > rule.max_size_bytes:
        limit_mb = rule.max_size_bytes // (1024 * 1024)
        raise UploadValidationError(
            "file_too_large",
            f"{safe_name} exceeds the {limit_mb} MB limit for {rule.kind.value} files.",
        )

    normalized_mime = (content_type or "").lower().strip()
    if normalized_mime not in GENERIC_MIMES and normalized_mime not in rule.accepted_mimes:
        raise UploadValidationError(
            "mime_mismatch",
            f"The browser reported {normalized_mime}, which does not match {extension}.",
        )
    if not _has_valid_signature(extension, content):
        raise UploadValidationError(
            "content_mismatch",
            f"The contents of {safe_name} do not match its file extension.",
        )

    return FileDescriptor(
        filename=safe_name,
        extension=extension,
        kind=rule.kind,
        content_type=rule.canonical_mime,
        size_bytes=len(content),
        content_hash=sha256(content).hexdigest(),
    )
