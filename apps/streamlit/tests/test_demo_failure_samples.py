from pathlib import Path

import pytest

from ingestion.models import PreviewGenerationError
from ingestion.pipeline import ingest_upload


REPOSITORY_ROOT = Path(__file__).parents[3]
FAILURE_SAMPLES = REPOSITORY_ROOT / "samples" / "demo" / "failures"


def test_committed_corrupt_pdf_fixture_fails_safely() -> None:
    content = (FAILURE_SAMPLES / "corrupt.pdf").read_bytes()

    with pytest.raises(PreviewGenerationError) as error:
        ingest_upload(
            filename="corrupt.pdf",
            content_type="application/pdf",
            content=content,
        )

    assert error.value.code == "invalid_pdf"
