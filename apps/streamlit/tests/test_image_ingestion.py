from io import BytesIO

import pytest
from PIL import Image

from ingestion.image import generate_image_preview
from ingestion.models import PreviewGenerationError


def image_bytes(
    *, width: int, height: int, image_format: str = "PNG", mode: str = "RGB"
) -> bytes:
    image = Image.new(mode, (width, height), color=(20, 108, 67))
    output = BytesIO()
    image.save(output, format=image_format)
    return output.getvalue()


def test_large_image_is_scaled_without_changing_aspect_ratio() -> None:
    preview = generate_image_preview(
        image_bytes(width=3_200, height=1_600),
        max_dimension=800,
    )

    assert (preview.source_width, preview.source_height) == (3_200, 1_600)
    assert (preview.width, preview.height) == (800, 400)
    assert preview.content.startswith(b"\x89PNG\r\n\x1a\n")


def test_small_image_is_not_upscaled() -> None:
    preview = generate_image_preview(image_bytes(width=320, height=200))

    assert (preview.width, preview.height) == (320, 200)


def test_jpeg_is_normalized_to_png() -> None:
    preview = generate_image_preview(
        image_bytes(width=640, height=480, image_format="JPEG")
    )

    assert preview.format == "PNG"
    assert preview.content.startswith(b"\x89PNG\r\n\x1a\n")


def test_transparent_image_preserves_alpha_channel() -> None:
    content = image_bytes(width=64, height=64, mode="RGBA")
    preview = generate_image_preview(content)

    with Image.open(BytesIO(preview.content)) as image:
        assert image.mode == "RGBA"


def test_invalid_image_has_stable_error_code() -> None:
    with pytest.raises(PreviewGenerationError) as error:
        generate_image_preview(b"\x89PNG\r\n\x1a\nnot a real image")

    assert error.value.code == "invalid_image"


def test_preview_dimension_must_be_positive() -> None:
    with pytest.raises(ValueError, match="positive"):
        generate_image_preview(image_bytes(width=10, height=10), max_dimension=0)
