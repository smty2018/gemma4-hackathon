from io import BytesIO
from warnings import catch_warnings, simplefilter

from PIL import Image, ImageOps, UnidentifiedImageError

from ingestion.models import PreviewGenerationError, PreviewImage


MAX_IMAGE_PIXELS = 40_000_000
DEFAULT_PREVIEW_DIMENSION = 1_600


def _preview_mode(image: Image.Image) -> str:
    if "A" in image.getbands():
        return "RGBA"
    return "RGB"


def generate_image_preview(
    content: bytes,
    *,
    max_dimension: int = DEFAULT_PREVIEW_DIMENSION,
) -> PreviewImage:
    if max_dimension < 1:
        raise ValueError("max_dimension must be positive")

    try:
        with catch_warnings():
            simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(content)) as opened_image:
                source_width, source_height = opened_image.size
                if source_width * source_height > MAX_IMAGE_PIXELS:
                    raise PreviewGenerationError(
                        "image_too_many_pixels",
                        "The image dimensions are too large to process safely.",
                    )

                opened_image.load()
                normalized = ImageOps.exif_transpose(opened_image)
                normalized = normalized.convert(_preview_mode(normalized))
                normalized.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)

                output = BytesIO()
                normalized.save(output, format="PNG", optimize=True)
                return PreviewImage(
                    content=output.getvalue(),
                    width=normalized.width,
                    height=normalized.height,
                    source_width=source_width,
                    source_height=source_height,
                )
    except PreviewGenerationError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
        raise PreviewGenerationError(
            "image_too_many_pixels",
            "The image dimensions are too large to process safely.",
        ) from error
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise PreviewGenerationError(
            "invalid_image",
            "The image could not be decoded. Try exporting it as PNG or JPEG.",
        ) from error
