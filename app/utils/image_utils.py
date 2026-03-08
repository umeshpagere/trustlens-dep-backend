"""
TrustLens Phase 4: Image Validation Utility

Provides a single reliable entry point for bytes → PIL Image conversion,
used by both hash_utils and metadata_utils to prevent duplication.

Design rationale
----------------
Centralising image loading means:
  - One place to add future preprocessing (resize, normalise, colour-space convert)
  - One place to handle format-specific quirks (HEIC, WebP, animated GIF)
  - Consumer code stays clean — no try/except around Image.open everywhere

Async readiness:
  PIL.Image.open and verify() are synchronous I/O + CPU. Safe for
  asyncio.to_thread() in Phase 5 with zero logic changes.
"""

import io
from typing import Any

from PIL import Image, UnidentifiedImageError

# Maximum image size to process (10 MB). Prevents memory exhaustion from
# adversarially large uploads while allowing high-resolution news photos.
MAX_IMAGE_BYTES = 10 * 1024 * 1024


def bytes_to_pil(image_bytes: bytes) -> Image.Image | None:
    """
    Convert raw image bytes to a PIL Image object.

    Returns None on:
      - Empty or non-bytes input
      - Images exceeding MAX_IMAGE_BYTES
      - Corrupt / unrecognised image format
      - Any other PIL error

    The returned image is NOT closed by this function; callers are
    responsible for managing the lifecycle (or using as context manager).
    """
    if not image_bytes or not isinstance(image_bytes, bytes):
        return None
    if len(image_bytes) > MAX_IMAGE_BYTES:
        return None
    try:
        img = Image.open(io.BytesIO(image_bytes))
        # verify() detects truncation and corruption; it resets the pointer so
        # the image can still be used after (re-open is not needed for metadata,
        # but callers that need pixel data should re-open from bytes themselves).
        img.verify()
        # Re-open after verify() because verify() exhausts the internal pointer
        img = Image.open(io.BytesIO(image_bytes))
        return img
    except (UnidentifiedImageError, Exception):
        return None


def is_valid_image(image_bytes: bytes) -> bool:
    """
    Return True if image_bytes can be opened and verified by Pillow.

    Faster than bytes_to_pil() when only validity is needed.
    """
    return bytes_to_pil(image_bytes) is not None


def get_image_format(image_bytes: bytes) -> str | None:
    """
    Return the detected image format string (e.g. 'JPEG', 'PNG', 'WEBP').
    Returns None for invalid or unrecognised input.
    """
    img = bytes_to_pil(image_bytes)
    if img is None:
        return None
    return img.format
