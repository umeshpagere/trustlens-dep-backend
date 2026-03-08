"""
TrustLens Phase 4: EXIF Metadata Extraction Utility

Extracts structured EXIF data from image bytes using Pillow.

Why EXIF matters for authenticity detection
--------------------------------------------
EXIF (Exchangeable Image File Format) stores metadata embedded by the camera
or processing software: timestamp, GPS, camera model, software used.

Signal value:
  - EXIF PRESENT: image likely direct from a camera (positive signal)
  - EXIF ABSENT:  common after WhatsApp/Twitter/Telegram (they strip it),
                  screenshots, or deliberate removal (weak negative signal)
  - EDITING SOFTWARE: Photoshop, GIMP, Stable Diffusion in the "Software"
                     field is a moderate-to-strong manipulation indicator

Why EXIF cannot alone prove manipulation:
  - Social media platforms strip EXIF from ALL images by default (GDPR/privacy)
  - EXIF is trivially forgeable using ExifTool or Pillow itself
  - A missing timestamp proves nothing; a present one doesn't prove authenticity

These signals are PROBABILISTIC — they adjust confidence, not determine truth.

Async readiness:
  Pillow _getexif() is synchronous CPU/IO. Wraps cleanly with asyncio.to_thread.
"""

import io
from typing import Any

from PIL import Image, ExifTags

# Software strings that suggest digital editing or AI generation.
# Case-insensitive substring match — order doesn't matter.
SUSPICIOUS_SOFTWARE_KEYWORDS: frozenset[str] = frozenset({
    "photoshop", "lightroom", "gimp", "affinity", "capture one",
    "darktable", "rawtherapee", "luminar",
    # AI generation tools
    "stable diffusion", "comfyui", "automatic1111", "midjourney",
    "dall-e", "firefly", "imagen", "kandinsky",
    # Generic editing hints
    "paint.net", "canva", "pixlr", "fotor",
})


def _exif_tag_name(tag_id: int) -> str:
    """Resolve a numeric EXIF tag ID to its human-readable name."""
    return ExifTags.TAGS.get(tag_id, f"Tag_{tag_id}")


def extract_exif(image_bytes: bytes) -> dict[str, Any]:
    """
    Extract EXIF metadata from image bytes.

    Returns a structured dict with:
      metadataPresent       : bool   — True if any EXIF data found
      cameraMake            : str    — e.g. "Apple", "Canon"
      cameraModel           : str    — e.g. "iPhone 14 Pro"
      software              : str    — e.g. "Adobe Photoshop CS6"
      dateTimeOriginal      : str    — capture timestamp if available
      gpsPresent            : bool   — any GPS tags found
      editingSoftwareDetected : bool — suspicious software keyword match
      rawTagCount           : int    — total EXIF tags found

    On failure (corrupt, no EXIF, non-JPEG format):
      Returns metadataPresent=False with all other fields at safe defaults.
      Never raises.
    """
    default: dict[str, Any] = {
        "metadataPresent": False,
        "cameraMake": "",
        "cameraModel": "",
        "software": "",
        "dateTimeOriginal": "",
        "gpsPresent": False,
        "editingSoftwareDetected": False,
        "rawTagCount": 0,
    }

    if not image_bytes or not isinstance(image_bytes, bytes):
        return default

    try:
        img = Image.open(io.BytesIO(image_bytes))

        # _getexif() returns None for non-JPEG or images with no EXIF block
        exif_method = getattr(img, "_getexif", None)
        if exif_method is None:
            return default

        raw_exif = exif_method()
        if not raw_exif or not isinstance(raw_exif, dict):
            return default

        # Build human-readable tag dict
        named: dict[str, Any] = {
            _exif_tag_name(k): v for k, v in raw_exif.items()
        }

        software = str(named.get("Software", "") or "").strip()
        make = str(named.get("Make", "") or "").strip()
        model = str(named.get("Model", "") or "").strip()
        dt_original = str(named.get("DateTimeOriginal", "") or "").strip()

        # GPS: tag 34853 is GPSInfo; present = has GPS coordinates
        gps_present = "GPSInfo" in named or 34853 in raw_exif

        # Detect suspicious editing software (case-insensitive)
        software_lower = software.lower()
        editing_detected = any(
            kw in software_lower for kw in SUSPICIOUS_SOFTWARE_KEYWORDS
        )

        return {
            "metadataPresent": True,
            "cameraMake": make[:100],
            "cameraModel": model[:100],
            "software": software[:200],
            "dateTimeOriginal": dt_original,
            "gpsPresent": gps_present,
            "editingSoftwareDetected": editing_detected,
            "rawTagCount": len(raw_exif),
        }

    except Exception:
        # EXIF extraction errors are expected (WebP/PNG/GIF have no EXIF,
        # truncated files, custom JPEG variants). Always degrade gracefully.
        return default
