"""
TrustLens Phase 4: Image Hashing Utilities

Provides perceptual hashing (pHash) for image reuse detection.

Why perceptual hash over SHA-256?
----------------------------------
SHA-256 (used for cache deduplication in hashing.py) is a cryptographic hash —
it changes completely with a single pixel edit. That makes it useless for finding
visually similar images that have been resized, cropped, JPEG-recompressed, or
lightly watermarked.

pHash works by:
  1. Shrinking the image to 32x32 pixels (removes high-frequency noise)
  2. Converting to greyscale (removes colour variation)
  3. Applying a Discrete Cosine Transform (DCT)
  4. Taking the top-left 8x8 DCT coefficients (low-frequency content = structure)
  5. Comparing each coefficient to the row mean → 64-bit binary string

Result: a 64-character hex fingerprint that is STABLE across common transformations.

Why Hamming distance for comparison?
--------------------------------------
Two pHashes that differ by ≤ 8 bits out of 64 (~12.5%) are visually near-identical.
This threshold was chosen to:
  - Tolerate JPEG artefacts, minor crops, and resize operations
  - Reject genuinely different images (distance > 10 for unrelated photos)
  - Not be so tight that routine social-media recompression creates false negatives

Async readiness:
  imagehash.phash() is CPU-bound. Safe for asyncio.to_thread() in Phase 5.
  No shared state — each call is completely independent.
"""

import io
from typing import Any

try:
    import imagehash
    from PIL import Image
    _IMAGEHASH_AVAILABLE = True
except ImportError:
    _IMAGEHASH_AVAILABLE = False

PHASH_SIMILARITY_THRESHOLD = 8   # bits out of 64; tune based on false-positive rate
PHASH_STRING_LENGTH = 16         # imagehash default hex length for pHash


def compute_phash(image_bytes: bytes) -> str | None:
    """
    Compute a perceptual hash (pHash) from raw image bytes.

    Returns a 16-character hex string on success, or None on any failure.
    Never raises — callers receive None as a graceful fallback signal.

    Parameters
    ----------
    image_bytes : bytes
        Raw binary content of any PIL-supported image format.

    Returns
    -------
    str | None
        Hex pHash string, or None if hashing failed.
    """
    if not _IMAGEHASH_AVAILABLE:
        return None
    if not image_bytes or not isinstance(image_bytes, bytes):
        return None
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return str(imagehash.phash(img))
    except Exception:
        return None


def phash_distance(hash_a: str, hash_b: str) -> int | None:
    """
    Compute Hamming distance between two pHash hex strings.

    Returns None if either hash is invalid/None, so callers can safely
    skip the comparison without crashing.
    """
    if not _IMAGEHASH_AVAILABLE:
        return None
    if not hash_a or not hash_b:
        return None
    try:
        h_a = imagehash.hex_to_hash(hash_a)
        h_b = imagehash.hex_to_hash(hash_b)
        return h_a - h_b   # imagehash.__sub__ returns Hamming distance
    except Exception:
        return None


def is_similar(
    hash_a: str | None,
    hash_b: str | None,
    threshold: int = PHASH_SIMILARITY_THRESHOLD,
) -> bool:
    """
    Return True if two pHash strings are within the Hamming distance threshold.

    Threshold semantics:
      0         — byte-exact perceptual match (same image)
      1–4       — near-identical (minor compression artefacts)
      5–8       — very similar (crop/resize/watermark tolerated)  ← default
      9–15      — similar but likely different content
      16+       — different images

    Returns False (not similar) when hashes are None or invalid — this is the
    safe fallback: we never want to falsely flag an image as reused.
    """
    dist = phash_distance(hash_a, hash_b)
    if dist is None:
        return False
    return dist <= threshold
