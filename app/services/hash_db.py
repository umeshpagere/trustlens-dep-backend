"""
TrustLens Phase 4: Known Image Hash Database (MVP In-Memory Store)

Maintains a registry of known image perceptual hashes paired with their
verified event context. Used to detect image reuse — a common misinformation
technique where a real photo from one event is recirculated with false captions
about a different event.

Architecture notes
-------------------
MVP implementation uses a static in-memory list. This is:
  - Zero-latency (no DB round-trip)
  - Thread-safe for reads (no mutation at runtime)
  - Immediately testable without infrastructure

Phase 5 migration path (MongoDB):
  Replace lookup_hash() body with:
    db.image_hashes.find_one(...)  using $expr and custom Hamming distance
  Or use Redis BITCOUNT on stored hashes for sub-millisecond lookup.

Why approximate matching is required:
  The same viral image may be JPEG-recompressed several times as it travels
  through WhatsApp, Twitter, Facebook, and Telegram. Each re-save changes
  the SHA-256 completely but the pHash stays within 3–5 bits of the original.
  Exact hash matching would miss 90% of real reuse cases.

Why small threshold is necessary:
  A threshold of 8 (out of 64 bits) allows:
    ✅ Minor JPEG compression artefacts
    ✅ Small crops and pad additions
    ✅ RGB → slightly different colour temperature
  But rejects:
    ❌ Different news photos from the same event
    ❌ Generic background images
    ❌ Stock photos on different topics
"""

from typing import Any

from app.utils.hash_utils import is_similar, PHASH_SIMILARITY_THRESHOLD

# ---------------------------------------------------------------------------
# Known image hash registry
#
# Format per entry:
#   hash             — pHash hex string of the original verified image
#   context          — human-readable description of the verified event
#   verifiedEventDate — ISO-8601 date the image was originally published
#   source           — organisation/outlet that verified the original
#
# Populate from a real fact-checking database (Snopes, AFP, Reuters) in prod.
# ---------------------------------------------------------------------------
KNOWN_IMAGE_HASHES: list[dict[str, str]] = [
    {
        "hash": "f8f0e0c0808080c0",
        "context": "Flood in Kerala, India — August 2019",
        "verifiedEventDate": "2019-08-12",
        "source": "The Hindu",
    },
    {
        "hash": "3c3c18181818183c",
        "context": "Syrian civil war — Aleppo bombing — 2016",
        "verifiedEventDate": "2016-09-24",
        "source": "BBC",
    },
    {
        "hash": "7e7e424242427e7e",
        "context": "COVID-19 hospital queue — New York — April 2020",
        "verifiedEventDate": "2020-04-07",
        "source": "Reuters",
    },
    {
        "hash": "0000007e7e000000",
        "context": "Australia bushfires — January 2020",
        "verifiedEventDate": "2020-01-10",
        "source": "AFP Fact Check",
    },
]


def lookup_hash(
    phash_str: str | None,
    threshold: int = PHASH_SIMILARITY_THRESHOLD,
) -> dict[str, Any] | None:
    """
    Search the known-hash registry for a visually similar image.

    Parameters
    ----------
    phash_str : str | None
        Perceptual hash of the image to look up.
    threshold : int
        Maximum Hamming distance to consider a match (default 8).

    Returns
    -------
    dict with keys {hash, context, verifiedEventDate, source} if a match
    is found within the threshold, or None if no match.

    Design: linear scan is fine for an in-memory MVP list of hundreds of
    entries. For thousands of hashes use a BK-tree or VP-tree structure.
    """
    if not phash_str:
        return None

    for entry in KNOWN_IMAGE_HASHES:
        stored_hash = entry.get("hash", "")
        if not stored_hash:
            continue
        try:
            if is_similar(phash_str, stored_hash, threshold=threshold):
                return dict(entry)   # return a copy, never expose mutable state
        except Exception:
            continue   # malformed stored hash — skip silently

    return None
