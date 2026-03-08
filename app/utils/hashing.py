import hashlib

def hash_image(image_bytes: bytes) -> str:
    """
    Generate a deterministic SHA-256 hash of image bytes.
    
    Privacy Note: We store only the hash, never the raw image data.
    This ensures user content cannot be reconstructed from our database.
    """
    return hashlib.sha256(image_bytes).hexdigest()


def hash_text(text: str) -> str:
    """
    Generate a deterministic SHA-256 hash of normalized text.
    
    Privacy Note: Text is normalized (lowercased, trimmed) before hashing
    to ensure consistent deduplication. Raw text is never stored.
    """
    normalized = text.strip().lower()
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()
