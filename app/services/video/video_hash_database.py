import logging
from typing import List, Dict, Any

try:
    from pymongo import MongoClient
    import certifi
except Exception:
    MongoClient = None
    certifi = None

from app.config.settings import Config

logger = logging.getLogger(__name__)

# Dedicated collection for video frame perceptual hashes
COLLECTION_NAME = "video_frame_hashes"

_hash_collection = None


def _get_hash_collection():
    """
    Lazily initialise and return the video_frame_hashes MongoDB collection.
    Separate from the main analysis_records collection intentionally.
    """
    global _hash_collection

    if _hash_collection is not None:
        return _hash_collection

    if MongoClient is None:
        print("⚠️ [HashDatabase] pymongo not installed.")
        return None

    if not Config.MONGODB_URI:
        print("⚠️ [HashDatabase] MONGODB_URI not configured.")
        return None

    try:
        client = MongoClient(
            Config.MONGODB_URI,
            serverSelectionTimeoutMS=5000,
            tlsAllowInvalidCertificates=True,
            tls=True,
        )
        db = client[Config.MONGODB_DATABASE or "trustlensDB"]
        _hash_collection = db[COLLECTION_NAME]
        # Index the hash field for O(1) lookups
        _hash_collection.create_index("hash")
        print(f"✅ [HashDatabase] Connected to MongoDB: {Config.MONGODB_DATABASE}/{COLLECTION_NAME}")
        return _hash_collection
    except Exception as e:
        print(f"❌ [HashDatabase] MongoDB connection failed: {e}")
        return None


def store_frame_hash(hash_val: str, metadata: Dict[str, Any]) -> bool:
    """
    Stores a single perceptual frame hash into the MongoDB collection.

    Args:
        hash_val: The computed pHash string.
        metadata: Dictionary with video_id, platform, timestamp, source_url.

    Returns:
        Boolean indicating success.
    """
    try:
        collection = _get_hash_collection()
        if collection is None:
            return False

        document = {"hash": hash_val, "metadata": metadata}
        collection.insert_one(document)
        return True

    except Exception as e:
        print(f"❌ [HashDatabase] Failed to store hash {hash_val[:8]}...: {e}")
        return False


def get_all_hashes() -> List[Dict[str, Any]]:
    """
    Retrieves the entire corpus of historical frame hashes for comparison.

    Returns:
        List of dicts with {\"hash\": \"...\", \"metadata\": {...}}
    """
    try:
        collection = _get_hash_collection()
        if collection is None:
            return []

        cursor = collection.find({}, {"_id": 0, "hash": 1, "metadata": 1})
        return list(cursor)

    except Exception as e:
        print(f"❌ [HashDatabase] Failed to retrieve historical hashes: {e}")
        return []
