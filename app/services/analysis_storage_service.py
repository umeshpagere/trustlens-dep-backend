"""
Analysis Storage Service for TrustLens (Imagine Cup)

This service stores analysis results in MongoDB using cryptographic hashes.

    Privacy & Responsible AI Design:
    - Only hash values are stored as unique identifiers; no raw images or text are persisted.
    - This approach protects user privacy by ensuring that raw sensitive content never leaves the ephemeral memory of the server for long-term storage.
    - By using cryptographic hashes (SHA-256), we can detect duplicates without knowing the original content.
    - We do not infer location, authorship, or personally identifiable information, adhering to Imagine Cup ethical standards.

"""

from datetime import datetime, timezone
try:
    from pymongo import MongoClient, errors
    import certifi
except Exception:
    MongoClient = None
    errors = None
    certifi = None
from app.config.settings import Config


_mongo_client = None
_collection = None


def _get_collection():
    """
    Lazily initialize and return the MongoDB collection.
    Uses a unique index on 'hash' for efficient lookups.
    """
    global _mongo_client, _collection
    
    if _collection is not None:
        return _collection
    
    if MongoClient is None:
        print("⚠️ pymongo not installed. Storage disabled.")
        return None
    
    if not Config.MONGODB_URI:
        print("⚠️ MongoDB URI not configured. Storage disabled.")
        return None
    
    try:
        connect_args = {
            "connect": True,
            "serverSelectionTimeoutMS": 2000,
            "socketTimeoutMS": 2000,
            "tlsAllowInvalidCertificates": getattr(Config, "MONGODB_TLS_ALLOW_INVALID_CERTIFICATES", False),
        }
        if certifi:
            connect_args["tlsCAFile"] = certifi.where()
            
        _mongo_client = MongoClient(Config.MONGODB_URI, **connect_args)
        db = _mongo_client[Config.MONGODB_DATABASE]
        _collection = db[Config.MONGODB_COLLECTION]
        _collection.create_index("hash", unique=True)
        print(f"✅ Connected to MongoDB: {Config.MONGODB_DATABASE}/{Config.MONGODB_COLLECTION}")
        return _collection
    except Exception as e:
        print(f"❌ MongoDB connection failed: {str(e)}")
        return None


def store_analysis(hash_value: str, data_type: str, analysis_result: dict) -> dict:
    """
    Store an analysis result in Azure DocumentDB.
    
    Args:
        hash_value: SHA-256 hash of the content (used as document id and partition key)
        data_type: "image" or "text"
        analysis_result: The analysis result object (no raw content)
    
    Returns:
        The stored document or error info
    
    Privacy Note: Only the hash is stored as the identifier.
    Raw user content is never persisted to protect privacy.
    """
    collection = _get_collection()
    if collection is None:
        return {"success": False, "error": "MongoDB not configured"}
    
    document = {
        "id": hash_value,
        "hash": hash_value,
        "type": data_type,
        "analysis": analysis_result,
        "createdAt": datetime.now(timezone.utc).isoformat()
    }
    
    try:
        collection.replace_one({"hash": hash_value}, document, upsert=True)
        print(f"✅ Stored analysis for hash: {hash_value[:16]}...")
        return {"success": True, "document": document}
    except errors.PyMongoError as e:
        print(f"❌ Failed to store analysis: {str(e)}")
        return {"success": False, "error": str(e)}


def get_analysis_by_hash(hash_value: str) -> dict:
    """
    Retrieve a previously stored analysis by its hash.
    
    Args:
        hash_value: SHA-256 hash of the content
    
    Returns:
        The stored document if found, None otherwise
    
    This enables efficient deduplication: if we've seen this content before,
    we return the cached analysis instead of re-processing.
    """
    collection = _get_collection()
    if collection is None:
        return None
    
    try:
        item = collection.find_one({"hash": hash_value})
        if item:
            print(f"✅ Found existing analysis for hash: {hash_value[:16]}...")
            item.pop("_id", None)
            return item
        return None
    except errors.PyMongoError as e:
        print(f"⚠️ Error retrieving analysis: {str(e)}")
        return None
