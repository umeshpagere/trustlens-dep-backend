import logging
import imagehash
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

def is_similar(hash1: str, hash2: str, threshold: int = 5) -> bool:
    """
    Computes the Hamming Distance between two perceptual hashes.
    
    Args:
        hash1: First pHash string
        hash2: Second pHash string
        threshold: The max distance to be considered "similar"
        
    Returns:
        True if distance <= threshold, else False
    """
    try:
        # Reconstruct the imagehash objects from the hex strings
        h1 = imagehash.hex_to_hash(hash1)
        h2 = imagehash.hex_to_hash(hash2)
        
        # Subtraction in imagehash computes the Hamming Distance
        distance = h1 - h2
        return bool(distance < threshold)
        
    except Exception as e:
        print(f"❌ [ContextDetector] Hash processing error for {hash1} vs {hash2}: {e}")
        return False

def detect_video_reuse(frame_hashes: List[str], database_hashes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compares the newly extracted hashes against the historical database hashes 
    to detect if the video has been seen/reused before.
    
    Args:
        frame_hashes: List of [hash] strings for the current video.
        database_hashes: List of [{"hash": "...", "metadata": {...}}] from historical videos.
        
    Returns:
        Structured context detection result dictionary.
    """
    if not frame_hashes:
        return {
            "contextReuseDetected": False,
            "matchedFrames": 0,
            "confidence": 0.0,
            "matchedSources": []
        }
        
    if not database_hashes:
        print("💡 [ContextDetector] Hash database is empty. No historical matches possible.")
        return {
            "contextReuseDetected": False,
            "matchedFrames": 0,
            "confidence": 0.0,
            "matchedSources": []
        }
        
    print(f"🔍 [ContextDetector] Comparing {len(frame_hashes)} new frames against {len(database_hashes)} known corpus hashes...")
    
    # We want to find how many of *our* frames hit a match.
    # We deduplicate the matches to prevent a single highly-repeated frame 
    # taking false credit across the db.
    matched_new_frames = set() 
    matched_sources = []
    
    for f_hash in frame_hashes:
        for db_record in database_hashes:
            db_hash = db_record.get("hash")
            
            # Skip empty or malformed
            if not db_hash or len(f_hash) != len(db_hash):
                continue
                
            if is_similar(f_hash, db_hash):
                matched_new_frames.add(f_hash)
                
                # Capture where this was from natively
                metadata = db_record.get("metadata", {})
                source_id = metadata.get("video_id") or metadata.get("source_url")
                if source_id and source_id not in matched_sources:
                    matched_sources.append(source_id)
                break # Move to testing the next frame immediately once a match is found
                
    num_matches = len(matched_new_frames)
    total_frames = len(frame_hashes)
    confidence = float(num_matches) / total_frames if total_frames > 0 else 0.0
    
    # If 3 or more of our frames match previous historical frames, it's flagged.
    # Or if confidence > 0.4 (meaning 40% of the video is repeated content)
    is_reused = num_matches >= 3 or confidence > 0.4
    
    if is_reused:
        print(f"🚨 [ContextDetector] Context Reuse Detected! {num_matches}/{total_frames} frames matched historical corpus.")
    else:
        print(f"✅ [ContextDetector] Unique content. {num_matches}/{total_frames} frames matched historical corpus.")
        
    return {
        "contextReuseDetected": is_reused,
        "matchedFrames": num_matches,
        "confidence": round(confidence, 3),
        "matchedSources": matched_sources[:5] # Limit to top 5 trace sources
    }
