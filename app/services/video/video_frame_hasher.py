import logging
from typing import List, Optional
from PIL import Image
import imagehash

logger = logging.getLogger(__name__)

def compute_frame_hash(frame_path: str) -> Optional[str]:
    """
    Computes the perceptual hash of a single image frame using imagehash.
    
    Args:
        frame_path: Absolute path to the extracted image frame.
        
    Returns:
        String representation of the perceptual hash (e.g. 'f0e1c2d3a4b5c6d7')
        or None if image reading fails.
    """
    try:
        image = Image.open(frame_path)
        # Convert to perceptual hash. phash is robust to color/scale changes
        phash = imagehash.phash(image)
        return str(phash)
    except Exception as e:
        print(f"❌ [VideoHasher] Failed to compute pHash for {frame_path}: {e}")
        return None

def compute_video_hashes(frames: List[str]) -> List[str]:
    """
    Computes perceptual hashes for a restricted number of representative frames.
    
    Args:
        frames: List of absolute file paths to extracted video frames.
        
    Returns:
        List of computed string hashes.
    """
    if not frames:
        return []

    # Target 5 to 10 frames to avoid over-hashing massive videos
    # but still capture enough visual state changes.
    num_frames = len(frames)
    max_frames = 10
    
    if num_frames <= max_frames:
        selected_frames = frames
    else:
        # Down-sample to exactly max_frames
        step = max(1, num_frames // max_frames)
        selected_frames = frames[::step][:max_frames]
        
    print(f"🔍 [VideoHasher] Computing pHashes for {len(selected_frames)} representative frames...")
    
    hashes = []
    for frame_path in selected_frames:
        h = compute_frame_hash(frame_path)
        if h:
            hashes.append(h)
            
    print(f"✅ [VideoHasher] Successfully computed {len(hashes)} perceptual hashes.")
    return hashes
