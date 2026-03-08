import os
import asyncio
import logging
from typing import List, Dict, Any
import requests
from app.config.settings import Config

# logger = logging.getLogger(__name__)

# Constants
SIGHTENGINE_API_URL = "https://api.sightengine.com/1.0/check.json"
MAX_FRAMES_TO_ANALYZE = 8

async def detect_ai_frame(frame_path: str) -> float:
    """
    Asynchronously sends a single video frame to the Sightengine GenAI API.
    
    Args:
        frame_path: Absolute path to the extracted JPEG frame.
        
    Returns:
        A float representing the probability [0.0 - 1.0] that the frame is AI-generated.
        Returns 0.0 on failure or missing credentials.
    """
    api_user = getattr(Config, "SIGHTENGINE_API_USER", None) or ""
    api_secret = getattr(Config, "SIGHTENGINE_API_SECRET", None) or ""

    if not api_user or not api_secret:
        print("⚠️ [Sightengine] Missing API credentials. Skipping AI detection for frame.")
        return 0.0

    try:
        # Run synchronous requests call in a thread pool
        score = await asyncio.to_thread(_sync_detect_ai_frame, frame_path, api_user, api_secret)
        return score
    except Exception as e:
        print(f"❌ [Sightengine] API task failed for {os.path.basename(frame_path)}: {e}")
        return 0.0

def _sync_detect_ai_frame(frame_path: str, api_user: str, api_secret: str) -> float:
    """Synchronous worker that pushes the image to Sightengine."""
    try:
        with open(frame_path, 'rb') as f:
            params = {
                'models': 'genai',
                'api_user': api_user,
                'api_secret': api_secret
            }
            files = {'media': f}
            
            # Timeout set to 30s to prevent hanging, especially with parallel uploads
            response = requests.post(SIGHTENGINE_API_URL, files=files, data=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                # Typical response: {"status": "success", "type": "genai", "ai_generated": 0.81, ...}
                if data.get("status") == "success" and "type" in data:
                    if "ai_generated" in data:
                        return float(data.get("ai_generated", 0))
                    t = data.get("type")
                    if isinstance(t, dict) and t.get("ai_generated") is not None:
                        return float(t.get("ai_generated", 0))
                    if "genai" in data and isinstance(data.get("genai"), dict):
                        return float(data["genai"].get("ai_generated", 0))
                else:
                    print(f"⚠️ [Sightengine] Unsuccessful status or missing 'type': {data}")
            else:
                print(f"❌ [Sightengine] API returned status code {response.status_code}: {response.text}")
                
    except requests.exceptions.RequestException as e:
        print(f"❌ [Sightengine] Network/Timeout Error: {e}")
    except Exception as e:
        print(f"❌ [Sightengine] Unexpected error: {e}")
        
    return 0.0

async def analyze_video_ai(frames: List[str]) -> Dict[str, Any]:
    """
    Selects representative frames, analyzes them for AI generation, and aggregates the results.
    
    Args:
        frames: List of absolute file paths to all extracted video frames.
        
    Returns:
        Structured dictionary containing AI generation probability and metadata.
    """
    if not frames:
        print("⚠️ [Sightengine] No frames provided for AI detection.")
        return {
            "aiGeneratedProbability": 0.0,
            "isLikelyAIGenerated": False,
            "framesAnalyzed": 0
        }
    
    # Select representative, evenly spaced frames (max 8)
    num_frames = len(frames)
    if num_frames <= MAX_FRAMES_TO_ANALYZE:
        selected_frames = frames
    else:
        # e.g., if 30 frames, take indices [0, 4, 8, 12, 17, 21, 25, 29]
        step = max(1, num_frames // MAX_FRAMES_TO_ANALYZE)
        selected_frames = frames[::step][:MAX_FRAMES_TO_ANALYZE]
        
    print(f"🤖 [Sightengine] Analyzing {len(selected_frames)} selective frames for AI generation...")
    
    # Process frames concurrently
    tasks = [detect_ai_frame(fp) for fp in selected_frames]
    scores = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Filter out exceptions from task failures
    valid_scores = [s for s in scores if isinstance(s, (int, float))]
    
    if not valid_scores:
        print("⚠️ [Sightengine] All frame detections failed. Returning null fallback.")
        return {
            "aiGeneratedProbability": None,
            "isLikelyAIGenerated": False,
            "framesAnalyzed": 0
        }
        
    average_score = sum(valid_scores) / len(valid_scores)
    
    print(f"🤖 [Sightengine] AI Generation Probability: {average_score:.2f} across {len(valid_scores)} frames")
    
    return {
        "aiGeneratedProbability": round(average_score, 3),
        "isLikelyAIGenerated": average_score > 0.6,
        "framesAnalyzed": len(valid_scores)
    }
