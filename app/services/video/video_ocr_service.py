import re
import asyncio
import logging
from typing import List
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials
from app.config.settings import Config

# logger = logging.getLogger(__name__)

# Compile regex patterns for filtering OCR noise
timestamp_pattern = re.compile(r"^\d{1,2}:\d{2}(:\d{2})?$") # Matches 12:35, 1:24:00
numeric_only_pattern = re.compile(r"^[\d\W]+$")             # Only numbers/punctuation
watermark_words = {"subscribe", "follow us", "like and share", "tiktok", "instagram"}

def _get_vision_client() -> ComputerVisionClient:
    """Initializes the Azure Computer Vision client using env credentials."""
    key = getattr(Config, "AZURE_VISION_KEY", None) or Config.AZURE_OPENAI_API_KEY
    raw_fallback = Config.AZURE_OPENAI_ENDPOINT
    fallback_match = re.search(r"^(https?://[^/]+)", raw_fallback or "")
    clean_fallback = fallback_match.group(1) if fallback_match else (raw_fallback or "")
    vision_endpoint = getattr(Config, "AZURE_VISION_ENDPOINT", None) or clean_fallback

    if not key or not vision_endpoint:
        raise ValueError("Azure Vision Credentials missing in environment.")
        
    credentials = CognitiveServicesCredentials(key)
    return ComputerVisionClient(vision_endpoint, credentials)


async def extract_text_from_frame(frame_path: str) -> List[str]:
    """
    Asynchronously extracts and filters OCR text from a single video frame.
    
    Args:
        frame_path: Absolute path to the extracted JPEG frame.
        
    Returns:
        A list of cleaned, verified text strings found in the frame.
    """
    extracted_lines = []
    
    try:
        # We must run the synchronous Azure SDK call in a thread pool so we don't block
        # the asyncio event loop during concurrent frame processing.
        lines = await asyncio.to_thread(_sync_extract_text, frame_path)
        
        for line in lines:
            cleaned = line.strip().lower()
            # print(f"🔍 [OCR Raw] {line}") # Uncomment for extreme verbosity
            
            # Apply Normalization & Filtration Rules
            if len(cleaned) < 4:
                print(f"🗑️ [OCR Dropped: Too short] {cleaned}")
                continue
            if numeric_only_pattern.match(cleaned):
                print(f"🗑️ [OCR Dropped: Numeric/Punctuation] {cleaned}")
                continue
            if timestamp_pattern.match(cleaned):
                print(f"🗑️ [OCR Dropped: Timestamp] {cleaned}")
                continue
                
            # Check for common social media watermarks
            is_watermark = False
            for w in watermark_words:
                if w in cleaned:
                    print(f"🗑️ [OCR Dropped: Watermark '{w}'] {cleaned}")
                    is_watermark = True
                    break
            if is_watermark:
                continue
                
            extracted_lines.append(cleaned)
            
    except Exception as e:
        print(f"❌ [OCR] failed for frame {os.path.basename(frame_path)}: {e}")
        
    return extracted_lines


def _sync_extract_text(frame_path: str) -> List[str]:
    """Synchronous worker that pushes the image to Azure CV API."""
    try:
        client = _get_vision_client()
        with open(frame_path, "rb") as img_stream:
            # We use recognize_printed_text for rapid OCR typical of banners/subtitles
            ocr_result = client.recognize_printed_text_in_stream(img_stream)
            
            lines = []
            for region in ocr_result.regions:
                for line in region.lines:
                    text_parts = [word.text for word in line.words]
                    lines.append(" ".join(text_parts))
            return lines
    except Exception as e:
        print(f"❌ [OCR] Azure API Exception: {e}")
        return []
