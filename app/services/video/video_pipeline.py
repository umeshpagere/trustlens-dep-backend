import os
import asyncio
import logging
import tempfile
from typing import Dict, Any, Tuple

from app.services.video.video_frame_extractor import extract_video_frames
from app.services.video.video_ocr_service import extract_text_from_frame
from app.services.video.video_text_aggregator import aggregate_ocr_text
from app.services.video.video_ai_detector import analyze_video_ai
from app.services.video.video_frame_hasher import compute_video_hashes
from app.services.video.video_hash_database import get_all_hashes, store_frame_hash
from app.services.video.video_context_detector import detect_video_reuse
from app.services.video_analysis import extract_transcript

async def process_video_text(video_url: str) -> Dict[str, Any]:
    """
    Complete Phase A Multi-modal pipeline:
      1. Synchronously extracts audio and transcribes (via old deepgram method).
      2. Synchronously extracts video frames using ffmpeg.
      3. Asynchronously runs Azure OCR across all frames in parallel.
      4. Aggregates, deduplicates, and limits OCR text.
      5. Combines outputs into a unified semantic payload.
      
    Args:
        video_url: Accessible URL or file path for the video.
        
    Returns:
        Structured JSON dictionary containing transcript, OCR components, and metadata.
    """
    print(f"🎬 [Pipeline] Starting Multi-modal Video Pipeline for: {video_url}")
    
    # 1. Run the existing Audio Extractor (which itself uses yt-dlp & Deepgram/Whisper)
    print("🎬 [Pipeline] Step 1: Extracting Audio Transcript")
    # Wrap synchronous transcript extraction in a thread pool to avoid blocking the event loop
    audio_result = await asyncio.to_thread(extract_transcript, video_url)
    spoken_transcript = audio_result.get("transcript") if audio_result.get("success") else None
    
    # 2. Extract Video Frames (OCR Pipeline Initialization)
    print("🎬 [Pipeline] Step 2: Extracting Video Frames via FFmpeg")
    frame_paths = []
    
    # We use a context manager to guarantee cleanup of the 30 images afterward
    with tempfile.TemporaryDirectory(prefix="trustlens_vid_frames_") as tmp_dir:
        # Runs synchronously (takes ~2 seconds for 30 frames locally)
        try:
            # We use an internal to_thread call so the ffmpeg process doesn't block the async loop
            frame_paths = await asyncio.to_thread(
                extract_video_frames,
                video_url, 
                tmp_dir, 
                fps=0.5, 
                max_frames=30
            )
        except Exception as e:
            print(f"❌ [Pipeline] Frame Extraction failed: {e}")
            
        # 2.5 Compute frame hashes for context reuse detection
        print("🎬 [Pipeline] Step 2.5: Computing perceptual hashes for context reuse")
        frame_hashes = []
        context_detection_result = {
            "contextReuseDetected": False,
            "matchedFrames": 0,
            "confidence": 0.0,
            "matchedSources": []
        }
        
        if frame_paths:
            try:
                frame_hashes = await asyncio.to_thread(compute_video_hashes, frame_paths)
                if frame_hashes:
                    database_hashes = await asyncio.to_thread(get_all_hashes)
                    context_detection_result = await asyncio.to_thread(
                        detect_video_reuse, frame_hashes, database_hashes
                    )
            except Exception as e:
                print(f"❌ [Pipeline] Context Reuse Detection failed: {e}")

        # 3. Process Frames in Parallel (OCR + AI Detection)
        print(f"🎬 [Pipeline] Step 3: Running Async OCR & AI Detection on {len(frame_paths)} frames")
        all_frame_texts = []
        ai_detection_result = {
            "aiGeneratedProbability": 0.0,
            "isLikelyAIGenerated": False,
            "framesAnalyzed": 0
        }
        
        if frame_paths:
            async def run_ocr():
                ocr_tasks = [extract_text_from_frame(fp) for fp in frame_paths]
                res = await asyncio.gather(*ocr_tasks, return_exceptions=True)
                return [r if not isinstance(r, Exception) else [] for r in res]
                
            # Execute both massive I/O operations concurrently
            ocr_coro = run_ocr()
            ai_coro = analyze_video_ai(frame_paths)
            
            results = await asyncio.gather(ocr_coro, ai_coro, return_exceptions=True)
            
            # Safe unpacking
            all_frame_texts = results[0] if not isinstance(results[0], Exception) else []
            if not isinstance(results[1], Exception):
                ai_detection_result = results[1]
            else:
                print(f"❌ [Pipeline] AI Detection gathering failed: {results[1]}")
            
        # 4. Synthesize & Structure
        print("🎬 [Pipeline] Step 4: Aggregating Multi-modal Outputs")
        capped_lines, ocr_metadata = aggregate_ocr_text(all_frame_texts)
        
        print(f"🎬 [Pipeline] Completed. Extracted {len(capped_lines)} unique text lines.")
        
        # Build combined string for the LLM injection
        combined_text = ""
        if spoken_transcript:
            combined_text += f"Transcript:\n[{spoken_transcript}]\n\n"
        if capped_lines:
            combined_text += "On-screen text detected in video frames:\n"
            for line in capped_lines:
                combined_text += f"[{line}]\n"
                
        if not combined_text:
            combined_text = "[No audible transcript or visible on-screen text detected in this video]"

        visual_text_detected = len(capped_lines) > 0
        
        # 5. Store new video hashes for future context detection
        if frame_hashes:
            print("🎬 [Pipeline] Step 5: Archiving video hashes to database")
            hash_metadata = {
                "video_id": audio_result.get("video_id"),
                "platform": audio_result.get("platform", "unknown"),
                "source_url": video_url
            }
            try:
                for h in frame_hashes:
                    await asyncio.to_thread(store_frame_hash, h, hash_metadata)
            except Exception as e:
                print(f"⚠️ [Pipeline] Hash storage failed (non-fatal): {e}")

        return {
            "success": audio_result.get("success", True) or visual_text_detected,
            "videoTranscript": spoken_transcript,
            "ocrText": capped_lines,
            "combinedVideoText": combined_text,
            "framesAnalyzed": len(frame_paths),
            "visualTextDetected": visual_text_detected,
            "ocrMetadata": ocr_metadata,
            # Pass through the old yt-dlp metadata for upstream compatibility
            "method": "multimodal_pipeline",
            "platform": audio_result.get("platform", "unknown"),
            "video_id": audio_result.get("video_id"),
            "title": audio_result.get("title", ""),
            "aiDetection": ai_detection_result,
            "contextDetection": context_detection_result,
            "error": audio_result.get("error") if not audio_result.get("success") else None
        }
