import os
import time
import ffmpeg
import logging

# logger = logging.getLogger(__name__)

def extract_video_frames(video_path: str, output_dir: str, fps: float = 0.5, max_frames: int = 30) -> list[str]:
    """
    Extracts frames from a video file using ffmpeg-python.
    
    Args:
        video_path: Absolute or accessible path/URL to the video file.
        output_dir: Directory where the extracted JPEG frames will be saved.
        fps: The rate at which to extract frames (e.g., 0.5 = 1 frame every 2 seconds).
        max_frames: The maximum number of frames to extract to prevent processing overload.
        
    Returns:
        List of absolute file paths to the extracted JPEG frames.
    """
    start_time = time.perf_counter()
    extracted_frames = []

    try:
        # Define output pattern
        output_pattern = os.path.join(output_dir, "frame_%04d.jpg")
        
        # 1) Use yt-dlp to resolve the raw media URL if needed
        import yt_dlp
        ydl_opts = {
            "quiet": True, 
            "no_warnings": True, 
            "format": "bestvideo/best"
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(video_path, download=False)
                resolved_url = info.get('url', video_path)
        except Exception as e:
            print(f"⚠️ yt-dlp could not resolve URL, falling back to original: {e}")
            resolved_url = video_path

        # 2) Build and run the FFmpeg command using the resolved URL
        # vframes limits the total number of frames output
        stream = ffmpeg.input(resolved_url)
        stream = ffmpeg.filter(stream, 'fps', fps=fps)
        stream = ffmpeg.output(stream, output_pattern, vframes=max_frames, **{'qscale:v': 2})
        
        # Run ffmpeg synchronously, capturing stdout/stderr silently
        print(f"🎬 [FFmpeg] Extracting frames from {video_path} at {fps} FPS (max {max_frames})")
        ffmpeg.run(stream, capture_stdout=True, capture_stderr=True)
        
        # Collect the actual files generated
        for filename in sorted(os.listdir(output_dir)):
            if filename.startswith("frame_") and filename.endswith(".jpg"):
                extracted_frames.append(os.path.join(output_dir, filename))
                
        duration = time.perf_counter() - start_time
        print(f"✅ [FFmpeg] Extracted {len(extracted_frames)} frames in {duration:.2f}s")
        
    except ffmpeg.Error as e:
        err_msg = e.stderr.decode('utf8') if e.stderr else str(e)
        print(f"❌ [FFmpeg] frame extraction failed: {err_msg}")
    except Exception as e:
        print(f"❌ [FFmpeg] Unexpected error during frame extraction: {str(e)}")

    return extracted_frames
