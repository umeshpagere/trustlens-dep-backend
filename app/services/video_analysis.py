"""
Multi-Platform Video Analysis Service for TrustLens

Supports transcript extraction from:
  Tier 1 (subtitles): YouTube, TikTok, Facebook, Twitter/X, Reddit, Vimeo, Dailymotion, Rumble
  Tier 2 (audio/Google STT): Instagram, LinkedIn, Twitch, Odysee, Bitchute, Snapchat + Tier-1 fallback
  Tier 3 (skipped): unknown/unsupported platforms

All transcripts are analyzed for misinformation via Azure OpenAI.
"""

import re
import json
import hashlib
import os
import tempfile

from app.config.azure import get_azure_client
from app.config.settings import Config
from app.utils.platform_detector import (
    detect_platform,
    get_extraction_tier,
    get_platform_display_name,
)

# Keep old import for backward compatibility
try:
    from youtube_transcript_api import YouTubeTranscriptApi
    _YTAPI_AVAILABLE = True
except ImportError:
    _YTAPI_AVAILABLE = False

try:
    import yt_dlp
    _YTDLP_AVAILABLE = True
except ImportError:
    _YTDLP_AVAILABLE = False

try:
    from deepgram import DeepgramClient, PrerecordedOptions, FileSource
    _DEEPGRAM_AVAILABLE = True
except ImportError:
    _DEEPGRAM_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def hash_video_url(video_url: str) -> str:
    """Generate a SHA-256 hash of the video URL for caching."""
    return hashlib.sha256(video_url.strip().encode("utf-8")).hexdigest()


def _clean_markdown_json(text: str) -> str:
    """Strip markdown code fences from an LLM response."""
    text = text.strip()
    if text.startswith("```json"):
        text = re.sub(r"^```json\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    elif text.startswith("```"):
        text = re.sub(r"^```\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text


# ─────────────────────────────────────────────────────────────────────────────
# Backward-compatible YouTube extractor (uses youtube-transcript-api directly)
# ─────────────────────────────────────────────────────────────────────────────

def _extract_video_id_youtube(video_url: str) -> str:
    patterns = [
        r"(?:youtube\.com/watch\?.*v=|youtu\.be/|youtube\.com/embed/|youtube\.com/shorts/)([a-zA-Z0-9_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, video_url)
        if match:
            return match.group(1)
    raise ValueError(f"Could not extract YouTube video ID from URL: {video_url}")


def extract_youtube_transcript(video_url: str) -> dict:
    """
    Legacy YouTube-specific extractor using youtube-transcript-api.
    Kept for backward compatibility; new code should call extract_transcript().
    """
    try:
        video_id = _extract_video_id_youtube(video_url)
        print(f"🎬 [YouTube] Extracting transcript for video ID: {video_id}")

        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        full_transcript = " ".join(seg["text"] for seg in transcript_list)

        if len(full_transcript) > 8000:
            full_transcript = full_transcript[:8000] + "... [transcript truncated]"

        print(f"✅ [YouTube] Transcript extracted: {len(full_transcript)} characters")
        print(f"\n{'─'*60}")
        print(f"📝 [YouTube] TRANSCRIPT OUTPUT:")
        print(f"{'─'*60}")
        print(full_transcript)
        print(f"{'─'*60}\n")
        return {
            "success": True,
            "transcript": full_transcript,
            "video_id": video_id,
            "segment_count": len(transcript_list),
            "method": "youtube_transcript_api",
        }
    except Exception as e:
        print(f"❌ [YouTube] Transcript extraction failed: {str(e)}")
        return {"success": False, "error": str(e), "transcript": None, "video_id": None}


# ─────────────────────────────────────────────────────────────────────────────
# Tier 1: yt-dlp subtitle/caption extraction (no audio download)
# ─────────────────────────────────────────────────────────────────────────────

def extract_subtitles_with_ytdlp(video_url: str, platform: str) -> dict:
    """
    Attempt to pull embedded or auto-generated subtitles using yt-dlp.
    No audio is downloaded — this is fast and lightweight.
    """
    if not _YTDLP_AVAILABLE:
        return {"success": False, "error": "yt-dlp not installed"}

    print(f"📄 [yt-dlp] Extracting subtitles for {platform} URL: {video_url}")

    with tempfile.TemporaryDirectory() as tmpdir:
        output_template = os.path.join(tmpdir, "%(id)s.%(ext)s")

        ydl_opts = {
            "skip_download": True,           # Don't download the video
            "writesubtitles": True,          # Download manual subtitles
            "writeautomaticsub": True,       # Download auto-generated subtitles
            "subtitlesformat": "vtt",        # VTT is easiest to parse
            "subtitleslangs": ["en", "en-US", "en-GB"],
            "outtmpl": output_template,
            "quiet": True,
            "no_warnings": True,
            "socket_timeout": 10,            # fail fast if server is slow
            "retries": 1,                    # only one retry
            "fragment_retries": 1,
            "ignoreerrors": False,
            # Avoid rate-limiting: use a real browser UA
            "http_headers": {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                              "AppleWebKit/537.36 (KHTML, like Gecko) "
                              "Chrome/120.0.0.0 Safari/537.36"
            },
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(video_url, download=True)
                if info is None:
                    return {"success": False, "error": "yt-dlp returned no info"}

            # Find downloaded .vtt files
            vtt_files = [
                os.path.join(tmpdir, f)
                for f in os.listdir(tmpdir)
                if f.endswith(".vtt")
            ]

            if not vtt_files:
                return {
                    "success": False,
                    "error": f"No subtitles found for this {get_platform_display_name(platform)} video",
                }

            # Parse first VTT file found
            transcript_text = _parse_vtt(vtt_files[0])
            if not transcript_text.strip():
                return {"success": False, "error": "Subtitle file was empty"}

            # Truncate to ~8000 chars
            if len(transcript_text) > 8000:
                transcript_text = transcript_text[:8000] + "... [transcript truncated]"

            print(f"✅ [yt-dlp subtitles] Extracted {len(transcript_text)} chars for {platform}")
            print(f"\n{'─'*60}")
            print(f"📝 [yt-dlp subtitles] TRANSCRIPT OUTPUT ({platform}):")
            print(f"{'─'*60}")
            print(transcript_text)
            print(f"{'─'*60}\n")
            return {
                "success": True,
                "transcript": transcript_text,
                "method": "yt_dlp_subtitles",
                "video_id": info.get("id"),
                "title": info.get("title", ""),
            }

        except yt_dlp.utils.DownloadError as e:
            err = str(e)
            print(f"⚠️ [yt-dlp subtitles] DownloadError for {platform}: {err}")
            return {"success": False, "error": err}
        except Exception as e:
            print(f"⚠️ [yt-dlp subtitles] Unexpected error for {platform}: {str(e)}")
            return {"success": False, "error": str(e)}


def _parse_vtt(vtt_path: str) -> str:
    """Parse a WebVTT subtitle file and return clean transcript text."""
    lines = []
    try:
        with open(vtt_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        # Split into cue blocks, strip timestamps and metadata
        blocks = re.split(r"\n\n+", content)
        seen = set()
        for block in blocks:
            block_lines = block.strip().splitlines()
            text_lines = []
            for line in block_lines:
                # Skip WEBVTT header, NOTE blocks, timestamp lines
                if (
                    line.startswith("WEBVTT")
                    or line.startswith("NOTE")
                    or line.startswith("STYLE")
                    or re.match(r"^\d{2}:\d{2}", line)
                    or "-->" in line
                    or re.match(r"^\d+$", line.strip())
                ):
                    continue
                # Strip inline VTT tags like <00:00:01.000><c>
                clean = re.sub(r"<[^>]+>", "", line).strip()
                if clean:
                    text_lines.append(clean)

            text = " ".join(text_lines).strip()
            if text and text not in seen:
                seen.add(text)
                lines.append(text)

    except Exception as e:
        print(f"⚠️ VTT parse error: {e}")

    return " ".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Tier 2: yt-dlp audio download + Deepgram transcription
# ─────────────────────────────────────────────────────────────────────────────

def transcribe_audio_with_deepgram(video_url: str, platform: str) -> dict:
    """
    Download the audio track with yt-dlp and transcribe it using Deepgram API.

    Requires:
      - yt-dlp installed
      - ffmpeg installed (for audio conversion)
      - DEEPGRAM_API_KEY set in .env
    """
    if not _YTDLP_AVAILABLE:
        return {"success": False, "error": "yt-dlp not installed"}

    if not _DEEPGRAM_AVAILABLE:
        return {"success": False, "error": "deepgram-sdk not installed. Run: pip install deepgram-sdk"}

    api_key = Config.DEEPGRAM_API_KEY
    if not api_key:
        return {
            "success": False,
            "error": "DEEPGRAM_API_KEY not set. Set it in .env to use Deepgram."
        }

    print(f"🎙️ [Deepgram] Downloading audio for {platform}: {video_url}")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Deepgram handles compressed audio very well, flac or mp3 is fine.
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": os.path.join(tmpdir, "audio.%(ext)s"),
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",  
                "preferredquality": "192",
            }],
            "quiet": True,
            "no_warnings": True,
            "socket_timeout": 30,
            "retries": 2,
            "http_headers": {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                              "AppleWebKit/537.36 (KHTML, like Gecko) "
                              "Chrome/120.0.0.0 Safari/537.36"
            },
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(video_url, download=True)

            # Find the mp3 audio file
            audio_files = [
                os.path.join(tmpdir, f)
                for f in os.listdir(tmpdir)
                if f.endswith(".mp3")
            ]

            if not audio_files:
                return {"success": False, "error": "Audio file not created by yt-dlp"}

            audio_file = audio_files[0]
            file_size_mb = os.path.getsize(audio_file) / (1024 * 1024)
            print(f"✅ [Deepgram] Audio downloaded: {file_size_mb:.1f} MB")

            # Read audio bytes
            with open(audio_file, "rb") as f:
                audio_bytes = f.read()

            # Call Deepgram API
            print(f"📡 [Deepgram] Sending API request...")
            from deepgram import DeepgramClientOptions
            config = DeepgramClientOptions(options={"timeout": 120})
            deepgram = DeepgramClient(api_key, config)

            payload: FileSource = {
                "buffer": audio_bytes,
            }

            options = PrerecordedOptions(
                model="nova-2",
                detect_language=True,
                smart_format=True,
                punctuate=True,
                utterances=False,
                diarize=False
            )

            # Transcribe
            response = deepgram.listen.prerecorded.v("1").transcribe_file(payload, options)

            # Defensive extraction: Deepgram response structure may vary
            transcript_text = ""
            try:
                results = getattr(response, "results", None)
                channels = getattr(results, "channels", None) if results else None
                if channels and len(channels) > 0:
                    ch0 = channels[0]
                    alts = getattr(ch0, "alternatives", None)
                    if alts and len(alts) > 0:
                        transcript_text = (getattr(alts[0], "transcript", None) or "").strip()
            except (AttributeError, IndexError, TypeError) as e:
                print(f"⚠️ [Deepgram] Unexpected response structure: {e}")

            if not transcript_text:
                return {"success": False, "error": "Deepgram STT returned empty or invalid transcript"}

            # Truncate to ~8000 chars
            if len(transcript_text) > 8000:
                transcript_text = transcript_text[:8000] + "... [transcript truncated]"

            print(f"✅ [Deepgram] Transcribed {len(transcript_text)} chars for {platform}")
            print(f"\n{'─'*60}")
            print(f"🎙️ [Deepgram] TRANSCRIPT OUTPUT ({platform}):")
            print(f"{'─'*60}")
            print(transcript_text)
            print(f"{'─'*60}\n")
            return {
                "success": True,
                "transcript": transcript_text,
                "method": "deepgram_audio",
                "video_id": info.get("id") if info else None,
                "title": info.get("title", "") if info else "",
            }

        except yt_dlp.utils.DownloadError as e:
            err = str(e)
            print(f"❌ [Deepgram] DownloadError for {platform}: {err}")
            return {"success": False, "error": err}
        except Exception as e:
            print(f"❌ [Deepgram] Error for {platform}: {str(e)}")
            return {"success": False, "error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# Main orchestrator: platform-aware transcript extractor
# ─────────────────────────────────────────────────────────────────────────────

def extract_transcript(video_url: str) -> dict:
    """
    Platform-aware transcript extractor.

    Routing logic:
      1. Detect platform from URL
      2. YouTube: try youtube-transcript-api first (fastest), then yt-dlp subtitles
      3. Tier 1 platforms: try yt-dlp subtitles
      4. Tier 2 platforms: try yt-dlp subtitles first, then Whisper audio fallback
      5. Unknown: return skipped

    Returns dict with:
      success, transcript, platform, platform_name, method, video_id, title, error (if any)
    """
    platform = detect_platform(video_url)
    platform_name = get_platform_display_name(platform)
    tier = get_extraction_tier(platform)

    print(f"🔍 Detected platform: {platform_name} (tier: {tier})")

    base = {"platform": platform, "platform_name": platform_name}

    if platform == "unknown" or tier == "unsupported":
        return {
            **base,
            "success": False,
            "error": f"Platform not supported for video analysis",
            "transcript": None,
        }

    # ── YouTube: fast path via youtube-transcript-api ──
    if platform == "youtube" and _YTAPI_AVAILABLE:
        result = extract_youtube_transcript(video_url)
        if result.get("success"):
            return {**base, **result}
        print(f"⚠️ [YouTube] youtube-transcript-api failed, trying yt-dlp subtitles…")

    # ── Tier 1: yt-dlp subtitle extraction ──
    if tier in ("subtitle",) or platform == "youtube":
        result = extract_subtitles_with_ytdlp(video_url, platform)
        if result.get("success"):
            return {**base, **result}
        print(f"⚠️ [{platform_name}] Subtitle extraction failed: {result.get('error')}")
        # For Tier 1 only platforms, we don't try audio
        if tier == "subtitle":
            return {
                **base,
                "success": False,
                "error": result.get("error", "No subtitles available"),
                "transcript": None,
            }

    # ── Tier 2: Audio + Deepgram Speech-to-Text ──
    # Only for audio-tier platforms (Instagram, LinkedIn, Twitch, Twitter etc.)
    # YouTube is intentionally excluded — it always has captions via Tier 1
    if tier == "audio":
        print(f"🎙️ [{platform_name}] Using Deepgram audio transcription…")
        result = transcribe_audio_with_deepgram(video_url, platform)
        if result.get("success"):
            return {**base, **result}
        return {
            **base,
            "success": False,
            "error": result.get("error", "Audio transcription failed"),
            "transcript": None,
        }

    return {
        **base,
        "success": False,
        "error": "No transcript extraction method available",
        "transcript": None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# LLM Analysis (platform-agnostic, unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def analyze_video_with_llm(combined_video_text: str, video_url: str, accompanying_text: str = "") -> dict:
    """
    Analyze the unified multimodal text string (transcript + OCR frames) for misinformation using Azure OpenAI.
    Works for all platforms.
    """
    if not combined_video_text or len(combined_video_text.strip()) < 10:
        raise ValueError("Video text payload is too short for analysis")

    try:
        client = get_azure_client()
        response = client.chat.completions.create(
            model=Config.AZURE_OPENAI_DEPLOYMENT,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a fact-checking analyst for video content. Your PRIMARY task is to identify "
                        "factual claims that can be verified—specific statements of fact, statistics, event "
                        "descriptions, and attributions. The goal is to give users correct information, not to "
                        "judge tone or sentiment.\n\n"
                        "== CLAIM EXTRACTION RULES (STRICT) ==\n"
                        "A valid claim MUST contain:\n"
                        "  \u2022 A clear subject: a person, organisation, government, event, or object\n"
                        "  \u2022 A specific action or event involving that subject\n"
                        "  \u2022 Optional: numbers, dates, or locations\n\n"
                        "Do NOT include opinions, emotional reactions, or vague descriptions.\n"
                        "If a statement has no subject OR no action, do NOT include it in claimsIdentified.\n\n"
                        "VALID examples: "
                        "\"WHO declared a global health emergency.\", "
                        "\"Government banned bank withdrawals.\", "
                        "\"Police arrested protesters in Paris.\", "
                        "\"NASA confirmed water was found on Mars.\"\n\n"
                        "INVALID examples: "
                        "\"People are suffering.\", "
                        "\"This situation is terrible.\", "
                        "\"Something big is happening.\", "
                        "\"The situation is getting worse.\"\n\n"
                        "PRIORITIZE:\n"
                        "- claimsIdentified: List up to 5 specific FACTUAL claims a fact-checker could verify.\n"
                        "- credibilityScore: Base on whether claims are checkable and evidence-supported, NOT on tone.\n\n"
                        "Respond ONLY in valid JSON format. No markdown, no code blocks, no extra text."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Extract factual claims and assess verifiability of this video. "
                        f"Return JSON only in this exact format:\n"
                        f"{{\n"
                        f'  "claimsIdentified": string[] (key factual claims to verify, max 5—MOST IMPORTANT),\n'
                        f'  "topicSummary": string (2-3 sentence summary of content),\n'
                        f'  "riskLevel": "low" | "medium" | "high",\n'
                        f'  "credibilityScore": number (0-100),\n'
                        f'  "verdict": "Reliable" | "Questionable" | "High Risk",\n'
                        f'  "riskKeywordsFound": string[],\n'
                        f'  "explanation": string (focus on claims and verifiability, not tone)\n'
                        f"}}\n\n"
                        f"Scoring guidelines (base on FACTS and EVIDENCE, not sentiment):\n"
                        f"- 75-100: Reliable – Verifiable claims, well-sourced or consistent with evidence\n"
                        f"- 40-74: Questionable – Some claims unverifiable or conflicting with known facts\n"
                        f"- 0-39: High Risk – Key claims demonstrably false or wholly unverifiable\n\n"
                        f"Video URL: {video_url}\n"
                        f"{'USER TEXT CONTEXT: The user shared this video with: ' + repr(accompanying_text) + '. Does the video content support or contradict this claim? Include in explanation.\\n\\n' if accompanying_text else ''}"
                        f"\n--- VIDEO CONTENT (MULTIMODAL) ---\n"
                        f"{combined_video_text}\n"
                        f"--- END VIDEO CONTENT ---"
                    ),
                },
            ],
            temperature=0.2,
            max_tokens=800,
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content
        if not content:
            raise ValueError("No response content from Azure OpenAI")

        result = json.loads(_clean_markdown_json(content))

        if not all(k in result for k in ["riskLevel", "credibilityScore", "verdict", "explanation"]):
            raise ValueError("Invalid response format from Azure OpenAI")

        if not isinstance(result.get("riskKeywordsFound"), list):
            result["riskKeywordsFound"] = []
        if not isinstance(result.get("claimsIdentified"), list):
            result["claimsIdentified"] = []

        print(f"\n────────────────────────────────────────────────────────────")
        print(f"🤖 [LLM] VIDEO ANALYSIS RESPONSE:")
        print(f"────────────────────────────────────────────────────────────")
        print(json.dumps(result, indent=2))
        print(f"────────────────────────────────────────────────────────────\n")

        return result

    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse LLM response: {str(e)}")
    except Exception as e:
        print(f"Azure OpenAI video analysis failed: {str(e)}")
        raise ValueError(f"LLM video analysis failed: {str(e)}")
