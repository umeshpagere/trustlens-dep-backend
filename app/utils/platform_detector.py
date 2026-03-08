"""
Platform Detector for TrustLens Multi-Platform Video Analysis

Detects which social media platform a video URL belongs to and
provides routing information for the transcript extractor.
"""

import re
from urllib.parse import urlparse

# Platform definitions: name -> list of domain patterns
PLATFORM_DOMAINS = {
    "youtube":   ["youtube.com", "youtu.be", "youtube-nocookie.com", "m.youtube.com"],
    "tiktok":    ["tiktok.com", "vm.tiktok.com", "vt.tiktok.com", "m.tiktok.com"],
    "instagram": ["instagram.com", "www.instagram.com"],
    "facebook":  ["facebook.com", "fb.com", "fb.watch", "www.facebook.com", "m.facebook.com", "web.facebook.com"],
    "twitter":   ["twitter.com", "x.com", "t.co", "www.twitter.com", "www.x.com"],
    "reddit":    ["reddit.com", "www.reddit.com", "old.reddit.com", "v.redd.it", "redd.it"],
    "linkedin":  ["linkedin.com", "www.linkedin.com", "lnkd.in"],
    "twitch":    ["twitch.tv", "www.twitch.tv", "clips.twitch.tv", "m.twitch.tv"],
    "vimeo":     ["vimeo.com", "www.vimeo.com", "player.vimeo.com"],
    "dailymotion": ["dailymotion.com", "dai.ly"],
    "rumble":    ["rumble.com"],
    "odysee":    ["odysee.com"],
    "bitchute":  ["bitchute.com"],
    "snapchat":  ["snapchat.com", "www.snapchat.com"],
}

# Platforms that support subtitle/caption extraction without audio download (Tier 1)
SUBTITLE_SUPPORTED = {"youtube", "tiktok", "facebook", "reddit", "vimeo", "dailymotion", "rumble"}

# Platforms that need audio download + Whisper transcription (Tier 2)
AUDIO_TRANSCRIPTION_SUPPORTED = {"instagram", "linkedin", "twitch", "odysee", "bitchute", "snapchat", "twitter"}

# All supported platforms
ALL_SUPPORTED_PLATFORMS = set(PLATFORM_DOMAINS.keys())


def detect_platform(url: str) -> str:
    """
    Detect the social media platform of a video URL.

    Returns:
        Platform name string (e.g. "youtube", "tiktok") or "unknown"
    """
    if not url:
        return "unknown"

    try:
        parsed = urlparse(url.lower().strip())
        hostname = parsed.netloc.lstrip("www.")

        for platform, domains in PLATFORM_DOMAINS.items():
            for domain in domains:
                # Strip leading www. from domain list for comparison
                clean_domain = domain.lstrip("www.")
                if hostname == clean_domain or hostname.endswith("." + clean_domain):
                    return platform
    except Exception:
        pass

    return "unknown"


def is_supported_video_url(url: str) -> bool:
    """
    Check if the URL is from a supported video platform.
    Used by Pydantic schema validation.
    """
    return detect_platform(url) in ALL_SUPPORTED_PLATFORMS


def get_extraction_tier(platform: str) -> str:
    """
    Get the extraction tier for a given platform.

    Returns:
        "subtitle"  - Tier 1: use yt-dlp to pull embedded subtitles/captions
        "audio"     - Tier 2: download audio and transcribe via Whisper
        "unsupported" - Platform not handled
    """
    if platform in SUBTITLE_SUPPORTED:
        return "subtitle"
    elif platform in AUDIO_TRANSCRIPTION_SUPPORTED:
        return "audio"
    return "unsupported"


def get_platform_display_name(platform: str) -> str:
    """Human-readable platform name for API responses."""
    display_names = {
        "youtube":     "YouTube",
        "tiktok":      "TikTok",
        "instagram":   "Instagram",
        "facebook":    "Facebook",
        "twitter":     "Twitter/X",
        "reddit":      "Reddit",
        "linkedin":    "LinkedIn",
        "twitch":      "Twitch",
        "vimeo":       "Vimeo",
        "dailymotion": "Dailymotion",
        "rumble":      "Rumble",
        "odysee":      "Odysee",
        "bitchute":    "Bitchute",
        "snapchat":    "Snapchat",
    }
    return display_names.get(platform, platform.capitalize())
