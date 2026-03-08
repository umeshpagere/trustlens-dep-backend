"""
TrustLens Phase 6: Async Image Downloader

Replaces the synchronous `requests.get()` calls with `httpx.AsyncClient`
so image downloads do not block the event loop.

Why httpx over requests:
  `requests` uses blocking socket I/O — when running inside an asyncio event loop
  (Flask async route under Hypercorn), a blocking call stalls the entire loop
  and eliminates all concurrency gains. `httpx` is API-compatible but uses
  non-blocking I/O throughout.

Why a 5-second timeout:
  Image URLs from social media can hang indefinitely on faulty CDNs.
  Without a timeout the event loop is blocked until the OS TCP timeout fires
  (~75 s on macOS). 5 s is aggressive enough to stay within API response
  budgets while allowing for slow CDNs.

Why follow_redirects=True:
  Social media image URLs frequently issue 301/302 redirects to CDN hosts.
  Without following redirects the download would silently fail.
"""

import io
import asyncio
import httpx
from PIL import Image
from bs4 import BeautifulSoup

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}
_TIMEOUT = 5.0
_MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MB upper bound


def _validate_pil(content: bytes) -> bool:
    """Return True if content is a valid PIL-openable image."""
    try:
        img = Image.open(io.BytesIO(content))
        img.verify()
        return True
    except Exception:
        return False


async def download_image(url: str) -> dict:
    """
    Async download of an image URL.

    Returns
    -------
    {"success": True,  "buffer": <bytes>}  on success
    {"success": False, "error":  <str>}    on failure

    Never raises — callers can safely ignore the error case.
    """
    try:
        async with httpx.AsyncClient(
            timeout=_TIMEOUT,
            follow_redirects=True,
            headers=_HEADERS,
        ) as client:

            # --- Instagram special case ---
            if "instagram.com" in url and ("/p/" in url or "/reels/" in url):
                try:
                    base_url = url.split("?")[0].rstrip("/") + "/"
                    media_url = base_url + "media/?size=l"
                    print(f"📸 Detected Instagram URL, attempting: {media_url}")
                    resp = await client.get(media_url)
                    if resp.status_code == 200 and "image" in resp.headers.get("content-type", ""):
                        if _validate_pil(resp.content):
                            return {"success": True, "buffer": resp.content}
                except Exception as ig_err:
                    print(f"⚠️ Instagram direct extraction failed: {ig_err}")
                # Fall through to normal download

            # --- Standard download ---
            response = await client.get(url)
            response.raise_for_status()

            content_type = response.headers.get("content-type", "").lower()

            # If not an image, try og:image scrape
            if "image" not in content_type:
                try:
                    soup = BeautifulSoup(response.content, "html.parser")
                    og_image = soup.find("meta", property="og:image")
                    if og_image and og_image.get("content"):
                        og_url = og_image["content"]
                        print(f"🔗 Found og:image: {og_url}")
                        return await download_image(og_url)
                except Exception as scrape_err:
                    print(f"⚠️ Scrape attempt failed: {scrape_err}")
                return {
                    "success": False,
                    "error": f"URL did not return an image (Content-Type: {content_type})",
                }

            # Size guard
            if len(response.content) > _MAX_IMAGE_BYTES:
                return {"success": False, "error": "Image exceeds 10 MB size limit"}

            if not _validate_pil(response.content):
                return {"success": False, "error": "Downloaded data is not a valid image"}

            return {"success": True, "buffer": response.content}

    except httpx.TimeoutException:
        error = "Image download timed out after 5 seconds"
    except httpx.HTTPStatusError as e:
        error = f"Failed to download image: HTTP {e.response.status_code}"
    except httpx.ConnectError as e:
        error = f"Failed to connect to image host: {e}"
    except Exception as e:
        error = f"Failed to download image: {e}"

    print(f"[Image Download Error] {url}: {error}")
    return {"success": False, "error": error}
