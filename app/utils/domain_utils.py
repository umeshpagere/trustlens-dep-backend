"""
TrustLens Phase 3: Domain Extraction Utility

Provides a pure, side-effect-free helper for extracting a normalised
bare domain from an arbitrary URL string.

Design rationale
----------------
URL parsing is necessary because raw user-supplied strings vary wildly:
  - "HTTP://EXAMPLE.COM/path?q=1"   → example.com
  - "www.bbc.co.uk/news"            → bbc.co.uk  (www stripped)
  - "not a url at all"              → None

Why defensive parsing matters:
  The credibility pipeline must never crash on bad input. extract_domain()
  returns None rather than raising; callers can then apply a neutral fallback.

Why domain normalisation matters:
  "BBC.COM", "bbc.com", and "www.bbc.com" must all resolve to the same
  whitelist / blacklist entry. Lower-casing and www-stripping ensure that.
"""

import urllib.parse


def extract_domain(url: str) -> str | None:
    """
    Parse *url* and return the lowercase bare domain (no 'www.' prefix).

    Returns None for:
      - non-string input
      - empty / whitespace-only strings
      - strings without a recognisable scheme (http/https)
      - strings that produce an empty netloc after parsing

    Examples
    --------
    >>> extract_domain("https://www.BBC.com/news/world")
    'bbc.com'
    >>> extract_domain("http://infowars.com/article?id=1")
    'infowars.com'
    >>> extract_domain("not-a-url")
    None
    >>> extract_domain("")
    None
    """
    if not url or not isinstance(url, str):
        return None

    raw = url.strip()
    if not raw:
        return None

    # urllib.parse requires a scheme to populate netloc correctly.
    # Prepend https:// when missing so bare domains like "bbc.com/news" parse.
    # Use lower() only for the scheme check — not for the full URL — so the
    # netloc normalisation step later still handles case folding uniformly.
    if not raw.lower().startswith(("http://", "https://", "ftp://")):
        raw = "https://" + raw

    try:
        parsed = urllib.parse.urlparse(raw)
        netloc = (parsed.netloc or "").lower().strip()
        if not netloc:
            return None

        # Strip port if present (e.g. "example.com:8080" → "example.com")
        domain = netloc.split(":")[0]

        # Normalise: remove leading "www." so whitelist/blacklist lookups
        # match regardless of whether the URL used the subdomain.
        if domain.startswith("www."):
            domain = domain[4:]

        return domain if domain else None
    except Exception:
        # Belt-and-suspenders: urlparse is very tolerant but guard anyway.
        return None
