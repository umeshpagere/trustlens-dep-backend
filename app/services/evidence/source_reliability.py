"""
TrustLens Source Reliability Scoring (Phase E)

Assigns a trust score (0.0 – 1.0) to any evidence source URL.

Algorithm:
  1. Extract domain from URL
  2. Look up against 3-tier domain lists (High / Medium / Low trust)
  3. Apply bonus signals:
     - HTTPS (+0.05)
     - Known fact-check publisher (+0.10)
  4. Clamp result to [0.0, 1.0]
"""

import re
from urllib.parse import urlparse
from typing import Optional

# ---------------------------------------------------------------------------
# 1. Domain trust tiers
# ---------------------------------------------------------------------------

HIGH_TRUST_DOMAINS = {
    # Government / Inter-governmental
    "who.int", "cdc.gov", "nih.gov", "fda.gov", "nasa.gov", "un.org",
    "europa.eu", "gov.uk",
    # Premium wire/broadcast news
    "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk",
    # Major newspapers (international)
    "nytimes.com", "theguardian.com", "washingtonpost.com",
    "wsj.com", "ft.com", "economist.com",
    # Science / Academic
    "nature.com", "sciencedirect.com", "pubmed.ncbi.nlm.nih.gov",
    "sciencemag.org", "cell.com", "thelancet.com", "bmj.com",
    "nejm.org", "plos.org",
    # Indian premium outlets
    "thehindu.com", "ndtv.com", "hindustantimes.com", "livemint.com",
    "indianexpress.com",
    # Other trusted outlets
    "aljazeera.com", "dw.com", "npr.org", "pbs.org",
    "nationalgeographic.com", "scientificamerican.com",
}

MEDIUM_TRUST_DOMAINS = {
    "wikipedia.org", "en.wikipedia.org", "britannica.com",
    "time.com", "newsweek.com", "theatlantic.com",
    "vox.com", "slate.com", "bloomberg.com",
    "cnbc.com", "cnn.com", "abc.net.au",
    "latimes.com", "usatoday.com", "cbsnews.com", "nbcnews.com",
    "forbes.com", "businessinsider.com",
    "gizmodo.com", "wired.com", "techcrunch.com",
    "vice.com", "buzzfeed.com",
    # Fact-check publishers get medium-trust as base (bonus applied on top)
    "politifact.com", "factcheck.org", "snopes.com",
    "fullfact.org", "boomlive.in", "altnews.in", "vishvasnews.com",
}

FACT_CHECK_DOMAINS = {
    # Dedicated fact checking
    "politifact.com": 0.10,
    "factcheck.org": 0.10,
    "snopes.com": 0.10,
    "fullfact.org": 0.10,
    "boomlive.in": 0.08,
    "factchecker.in": 0.08,
    "indiatoday.in": 0.05,   # partial — some fact checks
    "altnews.in": 0.08,
    "vishvasnews.com": 0.07,
}

# Domains whose scores should be heavily penalised
LOW_TRUST_DOMAINS = {
    "naturalnews.com", "infowars.com", "beforeitsnews.com",
    "activistpost.com", "newspunch.com", "worldnewsdailyreport.com",
    "empirenews.net", "theonion.com",   # satire
    "clickhole.com", "babylonbee.com",  # satire
}

# ---------------------------------------------------------------------------
# 2. Domain extraction
# ---------------------------------------------------------------------------

def extract_domain(url: str) -> Optional[str]:
    """
    Extracts the registered domain (e.g. 'bbc.com') from a full URL.
    Falls back to None if the URL is malformed.
    """
    if not url:
        return None
    try:
        parsed = urlparse(url if "://" in url else f"https://{url}")
        host = parsed.netloc or parsed.path
        # Strip 'www.' prefix
        host = re.sub(r"^www\.", "", host.lower())
        # Strip port if present
        host = host.split(":")[0]
        return host if host else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 3. Base trust score lookup
# ---------------------------------------------------------------------------

def _base_trust_score(domain: str) -> float:
    """Returns the baseline trust score for a domain (before bonuses)."""
    if not domain:
        return 0.30

    if domain in HIGH_TRUST_DOMAINS:
        return 0.90
    if domain in MEDIUM_TRUST_DOMAINS:
        return 0.70
    if domain in LOW_TRUST_DOMAINS:
        return 0.10

    # Match any wikipedia subdomain (e.g. en.wikipedia.org)
    if domain.endswith(".wikipedia.org"):
        return 0.70

    # TLD-based heuristics for unknown domains
    if domain.endswith(".gov") or domain.endswith(".edu"):
        return 0.85
    if domain.endswith(".org"):
        return 0.60
    if domain.endswith(".ac.uk") or domain.endswith(".edu.au"):
        return 0.80

    return 0.35


# ---------------------------------------------------------------------------
# 4. Full trust score with signals
# ---------------------------------------------------------------------------

def calculate_source_trust(url: str) -> float:
    """
    Computes a final trust score for a source URL.

    Score components:
      - Base domain tier score
      - +0.05 if HTTPS is used
      - +0.07–0.10 if known fact-check publisher

    Returns:
        float in [0.0, 1.0]
    """
    if not url:
        return 0.30

    domain = extract_domain(url)
    score = _base_trust_score(domain)

    # HTTPS bonus
    if url.startswith("https://"):
        score += 0.05

    # Fact-check publisher bonus
    if domain in FACT_CHECK_DOMAINS:
        score += FACT_CHECK_DOMAINS[domain]

    return round(min(1.0, max(0.0, score)), 4)


# ---------------------------------------------------------------------------
# 5. Score from source name alone (for News API articles that lack full URL)
# ---------------------------------------------------------------------------

_SOURCE_NAME_MAP = {
    "reuters": 0.92, "ap": 0.92, "associated press": 0.92,
    "bbc": 0.90, "bbc news": 0.90,
    "the guardian": 0.88, "guardian": 0.88,
    "nytimes": 0.87, "new york times": 0.87,
    "washington post": 0.86, "wall street journal": 0.85,
    "nasa": 0.95, "who": 0.95, "cdc": 0.93,
    "nature": 0.92, "science daily": 0.80,
    "al jazeera": 0.82, "al jazeera english": 0.82,
    "ndtv": 0.78, "the hindu": 0.82,
    "politifact": 0.88, "snopes": 0.88, "factcheck.org": 0.88,
    "wikipedia": 0.74,
    "cnn": 0.70, "cnbc": 0.70, "bloomberg": 0.72,
    "forbes": 0.68, "business insider": 0.65,
    "gizmodo": 0.66, "wired": 0.72,
    "naturalnews": 0.10, "infowars": 0.05, "activistpost": 0.12,
}

def calculate_source_trust_by_name(source_name: str) -> float:
    """
    Computes a trust score using just the publisher name string
    (used when a URL is unavailable, e.g. NewsAPI source field).
    """
    if not source_name:
        return 0.35
    normalised = source_name.lower().strip()
    return _SOURCE_NAME_MAP.get(normalised, 0.40)
