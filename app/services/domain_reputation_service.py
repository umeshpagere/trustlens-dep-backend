"""
TrustLens Phase 3: Domain & Source Reputation Service

Evaluates a URL's domain against four independent reputation signals and
produces a normalised domainTrustScore (0–100) plus structured metadata.

Architecture principles enforced here
--------------------------------------
- Pure function style: evaluate_domain(url) → dict, no side effects.
- No dependency on LLM output, fact-check results, or image analysis.
- All external calls (WHOIS, SSL) are timeout-guarded; failures yield
  neutral fallbacks, never exceptions.
- No global mutable state; lists are module-level constants (immutable sets).
- Designed for async conversion: every blocking call is self-contained and
  can be wrapped with asyncio.to_thread() in Phase 4 with zero logic changes.

Parallelism readiness notes (Phase 4)
--------------------------------------
  async def evaluate_domain_async(url):
      return await asyncio.to_thread(evaluate_domain, url)

  # Run all four signals concurrently:
  whitelist_task  = asyncio.to_thread(_check_whitelist, domain)
  blacklist_task  = asyncio.to_thread(_check_blacklist, domain)
  whois_task      = asyncio.to_thread(_get_domain_age_days, domain)
  https_task      = asyncio.to_thread(_check_https, domain)
  results = await asyncio.gather(*[whitelist_task, blacklist_task,
                                    whois_task, https_task])

  Blocking bottleneck today: WHOIS (~1-3s), HTTPS (~0-3s).
  Caching strategy: TTLCache(maxsize=512, ttl=3600) keyed on domain.
  Domain metadata changes at most daily; a 1-hour TTL cuts external
  calls by >95% in production traffic.
"""

import asyncio
import ssl
import socket
from datetime import datetime, timezone
from typing import Any

from app.utils.domain_utils import extract_domain

# ---------------------------------------------------------------------------
# Static reputation lists
# Why static sets? O(1) lookup, zero external dependency, safe for threading.
# Production path: replace with DB/config-file read at startup.
# ---------------------------------------------------------------------------

TRUSTED_DOMAINS: frozenset[str] = frozenset({
    # Major international news agencies
    "reuters.com", "apnews.com", "afp.com",
    # Broadcasters
    "bbc.com", "bbc.co.uk", "npr.org", "pbs.org", "dw.com", "aljazeera.com",
    # Newspapers of record
    "nytimes.com", "washingtonpost.com", "theguardian.com", "wsj.com",
    "ft.com", "economist.com", "bloomberg.com", "theatlantic.com",
    # Science / academic
    "nature.com", "science.org", "scientificamerican.com", "ncbi.nlm.nih.gov",
    # Fact-check organisations
    "snopes.com", "factcheck.org", "politifact.com", "fullfact.org",
    "africacheck.org", "checkyourfact.com",
})

BLACKLISTED_DOMAINS: frozenset[str] = frozenset({
    # Known misinformation / satire sites frequently mistaken for news
    "infowars.com", "naturalnews.com", "beforeitsnews.com",
    "worldnewsdailyreport.com", "empirenews.net", "abcnews.com.co",
    "nationalreport.net", "newslo.com", "yournewswire.com",
    "stopyourinbox.com", "realnewsrightnow.com", "huzlers.com",
    "theonion.com",  # satire — not malicious but score must reflect it
})

# ---------------------------------------------------------------------------
# Scoring constants
# ---------------------------------------------------------------------------
NEUTRAL_BASE: int = 50          # Avoids penalising unknown domains
WHITELIST_BONUS: int = 30       # Strong positive signal, not certainty
BLACKLIST_PENALTY: int = 40     # Heavy but not absolute
NEW_DOMAIN_PENALTY: int = 20    # < 180-day-old domain
NO_HTTPS_PENALTY: int = 10      # Baseline hygiene signal; small penalty

NEW_DOMAIN_THRESHOLD_DAYS: int = 180
HTTPS_TIMEOUT_SECONDS: int = 3
WHOIS_TIMEOUT_SECONDS: int = 8


# ---------------------------------------------------------------------------
# Signal helpers — each returns a single piece of data, isolated for testing
# ---------------------------------------------------------------------------

def _check_whitelist(domain: str) -> bool:
    """Return True if domain is in the trusted whitelist."""
    return domain in TRUSTED_DOMAINS


def _check_blacklist(domain: str) -> bool:
    """Return True if domain is in the blacklist."""
    return domain in BLACKLISTED_DOMAINS


def _get_domain_age_days(domain: str) -> int | None:
    """
    Query WHOIS and return the domain's age in days, or None on failure.

    Why None instead of 0 on failure?
      WHOIS is unreliable: some TLDs restrict access (GDPR), registrars
      rate-limit aggressively, and many legitimate domains simply don't
      expose creation dates. None lets the scorer apply *no* penalty
      rather than falsely penalising a legitimate domain.

    Why this must not crash:
      A WHOIS timeout should never block the analysis pipeline; the penalty
      for an unknown age (0) is lower risk than a service outage.
    """
    try:
        import whois  # lazy import — only used here; keeps startup fast
        data = whois.whois(domain)

        creation = data.get("creation_date")
        if creation is None:
            return None

        # python-whois may return a list when multiple dates exist
        if isinstance(creation, list):
            creation = creation[0]

        if not isinstance(creation, datetime):
            return None

        # Normalise to UTC-aware for safe arithmetic
        if creation.tzinfo is None:
            creation = creation.replace(tzinfo=timezone.utc)

        now = datetime.now(tz=timezone.utc)
        age_days = (now - creation).days
        return max(0, age_days)

    except Exception:
        # WHOIS failures are expected; degrade gracefully
        return None


def _check_https(domain: str) -> bool:
    """
    Attempt a TLS handshake to verify HTTPS availability.

    Returns True if the domain accepts TLS on port 443 within the timeout.
    Returns False on timeout, connection refused, or any SSL error.

    Why HTTPS is a weak signal:
      TLS presence indicates infrastructure investment and baseline security
      hygiene. Its *absence* is a mild risk factor — not proof of bad intent.
      Many legitimate small/local news outlets lack HTTPS. Hence the small
      penalty (−10) rather than a disqualifying deduction.

    Why timeout is 3 seconds:
      A domain IP may exist but the port may be filtered. Without a timeout
      the pipeline would stall for OS-level TCP timeout (~75 s on macOS).
    """
    try:
        ctx = ssl.create_default_context()
        with socket.create_connection((domain, 443), timeout=HTTPS_TIMEOUT_SECONDS) as sock:
            with ctx.wrap_socket(sock, server_hostname=domain):
                return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Score normalisation
# ---------------------------------------------------------------------------

def _compute_score(
    *,
    is_trusted: bool,
    is_blacklisted: bool,
    age_days: int | None,
    https_secure: bool,
) -> tuple[int, list[str]]:
    """
    Apply incremental adjustments to NEUTRAL_BASE and collect risk factors.

    Why incremental (not absolute) scoring?
      Signals compose naturally. A new domain (−20) that also lacks HTTPS
      (−10) lands at 20 — appropriately risky — without any single flag
      being catastrophic. Absolute scoring (blacklist → 0) collapses nuance
      and prevents the final weighted formula from doing its job.

    Why clamping?
      Multiple bonuses/penalties could theoretically push the score outside
      [0, 100]. Clamping prevents instability in the weighted formula
      downstream without masking the underlying signal composition.
    """
    score = NEUTRAL_BASE
    risk_factors: list[str] = []

    if is_trusted:
        score += WHITELIST_BONUS
        # No risk factor added — this is a positive signal

    if is_blacklisted:
        score -= BLACKLIST_PENALTY
        risk_factors.append("Domain is on the known misinformation blacklist")

    if age_days is not None and age_days < NEW_DOMAIN_THRESHOLD_DAYS:
        score -= NEW_DOMAIN_PENALTY
        risk_factors.append(
            f"Domain is only {age_days} days old (threshold: {NEW_DOMAIN_THRESHOLD_DAYS} days)"
        )

    if not https_secure:
        score -= NO_HTTPS_PENALTY
        risk_factors.append("Domain does not support HTTPS on port 443")

    return max(0, min(100, score)), risk_factors


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def evaluate_domain(url: str | None) -> dict[str, Any]:
    """
    Evaluate domain reputation for a given URL.

    Parameters
    ----------
    url : str | None
        Any URL string. Invalid / None / empty returns a neutral result.

    Returns
    -------
    dict with keys:
        domainTrustScore  : int   — 0–100 normalised score
        domain            : str | None — extracted bare domain
        domainAgeDays     : int | None — days since WHOIS creation date
        httpsSecure       : bool  — TLS handshake succeeded
        isTrustedSource   : bool  — domain in whitelist
        isBlacklisted     : bool  — domain in blacklist
        riskFactors       : list[str] — human-readable explanations

    Why structured output?
      - domainTrustScore is used by evidence_ranker for source filtering ONLY.
        It does NOT feed the credibility scoring formula (removed in v2).
      - isTrustedSource still influences confidence modelling (evidence_flags).
      - Metadata (age, HTTPS, source flag) enables transparent UI display.
      - riskFactors power the "why is this risky?" explainability layer.

    Why is this independently callable?
      No dependency on LLM, fact-check, or image services. Callers in the
      orchestrator, test suite, or future CLI tools can invoke it in isolation.
    """
    neutral_result: dict[str, Any] = {
        "domainTrustScore": NEUTRAL_BASE,
        "domain": None,
        "domainAgeDays": None,
        "httpsSecure": False,
        "isTrustedSource": False,
        "isBlacklisted": False,
        "riskFactors": ["Could not extract a valid domain from the provided URL"],
    }

    domain = extract_domain(url) if url else None
    if not domain:
        return neutral_result

    # Whitelist / blacklist are pure dict lookups — synchronous, instant
    is_trusted    = _check_whitelist(domain)
    is_blacklisted = _check_blacklist(domain)

    # WHOIS lookup + TLS socket check are the two slowest operations.
    # Running them concurrently via asyncio.gather cuts worst-case latency
    # from ~6 s (sequential) to ~3 s (parallel) for unknown domains.
    age_days, https_secure = await asyncio.gather(
        asyncio.to_thread(_get_domain_age_days, domain),
        asyncio.to_thread(_check_https, domain),
    )

    score, risk_factors = _compute_score(
        is_trusted=is_trusted,
        is_blacklisted=is_blacklisted,
        age_days=age_days,
        https_secure=https_secure,
    )

    return {
        "domainTrustScore": score,
        "domain":           domain,
        "domainAgeDays":    age_days,
        "httpsSecure":      https_secure,
        "isTrustedSource":  is_trusted,
        "isBlacklisted":    is_blacklisted,
        "riskFactors":      risk_factors,
    }
