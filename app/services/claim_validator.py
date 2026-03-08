"""
TrustLens Claim Validator

Pure CPU module — no I/O, fully synchronous.
Provides two public helpers used by the text and video analysis pipelines:

  is_valid_claim(claim)   -> bool
      Returns True only if the claim contains a clear subject + action/event
      and does not match known vague-statement patterns.

  normalize_claim(claim)  -> str
      Strips hashtags, BREAKING-style caps, emotional intensifiers,
      and trailing punctuation to improve fact-check API match rates.

Why synchronous?
  Both functions are pure string transforms / regex checks — no I/O of any
  kind. Making them async would add coroutine overhead with zero benefit.
"""

import re
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Vague-phrase patterns — claims matching any of these are rejected.
# All compared against claim.lower().
# ---------------------------------------------------------------------------
VAGUE_PATTERNS: list[str] = [
    r"people are suffering",
    r"things are getting worse",
    r"something (big |bad |terrible |major )?is happening",
    r"the situation is (bad|terrible|getting worse|deteriorating|dire)",
    r"this is (a )?(terrible|awful|horrible|shocking|sad|bad|disgraceful|outrageous|unbelievable) (situation|moment|time|crisis|disaster|catastrophe|shame|lie|scam|hoax)",
    r"this (is|was) (a )?(disaster|catastrophe|shame|lie|scam|hoax)",
    r"it('s| is) getting (worse|bad|out of control)",
    r"everyone (is|knows|can see)",
    r"we need to (talk|do something|act|wake up)",
    r"nothing will change",
    r"they (are|won't) (doing|do) (nothing|anything)",
    r"wake up",
    r"open your eyes",
    r"spread the word",
    r"you won't believe",
    r"this changes everything",
    r"nothing to see here",
    r"things will never be the same",
    r"^(this|it|that) is (a )?(terrible|awful|horrible|shocking|sad|bad|outrageous|disgusting)",
]

_COMPILED_VAGUE = [re.compile(p, re.IGNORECASE) for p in VAGUE_PATTERNS]

# ---------------------------------------------------------------------------
# Action verbs — one must be present for a claim to be valid.
# Covers common factual-reporting verbs.
# ---------------------------------------------------------------------------
ACTION_VERBS: frozenset[str] = frozenset({
    # Official/institutional actions
    "announced", "declared", "confirmed", "reported", "stated", "said",
    "admitted", "denied", "revealed", "warned", "claimed", "alleged",
    "accused", "charged", "arrested", "sentenced", "convicted", "acquitted",
    # Government / policy
    "banned", "approved", "signed", "passed", "enacted", "repealed",
    "blocked", "vetoed", "ordered", "imposed", "lifted", "implemented",
    "launched", "canceled", "delayed", "postponed", "suspended",
    # Events / actions
    "killed", "died", "injured", "attacked", "bombed", "invaded",
    "elected", "appointed", "resigned", "fired", "hired", "promoted",
    "won", "lost", "defeated", "surrendered", "withdrew", "deployed",
    # Discovery / science
    "discovered", "found", "detected", "identified", "developed", "created",
    "invented", "released", "published", "concluded", "showed", "proved",
    "demonstrated", "measured", "recorded",
    # Economy / finance
    "raised", "cut", "increased", "decreased", "collapsed", "surged",
    "fell", "rose", "reached", "exceeded", "broke",
    # Simple copula with factual context (e.g. "X is 50%", "X was convicted")
    "is", "was", "were", "has", "have", "had",
})

# Verb regex: match any action verb as a whole word
_VERB_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(v) for v in ACTION_VERBS) + r")\b",
    re.IGNORECASE,
)

# Named entity / subject signals — any capitalised word (≥2 chars) or number
_SUBJECT_PATTERN = re.compile(r"\b[A-Z][a-zA-Z]{1,}\b|\b\d{4}\b")

# Emotional intensifiers to strip during normalisation
_INTENSIFIERS: list[str] = [
    "completely", "absolutely", "totally", "utterly", "extremely",
    "incredibly", "unbelievably", "definitely", "certainly", "obviously",
    "clearly", "literally", "actually",
]
_INTENSIFIER_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in _INTENSIFIERS) + r")\b",
    re.IGNORECASE,
)

# BREAKING-style all-caps emphasis words at the start of a claim
_CAPS_LEAD = re.compile(r"^[A-Z][A-Z]+:\s*")

# Hashtags
_HASHTAG = re.compile(r"#\S+")

# Trailing/leading punctuation we want to strip
_EDGE_PUNCT = re.compile(r"^[!?.,'\";\-–—\s]+|[!?.,'\";\-–—\s]+$")

# Collapse multiple spaces
_MULTI_SPACE = re.compile(r" {2,}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def is_valid_claim(claim: str) -> bool:
    """
    Return True if *claim* is a clear, specific, verifiable factual claim.

    A valid claim must satisfy ALL of the following:
      1. At least 3 words
      2. Contains an action verb from the known verb set
      3. Does not match any vague-phrase pattern
      4. Contains at least one named entity or capitalized subject word
    """
    if not claim or not isinstance(claim, str):
        logger.info("Rejected claim (empty or non-string): %r", claim)
        return False

    words = claim.split()
    if len(words) < 3:
        logger.info("Rejected claim (too short, %d words): %r", len(words), claim)
        return False

    # Rule 3: vague pattern check
    lower = claim.lower()
    for pattern in _COMPILED_VAGUE:
        if pattern.search(lower):
            logger.info("Rejected vague claim (matched pattern %r): %r", pattern.pattern, claim)
            return False

    # Rule 2: must have an action verb
    if not _VERB_PATTERN.search(claim):
        logger.info("Rejected claim (no action verb): %r", claim)
        return False

    # Rule 4: at least one subject / named entity
    if not _SUBJECT_PATTERN.search(claim):
        logger.info("Rejected claim (no named subject): %r", claim)
        return False

    return True


def normalize_claim(claim: str) -> str:
    """
    Normalise *claim* for better fact-check API matching.

    Steps (in order):
      1. Strip BREAKING / ALERT -style all-caps lead words
      2. Remove hashtags
      3. Remove emotional intensifiers
      4. Collapse extra whitespace
      5. Strip leading/trailing punctuation
    """
    if not claim or not isinstance(claim, str):
        return ""

    result = claim.strip()
    result = _CAPS_LEAD.sub("", result)           # "BREAKING: ..." → "..."
    result = _HASHTAG.sub("", result)             # remove #tags
    result = _INTENSIFIER_PATTERN.sub("", result) # remove intensifiers
    result = result.replace("!", "").replace("...", "").replace("…", "")
    result = _MULTI_SPACE.sub(" ", result)        # collapse spaces
    result = _EDGE_PUNCT.sub("", result)          # trim edge punctuation
    result = result.strip()

    logger.debug("Normalized claim: %r -> %r", claim, result)
    return result


def filter_and_normalize_claims(raw_claims: list) -> list[str]:
    """
    Convenience helper: apply is_valid_claim then normalize_claim to a list.

    Returns a list of normalized, valid claims (max 5).
    Logs each rejection for prompt-tuning diagnostics.
    """
    validated: list[str] = []
    for raw in (raw_claims or []):
        claim = str(raw).strip()
        if is_valid_claim(claim):
            validated.append(normalize_claim(claim))
        if len(validated) == 5:
            break
    return validated
