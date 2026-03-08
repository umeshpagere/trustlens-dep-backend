import re
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

# Keywords strongly indicating a breaking or recent event
TEMPORAL_KEYWORDS = [
    "today",
    "breaking",
    "breaking news",
    "just now",
    "latest",
    "this morning",
    "recently",
    "now",
    "tonight",
    "yesterday"
]


def contains_temporal_keywords(claim: str) -> bool:
    """
    Checks if the claim string contains any temporal keywords indicating
    a recent or breaking event.
    """
    if not claim:
        return False
        
    lower_claim = claim.lower()
    # Use word boundary boundaries to ensure exact word matches (don't match 'snow' for 'now')
    for keyword in TEMPORAL_KEYWORDS:
        # For multi-word phrases, simple subset check is fine
        if " " in keyword:
            if keyword in lower_claim:
                return True
        else:
            # For single words, use regex boundaries
            if re.search(r'\b' + re.escape(keyword) + r'\b', lower_claim):
                return True
                
    return False


def detect_breaking_news(claim: str, fact_check_results: List[Dict[str, Any]]) -> bool:
    """
    Determine whether a claim likely refers to a recent event not yet fact-checked.
    
    A claim is considered breaking if:
    1. It contains temporal indicators.
    2. The fact-check API returned 0 results.
    """
    has_temporal = contains_temporal_keywords(claim)
    
    if has_temporal and len(fact_check_results) == 0:
        logger.info(f"🚨 [BreakingNewsDetector] Breaking news detected for claim: '{claim[:50]}...'")
        return True
        
    return False
