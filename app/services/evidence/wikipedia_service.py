import logging
import wikipedia
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Suppress wikipedia library's verbose logger
logging.getLogger("wikipedia").setLevel(logging.ERROR)

# Set a descriptive user agent as required by Wikipedia's policy
wikipedia.set_user_agent("TrustLensBot/1.0 (trustlens misinformation detector)")


def search_wikipedia(query: str) -> Optional[Dict[str, Any]]:
    """
    Searches Wikipedia for the most relevant article matching the query.
    Uses the `wikipedia` Python library which internally uses a different
    API endpoint than the REST v1 API, avoiding rate limiting issues.
    
    Args:
        query: A short, keyword-focused search string.
        
    Returns:
        Dictionary with title, summary, and url, or None if no match found.
    """
    if not query:
        return None

    try:
        # Step 1: Search for matching page titles
        search_results = wikipedia.search(query, results=3)
        if not search_results:
            print(f"💡 [WikiService] No Wikipedia results for: '{query}'")
            return None

        # Step 2: Try to get a summary for the first matching result
        # DisambiguationError means multiple pages match — try the next result
        for page_title in search_results:
            try:
                page = wikipedia.page(page_title, auto_suggest=False)
                summary = wikipedia.summary(page_title, sentences=5, auto_suggest=False)
                
                print(f"✅ [WikiService] Found: '{page.title}'")
                return {
                    "title": page.title,
                    "summary": summary,
                    "url": page.url
                }
                
            except wikipedia.exceptions.DisambiguationError:
                # Try the next result
                continue
            except wikipedia.exceptions.PageError:
                # This title wasn't a real article, try next
                continue

        return None

    except Exception as e:
        print(f"❌ [WikiService] Wikipedia search failed for '{query[:40]}': {e}")
        return None
