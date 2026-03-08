"""
TrustLens Query Generator

Fallback service for generating search queries from a primary claim
when claim decomposition fails.
"""

def generate_queries(claim: str) -> list[str]:
    """
    Generate 1-3 search queries from a raw claim string.
    
    Rules:
    - Primary query is the first 5 words (best for event-level search)
    - Full claim is included as secondary
    - Replace 'purchased' with 'bought' if present
    """
    if not claim or not claim.strip():
        return []
        
    claim = claim.strip()
    words = claim.split()
    
    if len(words) == 0:
        return []
        
    queries = [
        " ".join(words[:5]),
        claim
    ]
    
    if "purchased" in claim.lower():
        queries.append(claim.replace("purchased", "bought").replace("Purchased", "Bought"))
        
    # Deduplicate while preserving order
    seen = set()
    unique_queries = []
    for q in queries:
        if q and q not in seen:
            seen.add(q)
            unique_queries.append(q)
            
    return unique_queries[:3]
