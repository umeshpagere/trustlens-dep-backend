import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

def generate_anchored_queries(event_tuple: Dict[str, str]) -> List[str]:
    """
    Generate structured search queries from a core event tuple, ensuring that
    the query is anchored to the primary entity to prevent query drift.
    """
    entity = event_tuple.get("entity", "").strip()
    action = event_tuple.get("action", "").strip()
    obj    = event_tuple.get("object", "").strip()

    # Fallback if the tuple is extremely malformed
    if not entity and not action and not obj:
        logger.warning("[QueryAnchor] Event tuple entirely empty. Returning empty list.")
        return []

    # Permutations based on the event structure
    raw_queries = []
    
    # Standard full clause
    full_clause = " ".join(filter(None, [entity, action, obj]))
    if full_clause:
        raw_queries.append(full_clause)

    # Noun-heavy, verb-last (often good for news search engines)
    noun_heavy = " ".join(filter(None, [entity, obj, action]))
    if noun_heavy:
        raw_queries.append(noun_heavy)

    # Simplified entity-object
    entity_obj = " ".join(filter(None, [entity, obj]))
    if entity_obj:
        raw_queries.append(entity_obj)
        
    # Validation/Filtering Logic
    valid_queries = []
    lower_entity = entity.lower()
    
    for query in raw_queries:
        # Enforce Entity Anchoring Rule: 
        # The entity MUST be present in the query.
        # This prevents drift where we search only for the object (e.g. searching "Ayodhya land" 
        # instead of "Amitabh Bachchan Ayodhya land")
        if entity and lower_entity not in query.lower():
            logger.debug(f"[QueryAnchor] Discarded unanchored query: '{query}'")
            continue
            
        # Basic cleanup: remove double spaces
        clean_query = " ".join(query.split())
        if clean_query:
            valid_queries.append(clean_query)

    # Deduplicate queries while preserving an orderly format
    # Using dict.fromkeys to maintain insertion order while creating a unique set
    deduplicated_queries = list(dict.fromkeys(valid_queries))
    
    # If the validation somehow stripped all queries (e.g. no entity was extracted),
    # emit a basic join as a last resort
    if not deduplicated_queries:
        logger.warning("[QueryAnchor] Validation stripped all queries. Falling back to simple join.")
        fallback = " ".join(filter(None, [entity, obj]))
        if fallback:
            deduplicated_queries = [fallback]

    if deduplicated_queries:
        logger.info(
            "Anchored queries generated:\n%s",
            "\n".join([f"{i+1}. {q}" for i, q in enumerate(deduplicated_queries)])
        )

    return deduplicated_queries
