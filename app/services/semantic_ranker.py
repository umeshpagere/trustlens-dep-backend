from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Load the model globally so it stays warm in memory across requests
try:
    logger.info("Loading sentence-transformers model 'all-MiniLM-L6-v2'...")
    _model = SentenceTransformer("all-MiniLM-L6-v2")
    logger.info("Successfully loaded sentence-transformers model.")
except Exception as e:
    logger.error("Failed to load sentence-transformers model. Semantic ranking will fail. Error: %s", e)
    _model = None

def embed_text(text: str) -> np.ndarray:
    """Convert text into a vector embedding."""
    if not text or not _model:
        # Return a zero vector if there's no text or model
        # 384 is the hidden size for all-MiniLM-L6-v2
        return np.zeros(384)
    return _model.encode(text)

def compute_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate the cosine similarity between two vector embeddings."""
    return float(cosine_similarity([vec1], [vec2])[0][0])

def rank_articles_by_semantic_similarity(claim: str, articles: list) -> list:
    """
    Ranks a list of API article dictionaries by their deep semantic similarity to the core claim.
    Injects a 'semantic_similarity' float between 0.0 and 1.0 into each dictionary.
    """
    if not claim or not articles:
        return articles
        
    if not _model:
        logger.warning("[SemanticRanker] Model not loaded. Skipping semantic ranking.")
        for article in articles:
            article["semantic_similarity"] = 0.5
        return articles

    claim_vector = embed_text(claim)
    ranked_articles = []

    for article in articles:
        title = article.get("title", "")
        desc = article.get("description", "")
        # Embed the combined title and description for higher accuracy
        content_to_embed = f"{title} {desc}".strip()
        
        article_vector = embed_text(content_to_embed)
        similarity = compute_similarity(claim_vector, article_vector)
        
        # Clip to [0, 1] range to avoid floating point anomalies with cosine
        similarity = max(0.0, min(1.0, similarity))
        
        article["semantic_similarity"] = similarity
        ranked_articles.append(article)

    # Sort descending
    ranked_articles.sort(key=lambda x: x["semantic_similarity"], reverse=True)

    # Debug Log the top 5
    logger.info("Semantic similarity ranking results:")
    for a in ranked_articles[:5]:
        logger.info("%s | similarity=%.2f", a.get("title", "")[:60], a["semantic_similarity"])

    return ranked_articles
