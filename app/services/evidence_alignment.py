import logging
import nltk
from nltk.tokenize import sent_tokenize
from typing import List, Dict, Any

from app.services.semantic_ranker import embed_text, compute_similarity

logger = logging.getLogger(__name__)

_punkt_downloaded = False

def _ensure_punkt():
    global _punkt_downloaded
    if not _punkt_downloaded:
        try:
            nltk.data.find('tokenizers/punkt_tab')
            _punkt_downloaded = True
        except LookupError:
            logger.info("Downloading NLTK punkt_tab tokenizer...")
            try:
                nltk.download('punkt_tab', quiet=True)
                _punkt_downloaded = True
            except Exception as e:
                logger.warning(f"Failed to download punkt_tab: {e}")

SIMILARITY_THRESHOLD = 0.65

def split_into_sentences(text: str) -> List[str]:
    """Safely split English text into a list of sentences using NLTK."""
    if not text:
        return []
        
    try:
        _ensure_punkt()
        return sent_tokenize(text)
    except Exception as e:
        logger.warning("NLTK sent_tokenize failed, falling back to basic split: %s", e)
        # Fallback if NLTK totally fails
        return [s.strip() for s in text.replace("\n", ". ").split(".") if len(s.strip()) > 5]


def rank_evidence_sentences(claim: str, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Slices a list of articles into raw sentences, embeds each sentence, and compares
    it to the claim's semantic vector.
    Returns a list of the highly correlated single sentences descending by similarity.
    """
    if not claim or not articles:
        return []

    claim_vec = embed_text(claim)
    ranked_sentences = []

    for article in articles:
        # Prefer 'content' if it exists, otherwise fallback to description or title
        text = article.get("content") or article.get("description") or article.get("title") or ""
        
        sentences = split_into_sentences(text)

        for sentence in sentences:
            sentence = sentence.strip()
            # Skip noise that's too short to be meaningful
            if len(sentence) < 10:
                continue

            sentence_vec = embed_text(sentence)
            similarity = compute_similarity(claim_vec, sentence_vec)

            # Prevent duplication of exact same sentences from different URLs
            if any(s["sentence"] == sentence for s in ranked_sentences):
                continue

            ranked_sentences.append({
                "sentence": sentence,
                "similarity": similarity,
                "source": article.get("source", "Unknown Publisher"),
                "url": article.get("url", "")
            })

    # Filter out weak signals (< 0.65 similarity)
    filtered_sentences = [
        s for s in ranked_sentences
        if s["similarity"] >= SIMILARITY_THRESHOLD
    ]

    # Sort descending by exact similarity match
    filtered_sentences.sort(key=lambda x: x["similarity"], reverse=True)

    # Log the top alignments
    logger.info("Top aligned evidence sentences mapped to claim: '%s'", claim[:50])
    for s in filtered_sentences[:5]:
        logger.info("✓ '%s' | similarity=%.2f", s["sentence"][:80], s["similarity"])

    return filtered_sentences
