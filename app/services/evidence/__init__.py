# Evidence retrieval sub-package for Phase D: Retrieval-Augmented Evidence Verification
# Modules:
#   - wikipedia_service.py  : Queries Wikipedia API for topic summaries
#   - news_service.py       : Queries NewsAPI for recent articles
#   - evidence_aggregator.py: Concurrently fetches all 3 sources and aggregates output
#   - evidence_verifier.py  : Sends aggregated evidence to Azure LLM for comparison
