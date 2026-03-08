"""
Tests for Evidence Ranker (Phase E)

Focuses on the new relevance and recency scoring mechanisms,
alongside existing source trust ranking.
"""

import unittest
from datetime import datetime, timezone, timedelta
from app.services.evidence.evidence_ranker import (
    _score_recency,
    rank_evidence_sources,
    filter_evidence_sources,
    rank_and_filter,
)

class TestEvidenceRanker(unittest.TestCase):

    def test_recency_highly_recent(self):
        # 1 hour ago
        dt = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        self.assertEqual(_score_recency(dt), 1.0)

    def test_recency_recent(self):
        # 4 days ago
        dt = (datetime.now(timezone.utc) - timedelta(days=4)).isoformat()
        self.assertEqual(_score_recency(dt), 0.8)

    def test_recency_standard(self):
        # 15 days ago
        dt = (datetime.now(timezone.utc) - timedelta(days=15)).isoformat()
        self.assertEqual(_score_recency(dt), 0.5)

    def test_recency_older(self):
        # 40 days ago
        dt = (datetime.now(timezone.utc) - timedelta(days=40)).isoformat()
        self.assertEqual(_score_recency(dt), 0.2)

    def test_recency_invalid_or_missing(self):
        self.assertEqual(_score_recency(None), 0.5)
        self.assertEqual(_score_recency("not a date"), 0.5)

    def test_rank_evidence_sources_composite(self):
        claim = "NASA launches new rover to Mars"
        
        # High trust, excellent relevance, highly recent
        item_best = {
            "url": "https://www.nasa.gov/rover",
            "title": "NASA launches new rover to Mars today",
            "publishedAt": (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
        }
        
        # High trust, but totally irrelevant and old
        item_irrelevant = {
            "url": "https://www.bbc.com/news",
            "title": "Economy grows by 2%",
            "description": "GDP data released.",
            "publishedAt": (datetime.now(timezone.utc) - timedelta(days=50)).isoformat()
        }
        
        ranked = rank_evidence_sources(claim, [item_irrelevant, item_best])
        self.assertEqual(len(ranked), 2)
        
        # The relevant, recent NASA article should strongly beat the BBC economy article
        self.assertEqual(ranked[0]["domain"], "nasa.gov")
        self.assertGreater(ranked[0]["compositeScore"], ranked[1]["compositeScore"])
        
        # Check computed fields are attached
        self.assertIn("trustScore", ranked[0])
        self.assertIn("semanticScore", ranked[0])
        self.assertIn("recencyScore", ranked[0])

    def test_filter_evidence_sources_drops_old_news(self):
        claim = "Test claim"
        
        item_new = {
            "url": "https://www.reuters.com/a",
            "title": "Test claim happening now",
            "publishedAt": (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
        }
        item_old = {
            "url": "https://www.reuters.com/b",
            "title": "Test claim happened years ago",
            "publishedAt": (datetime.now(timezone.utc) - timedelta(days=40)).isoformat()
        }
        
        ranked = rank_evidence_sources(claim, [item_new, item_old])
        filtered = filter_evidence_sources(ranked, min_trust=0.6)
        
        # 40-day old news should be dropped
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["url"], item_new["url"])

    def test_filter_evidence_sources_keeps_old_fact_checks(self):
        claim = "Test claim about history"
        
        item_old_fc = {
            "url": "https://www.snopes.com/fact-check",
            "title": "Test claim about history",
            "type": "factcheck",
            # No publishedAt or an old one
            "publishedAt": (datetime.now(timezone.utc) - timedelta(days=400)).isoformat()
        }
        
        ranked = rank_evidence_sources(claim, [item_old_fc])
        filtered = filter_evidence_sources(ranked, min_trust=0.6)
        
        # Fact checks bypass the age filter
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["type"], "factcheck")

    def test_filter_evidence_fallback(self):
        # If all sources are too old, we still want to return at least the best one
        claim = "Test"
        item_old = {
            "url": "https://www.bbc.com",
            "title": "Test",
            "publishedAt": (datetime.now(timezone.utc) - timedelta(days=40)).isoformat()
        }
        ranked = rank_evidence_sources(claim, [item_old])
        filtered = filter_evidence_sources(ranked, min_trust=0.6)
        self.assertEqual(len(filtered), 1)

if __name__ == "__main__":
    unittest.main()
