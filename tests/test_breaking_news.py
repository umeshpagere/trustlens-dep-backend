"""
Tests for Breaking News Verification Layer
"""

import unittest
from app.services.breaking_news_detector import (
    contains_temporal_keywords,
    detect_breaking_news
)
from app.services.breaking_news_service import (
    calculate_source_agreement,
    compute_breaking_news_confidence,
    filter_recent_articles
)
from datetime import datetime, timezone, timedelta

class TestBreakingNewsDetector(unittest.TestCase):

    def test_contains_temporal_keywords_true(self):
        claims = [
            "A fire broke out in London today",
            "Breaking: Earthquake hits Japan",
            "The stock market crashed this morning",
            "Just now, the prime minister resigned"
        ]
        for c in claims:
            self.assertTrue(contains_temporal_keywords(c), f"Failed on '{c}'")

    def test_contains_temporal_keywords_false(self):
        claims = [
            "The Earth orbits the sun",
            "World War 2 ended in 1945",
            "Water boils at 100 degrees",
            "A snowy day in New York" # 'now' is inside 'snowy' -> should not match boundary
        ]
        for c in claims:
            self.assertFalse(contains_temporal_keywords(c), f"Failed on '{c}'")

    def test_detect_breaking_news(self):
        # Has temporal + no fact checks -> True
        self.assertTrue(detect_breaking_news("Fire in London today", []))
        
        # Has temporal + has fact checks -> False (already checked)
        self.assertFalse(detect_breaking_news("Fire in London today", [{"text": "Fact"}]))
        
        # No temporal + no fact checks -> False (just an obscure claim, not breaking)
        self.assertFalse(detect_breaking_news("Water is wet", []))


class TestBreakingNewsService(unittest.TestCase):

    def test_calculate_source_agreement(self):
        claim = "US fighter jet shot down Iranian aircraft"
        articles = [
            # Strong match
            {
                "title": "US fighter jet shoots down Iranian aircraft over Gulf",
                "source": "Reuters",
                "description": "..."
            },
            # Strong match different publisher
            {
                "title": "Iranian drone shot down by US forces",
                "source": "BBC News",
                "description": "A US fighter jet..."
            },
            # Irrelevant
            {
                "title": "Stock market update",
                "source": "CNN",
                "description": "..."
            }
        ]
        
        result = calculate_source_agreement(articles, claim)
        self.assertEqual(result["supporting_sources"], 2)
        self.assertCountEqual(result["supporting_publishers"], ["Reuters", "BBC News"])

    def test_compute_breaking_news_confidence(self):
        # 3 or more supporters -> 90
        self.assertEqual(compute_breaking_news_confidence({"supporting_sources": 4}), 90)
        self.assertEqual(compute_breaking_news_confidence({"supporting_sources": 3}), 90)
        
        # 2 supporters -> 70
        self.assertEqual(compute_breaking_news_confidence({"supporting_sources": 2}), 70)
        
        # 1 supporter -> 50
        self.assertEqual(compute_breaking_news_confidence({"supporting_sources": 1}), 50)
        
        # 0 supporters -> 20
        self.assertEqual(compute_breaking_news_confidence({"supporting_sources": 0}), 20)

    def test_filter_recent_articles(self):
        now = datetime.now(timezone.utc)
        
        articles = [
            {"title": "Now", "publishedAt": now.isoformat()},
            {"title": "1 day old", "publishedAt": (now - timedelta(days=1)).isoformat()},
            {"title": "20 days old", "publishedAt": (now - timedelta(days=20)).isoformat()},
            {"title": "40 days old", "publishedAt": (now - timedelta(days=40)).isoformat()},
            {"title": "No date", "publishedAt": None}
        ]
        
        filtered = filter_recent_articles(articles)
        
        # Should keep Now, 1 day, and 20 day. Drop 40 day and No date.
        self.assertEqual(len(filtered), 3)
        
        # Should be sorted freshest first
        self.assertEqual(filtered[0]["title"], "Now")
        self.assertEqual(filtered[1]["title"], "1 day old")
        self.assertEqual(filtered[2]["title"], "20 days old")


if __name__ == "__main__":
    unittest.main()
