import unittest
from app.services.semantic_ranker import rank_articles_by_semantic_similarity

class TestSemanticRanker(unittest.TestCase):

    def test_semantic_synonym_matching(self):
        # Even though "purchased" and "invests in" don't share keyword stems perfectly
        # the semantic model should rank them very high
        claim = "Amitabh Bachchan purchased land in Ayodhya"
        articles = [
            {"title": "Bollywood actor invests in Ayodhya property", "description": ""},
            {"title": "Amitabh Bachchan buys land in Ayodhya", "description": ""},
            {"title": "Ayodhya real estate project attracts investors", "description": ""},
            {"title": "Weather forecast for Ayodhya today", "description": "Sunny"}
        ]
        
        ranked = rank_articles_by_semantic_similarity(claim, articles)
        
        self.assertEqual(len(ranked), 4)
        
        # We expect "buys land" or "invests" to be top 2, and weather to be dead last.
        top_two_titles = [a["title"] for a in ranked[:2]]
        self.assertIn("Amitabh Bachchan buys land in Ayodhya", top_two_titles)
        self.assertIn("Bollywood actor invests in Ayodhya property", top_two_titles)
        
        # Verify scores are attached and scaled properly
        self.assertGreater(ranked[0]["semantic_similarity"], 0.70)
        self.assertLess(ranked[-1]["semantic_similarity"], 0.60) # Weather should have low similarity

    def test_military_synonyms(self):
        claim = "Iran fighter jet shot down"
        articles = [
            {"title": "Stock market surges in Tehran"},
            {"title": "Iranian aircraft downed in conflict"},
            {"title": "Iran military plane destroyed"}
        ]
        
        ranked = rank_articles_by_semantic_similarity(claim, articles)
        
        self.assertEqual(len(ranked), 3)
        self.assertEqual(ranked[-1]["title"], "Stock market surges in Tehran")
        self.assertGreater(ranked[0]["semantic_similarity"], 0.65)
        
    def test_empty_arguments(self):
        # Empty claim
        self.assertEqual(len(rank_articles_by_semantic_similarity("", [{"title": "A"}])), 1)
        # Empty articles
        self.assertEqual(len(rank_articles_by_semantic_similarity("Claim", [])), 0)


if __name__ == '__main__':
    unittest.main()
