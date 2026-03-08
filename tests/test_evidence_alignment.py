import unittest
from app.services.evidence_alignment import split_into_sentences, rank_evidence_sentences

class TestEvidenceAlignment(unittest.TestCase):

    def test_split_into_sentences_nltk(self):
        text = "Hello there! My name is Dr. Watson. I live in the U.K. today."
        sentences = split_into_sentences(text)
        
        # Ensures that Dr. and U.K. don't falsely split the sentences.
        self.assertEqual(len(sentences), 3)
        self.assertEqual(sentences[0], "Hello there!")
        self.assertEqual(sentences[1], "My name is Dr. Watson.")
        self.assertEqual(sentences[2], "I live in the U.K. today.")

    def test_rank_evidence_sentences_extracts_highest_match(self):
        claim = "Amitabh Bachchan purchased land in Ayodhya"
        
        # A mock article with a lot of noise but one specific matching sentence
        article = {
            "source": "News Network",
            "url": "http://example.com/news",
            "content": (
                "The real estate market in India is booming right now. "
                "Many celebrities are looking outside of Mumbai for properties. "
                "For example, Bollywood actor Amitabh Bachchan invested in a housing project in Ayodhya yesterday. "
                "The project spans approximately 2.7 acres and is being developed by the House of Abhinandan Lodha. "
                "Local residents are very excited about the development."
            )
        }
        
        aligned = rank_evidence_sentences(claim, [article])
        
        # We expect it to drop the noise and only keep the highly relevant sentences (>= 0.65 threshold)
        self.assertTrue(len(aligned) >= 1)
        top_sentence = aligned[0]["sentence"]
        
        # The specific investment sentence should be ranked #1
        self.assertIn("Bollywood actor Amitabh Bachchan invested in a housing project in Ayodhya", top_sentence)
        self.assertGreaterEqual(aligned[0]["similarity"], 0.65)

    def test_rank_evidence_drops_irrelevant_articles(self):
        claim = "Aliens landed in Iran today"
        
        article = {
            "source": "Boring News",
            "content": "The weather in Tehran is remarkably sunny today. Residents are enjoying the beautiful skies."
        }
        
        aligned = rank_evidence_sentences(claim, [article])
        
        # Should be entirely stripped by the 0.65 threshold
        self.assertEqual(len(aligned), 0)


if __name__ == "__main__":
    unittest.main()
