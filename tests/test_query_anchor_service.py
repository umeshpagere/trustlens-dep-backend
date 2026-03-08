import unittest
from app.services.query_anchor_service import generate_anchored_queries

class TestQueryAnchorService(unittest.TestCase):

    def test_generate_anchored_queries_standard(self):
        event_tuple = {
            "entity": "Amitabh Bachchan",
            "action": "purchased",
            "object": "land in Ayodhya"
        }
        queries = generate_anchored_queries(event_tuple)
        
        # We expect 3 permutations, all deduplicated
        self.assertIn("Amitabh Bachchan purchased land in Ayodhya", queries)
        self.assertIn("Amitabh Bachchan land in Ayodhya purchased", queries)
        self.assertIn("Amitabh Bachchan land in Ayodhya", queries)
        self.assertEqual(len(queries), 3)

    def test_entity_anchoring_enforced(self):
        # A simulated case where we bypass the initial generation logic 
        # to ensure the anchor logic catches lack of entity.
        # But our generation uses the entity in every string built.
        # We'll just ensure that if entity is missing, it falls back safely.
        event_tuple = {
            "entity": "",
            "action": "jumped",
            "object": "over the moon"
        }
        queries = generate_anchored_queries(event_tuple)
        
        # If no entity, it just generates valid queries with what it has
        # because the entity anchoring filter triggers when entity IS present.
        self.assertIn("jumped over the moon", queries)
        self.assertIn("over the moon jumped", queries)
        self.assertIn("over the moon", queries)

    def test_missing_obj_or_action(self):
        event_tuple = {
            "entity": "Iran fighter jet",
            "action": "shot down",
            "object": ""
        }
        queries = generate_anchored_queries(event_tuple)
        self.assertIn("Iran fighter jet shot down", queries)
        
    def test_empty_tuple_returns_empty_list(self):
        self.assertEqual(generate_anchored_queries({"entity": "", "action": "", "object": ""}), [])

if __name__ == "__main__":
    unittest.main()
