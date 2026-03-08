import unittest
from unittest.mock import patch, MagicMock
from app.services.event_tuple_extractor import extract_event_tuple
import asyncio
import json

class TestEventTupleExtractor(unittest.TestCase):

    def setUp(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def tearDown(self):
        self.loop.close()

    @patch('app.services.event_tuple_extractor.get_azure_client')
    @patch('app.services.event_tuple_extractor._sync_extract_event_tuple')
    def test_extract_event_tuple_success(self, mock_sync, mock_client):
        # Setup mock LLM response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "entity": "Amitabh Bachchan",
            "action": "purchased",
            "object": "land in Ayodhya"
        })
        mock_sync.return_value = mock_response

        # Execute
        result = self.loop.run_until_complete(extract_event_tuple("Amitabh Bachchan purchased land in Ayodhya"))
        
        # Verify
        self.assertEqual(result["entity"], "Amitabh Bachchan")
        self.assertEqual(result["action"], "purchased")
        self.assertEqual(result["object"], "land in Ayodhya")
        mock_sync.assert_called_once()

    @patch('app.services.event_tuple_extractor.get_azure_client')
    def test_extract_event_tuple_fallback_on_llm_failure(self, mock_client):
        # Force an exception when creating the client or calling the API
        mock_client.side_effect = Exception("API down")
        
        # In fallback mode, "A B C D E" split is: entity="A B", action="C", object="D E"
        claim="Iran fighter jet shot down"
        result = self.loop.run_until_complete(extract_event_tuple(claim))
        
        self.assertEqual(result["entity"], "Iran fighter")
        self.assertEqual(result["action"], "jet")
        self.assertEqual(result["object"], "shot down")

    def test_empty_claim(self):
        result = self.loop.run_until_complete(extract_event_tuple(""))
        self.assertEqual(result, {"entity": "", "action": "", "object": ""})
        
        result_whitespace = self.loop.run_until_complete(extract_event_tuple("   "))
        self.assertEqual(result_whitespace, {"entity": "", "action": "", "object": ""})


if __name__ == "__main__":
    unittest.main()
