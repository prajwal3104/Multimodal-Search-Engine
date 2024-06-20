import unittest
import requests
from src.utils.config import Config

class TestAPI(unittest.TestCase):
    def test_search_endpoint(self):
        """Tests the /search API endpoint."""

        config = Config()
        url = f"http://localhost:5000/search"
        data = {"queries": ["a cat sitting on a couch", "a dog playing in the park"]}
        response = requests.post(url, json=data)

        self.assertEqual(response.status_code, 200)
        self.assertIn("results", response.json())

if __name__ == '__main__':
    unittest.main()