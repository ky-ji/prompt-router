import json
import unittest

from prompt_router.embedding import OpenAIEmbeddingClient


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


class EmbeddingTests(unittest.TestCase):
    def test_openai_embedding_client_requires_api_key(self):
        with self.assertRaises(ValueError):
            OpenAIEmbeddingClient(api_key="")

    def test_openai_embedding_client_posts_embedding_request(self):
        captured = {}

        def opener(request, timeout):
            captured["timeout"] = timeout
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            captured["authorization"] = request.get_header("Authorization")
            return FakeResponse(
                {
                    "data": [
                        {"index": 0, "embedding": [1.0, 0.0]},
                        {"index": 1, "embedding": [0.0, 1.0]},
                    ]
                }
            )

        client = OpenAIEmbeddingClient(
            api_key="test-key",
            model="text-embedding-3-small",
            dimensions=256,
            timeout=7,
            opener=opener,
        )

        vectors = client.embed_many(["hello", "world"])

        self.assertEqual(vectors, [[1.0, 0.0], [0.0, 1.0]])
        self.assertEqual(captured["authorization"], "Bearer test-key")
        self.assertEqual(captured["timeout"], 7)
        self.assertEqual(
            captured["payload"],
            {
                "model": "text-embedding-3-small",
                "input": ["hello", "world"],
                "encoding_format": "float",
                "dimensions": 256,
            },
        )


if __name__ == "__main__":
    unittest.main()
