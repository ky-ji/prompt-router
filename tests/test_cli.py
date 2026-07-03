import tempfile
import unittest
from pathlib import Path

from prompt_router.cli import build_index_file, match_index_file, result_to_dict


class FakeEmbeddingClient:
    def __init__(self, vectors):
        self.vectors = vectors

    def embed_many(self, texts):
        return [self.vectors[text] for text in texts]


class CliTests(unittest.TestCase):
    def test_build_index_file_and_match_index_file(self):
        summarize = "Summarize the following text into concise bullet points."
        translate = "Translate the following text into English."
        embedder = FakeEmbeddingClient(
            {
                summarize: [1.0, 0.0],
                translate: [0.0, 1.0],
                "帮我总结一下": [0.99, 0.01],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            prompts_path = Path(tmpdir) / "prompts.json"
            index_path = Path(tmpdir) / "index.json"
            prompts_path.write_text(
                f'["{summarize}", "{translate}"]',
                encoding="utf-8",
            )

            build_index_file(prompts_path, index_path, embedder, model="fake-model")
            result = match_index_file(
                index_path,
                "帮我总结一下",
                embedder,
                threshold=0.3,
                margin=0.1,
                top_k=2,
            )

        self.assertEqual(result.prompt, summarize)
        self.assertTrue(result.confident)

    def test_result_to_dict_contains_candidates(self):
        summarize = "Summarize the following text into concise bullet points."
        embedder = FakeEmbeddingClient(
            {
                summarize: [1.0, 0.0],
                "帮我总结一下": [0.99, 0.01],
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "index.json"
            index_path.write_text(
                """
                {
                  "schema_version": 1,
                  "model": "fake-model",
                  "entries": [
                    {
                      "prompt": "Summarize the following text into concise bullet points.",
                      "vector": [1.0, 0.0]
                    }
                  ]
                }
                """,
                encoding="utf-8",
            )
            result = match_index_file(index_path, "帮我总结一下", embedder)

        payload = result_to_dict(result)

        self.assertEqual(payload["prompt"], summarize)
        self.assertIn("candidates", payload)
        self.assertEqual(payload["candidates"][0]["prompt"], summarize)


if __name__ == "__main__":
    unittest.main()
