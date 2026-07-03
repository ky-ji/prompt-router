import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from prompt_router.cli import build_index_file, main, match_index_file, result_to_dict
from prompt_router.index import PromptIndex


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

    def test_route_command_prints_best_prompt_and_builds_index(self):
        summarize = "Summarize the following text into concise bullet points."
        embedder = FakeEmbeddingClient(
            {
                summarize: [1.0, 0.0],
                "帮我总结一下": [0.99, 0.01],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            prompts_path = Path(tmpdir) / "prompts.json"
            index_path = Path(tmpdir) / "index.json"
            prompts_path.write_text(f'["{summarize}"]', encoding="utf-8")
            output = StringIO()

            with patch("prompt_router.cli.create_embedding_client", return_value=embedder):
                with redirect_stdout(output):
                    code = main(
                        [
                            "route",
                            str(prompts_path),
                            "帮我总结一下",
                            "--index",
                            str(index_path),
                        ]
                    )
            index_exists = index_path.exists()

        self.assertEqual(code, 0)
        self.assertEqual(output.getvalue().strip(), summarize)
        self.assertTrue(index_exists)

    def test_route_strict_returns_non_zero_for_uncertain_match(self):
        summarize = "Summarize the following text into concise bullet points."
        embedder = FakeEmbeddingClient(
            {
                summarize: [1.0, 0.0],
                "随便聊聊天": [0.1, 0.99],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            prompts_path = Path(tmpdir) / "prompts.json"
            index_path = Path(tmpdir) / "index.json"
            prompts_path.write_text(f'["{summarize}"]', encoding="utf-8")
            output = StringIO()

            with patch("prompt_router.cli.create_embedding_client", return_value=embedder):
                with redirect_stdout(output):
                    code = main(
                        [
                            "route",
                            str(prompts_path),
                            "随便聊聊天",
                            "--index",
                            str(index_path),
                            "--threshold",
                            "0.8",
                            "--strict",
                        ]
                    )

        self.assertEqual(code, 1)
        self.assertEqual(output.getvalue().strip(), summarize)

    def test_match_command_uses_provider_and_model_from_index(self):
        summarize = "Summarize the following text into concise bullet points."
        embedder = FakeEmbeddingClient(
            {
                "帮我总结一下": [0.99, 0.01],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "index.json"
            PromptIndex(
                provider="local",
                model="fake-local-model",
                entries=[
                    PromptIndex.Entry(prompt=summarize, vector=[1.0, 0.0]),
                ],
            ).save_json(index_path)
            output = StringIO()

            with patch("prompt_router.api.create_embedding_client", return_value=embedder) as factory:
                with redirect_stdout(output):
                    code = main(["match", str(index_path), "帮我总结一下"])

        self.assertEqual(code, 0)
        self.assertEqual(output.getvalue().strip(), summarize)
        factory.assert_called_once_with("local", model="fake-local-model", dimensions=None)


if __name__ == "__main__":
    unittest.main()
