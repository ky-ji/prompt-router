import tempfile
import unittest
from pathlib import Path

from prompt_router.api import default_index_path, route_command
from prompt_router.index import PromptIndex


class FakeEmbeddingClient:
    def __init__(self, vectors):
        self.vectors = vectors
        self.calls = []

    def embed_many(self, texts):
        self.calls.append(list(texts))
        return [self.vectors[text] for text in texts]


class ApiTests(unittest.TestCase):
    def test_default_index_path_sits_next_to_prompt_file(self):
        self.assertEqual(
            default_index_path(Path("examples/prompts.json")),
            Path("examples/prompts.prompt-index.json"),
        )

    def test_route_command_builds_missing_index_and_reuses_it(self):
        summarize = "Summarize the following text into concise bullet points."
        translate = "Translate the following text into English."
        embedder = FakeEmbeddingClient(
            {
                summarize: [1.0, 0.0],
                translate: [0.0, 1.0],
                "帮我总结一下": [0.99, 0.01],
                "帮我翻译成英文": [0.01, 0.99],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            prompts_path = Path(tmpdir) / "prompts.json"
            index_path = Path(tmpdir) / "index.json"
            prompts_path.write_text(
                f'["{summarize}", "{translate}"]',
                encoding="utf-8",
            )

            first = route_command(
                prompts_path,
                "帮我总结一下",
                index_path=index_path,
                embedding_client=embedder,
            )
            second = route_command(
                prompts_path,
                "帮我翻译成英文",
                index_path=index_path,
                embedding_client=embedder,
            )
            loaded = PromptIndex.load_json(index_path)

        self.assertEqual(first.prompt, summarize)
        self.assertEqual(second.prompt, translate)
        self.assertEqual(loaded.provider, "local")
        self.assertIsNotNone(loaded.source_hash)
        self.assertEqual(
            embedder.calls,
            [
                [summarize, translate],
                ["帮我总结一下"],
                ["帮我翻译成英文"],
            ],
        )

    def test_route_command_rebuilds_when_prompt_file_changes(self):
        summarize = "Summarize the following text into concise bullet points."
        rewrite = "Rewrite the following text to make it clearer and more professional."
        embedder = FakeEmbeddingClient(
            {
                summarize: [1.0, 0.0],
                rewrite: [0.0, 1.0],
                "帮我总结一下": [0.99, 0.01],
                "帮我润色一下": [0.01, 0.99],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            prompts_path = Path(tmpdir) / "prompts.json"
            index_path = Path(tmpdir) / "index.json"
            prompts_path.write_text(f'["{summarize}"]', encoding="utf-8")
            route_command(
                prompts_path,
                "帮我总结一下",
                index_path=index_path,
                embedding_client=embedder,
            )

            prompts_path.write_text(f'["{rewrite}"]', encoding="utf-8")
            result = route_command(
                prompts_path,
                "帮我润色一下",
                index_path=index_path,
                embedding_client=embedder,
            )

        self.assertEqual(result.prompt, rewrite)

    def test_route_command_rebuilds_when_index_provider_or_model_differs(self):
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
            PromptIndex(
                provider="openai",
                model="old-model",
                source_hash="stale",
                entries=[PromptIndex.Entry(prompt="Wrong prompt.", vector=[0.0, 1.0])],
            ).save_json(index_path)

            result = route_command(
                prompts_path,
                "帮我总结一下",
                index_path=index_path,
                embedding_client=embedder,
            )
            rebuilt = PromptIndex.load_json(index_path)

        self.assertEqual(result.prompt, summarize)
        self.assertEqual(rebuilt.provider, "local")


if __name__ == "__main__":
    unittest.main()
