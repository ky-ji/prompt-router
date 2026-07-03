import tempfile
import unittest
from pathlib import Path

from prompt_router.index import PromptIndex
from prompt_router.router import PromptLibrary, PromptRouter


class CountingEmbeddingClient:
    def __init__(self, vectors):
        self.vectors = vectors
        self.calls = []

    def embed_many(self, texts):
        self.calls.append(list(texts))
        return [self.vectors[text] for text in texts]


class IndexTests(unittest.TestCase):
    def test_saves_and_loads_prompt_index(self):
        summarize = "Summarize the following text into concise bullet points."
        translate = "Translate the following text into English."
        embedder = CountingEmbeddingClient(
            {
                summarize: [1.0, 0.0],
                translate: [0.0, 1.0],
            }
        )
        index = PromptIndex.from_library(
            PromptLibrary.from_prompts([summarize, translate]),
            embedder,
            model="fake-model",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "index.json"
            index.save_json(path)
            loaded = PromptIndex.load_json(path)

        self.assertEqual(loaded.model, "fake-model")
        self.assertEqual([entry.prompt for entry in loaded.entries], [summarize, translate])
        self.assertEqual([entry.vector for entry in loaded.entries], [[1.0, 0.0], [0.0, 1.0]])

    def test_router_from_index_only_embeds_query_at_match_time(self):
        summarize = "Summarize the following text into concise bullet points."
        translate = "Translate the following text into English."
        embedder = CountingEmbeddingClient(
            {
                "帮我总结一下": [0.99, 0.01],
            }
        )
        index = PromptIndex(
            model="fake-model",
            entries=[
                PromptIndex.Entry(prompt=summarize, vector=[1.0, 0.0]),
                PromptIndex.Entry(prompt=translate, vector=[0.0, 1.0]),
            ],
        )
        router = PromptRouter.from_index(index, embedder, threshold=0.3)

        result = router.match("帮我总结一下")

        self.assertEqual(result.prompt, summarize)
        self.assertEqual(embedder.calls, [["帮我总结一下"]])


if __name__ == "__main__":
    unittest.main()
