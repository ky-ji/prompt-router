import json
import tempfile
import unittest
from pathlib import Path

from prompt_router.router import (
    PromptLibrary,
    PromptRouter,
    cosine_similarity,
)


class FakeEmbeddingClient:
    def __init__(self, vectors):
        self.vectors = vectors

    def embed_many(self, texts):
        return [self.vectors[text] for text in texts]


class RouterTests(unittest.TestCase):
    def test_cosine_similarity_scores_related_vectors(self):
        self.assertAlmostEqual(cosine_similarity([1, 0], [1, 0]), 1.0)
        self.assertAlmostEqual(cosine_similarity([1, 0], [0, 1]), 0.0)

    def test_cosine_similarity_rejects_mismatched_dimensions(self):
        with self.assertRaises(ValueError):
            cosine_similarity([1, 2], [1, 2, 3])

    def test_loads_prompt_library_from_plain_string_list(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "prompts.json"
            path.write_text(
                json.dumps(
                    [
                        "Summarize the following text into concise bullet points.",
                        "Translate the following text into English.",
                    ]
                ),
                encoding="utf-8",
            )

            library = PromptLibrary.from_json(path)

        self.assertEqual(
            [record.prompt for record in library.records],
            [
                "Summarize the following text into concise bullet points.",
                "Translate the following text into English.",
            ],
        )

    def test_matches_chinese_command_to_best_prompt(self):
        summarize = "Summarize the following text into concise bullet points."
        translate = "Translate the following text into English."
        embedder = FakeEmbeddingClient(
            {
                summarize: [1.0, 0.0],
                translate: [0.0, 1.0],
                "帮我总结一下这段内容": [0.95, 0.05],
            }
        )
        router = PromptRouter.from_prompts(
            [summarize, translate],
            embedder,
            threshold=0.3,
            margin=0.1,
        )

        result = router.match("帮我总结一下这段内容")

        self.assertEqual(result.prompt, summarize)
        self.assertTrue(result.confident)
        self.assertGreater(result.score, result.second_score)

    def test_marks_result_uncertain_when_score_is_too_low(self):
        summarize = "Summarize the following text into concise bullet points."
        embedder = FakeEmbeddingClient(
            {
                summarize: [1.0, 0.0],
                "随便聊聊天": [0.1, 0.99],
            }
        )
        router = PromptRouter.from_prompts([summarize], embedder, threshold=0.8)

        result = router.match("随便聊聊天")

        self.assertEqual(result.prompt, summarize)
        self.assertFalse(result.confident)
        self.assertEqual(result.reason, "below_threshold")

    def test_marks_result_uncertain_when_top_candidates_are_too_close(self):
        summarize = "Summarize the following text into concise bullet points."
        rewrite = "Rewrite the following text to make it clearer."
        embedder = FakeEmbeddingClient(
            {
                summarize: [1.0, 0.0],
                rewrite: [0.98, 0.2],
                "帮我整理一下这段话": [0.99, 0.1],
            }
        )
        router = PromptRouter.from_prompts(
            [summarize, rewrite],
            embedder,
            threshold=0.3,
            margin=0.05,
        )

        result = router.match("帮我整理一下这段话")

        self.assertFalse(result.confident)
        self.assertEqual(result.reason, "low_margin")


if __name__ == "__main__":
    unittest.main()
