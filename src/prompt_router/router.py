from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence


Vector = Sequence[float]


class EmbeddingClient(Protocol):
    def embed_many(self, texts: Sequence[str]) -> list[list[float]]:
        ...


@dataclass(frozen=True)
class PromptRecord:
    prompt: str


@dataclass(frozen=True)
class PromptLibrary:
    records: list[PromptRecord]

    @classmethod
    def from_prompts(cls, prompts: Sequence[str]) -> "PromptLibrary":
        records = []
        for index, item in enumerate(prompts):
            if not isinstance(item, str):
                raise ValueError(
                    f"Prompt at index {index} must be a string; got {type(item).__name__}."
                )
            prompt = item.strip()
            if not prompt:
                raise ValueError(f"Prompt at index {index} is empty.")
            records.append(PromptRecord(prompt=prompt))

        if not records:
            raise ValueError("Prompt library must contain at least one prompt.")

        return cls(records=records)

    @classmethod
    def from_json(cls, path: str | Path) -> "PromptLibrary":
        with Path(path).open("r", encoding="utf-8") as file:
            raw = json.load(file)

        if not isinstance(raw, list):
            raise ValueError("Prompt library must be a JSON list of prompt strings.")

        return cls.from_prompts(raw)

    @property
    def prompts(self) -> list[str]:
        return [record.prompt for record in self.records]


@dataclass(frozen=True)
class IndexedPrompt:
    prompt: str
    vector: list[float]


@dataclass(frozen=True)
class MatchCandidate:
    prompt: str
    score: float


@dataclass(frozen=True)
class MatchResult:
    prompt: str
    score: float
    second_score: float
    confident: bool
    reason: str
    candidates: list[MatchCandidate]


def cosine_similarity(left: Vector, right: Vector) -> float:
    if len(left) != len(right):
        raise ValueError(
            f"Vector dimensions must match; got {len(left)} and {len(right)}."
        )

    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0

    dot_product = sum(left_value * right_value for left_value, right_value in zip(left, right))
    return dot_product / (left_norm * right_norm)


class PromptRouter:
    def __init__(
        self,
        index: Sequence[IndexedPrompt],
        embedding_client: EmbeddingClient,
        *,
        threshold: float = 0.45,
        margin: float = 0.03,
    ) -> None:
        if not index:
            raise ValueError("PromptRouter requires at least one indexed prompt.")
        if margin < 0:
            raise ValueError("margin must be non-negative.")

        self._index = list(index)
        self._embedding_client = embedding_client
        self.threshold = threshold
        self.margin = margin

    @classmethod
    def from_library(
        cls,
        library: PromptLibrary,
        embedding_client: EmbeddingClient,
        *,
        threshold: float = 0.45,
        margin: float = 0.03,
    ) -> "PromptRouter":
        return cls.from_prompts(
            library.prompts,
            embedding_client,
            threshold=threshold,
            margin=margin,
        )

    @classmethod
    def from_index(
        cls,
        prompt_index,
        embedding_client: EmbeddingClient,
        *,
        threshold: float = 0.45,
        margin: float = 0.03,
    ) -> "PromptRouter":
        index = [
            IndexedPrompt(prompt=entry.prompt, vector=list(entry.vector))
            for entry in prompt_index.entries
        ]
        return cls(index, embedding_client, threshold=threshold, margin=margin)

    @classmethod
    def from_prompts(
        cls,
        prompts: Sequence[str],
        embedding_client: EmbeddingClient,
        *,
        threshold: float = 0.45,
        margin: float = 0.03,
    ) -> "PromptRouter":
        clean_prompts = [prompt.strip() for prompt in prompts if prompt.strip()]
        if not clean_prompts:
            raise ValueError("At least one prompt is required.")

        vectors = embedding_client.embed_many(clean_prompts)
        if len(vectors) != len(clean_prompts):
            raise ValueError("Embedding client returned the wrong number of vectors.")

        index = [
            IndexedPrompt(prompt=prompt, vector=list(vector))
            for prompt, vector in zip(clean_prompts, vectors)
        ]
        return cls(index, embedding_client, threshold=threshold, margin=margin)

    def match(self, command: str, *, top_k: int = 3) -> MatchResult:
        command = command.strip()
        if not command:
            raise ValueError("Command cannot be empty.")
        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        query_vector = self._embedding_client.embed_many([command])[0]
        candidates = sorted(
            (
                MatchCandidate(
                    prompt=indexed.prompt,
                    score=cosine_similarity(indexed.vector, query_vector),
                )
                for indexed in self._index
            ),
            key=lambda candidate: candidate.score,
            reverse=True,
        )

        best = candidates[0]
        second_score = candidates[1].score if len(candidates) > 1 else 0.0
        reason = "ok"
        confident = True

        if best.score < self.threshold:
            confident = False
            reason = "below_threshold"
        elif len(candidates) > 1 and best.score - second_score < self.margin:
            confident = False
            reason = "low_margin"

        return MatchResult(
            prompt=best.prompt,
            score=best.score,
            second_score=second_score,
            confident=confident,
            reason=reason,
            candidates=candidates[:top_k],
        )
