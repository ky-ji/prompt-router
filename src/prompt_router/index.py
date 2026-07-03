from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from prompt_router.router import EmbeddingClient, PromptLibrary


@dataclass(frozen=True)
class PromptIndex:
    @dataclass(frozen=True)
    class Entry:
        prompt: str
        vector: list[float]

    model: str
    entries: list[Entry]

    @classmethod
    def from_library(
        cls,
        library: PromptLibrary,
        embedding_client: EmbeddingClient,
        *,
        model: str,
    ) -> "PromptIndex":
        vectors = embedding_client.embed_many(library.prompts)
        if len(vectors) != len(library.records):
            raise ValueError("Embedding client returned the wrong number of vectors.")

        entries = [
            cls.Entry(prompt=record.prompt, vector=list(vector))
            for record, vector in zip(library.records, vectors)
        ]
        return cls(model=model, entries=entries)

    def save_json(self, path: str | Path) -> None:
        payload = {
            "schema_version": 1,
            "model": self.model,
            "entries": [
                {"prompt": entry.prompt, "vector": entry.vector}
                for entry in self.entries
            ],
        }
        Path(path).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load_json(cls, path: str | Path) -> "PromptIndex":
        with Path(path).open("r", encoding="utf-8") as file:
            payload = json.load(file)

        if not isinstance(payload, dict):
            raise ValueError("Prompt index must be a JSON object.")
        if payload.get("schema_version") != 1:
            raise ValueError("Unsupported prompt index schema version.")
        model = payload.get("model")
        entries = payload.get("entries")
        if not isinstance(model, str) or not model:
            raise ValueError("Prompt index must include a model string.")
        if not isinstance(entries, list) or not entries:
            raise ValueError("Prompt index must include at least one entry.")

        parsed_entries = []
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                raise ValueError(f"Index entry {index} must be an object.")
            prompt = entry.get("prompt")
            vector = entry.get("vector")
            if not isinstance(prompt, str) or not prompt:
                raise ValueError(f"Index entry {index} must include a prompt string.")
            if not isinstance(vector, list) or not vector:
                raise ValueError(f"Index entry {index} must include a non-empty vector.")
            if not all(isinstance(value, (int, float)) for value in vector):
                raise ValueError(f"Index entry {index} vector must contain numbers.")
            parsed_entries.append(
                cls.Entry(prompt=prompt, vector=[float(value) for value in vector])
            )

        return cls(model=model, entries=parsed_entries)
