from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
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
    provider: str = "openai"
    source_hash: str | None = None
    created_at: str | None = None

    @classmethod
    def from_library(
        cls,
        library: PromptLibrary,
        embedding_client: EmbeddingClient,
        *,
        provider: str = "local",
        model: str,
        source_hash: str | None = None,
    ) -> "PromptIndex":
        vectors = embedding_client.embed_many(library.prompts)
        if len(vectors) != len(library.records):
            raise ValueError("Embedding client returned the wrong number of vectors.")

        entries = [
            cls.Entry(prompt=record.prompt, vector=list(vector))
            for record, vector in zip(library.records, vectors)
        ]
        return cls(
            provider=provider,
            model=model,
            source_hash=source_hash,
            created_at=_utc_now(),
            entries=entries,
        )

    def save_json(self, path: str | Path) -> None:
        payload = {
            "schema_version": 2,
            "provider": self.provider,
            "model": self.model,
            "source_hash": self.source_hash,
            "created_at": self.created_at or _utc_now(),
            "entries": [
                {"prompt": entry.prompt, "vector": entry.vector}
                for entry in self.entries
            ],
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
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
        schema_version = payload.get("schema_version")
        if schema_version == 1:
            return cls(
                provider="openai",
                model=_read_model(payload),
                source_hash=None,
                created_at=None,
                entries=_read_entries(payload),
            )
        if schema_version != 2:
            raise ValueError("Unsupported prompt index schema version.")

        provider = payload.get("provider")
        if not isinstance(provider, str) or provider not in {"local", "openai"}:
            raise ValueError("Prompt index must include provider 'local' or 'openai'.")

        model = _read_model(payload)
        source_hash = payload.get("source_hash")
        if source_hash is not None and not isinstance(source_hash, str):
            raise ValueError("Prompt index source_hash must be a string or null.")
        created_at = payload.get("created_at")
        if created_at is not None and not isinstance(created_at, str):
            raise ValueError("Prompt index created_at must be a string or null.")

        return cls(
            provider=provider,
            model=model,
            source_hash=source_hash,
            created_at=created_at,
            entries=_read_entries(payload),
        )


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _read_model(payload: dict) -> str:
    model = payload.get("model")
    if not isinstance(model, str) or not model:
        raise ValueError("Prompt index must include a model string.")
    return model


def _read_entries(payload: dict) -> list[PromptIndex.Entry]:
    entries = payload.get("entries")
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
            PromptIndex.Entry(prompt=prompt, vector=[float(value) for value in vector])
        )

    return parsed_entries
