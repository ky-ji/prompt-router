from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Callable, Sequence


OpenUrl = Callable[[urllib.request.Request, int], object]


class OpenAIEmbeddingClient:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "text-embedding-3-small",
        dimensions: int | None = None,
        endpoint: str = "https://api.openai.com/v1/embeddings",
        timeout: int = 30,
        opener: OpenUrl | None = None,
    ) -> None:
        resolved_api_key = api_key if api_key is not None else os.getenv("OPENAI_API_KEY")
        if not resolved_api_key:
            raise ValueError("OPENAI_API_KEY is required.")
        if dimensions is not None and dimensions <= 0:
            raise ValueError("dimensions must be positive.")

        self.api_key = resolved_api_key
        self.model = model
        self.dimensions = dimensions
        self.endpoint = endpoint
        self.timeout = timeout
        self._opener = opener or urllib.request.urlopen

    def embed_many(self, texts: Sequence[str]) -> list[list[float]]:
        inputs = list(texts)
        if not inputs:
            return []

        payload = {
            "model": self.model,
            "input": inputs,
            "encoding_format": "float",
        }
        if self.dimensions is not None:
            payload["dimensions"] = self.dimensions

        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with self._opener(request, self.timeout) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Embedding request failed: {detail}") from error
        except urllib.error.URLError as error:
            raise RuntimeError(f"Embedding request failed: {error.reason}") from error

        data = response_payload.get("data")
        if not isinstance(data, list):
            raise RuntimeError("Embedding response did not include a data list.")

        if all(isinstance(item, dict) and "index" in item for item in data):
            data = sorted(data, key=lambda item: item["index"])

        vectors = []
        for index, item in enumerate(data):
            if not isinstance(item, dict) or not isinstance(item.get("embedding"), list):
                raise RuntimeError(f"Embedding response item {index} is invalid.")
            vectors.append([float(value) for value in item["embedding"]])

        if len(vectors) != len(inputs):
            raise RuntimeError("Embedding response length did not match input length.")

        return vectors
