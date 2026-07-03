from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Callable, Sequence


OpenUrl = Callable[[urllib.request.Request, int], object]

DEFAULT_LOCAL_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_OPENAI_MODEL = "text-embedding-3-small"


def default_model_for_provider(provider: str) -> str:
    normalized = provider.lower()
    if normalized == "local":
        return DEFAULT_LOCAL_MODEL
    if normalized == "openai":
        return DEFAULT_OPENAI_MODEL
    raise ValueError("provider must be 'local' or 'openai'.")


class LocalEmbeddingClient:
    def __init__(
        self,
        *,
        model: str | None = None,
        model_loader: Callable[[str], object] | None = None,
    ) -> None:
        self.model = model or DEFAULT_LOCAL_MODEL
        self._model_loader = model_loader or self._load_sentence_transformer
        self._model = None

    def _load_sentence_transformer(self, model: str) -> object:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise RuntimeError(
                "Local embeddings require sentence-transformers. "
                'Install it with: pip install -e ".[local]"'
            ) from error

        return SentenceTransformer(model)

    def _get_model(self) -> object:
        if self._model is None:
            try:
                self._model = self._model_loader(self.model)
            except ImportError as error:
                raise RuntimeError(
                    "Local embeddings require sentence-transformers. "
                    'Install it with: pip install -e ".[local]"'
                ) from error
        return self._model

    def embed_many(self, texts: Sequence[str]) -> list[list[float]]:
        inputs = list(texts)
        if not inputs:
            return []

        model = self._get_model()
        raw_vectors = model.encode(inputs)
        vectors = []
        for vector in raw_vectors:
            if hasattr(vector, "tolist"):
                vector = vector.tolist()
            vectors.append([float(value) for value in vector])
        return vectors


class OpenAIEmbeddingClient:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = DEFAULT_OPENAI_MODEL,
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


def create_embedding_client(
    provider: str = "local",
    *,
    model: str | None = None,
    dimensions: int | None = None,
    model_loader: Callable[[str], object] | None = None,
) -> LocalEmbeddingClient | OpenAIEmbeddingClient:
    normalized = provider.lower()
    resolved_model = model or default_model_for_provider(normalized)

    if normalized == "local":
        if dimensions is not None:
            raise ValueError("dimensions is only supported for the openai provider.")
        return LocalEmbeddingClient(model=resolved_model, model_loader=model_loader)

    if normalized == "openai":
        return OpenAIEmbeddingClient(model=resolved_model, dimensions=dimensions)

    raise ValueError("provider must be 'local' or 'openai'.")
