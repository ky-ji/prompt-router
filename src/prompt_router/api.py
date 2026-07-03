from __future__ import annotations

import hashlib
from pathlib import Path

from prompt_router.embedding import create_embedding_client, default_model_for_provider
from prompt_router.index import PromptIndex
from prompt_router.router import MatchResult, PromptLibrary, PromptRouter


def default_index_path(prompts_path: str | Path) -> Path:
    path = Path(prompts_path)
    return path.with_name(f"{path.stem}.prompt-index.json")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def result_to_dict(result: MatchResult) -> dict:
    return {
        "prompt": result.prompt,
        "score": result.score,
        "second_score": result.second_score,
        "confident": result.confident,
        "reason": result.reason,
        "candidates": [
            {"prompt": candidate.prompt, "score": candidate.score}
            for candidate in result.candidates
        ],
    }


def build_index_file(
    prompts_path: str | Path,
    index_path: str | Path,
    embedding_client=None,
    *,
    provider: str = "local",
    model: str | None = None,
    dimensions: int | None = None,
) -> PromptIndex:
    prompts_path = Path(prompts_path)
    normalized_provider = _normalize_provider(provider)
    resolved_model = model or default_model_for_provider(normalized_provider)
    client = embedding_client or create_embedding_client(
        normalized_provider,
        model=resolved_model,
        dimensions=dimensions,
    )
    library = PromptLibrary.from_json(prompts_path)
    index = PromptIndex.from_library(
        library,
        client,
        provider=normalized_provider,
        model=resolved_model,
        source_hash=file_sha256(prompts_path),
    )
    index.save_json(index_path)
    return index


def match_index_file(
    index_path: str | Path,
    command: str,
    embedding_client=None,
    *,
    provider: str | None = None,
    model: str | None = None,
    dimensions: int | None = None,
    threshold: float = 0.45,
    margin: float = 0.03,
    top_k: int = 3,
) -> MatchResult:
    index = PromptIndex.load_json(index_path)
    normalized_provider = _normalize_provider(provider or index.provider)
    resolved_model = model or index.model

    if normalized_provider != index.provider:
        raise ValueError(
            f"Index was built with provider '{index.provider}', not '{normalized_provider}'. "
            "Rebuild the index or use the matching provider."
        )
    if resolved_model != index.model:
        raise ValueError(
            f"Index was built with model '{index.model}', not '{resolved_model}'. "
            "Rebuild the index or use the matching model."
        )

    client = embedding_client or create_embedding_client(
        normalized_provider,
        model=resolved_model,
        dimensions=dimensions,
    )
    router = PromptRouter.from_index(
        index,
        client,
        threshold=threshold,
        margin=margin,
    )
    return router.match(command, top_k=top_k)


def route_command(
    prompts_path: str | Path,
    command: str,
    *,
    provider: str = "local",
    model: str | None = None,
    index_path: str | Path | None = None,
    dimensions: int | None = None,
    threshold: float = 0.45,
    margin: float = 0.03,
    top_k: int = 3,
    embedding_client=None,
) -> MatchResult:
    prompts_path = Path(prompts_path)
    resolved_index_path = (
        Path(index_path) if index_path is not None else default_index_path(prompts_path)
    )
    normalized_provider = _normalize_provider(provider)
    resolved_model = model or default_model_for_provider(normalized_provider)
    source_hash = file_sha256(prompts_path)
    client = embedding_client or create_embedding_client(
        normalized_provider,
        model=resolved_model,
        dimensions=dimensions,
    )

    if _index_needs_rebuild(
        resolved_index_path,
        provider=normalized_provider,
        model=resolved_model,
        source_hash=source_hash,
    ):
        build_index_file(
            prompts_path,
            resolved_index_path,
            client,
            provider=normalized_provider,
            model=resolved_model,
            dimensions=dimensions,
        )

    return match_index_file(
        resolved_index_path,
        command,
        client,
        provider=normalized_provider,
        model=resolved_model,
        dimensions=dimensions,
        threshold=threshold,
        margin=margin,
        top_k=top_k,
    )


def _index_needs_rebuild(
    index_path: Path,
    *,
    provider: str,
    model: str,
    source_hash: str,
) -> bool:
    if not index_path.exists():
        return True

    index = PromptIndex.load_json(index_path)
    return not (
        index.provider == provider
        and index.model == model
        and index.source_hash == source_hash
    )


def _normalize_provider(provider: str) -> str:
    normalized = provider.lower()
    if normalized not in {"local", "openai"}:
        raise ValueError("provider must be 'local' or 'openai'.")
    return normalized
