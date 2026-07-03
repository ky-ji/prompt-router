from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from prompt_router.embedding import OpenAIEmbeddingClient
from prompt_router.index import PromptIndex
from prompt_router.router import MatchResult, PromptLibrary, PromptRouter


def build_index_file(
    prompts_path: str | Path,
    index_path: str | Path,
    embedding_client,
    *,
    model: str,
) -> PromptIndex:
    library = PromptLibrary.from_json(prompts_path)
    index = PromptIndex.from_library(library, embedding_client, model=model)
    index.save_json(index_path)
    return index


def match_index_file(
    index_path: str | Path,
    command: str,
    embedding_client,
    *,
    threshold: float = 0.45,
    margin: float = 0.03,
    top_k: int = 3,
) -> MatchResult:
    index = PromptIndex.load_json(index_path)
    router = PromptRouter.from_index(
        index,
        embedding_client,
        threshold=threshold,
        margin=margin,
    )
    return router.match(command, top_k=top_k)


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


def make_embedding_client(args) -> OpenAIEmbeddingClient:
    return OpenAIEmbeddingClient(
        model=args.model,
        dimensions=args.dimensions,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prompt-router",
        description="Route a Chinese command to the closest fixed English prompt.",
    )
    subparsers = parser.add_subparsers(dest="command_name", required=True)

    build = subparsers.add_parser("build", help="Build an embedding index for prompts.")
    build.add_argument("prompts", type=Path, help="JSON file containing a list of prompt strings.")
    build.add_argument("index", type=Path, help="Where to write the prompt index JSON.")
    build.add_argument("--model", default="text-embedding-3-small")
    build.add_argument("--dimensions", type=int, default=None)

    match = subparsers.add_parser("match", help="Match a command against a prompt index.")
    match.add_argument("index", type=Path, help="Prompt index JSON created by build.")
    match.add_argument("command", help="Chinese user command to route.")
    match.add_argument("--model", default="text-embedding-3-small")
    match.add_argument("--dimensions", type=int, default=None)
    match.add_argument("--threshold", type=float, default=0.45)
    match.add_argument("--margin", type=float, default=0.03)
    match.add_argument("--top-k", type=int, default=3)
    match.add_argument("--json", action="store_true", help="Print prompt, scores, and candidates as JSON.")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    embedding_client = make_embedding_client(args)

    if args.command_name == "build":
        build_index_file(args.prompts, args.index, embedding_client, model=args.model)
        print(args.index)
        return 0

    result = match_index_file(
        args.index,
        args.command,
        embedding_client,
        threshold=args.threshold,
        margin=args.margin,
        top_k=args.top_k,
    )
    if args.json:
        print(json.dumps(result_to_dict(result), ensure_ascii=False, indent=2))
    else:
        print(result.prompt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
