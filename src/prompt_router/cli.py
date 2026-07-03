from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from prompt_router.api import build_index_file, match_index_file, result_to_dict, route_command
from prompt_router.embedding import create_embedding_client


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prompt-router",
        description="Route a Chinese command to the closest fixed English prompt.",
    )
    subparsers = parser.add_subparsers(dest="command_name", required=True)

    build = subparsers.add_parser("build", help="Build an embedding index for prompts.")
    build.add_argument("prompts", type=Path, help="JSON file containing a list of prompt strings.")
    build.add_argument("index", type=Path, help="Where to write the prompt index JSON.")
    build.add_argument("--provider", choices=["local", "openai"], default="local")
    build.add_argument("--model", default=None)
    build.add_argument("--dimensions", type=int, default=None)

    match = subparsers.add_parser("match", help="Match a command against a prompt index.")
    match.add_argument("index", type=Path, help="Prompt index JSON created by build.")
    match.add_argument("command", help="Chinese user command to route.")
    match.add_argument("--provider", choices=["local", "openai"], default=None)
    match.add_argument("--model", default=None)
    match.add_argument("--dimensions", type=int, default=None)
    match.add_argument("--threshold", type=float, default=0.45)
    match.add_argument("--margin", type=float, default=0.03)
    match.add_argument("--top-k", type=int, default=3)
    match.add_argument("--json", action="store_true", help="Print prompt, scores, and candidates as JSON.")
    match.add_argument("--strict", action="store_true", help="Return exit code 1 when the match is uncertain.")

    route = subparsers.add_parser("route", help="Build or reuse an index, then route a command.")
    route.add_argument("prompts", type=Path, help="JSON file containing a list of prompt strings.")
    route.add_argument("command", help="Chinese user command to route.")
    route.add_argument("--index", type=Path, default=None, help="Index path. Defaults to PROMPTS.prompt-index.json.")
    route.add_argument("--provider", choices=["local", "openai"], default="local")
    route.add_argument("--model", default=None)
    route.add_argument("--dimensions", type=int, default=None)
    route.add_argument("--threshold", type=float, default=0.45)
    route.add_argument("--margin", type=float, default=0.03)
    route.add_argument("--top-k", type=int, default=3)
    route.add_argument("--json", action="store_true", help="Print prompt, scores, and candidates as JSON.")
    route.add_argument("--strict", action="store_true", help="Return exit code 1 when the match is uncertain.")

    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)

        if args.command_name == "build":
            embedding_client = create_embedding_client(
                args.provider,
                model=args.model,
                dimensions=args.dimensions,
            )
            build_index_file(
                args.prompts,
                args.index,
                embedding_client,
                provider=args.provider,
                model=args.model,
                dimensions=args.dimensions,
            )
            print(args.index)
            return 0

        if args.command_name == "route":
            embedding_client = create_embedding_client(
                args.provider,
                model=args.model,
                dimensions=args.dimensions,
            )
            result = route_command(
                args.prompts,
                args.command,
                provider=args.provider,
                model=args.model,
                index_path=args.index,
                dimensions=args.dimensions,
                threshold=args.threshold,
                margin=args.margin,
                top_k=args.top_k,
                embedding_client=embedding_client,
            )
            _print_result(result, as_json=args.json)
            return 1 if args.strict and not result.confident else 0

        result = match_index_file(
            args.index,
            args.command,
            provider=args.provider,
            model=args.model,
            dimensions=args.dimensions,
            threshold=args.threshold,
            margin=args.margin,
            top_k=args.top_k,
        )
        _print_result(result, as_json=args.json)
        return 1 if args.strict and not result.confident else 0
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


def _print_result(result, *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(result_to_dict(result), ensure_ascii=False, indent=2))
    else:
        print(result.prompt)


if __name__ == "__main__":
    sys.exit(main())
