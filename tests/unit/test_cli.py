from __future__ import annotations

from rag.cli import build_parser


def test_cli_exposes_complete_runtime_commands() -> None:
    parser = build_parser()
    for command in ("chunk", "validate", "ingest", "route", "retrieve", "ask"):
        args = parser.parse_args([command, "question"] if command in {"route", "retrieve", "ask"} else [command])
        assert callable(args.handler)


def test_ask_defaults_to_automatic_routing() -> None:
    args = build_parser().parse_args(["ask", "What is the current policy?"])
    assert args.route == "auto"
    assert args.model == "mistral:7b"
