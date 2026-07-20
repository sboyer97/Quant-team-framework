from __future__ import annotations

import argparse

import main
from agents import universe


def test_universe_agent_structures_questions(monkeypatch) -> None:
    monkeypatch.setattr(
        universe,
        "chat_json",
        lambda *args, **kwargs: {
            "questions": [
                {"id": "count", "question": "How many assets?", "default": "10"}
            ]
        },
    )

    assert universe.generate_universe_questions("crypto") == [
        {"id": "count", "question": "How many assets?", "default": "10"}
    ]


def test_universe_agent_normalizes_tickers(monkeypatch) -> None:
    monkeypatch.setattr(
        universe,
        "chat_json",
        lambda *args, **kwargs: {
            "universe_name": "Liquid crypto",
            "rationale": "Top liquid assets",
            "tickers": ["btc-usd", "ETH-USD", "BTC-USD"],
        },
    )

    result = universe.resolve_universe_request("crypto", {"count": "10"})

    assert result["tickers"] == ["BTC-USD", "ETH-USD"]


def test_non_interactive_agent_uses_default_answers(monkeypatch) -> None:
    monkeypatch.setattr(
        main,
        "generate_universe_questions",
        lambda request: [
            {"id": "count", "question": "How many assets?", "default": "10"}
        ],
    )
    monkeypatch.setattr(
        main,
        "resolve_universe_request",
        lambda request, answers: {
            "universe_name": "Crypto 10",
            "rationale": answers["count"],
            "tickers": ["BTC-USD", "ETH-USD"],
        },
    )
    args = argparse.Namespace(
        universe_request="crypto",
        universe_answer=[],
        non_interactive=True,
    )

    result = main._resolve_agent_universe(args)

    assert result["rationale"] == "10"
