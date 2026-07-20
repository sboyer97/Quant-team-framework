from __future__ import annotations

from agents import research


def test_semantic_scholar_results_are_structured(monkeypatch) -> None:
    response = type(
        "Response",
        (),
        {
            "raise_for_status": lambda self: None,
            "json": lambda self: {
                "data": [
                    {
                        "title": "A Real Paper",
                        "url": "https://www.semanticscholar.org/paper/123",
                        "year": 2020,
                        "authors": [{"name": "Researcher"}],
                        "abstract": "An empirical study.",
                    }
                ]
            },
        },
    )()
    monkeypatch.setattr(research.requests, "get", lambda *args, **kwargs: response)

    results = research._search_literature("pairs trading")

    assert results[0]["title"] == "A Real Paper"
    assert results[0]["source"] == "Semantic Scholar"


def test_research_references_cannot_be_invented(monkeypatch) -> None:
    verified = [{"title": "Verified", "url": "https://example.com/paper"}]
    monkeypatch.setattr(research, "_search_literature", lambda query: verified)
    monkeypatch.setattr(
        research,
        "chat_json",
        lambda *args, **kwargs: {
            "strategy_name": "Test",
            "strategy_logic": "Logic",
            "key_parameters": {},
            "signals": [],
            "references": ["Invented citation"],
        },
    )

    result = research.run_research_agent(
        "pairs",
        {"universe": "banks", "fields": ["close"]},
    )

    assert result["references"] == ["Verified — https://example.com/paper"]
