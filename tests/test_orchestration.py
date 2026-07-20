from __future__ import annotations

import argparse

import pandas as pd
import pytest

import main


def test_non_interactive_selection_has_research_objective_default() -> None:
    args = argparse.Namespace(
        non_interactive=True,
        source="yahoo",
        universe="custom",
        model_type="price",
        idea=None,
        tickers="AAA,BBB",
        universe_request=None,
        csv_path=None,
        runs=1,
        start="2020-01-01",
        end="2024-01-01",
    )

    selected = main._complete_selection(args)

    assert selected.idea == "pairs trading on the selected universe"


def test_interactive_selection_only_asks_idea_runs_and_dates(monkeypatch) -> None:
    args = argparse.Namespace(
        non_interactive=False,
        source=None,
        universe=None,
        model_type=None,
        idea=None,
        tickers=None,
        universe_request=None,
        csv_path=None,
        runs=None,
        start=None,
        end=None,
    )
    prompts: list[str] = []

    def fake_input(prompt: str) -> str:
        prompts.append(prompt)
        return ""

    monkeypatch.setattr(main.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", fake_input)

    selected = main._complete_selection(args)

    assert len(prompts) == 4  # idea, candidates, start date, end date
    assert "Research objective" in prompts[0]
    assert selected.source == "yahoo"
    assert selected.universe == "sp500"
    assert selected.model_type == "price"
    assert selected.idea == "pairs trading on the S&P 500"
    assert selected.runs == 10


def test_backtest_date_range_validation() -> None:
    main._validate_date_range("2020-01-01", "2024-01-01")

    with pytest.raises(ValueError, match="earlier"):
        main._validate_date_range("2024-01-01", "2020-01-01")
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        main._validate_date_range("01/01/2020", "2024-01-01")


def test_verification_feedback_loops_to_implementation(monkeypatch) -> None:
    calls: list[dict | None] = []
    research = {
        "strategy_name": "Test",
        "strategy_logic": "Test logic",
        "key_parameters": {},
        "signals": [],
        "references": [],
    }
    code = "def generate_signals(data):\n    return data['close'] * 0\n"

    monkeypatch.setattr(main, "run_research_agent", lambda idea, context: research)

    def implement(summary, context, verification_feedback=None, previous_code=None):
        calls.append(verification_feedback)
        return code

    reviews = iter(
        [
            {"passed": False, "issues": [{"severity": "critical", "description": "fix"}]},
            {"passed": True, "issues": []},
        ]
    )
    monkeypatch.setattr(main, "run_implementation_agent", implement)
    monkeypatch.setattr(main, "run_verification_agent", lambda *args: next(reviews))
    monkeypatch.setattr(main, "run_backtest_agent", lambda *args, **kwargs: {"sharpe_ratio": 1.2})

    close = pd.DataFrame({"AAA": [1, 2], "BBB": [2, 3]})
    result = main.run_candidate(
        "idea",
        {"close": close},
        {"universe": "custom", "fields": ["close"]},
        5.0,
    )

    assert calls == [None, [{"severity": "critical", "description": "fix"}]]
    assert result["metrics"]["sharpe_ratio"] == 1.2


def test_qualitative_review_gets_one_revision_then_becomes_advisory(monkeypatch) -> None:
    research = {
        "strategy_name": "Test",
        "strategy_logic": "Logic",
        "key_parameters": {},
        "signals": [],
    }
    implementation_calls = 0
    monkeypatch.setattr(main, "run_research_agent", lambda *args: research)

    def implement(*args, **kwargs):
        nonlocal implementation_calls
        implementation_calls += 1
        return "def generate_signals(data): return data['close'] * 0"

    monkeypatch.setattr(main, "run_implementation_agent", implement)
    monkeypatch.setattr(
        main,
        "run_verification_agent",
        lambda *args: {
            "passed": True,
            "needs_revision": True,
            "issues": [{"severity": "warning", "description": "improve semantics"}],
        },
    )
    monkeypatch.setattr(main, "run_backtest_agent", lambda *args, **kwargs: {"sharpe_ratio": 0})

    result = main.run_candidate(
        "idea",
        {"close": pd.DataFrame({"AAA": [1], "BBB": [1]})},
        {"universe": "custom", "fields": ["close"]},
        5.0,
    )

    assert implementation_calls == 2
    assert result["verification"]["advisory_issues_remaining"] is True


def test_result_files_keep_ranked_runs(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    ranked = [
        {
            "run": 2,
            "code": "def generate_signals(data): pass\n",
            "research": {"strategy_name": "Winner"},
            "verification": {"passed": True, "issues": []},
            "metrics": {"sharpe_ratio": 2.0},
        }
    ]

    main._save_results(ranked, [{"run": 1, "error": "failed"}], {"runs": 2})

    assert (tmp_path / "best_strategy.py").exists()
    report = __import__("json").loads((tmp_path / "research_runs_report.json").read_text())
    assert report["ranked_runs"][0]["run"] == 2
    assert report["failures"][0]["run"] == 1
