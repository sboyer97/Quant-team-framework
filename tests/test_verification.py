from __future__ import annotations

from agents import verification


def test_warnings_do_not_block_backtest(monkeypatch) -> None:
    monkeypatch.setattr(
        verification,
        "chat_json",
        lambda *args, **kwargs: {
            "passed": False,
            "issues": [
                {
                    "severity": "warning",
                    "category": "logic_error",
                    "description": "parameter differs from research",
                }
            ],
        },
    )

    result = verification.run_verification_agent(
        "def generate_signals(data): return data['close'] * 0",
        {"strategy_name": "test"},
        {"fields": ["close"]},
    )

    assert result["passed"] is True
    assert result["needs_revision"] is False


def test_material_llm_issue_requests_one_revision_without_blocking(monkeypatch) -> None:
    monkeypatch.setattr(
        verification,
        "chat_json",
        lambda *args, **kwargs: {
            "passed": False,
            "issues": [
                {
                    "severity": "critical",
                    "category": "logic_error",
                    "description": "Exit rule can never trigger.",
                }
            ],
        },
    )

    result = verification.run_verification_agent(
        "def generate_signals(data): return data['close'] * 0",
        {"strategy_name": "test"},
        {"fields": ["close"]},
    )

    assert result["passed"] is True
    assert result["needs_revision"] is True


def test_current_day_warning_is_non_blocking_with_execution_lag(monkeypatch) -> None:
    monkeypatch.setattr(
        verification,
        "chat_json",
        lambda *args, **kwargs: {
            "passed": False,
            "issues": [
                {
                    "severity": "critical",
                    "category": "lookahead_bias",
                    "description": "The current day's rank is used for the same day.",
                }
            ],
        },
    )

    result = verification.run_verification_agent(
        "def generate_signals(data): return data['close'].astype(float) * 0.0",
        {"strategy_name": "current close"},
        {"fields": ["close"]},
    )

    assert result["passed"] is True
    assert result["needs_revision"] is False


def test_prefix_check_rejects_full_sample_signal(monkeypatch) -> None:
    monkeypatch.setattr(
        verification,
        "chat_json",
        lambda *args, **kwargs: {"passed": True, "issues": []},
    )
    code = """\
import pandas as pd
import numpy as np
def generate_signals(data):
    close = data["close"]
    return close / close.mean() - 1.0
"""

    result = verification.run_verification_agent(
        code,
        {"strategy_name": "leaky"},
        {"fields": ["close"]},
    )

    assert result["passed"] is False
    assert any(issue["category"] == "lookahead_bias" for issue in result["issues"])


def test_integer_weight_failure_returns_concrete_fix(monkeypatch) -> None:
    monkeypatch.setattr(
        verification,
        "chat_json",
        lambda *args, **kwargs: {"passed": True, "issues": []},
    )
    code = """\
import pandas as pd
import numpy as np
def generate_signals(data):
    weights = pd.DataFrame(0, index=data["close"].index, columns=data["close"].columns)
    weights.iloc[20:, 0] = 0.5
    return weights
"""

    result = verification.run_verification_agent(
        code,
        {"strategy_name": "dtype failure"},
        {"fields": ["close"]},
    )

    assert result["passed"] is False
    assert "floating-point zeros" in result["issues"][-1]["suggested_fix"]


def test_stateful_exit_and_placeholder_cointegration_are_rejected(monkeypatch) -> None:
    monkeypatch.setattr(
        verification,
        "chat_json",
        lambda *args, **kwargs: {"passed": True, "issues": []},
    )
    code = """\
import pandas as pd
import numpy as np
def cointegration_test(left, right):
    return True
def generate_signals(data):
    exit_threshold = 0.0
    return data["close"].astype(float) * exit_threshold
"""

    result = verification.run_verification_agent(
        code,
        {"strategy_name": "invalid pairs"},
        {"fields": ["close"]},
    )

    assert result["passed"] is False
    descriptions = " ".join(issue["description"] for issue in result["issues"])
    assert "stateless" in descriptions
    assert "cointegration" in descriptions


def test_removed_pandas_frequency_alias_is_rejected_before_llm(monkeypatch) -> None:
    def unexpected_llm_call(*args, **kwargs):
        raise AssertionError("LLM reviewer should not run after objective failure")

    monkeypatch.setattr(verification, "chat_json", unexpected_llm_call)
    code = """\
import pandas as pd
import numpy as np
def generate_signals(data):
    monthly = data["close"].resample("M").last()
    return monthly.reindex(data["close"].index).fillna(0.0)
"""

    result = verification.run_verification_agent(
        code,
        {"strategy_name": "old pandas"},
        {"fields": ["close"]},
    )

    assert result["passed"] is False
    assert "frequency alias" in result["issues"][0]["description"]
