"""Verification Agent.

Reviews strategy code for the failure modes that most often invalidate a
backtest — lookahead bias, data leakage, and logic errors — before any
capital (or compute) is spent running it.
"""

from __future__ import annotations

import ast
import json
import logging
import re

import numpy as np
import pandas as pd

from utils.llm import chat_json

logger = logging.getLogger(__name__)

FORBIDDEN_PATTERN_MESSAGES = {
    r"\.shift\(\s*-\d+": "Negative shift uses future observations.",
    r"center\s*=\s*True": "Centered rolling windows use future observations.",
}

SYSTEM_PROMPT = """\
You are a rigorous quantitative code reviewer at a systematic hedge fund.
Your sole job is to decide whether strategy code is safe to backtest.

Review the provided implementation of `generate_signals(data)` for:

1. LOOKAHEAD BIAS — any use of future information in today's weight:
   .shift(-n), rolling(center=True), statistics fit on the full sample
   (e.g. z-scores using the whole series' mean/std), min/max over the full
   history, labels or thresholds derived from future data.
2. DATA LEAKAGE — normalizing, ranking or fitting across the entire dataset
   before generating per-date signals; using the test period to select
   parameters inside the function.
3. LOGIC ERRORS — runtime failures, NaN propagation, division by zero,
   unbounded leverage, or interface violations (wrong signature, wrong return
   shape, use of unavailable fields, forbidden imports or I/O).

Judge only genuine problems. Do not flag style preferences, and do not flag
the absence of next-day execution lag — the backtest engine applies a one-day
lag to all weights itself.

Important execution context:
- Every field in `data` has already been restricted to the requested universe.
  Using all supplied columns is correct.
- Rolling calculations ending at the current row are backward-looking and
  valid. A current-row signal may use the current close because the backtest
  trades it one day later.
- For pairs strategies, a rolling correlation/hedge-ratio/spread approach is
  an acceptable implementable proxy for cointegration when only pandas and
  numpy are allowed. Do not require a separate statistical package or a
  full-sample cointegration test. Never flag the absence of Engle-Granger,
  ADF or Johansen when the implementation uses this rolling proxy.
- Stateless target weights do not need an exit threshold or knowledge of a
  previous position. Judge the current target signal, not stateful trade
  lifecycle semantics.
- Portfolio weights are dollar fractions. Equal positive and negative weights
  are dollar-neutral even when stock prices differ.
- Gross normalization by the sum of absolute weights is valid; zero rows may
  safely become zero after replacing a zero denominator or filling NaNs.
- Treat imperfect entry/exit choices, parameter differences, missed trading
  opportunities and deviations from the research narrative as warnings.
  They are critical only if they cause a runtime/interface failure, future
  leakage, non-finite output or uncontrolled leverage.

Respond with a JSON object:
{
  "passed": true/false,
  "issues": [
    {
      "severity": "critical" | "warning",
      "category": "lookahead_bias" | "data_leakage" | "logic_error",
      "description": "what is wrong and where",
      "suggested_fix": "concrete fix"
    }
  ]
}
`passed` must be false if and only if there is at least one critical issue.
The purpose is to reject unsafe or non-runnable code, not to demand a perfect
trading strategy.
"""


def _issue(category: str, description: str, suggested_fix: str) -> dict:
    return {
        "severity": "critical",
        "category": category,
        "description": description,
        "suggested_fix": suggested_fix,
    }


def _synthetic_data(fields: list[str]) -> dict[str, pd.DataFrame]:
    # Long enough that a 252-row selection window fits strictly inside every
    # prefix cut below, so backward-looking selection stays prefix-invariant.
    index = pd.date_range("2020-01-01", periods=600, freq="B")
    columns = ["AAA", "BBB", "CCC", "DDD"]
    time = np.arange(len(index), dtype=float)
    close = pd.DataFrame(
        {
            column: 100.0 + offset + (0.03 + offset / 1000.0) * time
            + np.sin(time / (8.0 + offset))
            for offset, column in enumerate(columns, start=1)
        },
        index=index,
    )
    data = {"close": close}
    if "volume" in fields:
        data["volume"] = pd.DataFrame(
            {
                column: 1_000_000.0 + offset * 10_000.0 + 1000.0 * np.cos(time / 5.0)
                for offset, column in enumerate(columns, start=1)
            },
            index=index,
        )
    return data


def _objective_review(code: str, dataset_context: dict) -> list[dict]:
    """Reproduce safety failures instead of trusting reviewer assertions."""
    issues: list[dict] = []
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [_issue("logic_error", f"Strategy has invalid syntax: {exc}", "Return valid Python.")]

    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
    forbidden = imports - {"pandas", "numpy"}
    if forbidden:
        issues.append(
            _issue(
                "logic_error",
                f"Forbidden imports: {sorted(forbidden)}.",
                "Use only pandas and numpy.",
            )
        )
    for pattern, message in FORBIDDEN_PATTERN_MESSAGES.items():
        if re.search(pattern, code):
            issues.append(_issue("lookahead_bias", message, "Use backward-looking operations."))
    if re.search(r"pd\.DataFrame\(\s*0\s*,", code):
        issues.append(
            _issue(
                "logic_error",
                "Weights DataFrame is initialized with integer zeros.",
                "Initialize fractional weights with floating-point zeros: "
                "pd.DataFrame(0.0, ...).",
            )
        )
    if re.search(r"(?:resample\(\s*|freq\s*=\s*)[\"'](?:M|Q|Y)[\"']", code):
        issues.append(
            _issue(
                "logic_error",
                "Removed pandas frequency alias is used.",
                "Use pandas 3.x aliases ME, QE, or YE; prefer daily targets when possible.",
            )
        )
    identifiers = {
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
    } | {
        node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    }
    if identifiers & {"exit_signal", "exit_threshold"}:
        issues.append(
            _issue(
                "logic_error",
                "Stateful exit logic violates the stateless target-weight contract.",
                "Remove exit_signal/exit_threshold and derive each day's target "
                "directly from its backward-looking z-score.",
            )
        )
    if any(
        name.lower() in {"engle_granger", "cointegration_test", "adfuller"}
        for name in identifiers
    ):
        issues.append(
            _issue(
                "logic_error",
                "Formal or placeholder cointegration logic is not supported by the allowed imports.",
                "Use rolling correlation and a rolling covariance/variance hedge-ratio proxy.",
            )
        )
    if issues:
        return issues

    namespace = {"pd": pd, "np": np}
    try:
        exec(code, namespace)  # noqa: S102 - local deterministic validation
        generate_signals = namespace.get("generate_signals")
        if not callable(generate_signals):
            raise TypeError("generate_signals is missing")
        data = _synthetic_data(dataset_context.get("fields", ["close"]))
        full = generate_signals({key: value.copy() for key, value in data.items()})
        if not isinstance(full, pd.DataFrame):
            raise TypeError("generate_signals must return a DataFrame")
        if full.shape != data["close"].shape:
            raise ValueError("returned weights do not match the close data shape")
        numeric = full.apply(pd.to_numeric, errors="coerce")
        if not np.isfinite(numeric.to_numpy()).all():
            raise ValueError("returned weights contain NaN or infinite values")

        # Past weights must not change when future rows are removed.
        for cut in (350, 500):
            prefix_data = {key: value.iloc[:cut].copy() for key, value in data.items()}
            prefix = generate_signals(prefix_data)
            expected = numeric.iloc[:cut].to_numpy()
            actual = prefix.apply(pd.to_numeric, errors="coerce").to_numpy()
            if expected.shape != actual.shape or not np.allclose(
                expected, actual, rtol=1e-9, atol=1e-12, equal_nan=False
            ):
                issues.append(
                    _issue(
                        "lookahead_bias",
                        "Past signals change when future rows are removed.",
                        "Replace full-sample calculations with rolling or expanding calculations.",
                    )
                )
                break
    except Exception as exc:
        error = str(exc)
        suggested_fix = (
            "Initialize weights with floating-point zeros, e.g. "
            "pd.DataFrame(0.0, index=..., columns=...), before assigning fractions."
            if "dtype 'int64'" in error
            else "Make generate_signals execute on the declared fields and return finite weights."
        )
        issues.append(
            _issue(
                "logic_error",
                f"Objective execution check failed: {error}",
                suggested_fix,
            )
        )
    return issues


def run_verification_agent(
    code: str,
    research_summary: dict,
    dataset_context: dict,
) -> dict:
    """Review `code` against `research_summary`. Returns
    {"passed": bool, "issues": [...]}."""
    logger.info("Verification Agent: reviewing implementation")

    # Run reproducible checks first. There is no value in spending another LLM
    # call to diagnose code that already fails deterministically.
    objective_issues = _objective_review(code, dataset_context)
    if objective_issues:
        verification = {
            "passed": False,
            "needs_revision": True,
            "issues": objective_issues,
        }
        _log_verification(verification)
        return verification

    user_prompt = (
        "Research summary the code should implement:\n"
        + json.dumps(research_summary, indent=2)
        + "\n\nDataset context (the only available fields):\n"
        + json.dumps(dataset_context, indent=2)
        + "\n\nImplementation to review:\n```python\n"
        + code
        + "\n```"
    )

    result = chat_json(SYSTEM_PROMPT, user_prompt)
    llm_issues = result.get("issues", [])
    needs_revision = any(
        issue.get("severity") == "critical"
        and not _is_execution_lag_false_positive(issue)
        for issue in llm_issues
    )
    # The reviewer supplies useful qualitative feedback, but a probabilistic
    # assertion alone must not reject executable code. Objective checks decide.
    issues = [
        {**issue, "severity": "warning", "description": f"LLM review: {issue.get('description', '')}"}
        for issue in llm_issues
    ]
    verification = {
        "passed": True,
        "needs_revision": needs_revision,
        "issues": issues,
    }
    _log_verification(verification)
    return verification


def _is_execution_lag_false_positive(issue: dict) -> bool:
    """Current-close data is safe because positions are applied next day."""
    if issue.get("category") != "lookahead_bias":
        return False
    description = str(issue.get("description", "")).lower()
    markers = (
        "current date",
        "current day",
        "same day",
        "today's",
        "last trading day",
        "end of the month",
    )
    return any(marker in description for marker in markers)


def _log_verification(verification: dict) -> None:
    passed = verification["passed"]
    needs_revision = verification["needs_revision"]
    issues = verification["issues"]
    logger.info(
        "Verification Agent: %s (%d issue(s))",
        "FAILED" if not passed else ("REVISION ADVISED" if needs_revision else "PASSED"),
        len(issues),
    )
    for issue in issues:
        logger.info(
            "  [%s/%s] %s",
            issue.get("severity", "?"),
            issue.get("category", "?"),
            issue.get("description", ""),
        )
