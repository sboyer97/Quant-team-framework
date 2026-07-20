"""Implementation Agent.

Turns a research summary into executable Python strategy code that conforms
to the fixed interface the backtest engine calls:

    def generate_signals(data: dict[str, pd.DataFrame]) -> pd.DataFrame

Each data field is a wide DataFrame (index: dates, columns: tickers). The
function returns weights matching `data["close"]`. The backtester applies a
one-day execution lag.
"""

from __future__ import annotations

import json
import logging
import re

from utils.llm import chat_json

logger = logging.getLogger(__name__)

STRATEGY_INTERFACE = """\
def generate_signals(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    \"\"\"Available keys are declared in the dataset context.
    Every value is a wide daily DataFrame (index=dates, columns=tickers).
    Returns target weights shaped exactly like data["close"].
    Positive = long, negative = short, 0 = flat.\"\"\"
"""

SYSTEM_PROMPT = f"""\
You are a senior quantitative developer at a systematic hedge fund.

Given a structured research summary, write clean, production-quality Python
code implementing the strategy. The code MUST define exactly this function:

{STRATEGY_INTERFACE}

Hard requirements:
- Import only pandas as pd and numpy as np. No other imports, no I/O, no
  network access, no data downloads.
- Use only fields listed in the dataset context. Access prices through
  data["close"] and volume through data["volume"] when available.
- No lookahead bias: the weight on any date may depend only on data up to
  and including that date. Use expanding/rolling windows, never center=True,
  never .shift(-n), never statistics computed over the full sample.
- A signal computed from the current row is valid because the backtester
  applies a one-day execution lag. Do not shift solely to satisfy execution.
- Target pandas 3.x. Never use removed frequency aliases "M", "Q", or "Y";
  use "ME", "QE", or "YE". Prefer daily vectorized targets over resampling.
- The columns are already the selected universe. Use them directly; do not
  validate membership or hardcode ticker symbols.
- Return stateless target weights recalculated independently for every date.
  Use boolean masks to set inactive signals to zero; do not build a position
  state machine with entry/exit mutations.
- For pairs strategies, do not implement or claim to implement Engle-Granger,
  ADF or cointegration without a statistical package. Use rolling correlation
  and a rolling covariance/variance hedge ratio as the research-approved
  proxy. Never add a placeholder test that always returns True.
- Do not use an exit threshold or persistent entry/exit state. Express the
  daily target directly from the current backward-looking z-score, setting it
  to zero when the signal is inactive.
- Initialize every weights/signals DataFrame with floating-point values, for
  example `pd.DataFrame(0.0, ...)`, before assigning fractional weights.
- Replace zero denominators with np.nan before division and finish with
  replace([np.inf, -np.inf], np.nan).fillna(0.0).
- Weights are dollar portfolio fractions, not share counts. A pair weighted
  +0.5 and -0.5 is dollar-neutral regardless of the two stock prices.
- Handle NaNs defensively (e.g. min_periods on rolling windows) and return
  a fully numeric DataFrame with NaNs replaced by 0.
- Keep gross exposure reasonable (sum of absolute weights per day <= 1 is a
  good default; the backtester also enforces this).
- Vectorized pandas/numpy; loops only where genuinely necessary (e.g. over
  pairs). Concise comments explaining non-obvious choices.
- The universe may contain hundreds of tickers and the whole function must
  finish in well under two minutes on a laptop. Never enumerate every ticker
  pair or compute rolling statistics over the full pairwise cross product.
  For pairs strategies, first select a small candidate set (at most ~20
  pairs) from ONE correlation matrix computed on returns of exactly the
  first 252 rows (`close.iloc[:252]` — a fixed row count, never a fraction
  of the sample length), keep weights at zero during those first 252 rows,
  trade the selected pairs only afterwards, and compute rolling statistics
  only for them. Selection must depend solely on those first rows so that
  removing future rows never changes past signals.
- The module must be self-contained and importable: no code should execute
  at import time other than function/constant definitions.

If verification feedback is provided, fix every issue it raises.

Respond with a JSON object:
{{
  "explanation": "1-2 sentences on how the code implements the strategy",
  "code": "the complete Python module as a string"
}}
"""


def _strip_code_fences(code: str) -> str:
    """Remove markdown code fences if the model wrapped its code in them."""
    match = re.search(r"```(?:python)?\s*\n(.*?)```", code, re.DOTALL)
    return match.group(1) if match else code


def run_implementation_agent(
    research_summary: dict,
    dataset_context: dict,
    verification_feedback: list[dict] | None = None,
    previous_code: str | None = None,
) -> str:
    """Return Python source implementing the strategy described in
    `research_summary`. On revision rounds, `previous_code` and
    `verification_feedback` are supplied so the agent patches rather than
    rewrites from scratch."""
    logger.info(
        "Implementation Agent: %s",
        "revising code after verification feedback" if verification_feedback else "writing initial code",
    )

    user_prompt_parts = [
        "Research summary:\n" + json.dumps(research_summary, indent=2),
        "Dataset context:\n" + json.dumps(dataset_context, indent=2),
    ]
    if previous_code and verification_feedback:
        user_prompt_parts.append("Previous implementation:\n```python\n" + previous_code + "\n```")
        user_prompt_parts.append(
            "Verification feedback — fix all of these issues:\n"
            + json.dumps(verification_feedback, indent=2)
        )

    result = chat_json(SYSTEM_PROMPT, "\n\n".join(user_prompt_parts), temperature=0.1)

    code = _strip_code_fences(result.get("code", "")).strip()
    if "def generate_signals" not in code:
        raise ValueError("Implementation Agent did not produce a generate_signals function.")

    # Fail fast on syntax errors instead of passing broken code downstream.
    compile(code, "<strategy>", "exec")

    return code
