"""Backtest Agent.

Unlike the other agents, this one is deliberately NOT an LLM: performance
numbers must be computed deterministically. It executes the verified strategy
code, applies a one-day execution lag and a gross-leverage cap, and reports
standard metrics.
"""

from __future__ import annotations

import logging
import signal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TRADING_DAYS_PER_YEAR = 252
MAX_GROSS_LEVERAGE = 1.0
# One-way transaction cost applied to turnover: commission + slippage.
# 5 bps is a reasonable all-in figure for liquid S&P 500 large caps.
DEFAULT_COST_BPS = 5.0
# Wall-clock budget for one generate_signals call. Generated code that scans
# every ticker pair on a large universe can otherwise hang the whole loop.
STRATEGY_TIMEOUT_SECONDS = 120


class BacktestError(Exception):
    """Raised when the strategy code fails to execute or produces unusable output."""


class _StrategyTimeout(Exception):
    """Raised inside the SIGALRM handler when the strategy runs too long."""


def _call_with_timeout(func, data: dict, timeout_seconds: int):
    """Run func(data) with a SIGALRM-based wall-clock limit.

    SIGALRM only exists on POSIX and only fires in the main thread; when it
    is unavailable the strategy simply runs without a budget.
    """
    use_alarm = hasattr(signal, "SIGALRM")
    if use_alarm:
        def _handler(signum, frame):
            raise _StrategyTimeout()

        previous = signal.signal(signal.SIGALRM, _handler)
        signal.alarm(timeout_seconds)
    try:
        return func(data)
    finally:
        if use_alarm:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous)


def _load_strategy(code: str):
    """Execute strategy source and return its generate_signals function.

    NOTE: `exec` of model-generated code is acceptable for a local demo; a
    production system would run this in an isolated sandbox with resource
    limits (see README).
    """
    namespace: dict = {"pd": pd, "np": np}
    try:
        exec(code, namespace)  # noqa: S102 — reviewed by the Verification Agent
    except Exception as exc:
        raise BacktestError(f"Strategy code failed to execute: {exc}") from exc

    func = namespace.get("generate_signals")
    if not callable(func):
        raise BacktestError("Strategy code does not define generate_signals().")
    return func


def _clean_weights(raw, prices: pd.DataFrame) -> pd.DataFrame:
    """Align, sanitize and leverage-cap the strategy's raw weights."""
    if isinstance(raw, pd.Series):
        raw = raw.to_frame()
    if not isinstance(raw, pd.DataFrame):
        raise BacktestError(f"generate_signals returned {type(raw).__name__}, expected DataFrame.")

    weights = (
        raw.reindex(index=prices.index, columns=prices.columns)
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    # Cap gross exposure at MAX_GROSS_LEVERAGE per day.
    gross = weights.abs().sum(axis=1)
    scale = (MAX_GROSS_LEVERAGE / gross).clip(upper=1.0).fillna(0.0)
    return weights.mul(scale, axis=0)


def run_backtest_agent(
    code: str,
    data: dict[str, pd.DataFrame],
    cost_bps: float = DEFAULT_COST_BPS,
    timeout_seconds: int = STRATEGY_TIMEOUT_SECONDS,
) -> dict:
    """Run one in-sample backtest on a prepared market dataset.

    Weights are lagged one day so that a signal computed at close t is traded
    at close t+1 — the engine-level guarantee against execution lookahead.

    Transaction costs: each day the strategy pays `cost_bps` (one-way, in
    basis points) on its turnover, i.e. the sum of absolute weight changes.
    All reported metrics are net of costs; the gross Sharpe and the cost
    drag are reported alongside for transparency.
    """
    if "close" not in data:
        raise BacktestError("Market data must contain a 'close' field.")
    prices = data["close"]
    logger.info(
        "Backtest Agent: running on %d tickers, %s to %s",
        prices.shape[1],
        prices.index[0].date(),
        prices.index[-1].date(),
    )

    generate_signals = _load_strategy(code)
    try:
        strategy_data = {field: frame.copy() for field, frame in data.items()}
        raw_weights = _call_with_timeout(generate_signals, strategy_data, timeout_seconds)
    except _StrategyTimeout:
        raise BacktestError(
            f"generate_signals exceeded the {timeout_seconds}s execution budget; "
            "the strategy is too slow for this universe size."
        ) from None
    except Exception as exc:
        raise BacktestError(f"generate_signals raised: {exc}") from exc

    weights = _clean_weights(raw_weights, prices)
    held = weights.shift(1).fillna(0.0)  # one-day execution lag

    asset_returns = prices.pct_change().fillna(0.0)
    gross_returns = (held * asset_returns).sum(axis=1)

    if held.abs().sum().sum() == 0:
        raise BacktestError("Strategy never takes a position.")

    # Transaction costs: pay cost_bps on daily turnover (sum of |weight changes|).
    turnover = held.diff().abs().sum(axis=1).fillna(held.iloc[0].abs().sum())
    daily_costs = turnover * cost_bps / 10_000.0
    net_returns = gross_returns - daily_costs

    equity = (1.0 + net_returns).cumprod()
    n_days = len(net_returns)

    annualized_return = equity.iloc[-1] ** (TRADING_DAYS_PER_YEAR / n_days) - 1.0
    vol = net_returns.std()
    sharpe = (
        float(net_returns.mean() / vol * np.sqrt(TRADING_DAYS_PER_YEAR))
        if vol > 0
        else 0.0
    )
    gross_vol = gross_returns.std()
    gross_sharpe = (
        float(gross_returns.mean() / gross_vol * np.sqrt(TRADING_DAYS_PER_YEAR))
        if gross_vol > 0
        else 0.0
    )
    max_drawdown = float((equity / equity.cummax() - 1.0).min())

    # A "trade" is any change in a name's held position.
    position_changes = held.diff().abs() > 1e-9
    num_trades = int(position_changes.sum().sum())

    metrics = {
        "sharpe_ratio": round(sharpe, 3),  # net of costs — used for ranking
        "gross_sharpe_ratio": round(gross_sharpe, 3),
        "annualized_return_pct": round(100 * float(annualized_return), 2),
        "annualized_cost_drag_pct": round(
            100 * float(daily_costs.mean()) * TRADING_DAYS_PER_YEAR, 2
        ),
        "max_drawdown_pct": round(100 * max_drawdown, 2),
        "num_trades": num_trades,
        "avg_daily_turnover": round(float(turnover.mean()), 3),
        "cost_bps": cost_bps,
        "num_days": n_days,
        "avg_gross_exposure": round(float(held.abs().sum(axis=1).mean()), 3),
        "period": f"{prices.index[0].date()} to {prices.index[-1].date()}",
    }
    logger.info("Backtest Agent: %s", metrics)
    return metrics
