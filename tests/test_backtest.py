from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from agents.backtest import BacktestError, run_backtest_agent


STRATEGY = """\
import numpy as np
import pandas as pd

def generate_signals(data):
    close = data["close"]
    momentum = close.pct_change(5)
    raw = momentum.sub(momentum.mean(axis=1), axis=0)
    gross = raw.abs().sum(axis=1).replace(0.0, np.nan)
    return raw.div(gross, axis=0).fillna(0.0)
"""


def test_backtest_accepts_market_data_dictionary() -> None:
    index = pd.date_range("2023-01-01", periods=100, freq="B")
    close = pd.DataFrame(
        {
            "AAA": 100 * np.cumprod(np.full(100, 1.001)),
            "BBB": 100 * np.cumprod(np.full(100, 0.999)),
        },
        index=index,
    )

    metrics = run_backtest_agent(STRATEGY, {"close": close}, cost_bps=0)

    assert metrics["num_days"] == 100
    assert metrics["num_trades"] > 0
    assert metrics["avg_gross_exposure"] <= 1.0


SLOW_STRATEGY = """\
import numpy as np
import pandas as pd

def generate_signals(data):
    while True:
        pass
"""


def test_backtest_aborts_strategies_exceeding_time_budget() -> None:
    index = pd.date_range("2023-01-01", periods=10, freq="B")
    close = pd.DataFrame({"AAA": np.ones(10), "BBB": np.ones(10)}, index=index)

    with pytest.raises(BacktestError, match="execution budget"):
        run_backtest_agent(SLOW_STRATEGY, {"close": close}, timeout_seconds=1)
