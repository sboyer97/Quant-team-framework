from __future__ import annotations

import pandas as pd

from utils.data import prepare_market_data, resolve_universe


def test_custom_universe_is_normalized_and_deduplicated() -> None:
    assert resolve_universe("custom", " jpm, BAC,jpm ") == ["JPM", "BAC"]


def test_index_universe_converts_symbols_for_yahoo(monkeypatch) -> None:
    constituents = pd.DataFrame(
        {"Ticker": ["MT.AS", *[f"TEST{i}" for i in range(39)]]}
    )
    response = type(
        "Response",
        (),
        {"text": "<html></html>", "raise_for_status": lambda self: None},
    )()
    monkeypatch.setattr("utils.data.requests.get", lambda *args, **kwargs: response)
    monkeypatch.setattr(pd, "read_html", lambda url: [constituents])

    tickers = resolve_universe("cac40")

    assert len(tickers) == 40
    assert tickers[:2] == ["MT.AS", "TEST0.PA"]


def test_nasdaq100_uses_official_constituent_api(monkeypatch) -> None:
    rows = [{"symbol": f"TEST{i}"} for i in range(100)]
    response = type(
        "Response",
        (),
        {
            "raise_for_status": lambda self: None,
            "json": lambda self: {"data": {"data": {"rows": rows}}},
        },
    )()
    monkeypatch.setattr("utils.data.requests.get", lambda *args, **kwargs: response)

    tickers = resolve_universe("nasdaq100")

    assert len(tickers) == 100
    assert tickers[0] == "TEST0"


def test_csv_price_volume_dataset_is_prepared(tmp_path) -> None:
    dates = pd.date_range("2024-01-01", periods=40, freq="B")
    rows = [
        {
            "date": date,
            "ticker": ticker,
            "close": 100 + index + offset,
            "volume": 1_000_000 + index,
        }
        for index, date in enumerate(dates)
        for ticker, offset in (("AAA", 0), ("BBB", 10))
    ]
    path = tmp_path / "market.csv"
    pd.DataFrame(rows).to_csv(path, index=False)

    data = prepare_market_data(
        source="csv",
        tickers=["AAA", "BBB"],
        start="2024-01-01",
        end="2024-12-31",
        model_type="price_volume",
        csv_path=str(path),
        use_cache=False,
    )

    assert set(data) == {"close", "volume"}
    assert data["close"].shape == (40, 2)
    assert data["volume"].columns.tolist() == ["AAA", "BBB"]
