"""Dataset catalog and local preparation for the research pipeline."""

from __future__ import annotations

import hashlib
import logging
import time
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

logger = logging.getLogger(__name__)

MarketData = dict[str, pd.DataFrame]
CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"

SECTOR_UNIVERSES: dict[str, list[str]] = {
    "banking": ["JPM", "BAC", "WFC", "C", "USB", "PNC", "TFC", "GS", "MS", "SCHW"],
    "technology": ["AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CRM", "AMD", "ADBE", "CSCO", "INTC"],
    "energy": ["XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "OXY", "WMB"],
    "healthcare": ["UNH", "JNJ", "LLY", "PFE", "ABBV", "MRK", "TMO", "ABT", "DHR", "BMY"],
    "consumer": ["PG", "KO", "PEP", "COST", "WMT", "MCD", "NKE", "SBUX", "TGT", "HD"],
}
INDEX_UNIVERSES: dict[str, dict[str, object]] = {
    "sp500": {
        "url": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        "columns": ("Symbol", "Ticker"),
        "suffix": "",
    },
    "nasdaq100": {
        "url": "https://en.wikipedia.org/wiki/Nasdaq-100",
        "columns": ("Ticker", "Symbol"),
        "suffix": "",
    },
    "cac40": {
        "url": "https://en.wikipedia.org/wiki/CAC_40",
        "columns": ("Ticker", "Symbol"),
        "suffix": ".PA",
    },
    "dax40": {
        "url": "https://en.wikipedia.org/wiki/DAX",
        "columns": ("Ticker symbol", "Ticker", "Symbol"),
        "suffix": ".DE",
    },
    "ftse100": {
        "url": "https://en.wikipedia.org/wiki/FTSE_100_Index",
        "columns": ("Ticker", "EPIC", "Symbol"),
        "suffix": ".L",
    },
    "nikkei225": {
        "url": "https://en.wikipedia.org/wiki/Nikkei_225",
        "columns": ("Code", "Ticker", "Symbol"),
        "suffix": ".T",
    },
}
UNIVERSE_CHOICES = (
    *INDEX_UNIVERSES.keys(),
    *SECTOR_UNIVERSES.keys(),
    "agent",
    "custom",
)
MODEL_TYPES = ("price", "price_volume")


def _resolve_index_universe(name: str) -> list[str]:
    if name == "nasdaq100":
        try:
            response = requests.get(
                "https://api.nasdaq.com/api/quote/list-type/nasdaq100",
                headers={
                    "User-Agent": "Mozilla/5.0",
                    "Accept": "application/json, text/plain, */*",
                    "Origin": "https://www.nasdaq.com",
                    "Referer": "https://www.nasdaq.com/",
                },
                timeout=15,
            )
            response.raise_for_status()
            rows = response.json()["data"]["data"]["rows"]
            tickers = [
                str(row["symbol"]).strip().upper().replace(".", "-")
                for row in rows
                if row.get("symbol")
            ]
            if len(tickers) < 90:
                raise ValueError("Nasdaq API returned too few constituents")
            return list(dict.fromkeys(tickers))
        except Exception as exc:
            raise RuntimeError(
                "Could not retrieve the nasdaq100 constituents. "
                "Use a sector or custom universe, then retry."
            ) from exc

    config = INDEX_UNIVERSES[name]
    try:
        response = requests.get(
            str(config["url"]),
            headers={"User-Agent": "quant-team-framework/1.0"},
            timeout=15,
        )
        response.raise_for_status()
        tables = pd.read_html(StringIO(response.text))
        series = None
        for table in tables:
            for column in config["columns"]:
                if column in table.columns:
                    candidate = table[column].dropna().astype(str).str.strip()
                    if len(candidate) >= 20:
                        series = candidate
                        break
            if series is not None:
                break
        if series is None:
            raise ValueError("No constituent table found")

        suffix = str(config["suffix"])
        tickers = []
        for raw_symbol in series:
            symbol = raw_symbol.split()[0].upper()
            if suffix:
                if "." not in symbol:
                    symbol += suffix
            else:
                symbol = symbol.replace(".", "-")
            tickers.append(symbol)
        return list(dict.fromkeys(tickers))
    except Exception as exc:
        raise RuntimeError(
            f"Could not retrieve the {name} constituents. "
            "Use a sector or custom universe, then retry."
        ) from exc


def resolve_universe(name: str, custom_tickers: str | None = None) -> list[str]:
    """Resolve a menu selection to ticker symbols before agents are started."""
    name = name.lower().strip()
    if name == "custom":
        tickers = [ticker.strip().upper() for ticker in (custom_tickers or "").split(",")]
        tickers = [ticker for ticker in tickers if ticker]
        if not tickers:
            raise ValueError("Custom universe requires at least one ticker.")
        return list(dict.fromkeys(tickers))
    if name in SECTOR_UNIVERSES:
        return SECTOR_UNIVERSES[name].copy()
    if name in INDEX_UNIVERSES:
        return _resolve_index_universe(name)
    raise ValueError(f"Unknown universe '{name}'. Choose from: {', '.join(UNIVERSE_CHOICES)}")


def _cache_key(source: str, tickers: list[str], start: str, end: str, model_type: str) -> str:
    payload = "|".join([source, ",".join(sorted(tickers)), start, end, model_type])
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _cache_paths(key: str, fields: tuple[str, ...]) -> dict[str, Path]:
    return {field: CACHE_DIR / key / f"{field}.parquet" for field in fields}


def _clean_market_data(data: MarketData) -> MarketData:
    close = data["close"].sort_index().apply(pd.to_numeric, errors="coerce")
    valid = close.columns[close.isna().mean() < 0.10]
    if len(valid) < 2:
        raise RuntimeError("The selected dataset has fewer than two usable tickers.")
    close = close[valid].dropna()
    cleaned: MarketData = {"close": close}
    for field, frame in data.items():
        if field == "close":
            continue
        cleaned[field] = (
            frame.reindex(index=close.index, columns=close.columns)
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
        )
    return cleaned


def _download_yahoo(tickers: list[str], start: str, end: str, model_type: str) -> MarketData:
    logger.info("Preparing Yahoo dataset for %d tickers (%s to %s)", len(tickers), start, end)
    raw = pd.DataFrame()
    for attempt in range(1, 4):
        raw = yf.download(
            tickers,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            group_by="column",
            threads=True,
        )
        if not raw.empty:
            break
        wait = 5 * attempt
        logger.warning("Yahoo returned no data (attempt %d/3); retrying in %ds", attempt, wait)
        time.sleep(wait)
    if raw.empty:
        raise RuntimeError("Yahoo Finance returned no data.")

    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
        volume = raw["Volume"]
    else:
        close = raw[["Close"]].rename(columns={"Close": tickers[0]})
        volume = raw[["Volume"]].rename(columns={"Volume": tickers[0]})
    data: MarketData = {"close": close}
    if model_type == "price_volume":
        data["volume"] = volume
    return _clean_market_data(data)


def _load_csv(path: Path, tickers: list[str], model_type: str) -> MarketData:
    """Load long-form CSV columns: date,ticker,close[,volume]."""
    if not path.exists():
        raise FileNotFoundError(f"CSV dataset not found: {path}")
    raw = pd.read_csv(path)
    raw.columns = [str(column).strip().lower() for column in raw.columns]
    required = {"date", "ticker", "close"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")
    if model_type == "price_volume" and "volume" not in raw.columns:
        raise ValueError("price_volume models require a 'volume' CSV column.")

    raw["date"] = pd.to_datetime(raw["date"], errors="raise")
    raw["ticker"] = raw["ticker"].astype(str).str.upper()
    raw = raw[raw["ticker"].isin(tickers)]
    data: MarketData = {
        "close": raw.pivot(index="date", columns="ticker", values="close")
    }
    if model_type == "price_volume":
        data["volume"] = raw.pivot(index="date", columns="ticker", values="volume")
    return _clean_market_data(data)


def prepare_market_data(
    *,
    source: str,
    tickers: list[str],
    start: str,
    end: str,
    model_type: str,
    csv_path: str | None = None,
    use_cache: bool = True,
) -> MarketData:
    """Prepare selected data locally before the agent loop starts."""
    if model_type not in MODEL_TYPES:
        raise ValueError(f"Unknown model type '{model_type}'.")
    source = source.lower()
    fields = ("close", "volume") if model_type == "price_volume" else ("close",)
    source_id = str(Path(csv_path).resolve()) if source == "csv" and csv_path else source
    key = _cache_key(source_id, tickers, start, end, model_type)
    paths = _cache_paths(key, fields)
    if use_cache and all(path.exists() for path in paths.values()):
        logger.info("Loading prepared dataset from %s", paths["close"].parent)
        return {field: pd.read_parquet(path) for field, path in paths.items()}

    if source == "yahoo":
        data = _download_yahoo(tickers, start, end, model_type)
    elif source == "csv":
        if not csv_path:
            raise ValueError("CSV source requires --csv-path.")
        data = _load_csv(Path(csv_path), tickers, model_type)
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
        data = {field: frame.loc[start_ts:end_ts] for field, frame in data.items()}
        data = _clean_market_data(data)
    else:
        raise ValueError("Source must be 'yahoo' or 'csv'.")

    if use_cache:
        for field, path in paths.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            data[field].to_parquet(path)
    return data
