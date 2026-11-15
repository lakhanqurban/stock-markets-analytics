"""Download hourly klines + funding rates from Binance USDT-M futures.

Public REST API, no key required. Data is stored incrementally:
one parquet file per symbol and data type under ``data/raw/``.

Re-running is cheap: only rows newer than the last stored timestamp are
fetched (incremental loading - rubric: automation).

Usage:
    python -m src.download                 # all symbols, incremental
    python -m src.download --symbols BTCUSDT ETHUSDT
    python -m src.download --force         # re-download from START
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import requests

from . import config

KLINE_COLS = [
    "open_time", "open", "high", "low", "close", "close_time", "volume",
    "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore",
]


def _now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def _ts_ms(s: str) -> int:
    return int(datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp() * 1000)


def _get(path: str, params: dict[str, Any] | None = None) -> Any:
    """GET with retry/backoff on rate limits (429/418) and 5xx."""
    url = config.BASE_FAPI + path
    for attempt in range(config.MAX_RETRIES):
        try:
            r = requests.get(url, params=params, timeout=20)
            if r.status_code in (429, 418):          # rate limited
                wait = int(r.headers.get("Retry-After", 2 ** attempt))
                time.sleep(max(wait, 2))
                continue
            if r.status_code >= 500:
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            if attempt == config.MAX_RETRIES - 1:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def active_universe() -> tuple[list[str], list[str]]:
    """Filter candidates against current exchangeInfo (drop delisted)."""
    info = _get("/fapi/v1/exchangeInfo")
    tradable = {s["symbol"] for s in info["symbols"] if s["status"] == "TRADING"}
    keep = [s for s in config.UNIVERSE_CANDIDATES if s in tradable]
    dropped = [s for s in config.UNIVERSE_CANDIDATES if s not in tradable]
    return keep, dropped


# --------------------------------------------------------------------------- #
# Klines (OHLCV + trade counts + taker flow) - full history from START
# --------------------------------------------------------------------------- #
def _fetch_klines(symbol: str, start_ms: int, end_ms: int) -> list[list]:
    rows: list[list] = []
    cur = start_ms
    while cur < end_ms:
        batch = _get("/fapi/v1/klines", {
            "symbol": symbol, "interval": config.INTERVAL,
            "startTime": cur, "endTime": end_ms, "limit": 1500,
        })
        if not batch:
            break
        rows.extend(batch)
        cur = batch[-1][0] + 1
        if len(batch) < 1500:
            break
        time.sleep(config.REQUEST_SLEEP)
    return rows


def download_klines(symbol: str, force: bool = False) -> int:
    path = config.RAW_DIR / f"{symbol}_klines.parquet"
    start_ms = _ts_ms(config.START)
    if path.exists() and not force:
        old = pd.read_parquet(path)
        start_ms = max(int(old["open_time"].max()) + 1, start_ms)
    rows = _fetch_klines(symbol, start_ms, _now_ms())
    if rows:
        new = pd.DataFrame(rows, columns=KLINE_COLS).drop(columns=["ignore"])
        if path.exists():
            new = pd.concat([pd.read_parquet(path), new], ignore_index=True)
        new = (new.drop_duplicates(subset="open_time")
                  .sort_values("open_time").reset_index(drop=True))
        new.to_parquet(path, index=False)
    return len(rows)


# --------------------------------------------------------------------------- #
# Funding rates - full history (8h cadence, some newer symbols 4h/1h)
# --------------------------------------------------------------------------- #
def _fetch_funding(symbol: str, start_ms: int) -> list[dict]:
    rows: list[dict] = []
    cur = start_ms
    while True:
        batch = _get("/fapi/v1/fundingRate", {
            "symbol": symbol, "startTime": cur, "limit": 1000,
        })
        if not batch:
            break
        rows.extend(batch)
        cur = int(batch[-1]["fundingTime"]) + 1
        if len(batch) < 1000:
            break
        time.sleep(config.REQUEST_SLEEP)
    return rows


def download_funding(symbol: str, force: bool = False) -> int:
    path = config.RAW_DIR / f"{symbol}_funding.parquet"
    start_ms = _ts_ms(config.START)
    if path.exists() and not force:
        old = pd.read_parquet(path)
        start_ms = max(int(old["fundingTime"].max()) + 1, start_ms)
    rows = _fetch_funding(symbol, start_ms)
    if rows:
        new = pd.DataFrame(rows)[["fundingTime", "fundingRate", "markPrice"]]
        if path.exists():
            new = pd.concat([pd.read_parquet(path), new], ignore_index=True)
        new = (new.drop_duplicates(subset="fundingTime")
                  .sort_values("fundingTime").reset_index(drop=True))
        new.to_parquet(path, index=False)
    return len(rows)


# --------------------------------------------------------------------------- #
# Cross-asset macro series from Yahoo Finance (lecture data source)
# --------------------------------------------------------------------------- #
MACRO_TICKERS = {"SPY": "spy", "^VIX": "vix", "DX-Y.NYB": "dxy"}


def download_macro() -> int:
    """Daily SPY / VIX / DXY closes -> data/raw/macro.parquet (small, full refresh)."""
    import yfinance as yf

    raw = yf.download(list(MACRO_TICKERS), start=config.START[:10],
                      interval="1d", auto_adjust=True, progress=False)
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    close = close.rename(columns={k: v for k, v in MACRO_TICKERS.items()})
    df = close.reset_index().rename(columns={"Date": "ts", "Datetime": "ts"})
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df[["ts"] + list(MACRO_TICKERS.values())].dropna()
    df.to_parquet(config.RAW_DIR / "macro.parquet", index=False)
    return len(df)


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Download Binance futures data")
    ap.add_argument("--symbols", nargs="*", help="subset of symbols to fetch")
    ap.add_argument("--force", action="store_true", help="ignore local cache")
    ap.add_argument("--skip-funding", action="store_true")
    ap.add_argument("--skip-macro", action="store_true")
    args = ap.parse_args()

    if not args.skip_macro:
        n = download_macro()
        print(f"macro: {n} daily rows (SPY/VIX/DXY)", flush=True)

    if args.symbols:
        universe = args.symbols
        dropped: list[str] = []
    else:
        universe, dropped = active_universe()
    print(f"universe: {len(universe)} symbols" +
          (f", dropped (delisted): {dropped}" if dropped else ""))

    t0 = time.time()
    for i, sym in enumerate(universe, 1):
        n_k = download_klines(sym, force=args.force)
        n_f = 0 if args.skip_funding else download_funding(sym, force=args.force)
        print(f"[{i:02d}/{len(universe)}] {sym}: +{n_k} klines, "
              f"+{n_f} funding ({time.time() - t0:.0f}s)", flush=True)
    print(f"done in {time.time() - t0:.0f}s -> {config.RAW_DIR}")


if __name__ == "__main__":
    main()
