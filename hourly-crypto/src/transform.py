"""Combine raw per-symbol dumps into ONE modeling dataset (data/combined.parquet).

Pipeline stage: download -> **transform** -> train -> simulate

What it does
------------
1. Loads all ``data/raw/*_klines.parquet`` + ``*_funding.parquet`` files.
2. Derives ~34 new numeric features per symbol (returns, ranges, volatility,
   volume/taker flow, RSI/MACD, funding-rate features, BTC beta/correlation).
3. Adds cross-sectional features (hourly volume share, hourly return rank).
4. As-of merges funding rates (8h events -> hourly grid, backward fill).
5. Builds forward-looking targets: growth_future_24h/72h + binary target_24h.
6. Writes one combined parquet + prints dataset summary.

Usage:
    python -m src.transform
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from . import config

RET = "ret_1h"


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #
def load_klines() -> pd.DataFrame:
    files = sorted(config.RAW_DIR.glob("*_klines.parquet"))
    if not files:
        raise FileNotFoundError(f"no kline files in {config.RAW_DIR} - run src.download first")
    parts = []
    for f in files:
        sym = f.name.replace("_klines.parquet", "")
        df = pd.read_parquet(f)
        df["symbol"] = sym
        parts.append(df)
    df = pd.concat(parts, ignore_index=True)
    df["ts"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for c in ["open", "high", "low", "close", "volume", "quote_volume",
              "taker_buy_base", "taker_buy_quote"]:
        df[c] = df[c].astype("float64")
    df["trades"] = df["trades"].astype("int64")
    return df.sort_values(["symbol", "ts"], kind="stable").reset_index(drop=True)


def load_funding() -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for f in sorted(config.RAW_DIR.glob("*_funding.parquet")):
        sym = f.name.replace("_funding.parquet", "")
        df = pd.read_parquet(f)
        df["funding_ts"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
        df["fundingRate"] = df["fundingRate"].astype("float64")
        out[sym] = df[["funding_ts", "fundingRate"]].sort_values("funding_ts").reset_index(drop=True)
    return out


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def _market_frame(df: pd.DataFrame) -> pd.DataFrame:
    """BTC series used as the market factor for every other coin."""
    btc = df.loc[df["symbol"] == "BTCUSDT", ["ts", RET]].copy()
    btc = btc.rename(columns={RET: "mkt_ret_1h"})
    btc["mkt_ret_24h"] = btc["mkt_ret_1h"].rolling(24, min_periods=24).sum()
    btc["btc_rv_24"] = btc["mkt_ret_1h"].rolling(24, min_periods=24).std() * np.sqrt(24)
    return btc.reset_index(drop=True)


def symbol_features(s: pd.DataFrame, funding: pd.DataFrame | None,
                    market: pd.DataFrame) -> pd.DataFrame:
    """All per-symbol features for one symbol (rows sorted by ts, index kept)."""
    out = pd.DataFrame(index=s.index)
    c, o, h, l = s["close"], s["open"], s["high"], s["low"]
    vol, qv, trades = s["volume"], s["quote_volume"], s["trades"]

    # --- returns / candle shape -------------------------------------------- #
    out["ret_1h"] = np.log(c / c.shift(1))
    out["ret_24h"] = np.log(c / c.shift(24))
    out["ret_168h"] = np.log(c / c.shift(168))
    out["hl_range"] = (h - l) / c
    out["body"] = (c - o) / c
    out["body_sign"] = np.sign(c - o)

    # --- volume / flow ------------------------------------------------------ #
    out["vol_z24"] = (vol - vol.rolling(24).mean()) / vol.rolling(24).std()
    out["taker_buy_ratio"] = s["taker_buy_base"] / vol.replace(0.0, np.nan)
    vwap24 = qv.rolling(24).sum() / vol.rolling(24).sum().replace(0.0, np.nan)
    out["vwap_dev_24"] = c / vwap24 - 1.0
    out["trades_z24"] = (trades - trades.rolling(24).mean()) / trades.rolling(24).std()

    # --- volatility regime -------------------------------------------------- #
    out["rv_24"] = out["ret_1h"].rolling(24).std() * np.sqrt(24)
    out["rv_168"] = out["ret_1h"].rolling(168).std() * np.sqrt(24)
    out["vol_of_vol"] = out["rv_24"].rolling(168).std()
    out["ret_skew_24"] = out["ret_1h"].rolling(24).skew()

    # --- momentum / oscillinators ------------------------------------------ #
    out["rsi_14"] = _rsi(c, 14)
    out["rsi_48"] = _rsi(c, 48)
    e12, e26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    macd = e12 - e26
    out["macd_hist"] = (macd - macd.ewm(span=9, adjust=False).mean()) / c
    out["dist_high_168"] = c / c.rolling(168).max() - 1.0
    out["dist_low_168"] = c / c.rolling(168).min() - 1.0
    out["price_z_168"] = (c - c.rolling(168).mean()) / c.rolling(168).std()

    # --- calendar (cyclical encodings) -------------------------------------- #
    hour, dow = s["ts"].dt.hour, s["ts"].dt.dayofweek
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    out["log_days_listed"] = np.log1p((s["ts"] - s["ts"].iloc[0]).dt.total_seconds() / 86400)

    # --- funding-rate features (8h events as-of merged onto hourly grid) ---- #
    if funding is not None and len(funding):
        m = pd.merge_asof(s[["ts"]].reset_index(drop=True), funding,
                          left_on="ts", right_on="funding_ts", direction="backward")
        fr = m["fundingRate"]
        out["funding_rate"] = fr.to_numpy()
        out["funding_ma_24"] = fr.rolling(24, min_periods=1).mean().to_numpy()
        mu = fr.rolling(720, min_periods=6).mean()
        sd = fr.rolling(720, min_periods=6).std().replace(0.0, np.nan)
        out["funding_z_720"] = ((fr - mu) / sd).to_numpy()
        out["funding_pos"] = (fr > 0).astype("float64").to_numpy()
    else:
        out["funding_rate"] = np.nan
        out["funding_ma_24"] = np.nan
        out["funding_z_720"] = np.nan
        out["funding_pos"] = np.nan

    # --- market (BTC) features, as-of merged -------------------------------- #
    m = pd.merge_asof(s[["ts"]].reset_index(drop=True), market, left_on="ts",
                      right_on="ts", direction="backward")
    out["mkt_ret_24h"] = m["mkt_ret_24h"].to_numpy()
    out["btc_rv_24"] = m["btc_rv_24"].to_numpy()
    mkt_ret = pd.Series(m["mkt_ret_1h"].to_numpy(), index=s.index)
    out["corr_btc_168"] = out["ret_1h"].rolling(168).corr(mkt_ret)
    cov = out["ret_1h"].rolling(168).cov(mkt_ret)
    out["beta_btc_168"] = cov / mkt_ret.rolling(168).var()

    return out


# --------------------------------------------------------------------------- #
# Cross-asset macro (Yahoo Finance daily -> hourly as-of merged)
# --------------------------------------------------------------------------- #
def load_macro() -> pd.DataFrame | None:
    path = config.RAW_DIR / "macro.parquet"
    if not path.exists():
        return None
    m = pd.read_parquet(path).sort_values("ts").reset_index(drop=True)
    # a daily bar stamped day D is only known after the US close (~21:00 UTC
    # day D); make it available from D+1 00:00 UTC to avoid lookahead
    m["ts"] = m["ts"] + pd.Timedelta(days=1)
    m["spy_ret_5d"] = m["spy"].pct_change(5)
    m["spy_ret_20d"] = m["spy"].pct_change(20)
    m["vix_log"] = np.log(m["vix"])
    m["vix_chg_5d"] = m["vix"].pct_change(5)
    m["dxy_ret_10d"] = m["dxy"].pct_change(10)
    return m[["ts", "spy_ret_5d", "spy_ret_20d", "vix_log", "vix_chg_5d",
              "dxy_ret_10d"]]


# --------------------------------------------------------------------------- #
# Main transform
# --------------------------------------------------------------------------- #
def build() -> pd.DataFrame:
    t0 = time.time()
    df = load_klines()
    # market factor needs hourly returns before per-symbol features exist
    g_close = df.groupby("symbol")["close"]
    df["ret_1h"] = np.log(g_close.transform(lambda s: s / s.shift(1)))
    funding_map = load_funding()
    market = _market_frame(df)
    print(f"loaded {len(df):,} rows / {df['symbol'].nunique()} symbols "
          f"({time.time() - t0:.0f}s)")

    feats = [symbol_features(s, funding_map.get(sym), market)
             for sym, s in df.groupby("symbol", sort=False)]
    feat = pd.concat(feats)
    out = pd.concat([df[["symbol", "ts", "open", "high", "low", "close", "volume",
                         "quote_volume", "trades"]], feat], axis=1)

    # --- cross-sectional features (within the same hour) -------------------- #
    out["xs_quote_share"] = out.groupby("ts")["quote_volume"].transform(lambda s: s / s.sum())
    out["xs_ret_rank"] = out.groupby("ts")["ret_1h"].rank(pct=True)
    out["hour_bucket"] = (out["ts"].dt.hour // 6).astype("int8")

    # --- cross-asset macro, daily values as-of merged onto hourly grid ------ #
    macro = load_macro()
    if macro is not None:
        ts_u = out[["ts"]].drop_duplicates().sort_values("ts")
        macro["ts"] = macro["ts"].astype(ts_u["ts"].dtype)  # ms vs us res
        ts_u = pd.merge_asof(ts_u, macro, on="ts", direction="backward")
        out = out.merge(ts_u, on="ts", how="left")

    # --- targets ------------------------------------------------------------ #
    g = out.groupby("symbol")["close"]
    out["growth_future_24h"] = g.shift(-config.HOLD_HOURS) / out["close"]
    out["growth_future_72h"] = g.shift(-72) / out["close"]
    valid = out["growth_future_24h"].notna()
    out["target_24h"] = np.where(
        valid, (out["growth_future_24h"] > config.TARGET_THRESHOLD).astype("float64"),
        np.nan)

    print(f"combined: {len(out):,} rows x {len(out.columns)} cols "
          f"({time.time() - t0:.0f}s)")
    return out


TO_PREDICT = "growth_future_24h"
TARGET = "target_24h"
NUMERIC_FEATURES = [
    # returns / candle shape
    "ret_1h", "ret_24h", "ret_168h", "hl_range", "body", "body_sign",
    # volume / flow
    "vol_z24", "taker_buy_ratio", "vwap_dev_24", "trades_z24",
    # volatility
    "rv_24", "rv_168", "vol_of_vol", "ret_skew_24",
    # momentum
    "rsi_14", "rsi_48", "macd_hist", "dist_high_168", "dist_low_168",
    "price_z_168",
    # calendar
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "log_days_listed",
    # funding (new data source)
    "funding_rate", "funding_ma_24", "funding_z_720", "funding_pos",
    # market (BTC factor)
    "mkt_ret_24h", "btc_rv_24", "corr_btc_168", "beta_btc_168",
    # cross-sectional
    "xs_quote_share", "xs_ret_rank",
    # cross-asset macro (Yahoo Finance - lecture data source)
    "spy_ret_5d", "spy_ret_20d", "vix_log", "vix_chg_5d", "dxy_ret_10d",
]
DUMMY_FEATURES = ["symbol", "hour_bucket"]


def main() -> None:
    df = build()
    path = config.DATA_DIR / "combined.parquet"
    df.to_parquet(path, index=False)

    train_mask = df["ts"] < pd.Timestamp(config.TRAIN_END, tz="UTC")
    y_train = df.loc[train_mask, TARGET]
    print(f"\nsaved -> {path}")
    print(f"window: {df['ts'].min()} .. {df['ts'].max()}")
    print(f"features: {len(NUMERIC_FEATURES)} numeric + dummies {DUMMY_FEATURES}")
    print(f"target {TARGET} (growth>{config.TARGET_THRESHOLD}): "
          f"train class share = {y_train.mean():.3f}, "
          f"all = {df[TARGET].mean():.3f}")
    print(f"rows with target = {df[TARGET].notna().sum():,} / {len(df):,}")


if __name__ == "__main__":
    main()
