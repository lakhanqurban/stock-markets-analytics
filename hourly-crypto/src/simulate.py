"""Trading simulations driven by model predictions (vector + exact).

Design notes (debugging history)
--------------------------------
v1 (hourly rebalance) lost money: the signal is positive-EV
(+0.57% per 24h) but hourly re-weighting churned ~968 full-equivalent
rebalances -> fees (1.45) dwarfed gross PnL (0.63). Fix: decisions are made
once a day (00:00 UTC) and held for 24h -> turnover collapses to entry/exit
events only.

Strategies (costs = FEE_BPS + SLIPPAGE_BPS per side):
    sim1_rf_long       vector, daily long-only, equal weight among signals
    sim2_rf_probasize  vector, daily, position size = model probability
    sim3_rf_longs      vector, daily dollar-neutral long/short
    sim4_rf_advanced   exact (iterative) simulation WITH reinvestment:
                       top-5 concurrent, probability sizing, optional
                       stop-loss / max-hold / vol-regime filter (selected
                       on the VALIDATION set, never on test)
    benchmark_btc      buy & hold BTCUSDT over the same window.

Signals decided at hour t are executed at the close of t+1 (one-hour shift).

Usage:
    python -m src.simulate --split valid     # strategy selection
    python -m src.simulate --split test      # final report
"""
from __future__ import annotations

import argparse
import itertools
import json

import numpy as np
import pandas as pd

from . import config, transform

COST = (config.FEE_BPS + config.SLIPPAGE_BPS) / 1e4   # one-way, decimal
HOURS_PER_YEAR = 24 * 365.25
CONFIG_PATH = config.OUTPUT_DIR / "strategy_config.json"


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_preds(split: str) -> pd.DataFrame:
    df = pd.read_parquet(config.OUTPUT_DIR / f"predictions_{split}.parquet")
    btc = pd.read_parquet(config.DATA_DIR / "combined.parquet",
                          columns=["symbol", "ts", "btc_rv_24"])
    btc = btc.loc[btc["symbol"] == "BTCUSDT", ["ts", "btc_rv_24"]]
    df = df.merge(btc, on="ts", how="left")
    return df.sort_values(["ts", "symbol"]).reset_index(drop=True)


def wide(df: pd.DataFrame, col: str) -> pd.DataFrame:
    return df.pivot(index="ts", columns="symbol", values=col)


# --------------------------------------------------------------------------- #
# Daily-held weights (decide at 00:00, hold 24h, shift 1h for execution)
# --------------------------------------------------------------------------- #
def _daily_weights(day_proba: pd.DataFrame, thr: float, mode: str,
                   rv_min: float | None) -> pd.DataFrame:
    if mode == "long":
        act = day_proba >= thr
        n = act.sum(axis=1)
        return act.astype("float64").div(n.replace(0, np.nan), axis=0).fillna(0.0)
    if mode == "probasize":
        act = day_proba.where(day_proba >= thr, 0.0)
        s = act.sum(axis=1)
        return act.div(s.replace(0, np.nan), axis=0).fillna(0.0)
    if mode == "longshort":
        long = _daily_weights(day_proba, thr, "long", None)
        short = _daily_weights(1.0 - day_proba, 1.0 - thr, "long", None)
        return long - short
    if mode == "advanced":
        act = day_proba.where(day_proba >= thr, 0.0)
        keep = act.apply(
            lambda row: row.nlargest(config.MAX_CONCURRENT_POSITIONS).index,
            axis=1)
        mask = pd.DataFrame(0.0, index=act.index, columns=act.columns)
        for i, cols in enumerate(keep):
            mask.iloc[i, [act.columns.get_loc(c) for c in cols]] = 1.0
        act = act * mask
        s = act.sum(axis=1)
        return act.div(s.replace(0, np.nan), axis=0).fillna(0.0)
    raise KeyError(mode)


def daily_held_weights(proba_w: pd.DataFrame, thr: float, mode: str,
                       rv: pd.Series | None = None,
                       rv_min: float | None = None) -> pd.DataFrame:
    """Sample decisions at 00:00 UTC, hold for 24h, execute 1h later."""
    day_mask = proba_w.index.hour == 0
    day_proba = proba_w.loc[day_mask]
    w = _daily_weights(day_proba, thr, mode, rv_min)
    if mode == "advanced" and rv is not None and rv_min is not None:
        ok = rv.reindex(day_proba.index) >= rv_min
        w = w.mul(ok.astype("float64"), axis=0).fillna(0.0)
    w = w.reindex(proba_w.index).ffill().shift(1).fillna(0.0)
    return w


def vector_sim(df: pd.DataFrame, prob_col: str, thr: float, mode: str,
               rv_min: float | None = None
               ) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    proba_w = wide(df, prob_col)
    close_w = wide(df, "close")
    ret = close_w.pct_change()
    rv = df.groupby("ts")["btc_rv_24"].first()
    w = daily_held_weights(proba_w, thr, mode, rv, rv_min)

    gross = (w * ret).sum(axis=1)
    fees = w.diff().abs().sum(axis=1) * COST
    net = (gross - fees).fillna(0.0)
    equity = (1.0 + net).cumprod()
    contrib = (w * ret).stack()
    contrib.index.names = ["ts", "symbol"]
    return equity, contrib, w


# --------------------------------------------------------------------------- #
# Exact simulation with reinvestment
# --------------------------------------------------------------------------- #
def exact_sim(df: pd.DataFrame, prob_col: str, thr: float,
              regime_min: float | None = None,
              stop_loss: float | None = None,
              max_hold: int | None = None) -> tuple[pd.Series, pd.DataFrame]:
    """Iterative simulation with actual positions, fees and reinvestment.

    Mirrors the vector strategies: decisions are taken from the 00:00 UTC
    signal and executed at 01:00 (rebalance to target weights), positions
    are held in between. Hourly stop-loss checks force liquidation; a
    stopped symbol may not be re-bought until its signal resets below thr.
    """
    by_hour: dict[pd.Timestamp, list] = {}
    for (ts, sym), r in df.groupby(["ts", "symbol"], sort=True):
        by_hour.setdefault(ts, []).append(
            (sym, r[prob_col].iloc[0], r["close"].iloc[0], r["btc_rv_24"].iloc[0]))

    hours = sorted(by_hour)
    cash = config.START_CAPITAL
    positions: dict[str, dict] = {}          # sym -> {qty, entry, entry_i}
    blocked: set[str] = set()                # stopped, awaiting signal reset
    equity_curve, trades = [], []
    prev_probs: dict[str, float] = {}
    prev_hour_of_day: int | None = None

    def _close_position(sym: str, px: float, ts, reason: str) -> None:
        nonlocal cash
        pos = positions.pop(sym)
        proceeds = pos["qty"] * px * (1 - COST)
        cash += proceeds
        trades.append({
            "symbol": sym, "entry_ts": pos["entry_ts"], "exit_ts": ts,
            "entry_px": pos["entry"], "exit_px": px,
            "pnl": proceeds - pos["qty"] * pos["entry"] * (1 + COST),
            "reason": reason})

    for i, ts in enumerate(hours):
        hour_rows = by_hour[ts]
        closes = {sym: c for sym, _, c, _ in hour_rows}
        hour_of_day = ts.hour

        # 1) hourly stop-loss checks
        if stop_loss is not None:
            for sym in list(positions):
                px = closes.get(sym)
                if (px is not None
                        and px <= positions[sym]["entry"] * (1 - stop_loss)):
                    _close_position(sym, px, ts, "stop_loss")
                    blocked.add(sym)

        # unblock symbols whose signal has reset
        blocked = {s for s in blocked if prev_probs.get(s, 0.0) >= thr}

        # 2) daily rebalance: decision from 00:00 signal, executed at 01:00
        if prev_hour_of_day == 0:
            regime_ok = (regime_min is None) or all(
                rv >= regime_min for _, _, _, rv in hour_rows)
            active = ({s: p for s, p in prev_probs.items()
                       if p >= thr and s in closes and s not in blocked}
                      if regime_ok else {})
            if active and cash + sum(
                    p["qty"] * closes.get(s, p["entry"])
                    for s, p in positions.items()) > 1.0:
                top = sorted(active.items(), key=lambda kv: -kv[1])[
                    :config.MAX_CONCURRENT_POSITIONS]
                total = sum(p for _, p in top)
                targets = {s: p / total for s, p in top}   # full utilisation
            else:
                targets = {}                                # go flat

            eq = cash + sum(p["qty"] * closes.get(s, p["entry"])
                            for s, p in positions.items())
            # force exits first (max_hold), then trade the diffs
            for sym in list(positions):
                if max_hold is not None and i - positions[sym]["entry_i"] >= max_hold:
                    px = closes.get(sym, positions[sym]["entry"])
                    _close_position(sym, px, ts, "max_hold")
            for sym in set(positions) | set(targets):
                px = closes.get(sym)
                if px is None:
                    continue
                want = eq * targets.get(sym, 0.0) * (1 - COST) / px
                have = positions[sym]["qty"] if sym in positions else 0.0
                delta = want - have
                if abs(delta) * px < 1.0:                   # skip dust trades
                    continue
                cash -= delta * px                          # +ve buy / -ve sell
                cash -= abs(delta) * px * COST
                if delta > 0 and sym not in positions:
                    positions[sym] = {"qty": want, "entry": px, "entry_ts": ts,
                                      "entry_i": i}
                elif delta > 0:
                    pos = positions[sym]
                    # weighted average cost basis for later PnL attribution
                    new_qty = pos["qty"] + delta      # == want
                    pos["entry"] = (pos["qty"] * pos["entry"] + delta * px) / new_qty
                    pos["qty"] = new_qty
                elif sym in positions:
                    if want <= 0:
                        pos = positions.pop(sym)
                        trades.append({
                            "symbol": sym, "entry_ts": pos["entry_ts"],
                            "exit_ts": ts, "entry_px": pos["entry"],
                            "exit_px": px,
                            "pnl": (pos["qty"] * px * (1 - COST)
                                    - pos["qty"] * pos["entry"] * (1 + COST)),
                            "reason": "signal_off"})
                    else:
                        positions[sym]["qty"] = want        # partial trim

        # 3) mark to market
        eq = cash + sum(p["qty"] * closes.get(s, p["entry"])
                        for s, p in positions.items())
        equity_curve.append((ts, eq))
        prev_probs = {sym: p for sym, p, _, _ in hour_rows}
        prev_hour_of_day = hour_of_day

    if positions and equity_curve:
        last_ts = hours[-1]
        closes = {sym: c for sym, _, c, _ in by_hour[last_ts]}
        for sym in list(positions):
            _close_position(sym, closes.get(sym, positions[sym]["entry"]),
                            last_ts, "end_of_data")
        equity_curve[-1] = (last_ts, cash)

    eq = pd.Series([e for _, e in equity_curve],
                   index=pd.DatetimeIndex([t for t, _ in equity_curve], name="ts"))
    return eq, pd.DataFrame(trades)


# --------------------------------------------------------------------------- #
# Performance metrics
# --------------------------------------------------------------------------- #
def perf_stats(equity: pd.Series, n_trades: int = 0) -> dict:
    equity = equity.dropna()
    if len(equity) < 2:
        return {}
    years = len(equity) / HOURS_PER_YEAR
    total = equity.iloc[-1] / equity.iloc[0]
    rets = equity.pct_change().dropna()
    return {
        "total_return": round(float(total - 1), 4),
        "cagr": round(float(total ** (1 / years) - 1), 4),
        "sharpe": round(float(rets.mean() / rets.std() * np.sqrt(HOURS_PER_YEAR))
                        if rets.std() > 0 else 0.0, 3),
        "max_drawdown": round(float((equity / equity.cummax() - 1).min()), 4),
        "n_trades": int(n_trades),
        "trades_per_day": round(n_trades / max(len(equity) / 24, 1e-9), 2),
    }


# --------------------------------------------------------------------------- #
# Exact-sim configuration selection on VALID (never on test)
# --------------------------------------------------------------------------- #
def select_config(df_valid: pd.DataFrame, prob_col: str, thr: float) -> dict:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            return json.load(f)
    rv_min_grid = [None, float(df_valid["btc_rv_24"].dropna().quantile(1 / 3))]
    rows = []
    for stop, hold, regime in itertools.product([None, 0.15], [None, 48],
                                                rv_min_grid):
        eq, tr = exact_sim(df_valid, prob_col, thr, regime_min=regime,
                           stop_loss=stop, max_hold=hold)
        st = perf_stats(eq, len(tr))
        rows.append({"stop_loss": stop, "max_hold": hold,
                     "regime_min": regime, **st})
        print(f"  valid grid stop={stop} hold={hold} regime={regime}: {st}")
    tab = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
    best = tab.iloc[0]
    cfg = {
        "stop_loss": None if pd.isna(best["stop_loss"]) else float(best["stop_loss"]),
        "max_hold": None if pd.isna(best["max_hold"]) else int(best["max_hold"]),
        "regime_min": None if pd.isna(best["regime_min"]) else float(best["regime_min"]),
        "selected_on": "valid", "valid_sharpe": float(best["sharpe"]),
    }
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"selected exact-sim config: {cfg}")
    return cfg


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["valid", "test"])
    ap.add_argument("--model", default="rf")
    ap.add_argument("--skip-select", action="store_true",
                    help="skip valid-set config selection (use saved config)")
    args = ap.parse_args()

    df = load_preds(args.split)
    prob_col = f"proba_{args.model}"
    thr_col = f"pred_{args.model}"
    with open(config.OUTPUT_DIR / "metrics.json") as f:
        thr = json.load(f)["models"][args.model]["threshold"]

    if args.split == "valid" and not args.skip_select:
        print("selecting exact-sim config on VALID...")
        cfg = select_config(load_preds("valid"), prob_col, thr)
    else:
        cfg = (json.loads(CONFIG_PATH.read_text()) if CONFIG_PATH.exists()
               else {"stop_loss": None, "max_hold": None, "regime_min": None})
    cfg = {k: cfg.get(k) for k in ("stop_loss", "max_hold", "regime_min")}

    results, equities, contribs = {}, {}, {}
    strategies = {
        "sim1_rf_long": dict(thr=thr, mode="long"),
        "sim2_rf_probasize": dict(thr=thr, mode="probasize"),
        "sim3_rf_longs": dict(thr=thr, mode="longshort"),
        "sim4_rf_advanced": dict(thr=thr, mode="advanced"),
    }
    for name, kw in strategies.items():
        eq, contrib, w = vector_sim(df, prob_col, **kw)
        entered = (w.abs() > 0) & (w.shift(1).abs() == 0)
        n_trades = int(entered.to_numpy().sum())
        results[name] = perf_stats(eq, n_trades)
        equities[name] = eq
        contribs[name] = contrib
        print(f"{name}: {results[name]}")

    eq_ex, trades = exact_sim(df, prob_col, thr, **cfg)
    results["sim5_rf_exact"] = perf_stats(eq_ex, len(trades))
    equities["sim5_rf_exact"] = eq_ex
    print(f"sim5_rf_exact {cfg}: {results['sim5_rf_exact']}")

    btc = df[df["symbol"] == "BTCUSDT"].set_index("ts")["close"]
    bench = btc / btc.iloc[0]
    results["benchmark_btc"] = perf_stats(bench, 1)
    equities["benchmark_btc"] = bench
    print(f"benchmark_btc: {results['benchmark_btc']}")

    # ------------------------------------------------------------------ #
    # Deep analysis: precision by ticker
    # ------------------------------------------------------------------ #
    y = df[transform.TARGET]
    sig = df[thr_col] == 1
    ticker = df.assign(_y=y, _s=sig).groupby("symbol").apply(
        lambda g: pd.Series({
            "n_rows": len(g),
            "base_rate": g["_y"].mean(),
            "signal_share": g["_s"].mean(),
            "precision": (g.loc[g["_s"], "_y"].mean() if g["_s"].any()
                          else np.nan),
        }), include_groups=False)
    ticker.to_csv(config.OUTPUT_DIR / f"ticker_precision_{args.split}.csv")

    # ------------------------------------------------------------------ #
    # Persist
    # ------------------------------------------------------------------ #
    pd.DataFrame(results).T.to_csv(
        config.OUTPUT_DIR / f"simulation_results_{args.split}.csv")
    pd.DataFrame(equities).to_parquet(
        config.OUTPUT_DIR / f"equity_curves_{args.split}.parquet")
    trades.to_csv(config.OUTPUT_DIR / f"trades_exact_{args.split}.csv",
                  index=False)
    out = df[["ts", "symbol", transform.TARGET, prob_col, thr_col]].copy()
    for name, contrib in contribs.items():
        out = out.merge(contrib.rename(f"{name}_r").reset_index(),
                        on=["ts", "symbol"], how="left")
    out.to_parquet(
        config.OUTPUT_DIR / f"predictions_with_sims_{args.split}.parquet",
        index=False)
    print(f"\nwrote simulation_results_{args.split}.csv, equity curves, "
          f"trades, ticker precision")


if __name__ == "__main__":
    main()
