"""Turn outputs/latest_trades.csv into broker orders (Binance USDT-M futures).

Dry run by default - builds orders, prints them, writes outputs/orders_latest.json:

    python -m src.broker                      # preview only (safe, no keys needed)
    python -m src.broker --live --testnet     # place orders on the Binance futures testnet
    python -m src.broker --live --mainnet     # REAL MONEY - must be stated explicitly

Credentials come from env vars BINANCE_API_KEY / BINANCE_API_SECRET
(never hard-coded, never printed). This module is a *bonus* integration
stub - the project itself trades only in simulation.
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import time
from datetime import datetime, timezone
from urllib.parse import urlencode

import pandas as pd
import requests

from . import config

FAPI = {
    "mainnet": "https://fapi.binance.com",
    "testnet": "https://testnet.binancefuture.com",
}


def load_trades() -> pd.DataFrame:
    """Newest hour's BUY signals written by src/pipeline.py."""
    path = config.OUTPUT_DIR / "latest_trades.csv"
    df = pd.read_csv(path) if path.exists() else pd.DataFrame()
    return df


def build_orders(trades: pd.DataFrame, budget_usdt: float) -> list[dict]:
    """One market order per signal, fixed USDT budget (quoteOrderQty)."""
    orders = []
    cols = ["symbol", "action", "signal_probability"]
    for sym, action, prob in trades[cols].itertuples(index=False):
        orders.append({
            "symbol": sym,
            "side": str(action).upper(),
            "type": "MARKET",
            "quoteOrderQty": round(budget_usdt, 2),
            "signal_probability": round(float(prob), 4),
        })
    return orders


def _sign(query: str, secret: str) -> str:
    return hmac.new(secret.encode(), query.encode(),
                    hashlib.sha256).hexdigest()


def place_order(order: dict, key: str, secret: str, base_url: str) -> dict:
    """POST one signed order to /fapi/v1/order (live path)."""
    params = {k: order[k] for k in ("symbol", "side", "type", "quoteOrderQty")}
    params["recvWindow"] = 5000
    params["timestamp"] = int(time.time() * 1000)
    query = urlencode(params)
    query += "&signature=" + _sign(query, secret)
    r = requests.post(f"{base_url}/fapi/v1/order?{query}",
                      headers={"X-MBX-APIKEY": key}, timeout=30)
    return {"http": r.status_code, "response": r.json()}


def preview(budget_usdt: float | None = None) -> pd.DataFrame:
    """Build + persist + print the order list (always a dry run)."""
    budget = budget_usdt or config.START_CAPITAL / config.MAX_CONCURRENT_POSITIONS
    trades = load_trades()
    orders = build_orders(trades, budget)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = config.OUTPUT_DIR / "orders_latest.json"
    payload = {"created_utc": stamp, "budget_usdt": budget, "orders": orders}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[broker] {len(orders)} order(s), budget {budget:.0f} USDT each "
          f"-> {path.name} (dry run)")
    for o in orders:
        print("   ", json.dumps(o))
    if not orders:
        print("[broker] no open signals this hour - nothing to do")
        return pd.DataFrame(columns=["symbol", "side", "type",
                                     "quoteOrderQty", "signal_probability"])
    return pd.DataFrame(orders)


def main() -> None:
    ap = argparse.ArgumentParser(description="latest_trades.csv -> orders")
    ap.add_argument("--live", action="store_true",
                    help="actually place the orders (default: dry run)")
    ap.add_argument("--testnet", action="store_true",
                    help="with --live: use the futures testnet")
    ap.add_argument("--mainnet", action="store_true",
                    help="with --live: use the real futures market (real money)")
    ap.add_argument("--budget", type=float, default=None,
                    help="USDT per order (default: capital / max positions)")
    args = ap.parse_args()

    orders = preview(args.budget)
    if not args.live:
        return
    if not args.testnet and not args.mainnet:
        raise SystemExit("--live requires --testnet or --mainnet")

    key = os.environ.get("BINANCE_API_KEY")
    secret = os.environ.get("BINANCE_API_SECRET")
    if not key or not secret:
        raise SystemExit("set BINANCE_API_KEY / BINANCE_API_SECRET first")
    base = FAPI["testnet" if args.testnet else "mainnet"]
    print(f"[broker] placing {len(orders)} order(s) on {base}")

    results = [place_order(o, key, secret, base) for o in orders]
    out = config.OUTPUT_DIR / "orders_placed.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[broker] results -> {out}")


if __name__ == "__main__":
    main()
