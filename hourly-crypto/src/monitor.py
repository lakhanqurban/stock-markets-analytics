"""Monitoring: one-command health summary of data, model and portfolio.

    python -m src.monitor              # print summary + write outputs/monitor.html
    python -m src.monitor --telegram   # also push the summary to Telegram

To make it live: put it in cron right after ``src.pipeline`` and export
``TELEGRAM_BOT_TOKEN`` + ``TELEGRAM_CHAT_ID`` (create a bot with @BotFather
and get your chat id from ``getUpdates``).
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

import pandas as pd
import requests

from . import config
from .simulate import perf_stats


def summary() -> dict:
    """Latest health numbers: data freshness, model quality, portfolio."""
    with open(config.OUTPUT_DIR / "metrics.json") as f:
        m = json.load(f)["models"]["rf"]

    eq = pd.read_parquet(config.OUTPUT_DIR / "equity_curves_test.parquet")
    pred = pd.read_parquet(config.OUTPUT_DIR / "predictions_test.parquet",
                           columns=["ts"])
    trades = pd.read_csv(config.OUTPUT_DIR / "latest_trades.csv")

    strat = perf_stats(eq["sim5_rf_exact"])
    bench = perf_stats(eq["benchmark_btc"])
    return {
        "updated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
        "data_latest_hour": str(pred["ts"].max()),
        "rf_test_auc": round(m["test_auc"], 4),
        "rf_threshold": m["threshold"],
        "rf_test_precision": round(m["test_precision"], 4),
        "open_signals_now": len(trades),
        "test_total_return": round(strat["total_return"], 4),
        "test_sharpe": round(strat["sharpe"], 2),
        "test_max_drawdown": round(strat["max_drawdown"], 4),
        "benchmark_total_return": round(bench["total_return"], 4),
    }


def write_html(s: dict) -> str:
    """Tiny static dashboard snapshot (open outputs/monitor.html in a browser;
    make it live by regenerating it from cron after every pipeline run)."""
    rows = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in s.items())
    html = (
        "<html><head><title>hourly-crypto monitor</title>"
        "<style>body{font-family:sans-serif;margin:2em}"
        "td{padding:4px 12px;border-bottom:1px solid #eee}</style></head>"
        f"<body><h2>hourly-crypto monitor @ {s['updated_utc']} UTC</h2>"
        f"<table>{rows}</table></body></html>"
    )
    path = config.OUTPUT_DIR / "monitor.html"
    path.write_text(html, encoding="utf-8")
    return str(path)


def send_telegram(text: str) -> bool:
    """Push *text* to Telegram; returns False when not configured."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat:
        print("[monitor] TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set - "
              "skipping Telegram")
        return False
    r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                      json={"chat_id": chat, "text": text}, timeout=30)
    print(f"[monitor] telegram HTTP {r.status_code}")
    return r.ok


def main() -> None:
    ap = argparse.ArgumentParser(description="Portfolio/model health summary")
    ap.add_argument("--telegram", action="store_true",
                    help="also send the summary to Telegram")
    args = ap.parse_args()

    s = summary()
    width = max(len(k) for k in s)
    for k, v in s.items():
        print(f"{k:>{width}} : {v}")
    print(f"\n[monitor] wrote {write_html(s)}")
    if args.telegram:
        send_telegram(
            "hourly-crypto: "
            + ", ".join(f"{k}={v}" for k, v in s.items()))


if __name__ == "__main__":
    main()
