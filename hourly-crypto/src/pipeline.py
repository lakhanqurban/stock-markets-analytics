"""End-to-end workflow orchestrator (the notebook calls into this module).

Steps: download -> transform -> train -> simulate (valid, then test) ->
latest trade list. Two operating modes:

    --source file      use local parquet files only (fast, reproducible,
                       no API calls)  [default]
    --source download  first pull latest data from Binance/Yahoo APIs
                       (incremental - only missing rows are fetched)

Usage:
    python -m src.pipeline                     # file mode, full run
    python -m src.pipeline --source download   # update data first
    python -m src.pipeline --skip-train        # reuse fitted models

Cron (daily, 01:15 UTC): see README.md
"""
from __future__ import annotations

import argparse
import sys
import time

import pandas as pd

from . import config


def _run(main_fn, argv: list[str]) -> None:
    """Call a module main() that parses sys.argv, with a custom argv."""
    old = sys.argv
    sys.argv = argv
    try:
        main_fn()
    finally:
        sys.argv = old


def latest_trades() -> pd.DataFrame:
    """Most recent hour's BUY signals (top by probability) for execution."""
    p = pd.read_parquet(config.OUTPUT_DIR / "predictions_test.parquet")
    last_ts = p["ts"].max()
    sel = p[(p["ts"] == last_ts) & (p["pred_rf"] == 1)].nlargest(
        5, "proba_rf")[["ts", "symbol", "proba_rf"]].copy()
    sel = sel.rename(columns={"proba_rf": "signal_probability"})
    sel["action"] = "BUY"
    sel.to_csv(config.OUTPUT_DIR / "latest_trades.csv", index=False)
    print(f"\nlatest trades @ {last_ts} "
          f"({len(sel)} signals) -> outputs/latest_trades.csv")
    if not sel.empty:
        print(sel.to_string(index=False))
    return sel


def run(source: str = "file", skip_train: bool = False) -> None:
    t0 = time.time()
    from . import download, simulate, transform, train

    if source == "download":
        _run(download.main, ["src.download"])
    else:
        print("[pipeline] source=file: using local data (no API calls)")

    _run(transform.main, ["src.transform"])

    if skip_train:
        print("[pipeline] --skip-train: reusing models/ + predictions")
    else:
        _run(train.main, ["src.train"])

    # strategy config selection happens on valid; test run reuses it
    _run(simulate.main, ["src.simulate", "--split", "valid"])
    _run(simulate.main, ["src.simulate", "--split", "test", "--skip-select"])

    latest_trades()

    res = pd.read_csv(config.OUTPUT_DIR / "simulation_results_test.csv",
                      index_col=0)
    print("\n=== test-set strategy summary ===")
    print(res.to_string())
    print(f"\n[pipeline] done in {time.time() - t0:.0f}s")


def main() -> None:
    ap = argparse.ArgumentParser(description="End-to-end pipeline")
    ap.add_argument("--source", choices=["file", "download"], default="file")
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()
    run(source=args.source, skip_train=args.skip_train)


if __name__ == "__main__":
    main()
