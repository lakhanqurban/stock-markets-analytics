"""Train models on the combined dataset and emit validation/test predictions.

Models (all classification on target_24h = forward-24h return > 1.0):
    logreg  - LogisticRegression baseline (scaled numerics)
    dt      - DecisionTreeClassifier            (lecture model)
    rf      - RandomForestClassifier            (lecture model)
    xgb     - XGBoostClassifier                 (new model - rubric)

Workflow
--------
1. Chronological split: train < 2024-07-01 <= valid < 2025-07-01 <= test.
2. Small manual hyper-parameter grids, chosen by VALIDATION ROC-AUC
   (never touches the test set).
3. Probability-threshold sweep on validation -> chosen rule per model
   (max precision subject to >= 2% of rows being signals).
4. Writes: outputs/predictions_{valid,test}.parquet,
           outputs/metrics.json, outputs/feature_importance.csv,
           outputs/threshold_sweep.csv, models/*.joblib

Usage:
    python -m src.train
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from . import config, transform

MODEL_DIR = config.PROJECT_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

MIN_SIGNAL_SHARE = 0.02   # chosen threshold must flag >= 2% of validation rows


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_splits() -> dict[str, pd.DataFrame]:
    df = pd.read_parquet(config.DATA_DIR / "combined.parquet")
    cols = transform.NUMERIC_FEATURES + transform.DUMMY_FEATURES + \
        [transform.TARGET, "ts", "symbol", "close"]
    df = df.dropna(subset=cols).reset_index(drop=True)
    symbol_raw = df["symbol"]           # get_dummies would consume this column
    df = pd.get_dummies(df, columns=transform.DUMMY_FEATURES, dtype="float64")
    df["symbol"] = symbol_raw
    train_end = pd.Timestamp(config.TRAIN_END, tz="UTC")
    valid_end = pd.Timestamp(config.VALID_END, tz="UTC")
    return {
        "train": df[df["ts"] < train_end].reset_index(drop=True),
        "valid": df[(df["ts"] >= train_end) & (df["ts"] < valid_end)].reset_index(drop=True),
        "test": df[df["ts"] >= valid_end].reset_index(drop=True),
    }


def feature_cols(df: pd.DataFrame) -> list[str]:
    # exclude identifiers, targets AND all forward-looking growth columns
    # (growth_future_* would be target leakage)
    skip = {transform.TARGET, transform.TO_PREDICT, "ts", "symbol", "close"}
    return [c for c in df.columns
            if c not in skip and not c.startswith("growth_future")]


def xy(df: pd.DataFrame, feats: list[str], fit_scaler: bool = False,
       scaler: StandardScaler | None = None):
    """Numeric features scaled + dummy features raw (scaler only on numerics)."""
    num_idx = [feats.index(c) for c in transform.NUMERIC_FEATURES]
    X = df[feats].to_numpy(dtype="float64")
    if fit_scaler:
        scaler = StandardScaler().fit(X[:, num_idx])
    assert scaler is not None
    X[:, num_idx] = scaler.transform(X[:, num_idx])
    return X, df[transform.TARGET].to_numpy(dtype="float64"), scaler


# --------------------------------------------------------------------------- #
# Models and tuning
# --------------------------------------------------------------------------- #
def grids() -> dict[str, list[dict]]:
    return {
        "logreg": [{"C": 0.1}, {"C": 1.0}],
        "dt": [{"max_depth": d, "min_samples_leaf": m}
               for d in (6, 8, 10, 12) for m in (50, 200)],
        "rf": [{"max_depth": d, "min_samples_leaf": m}
               for d in (8, 10, 13) for m in (50, 200)],
        "xgb": [{"max_depth": d, "learning_rate": lr}
                for d in (4, 6, 8) for lr in (0.05, 0.1)],
    }


def make_model(name: str, params: dict):
    if name == "logreg":
        return LogisticRegression(max_iter=1000, **params)
    if name == "dt":
        return DecisionTreeClassifier(random_state=42, **params)
    if name == "rf":
        return RandomForestClassifier(n_estimators=100, n_jobs=-1,
                                      random_state=42, **params)
    if name == "xgb":
        return XGBClassifier(n_estimators=300, subsample=0.8,
                             colsample_bytree=0.8, objective="binary:logistic",
                             eval_metric="logloss", n_jobs=-1, random_state=42,
                             **params)
    raise KeyError(name)


def tune(name: str, X_tr, y_tr, X_va, y_va) -> tuple[object, dict, list[dict]]:
    """Pick hyper-parameters by validation AUC; return fitted best model."""
    table, best_model, best_params, best_auc = [], None, None, -1.0
    for params in grids()[name]:
        t0 = time.time()
        model = make_model(name, params)
        model.fit(X_tr, y_tr)
        auc = roc_auc_score(y_va, model.predict_proba(X_va)[:, 1])
        table.append({"model": name, **params, "val_auc": auc,
                      "fit_s": round(time.time() - t0, 1)})
        if auc > best_auc:
            best_model, best_params, best_auc = model, params, auc
    print(f"  {name}: best val AUC={best_auc:.4f} @ {best_params}")
    return best_model, best_params, table


# --------------------------------------------------------------------------- #
# Threshold rule (custom decision rule on higher-probability events)
# --------------------------------------------------------------------------- #
def sweep_thresholds(y: np.ndarray, prob: np.ndarray) -> pd.DataFrame:
    rows = []
    for thr in np.arange(0.50, 0.80, 0.01):
        sig = prob >= thr
        if sig.sum() == 0:
            continue
        rows.append({
            "threshold": round(float(thr), 2),
            "signal_share": float(sig.mean()),
            "n_signals": int(sig.sum()),
            "precision": float(y[sig].mean()),
            "hit_rate_lift": float(y[sig].mean() / max(y.mean(), 1e-9)),
        })
    return pd.DataFrame(rows)


def choose_threshold(sweep: pd.DataFrame) -> float:
    ok = sweep[sweep["signal_share"] >= MIN_SIGNAL_SHARE]
    if ok.empty:
        ok = sweep
    return float(ok.loc[ok["precision"].idxmax(), "threshold"])


# --------------------------------------------------------------------------- #
def main() -> None:
    t0 = time.time()
    splits = load_splits()
    feats = feature_cols(splits["train"])
    print(f"rows: train={len(splits['train']):,} "
          f"valid={len(splits['valid']):,} test={len(splits['test']):,} "
          f"| features={len(feats)}")

    X_tr, y_tr, scaler = xy(splits["train"], feats, fit_scaler=True)
    X_va, y_va, _ = xy(splits["valid"], feats, scaler=scaler)
    X_te, y_te, _ = xy(splits["test"], feats, scaler=scaler)

    metrics: dict = {"rows": {k: len(v) for k, v in splits.items()},
                     "n_features": len(feats), "models": {}}
    sweep_frames, proba_test = [], {}
    proba_valid = {}
    fitted: dict[str, object] = {}

    for name in grids():
        print(f"tuning {name}...")
        model, params, table = tune(name, X_tr, y_tr, X_va, y_va)
        sweep_frames.append(pd.DataFrame(table))
        fitted[name] = model

        pv, pt = model.predict_proba(X_va)[:, 1], model.predict_proba(X_te)[:, 1]
        proba_valid[name], proba_test[name] = pv, pt

        sweep = sweep_thresholds(y_va, pv)
        thr = choose_threshold(sweep)
        sweep.insert(0, "model", name)
        sweep_frames.append(sweep.assign(kind="threshold"))

        m = {
            "params": params,
            "val_auc": float(roc_auc_score(y_va, pv)),
            "test_auc": float(roc_auc_score(y_te, pt)),
            "threshold": thr,
            "val_precision": float(y_va[pv >= thr].mean()),
            "test_precision": float(y_te[pt >= thr].mean()),
            "test_signal_share": float((pt >= thr).mean()),
            "test_n_signals": int((pt >= thr).sum()),
        }
        metrics["models"][name] = m

        joblib.dump({"model": model, "scaler": scaler, "features": feats,
                     "threshold": thr}, MODEL_DIR / f"{name}.joblib")
        print(f"  -> test AUC={m['test_auc']:.4f} thr={thr:.2f} "
              f"test precision={m['test_precision']:.4f} "
              f"share={m['test_signal_share']:.3f}")

    # ------------------------------------------------------------------ #
    # Persist predictions for simulate.py
    # ------------------------------------------------------------------ #
    for split_name, proba in (("valid", proba_valid), ("test", proba_test)):
        out = splits[split_name][["ts", "symbol", "close",
                                  transform.TARGET]].copy()
        for name, p in proba.items():
            out[f"proba_{name}"] = p
            thr = metrics["models"][name]["threshold"]
            out[f"pred_{name}"] = (p >= thr).astype("int8")
        out.to_parquet(config.OUTPUT_DIR / f"predictions_{split_name}.parquet",
                       index=False)

    pd.DataFrame([m for f in sweep_frames if "kind" not in f.columns
                  for m in f.to_dict("records")]).to_csv(
        config.OUTPUT_DIR / "tuning_results.csv", index=False)
    pd.concat([f for f in sweep_frames if "kind" in f.columns],
              ignore_index=True).to_csv(
        config.OUTPUT_DIR / "threshold_sweep.csv", index=False)

    # feature importance (tree models) for the EDA section
    imp_rows = []
    for name in ("dt", "rf", "xgb"):
        model = fitted[name]
        importances = (model.feature_importances_ if hasattr(model, "feature_importances_")
                       else np.abs(model.coef_).ravel())
        imp_rows.append(pd.DataFrame({"feature": feats, "importance": importances,
                                      "model": name}))
    pd.concat(imp_rows).to_csv(config.OUTPUT_DIR / "feature_importance.csv",
                               index=False)

    with open(config.OUTPUT_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nmetrics -> {config.OUTPUT_DIR / 'metrics.json'} "
          f"({time.time() - t0:.0f}s total)")


if __name__ == "__main__":
    main()
