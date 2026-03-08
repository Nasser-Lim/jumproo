# -*- coding: utf-8 -*-
"""
V8 조합 백테스트 — Stats 70% + PatchTST 30%

V7 stat 백테스트 CSV (ticker, date, stat_score, actual_max_return, is_actual_surge)를
기반으로, 각 (ticker, date)에 대해 V8 PatchTST predicted_return을 계산하고 합산.

Usage:
  cd c:/Users/user/source/repos/jumproo
  python -X utf8 stock_prediction_v8/scripts/backtest_v8_combined.py
"""
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from stock_prediction_v8.src.model.patchtst_inference import PatchTSTPredictor
from stock_prediction_v8.src.data.create_dataset_v8 import (
    compute_rsi, compute_bollinger, compute_macd
)
from stock_prediction_v7.src.data.create_dataset_v7 import load_raw_csv

V8_ROOT = Path(__file__).parent.parent
V7_ROOT = V8_ROOT.parent / "stock_prediction_v7"
RAW_DIR = V8_ROOT.parent / "stock_prediction" / "data" / "raw"


def load_market_index():
    path = V8_ROOT / "data" / "market_index.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df.set_index("Date").sort_index()


def build_8channel_at_date(df, market_df, target_date, context_length=60):
    """Build 8-channel context ending at target_date (exclusive of future)."""
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df[df["Date"] <= pd.Timestamp(target_date)].reset_index(drop=True)

    if len(df) < context_length + 5:
        return None

    close = df["Close"].values.astype(float)
    volume = df["Volume"].values.astype(float)
    dates = df["Date"].values

    log_vol = np.log1p(volume)
    rsi = compute_rsi(close, period=14)
    boll_pos = compute_bollinger(close, window=20)
    macd_norm = compute_macd(close)

    if market_df is not None:
        kospi_ret = np.zeros(len(close))
        kosdaq_ret = np.zeros(len(close))
        mkt_vol = np.zeros(len(close))
        for i, d in enumerate(dates):
            d = pd.Timestamp(d)
            if d in market_df.index:
                row = market_df.loc[d]
                kospi_ret[i] = float(row.get("kospi_return", 0) or 0)
                kosdaq_ret[i] = float(row.get("kosdaq_return", 0) or 0)
                mkt_vol[i] = float(row.get("market_vol", 0) or 0)
    else:
        kospi_ret = np.zeros(len(close))
        kosdaq_ret = np.zeros(len(close))
        mkt_vol = np.zeros(len(close))

    t = len(close)
    current_price = close[t - 1]
    if current_price <= 0:
        return None

    ctx_close = close[t - context_length: t] / current_price

    ctx_log_vol = log_vol[t - context_length: t]
    lv_mean = np.nanmean(ctx_log_vol)
    lv_std = np.nanstd(ctx_log_vol)
    ctx_log_vol = (ctx_log_vol - lv_mean) / lv_std if lv_std > 0 else ctx_log_vol - lv_mean

    sample = np.stack([
        ctx_close,
        ctx_log_vol,
        rsi[t - context_length: t],
        boll_pos[t - context_length: t],
        macd_norm[t - context_length: t],
        kospi_ret[t - context_length: t],
        kosdaq_ret[t - context_length: t],
        mkt_vol[t - context_length: t],
    ], axis=-1)  # (60, 8)

    sample = np.nan_to_num(sample, nan=0.0, posinf=5.0, neginf=-5.0)
    sample = np.clip(sample, -10.0, 10.0)
    return sample


def combine_scores(stat_score, pt_return, surge_threshold=0.15,
                   stat_weight=0.7, pt_weight=0.3):
    pt_score = min(max(pt_return / (surge_threshold * 2), 0.0), 1.0)
    final = stat_weight * stat_score + pt_weight * pt_score
    final = float(np.clip(final, 0, 1))
    if final >= 0.6:
        signal = "STRONG_BUY"
    elif final >= 0.4:
        signal = "BUY"
    elif final >= 0.2:
        signal = "WATCH"
    else:
        signal = "NEUTRAL"
    return final, signal, pt_score


def main():
    print("=" * 65)
    print("  V8 Combined Backtest: Stats 70% + PatchTST 30%")
    print("=" * 65)

    # Load V7 stat backtest CSV
    stat_csv = V7_ROOT / "outputs" / "backtest_v7_stat_only.csv"
    if not stat_csv.exists():
        print(f"ERROR: {stat_csv} not found")
        sys.exit(1)
    stat_df = pd.read_csv(stat_csv)
    stat_df["date"] = pd.to_datetime(stat_df["date"])
    print(f"  V7 stat backtest: {len(stat_df):,} rows, {stat_df['ticker'].nunique()} tickers")
    print(f"  Period: {stat_df['date'].min().date()} ~ {stat_df['date'].max().date()}")

    # Load sector map
    sector_map_path = V8_ROOT / "configs" / "sector_map.json"
    sector_map = {}
    if sector_map_path.exists():
        with open(sector_map_path, "r", encoding="utf-8") as f:
            sm = json.load(f)
        sector_map = {k: v.get("sector_id", 0) for k, v in sm.get("stocks", {}).items()}
    print(f"  Sector map: {len(sector_map)} stocks")

    # Load market index
    market_df = load_market_index()
    print(f"  Market index: {'OK' if market_df is not None else 'MISSING'}")

    # Load PatchTST model
    model_path = V8_ROOT / "models" / "patchtst" / "best_model.pt"
    predictor = PatchTSTPredictor(model_path=model_path, device="cpu")
    if predictor.model is None:
        print("ERROR: PatchTST model not found")
        sys.exit(1)

    # Cache raw CSVs per ticker
    raw_cache = {}

    # Run inference for each row
    print(f"\n  Running PatchTST on {len(stat_df):,} rows...")
    pt_returns = []
    errors = 0

    tickers = stat_df["ticker"].unique()
    total_tickers = len(tickers)

    for ti, ticker in enumerate(tickers):
        # Load CSV once per ticker
        # ticker is like "005930.KS" — strip suffix to find file
        code = ticker.rsplit(".", 1)[0]
        csv_path = RAW_DIR / f"{code}.csv"

        if not csv_path.exists():
            # all rows for this ticker → 0
            mask = stat_df["ticker"] == ticker
            pt_returns.extend([0.0] * mask.sum())
            errors += mask.sum()
            continue

        if code not in raw_cache:
            try:
                raw_cache[code] = load_raw_csv(str(csv_path))
            except Exception:
                mask = stat_df["ticker"] == ticker
                pt_returns.extend([0.0] * mask.sum())
                errors += mask.sum()
                continue

        df_raw = raw_cache[code]
        sector_id = sector_map.get(code, 0)

        rows_for_ticker = stat_df[stat_df["ticker"] == ticker]

        for _, row in rows_for_ticker.iterrows():
            target_date = row["date"]
            context = build_8channel_at_date(df_raw, market_df, target_date)
            if context is None:
                pt_returns.append(0.0)
                errors += 1
                continue

            result = predictor.predict(context, sector_id=sector_id, n_samples=10)
            pt_returns.append(result.get("predicted_return", 0.0))

        if (ti + 1) % 50 == 0 or (ti + 1) == total_tickers:
            print(f"  [{ti+1}/{total_tickers}] {ticker}: done (errors so far: {errors})")

    stat_df["pt_return"] = pt_returns
    stat_df["pt_score"] = stat_df["pt_return"].apply(
        lambda r: min(max(r / (0.15 * 2), 0.0), 1.0)
    )
    stat_df["combined_score"] = 0.7 * stat_df["stat_score"] + 0.3 * stat_df["pt_score"]
    stat_df["combined_score"] = stat_df["combined_score"].clip(0, 1)

    def to_signal(s):
        if s >= 0.6: return "STRONG_BUY"
        if s >= 0.4: return "BUY"
        if s >= 0.2: return "WATCH"
        return "NEUTRAL"

    stat_df["combined_signal"] = stat_df["combined_score"].apply(to_signal)

    # Save
    out_path = V8_ROOT / "data" / "processed" / "backtest_v8_combined.csv"
    stat_df.to_csv(out_path, index=False)
    print(f"\n  Saved to {out_path}")

    # ─── Evaluation ───
    base_rate = stat_df["is_actual_surge"].mean()
    total_surge = stat_df["is_actual_surge"].sum()
    print(f"\n{'='*65}")
    print(f"  Val Set: N={len(stat_df):,}, surge_count={total_surge}, base_rate={base_rate:.2%}")
    print(f"{'='*65}")

    print(f"\n  [V7 Stat-Only]")
    print(f"  {'Threshold':>10} | {'Signals':>7} | {'Precision':>10} | {'Recall':>8} | {'AvgRet':>8}")
    print(f"  {'-'*10}-+-{'-'*7}-+-{'-'*10}-+-{'-'*8}-+-{'-'*8}")
    for th in [0.2, 0.3, 0.4, 0.5, 0.6]:
        sig = stat_df[stat_df["stat_score"] >= th]
        if len(sig) == 0: continue
        prec = sig["is_actual_surge"].mean()
        rec = sig["is_actual_surge"].sum() / max(total_surge, 1)
        avg_ret = sig["actual_max_return"].mean()
        print(f"  {th:>10.2f} | {len(sig):>7,} | {prec:>10.2%} | {rec:>8.2%} | {avg_ret:>+8.2%}")

    print(f"\n  [V8 Combined: Stats 70% + PatchTST 30%]")
    print(f"  {'Threshold':>10} | {'Signals':>7} | {'Precision':>10} | {'Recall':>8} | {'AvgRet':>8} | {'vs V7':>8}")
    print(f"  {'-'*10}-+-{'-'*7}-+-{'-'*10}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")

    # Store v7 precision for delta
    v7_prec = {}
    for th in [0.2, 0.3, 0.4, 0.5, 0.6]:
        sig = stat_df[stat_df["stat_score"] >= th]
        v7_prec[th] = sig["is_actual_surge"].mean() if len(sig) > 0 else 0.0

    for th in [0.2, 0.3, 0.4, 0.5, 0.6]:
        sig = stat_df[stat_df["combined_score"] >= th]
        if len(sig) == 0: continue
        prec = sig["is_actual_surge"].mean()
        rec = sig["is_actual_surge"].sum() / max(total_surge, 1)
        avg_ret = sig["actual_max_return"].mean()
        delta = prec - v7_prec.get(th, 0.0)
        delta_str = f"{delta:+.2%}"
        print(f"  {th:>10.2f} | {len(sig):>7,} | {prec:>10.2%} | {rec:>8.2%} | {avg_ret:>+8.2%} | {delta_str:>8}")

    print(f"\n  errors (no context/CSV): {errors}")
    print(f"{'='*65}")

    # Save summary JSON
    summary = {
        "base_rate": float(base_rate),
        "n_samples": len(stat_df),
        "v7_stat_only": {},
        "v8_combined": {},
    }
    for th in [0.2, 0.3, 0.4, 0.5, 0.6]:
        sig_v7 = stat_df[stat_df["stat_score"] >= th]
        sig_v8 = stat_df[stat_df["combined_score"] >= th]
        summary["v7_stat_only"][str(th)] = {
            "signals": len(sig_v7),
            "precision": float(sig_v7["is_actual_surge"].mean()) if len(sig_v7) > 0 else 0,
            "avg_ret": float(sig_v7["actual_max_return"].mean()) if len(sig_v7) > 0 else 0,
        }
        summary["v8_combined"][str(th)] = {
            "signals": len(sig_v8),
            "precision": float(sig_v8["is_actual_surge"].mean()) if len(sig_v8) > 0 else 0,
            "avg_ret": float(sig_v8["actual_max_return"].mean()) if len(sig_v8) > 0 else 0,
        }

    json_out = V8_ROOT / "data" / "processed" / "backtest_v8_combined_summary.json"
    with open(json_out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary JSON: {json_out}")


if __name__ == "__main__":
    main()
