"""
V7 Walk-Forward Backtest — Strictly Out-of-Sample Evaluation

Key principles:
- All model fitting uses data BEFORE the prediction point
- Walk-forward: refit every N days on rolling window
- Supports stat-only mode (no PatchTST) and hybrid mode
"""
import numpy as np
import pandas as pd
from pathlib import Path
import yaml
import json
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.model.predictor_v7 import SurgePredictor
from src.model.patchtst_inference import PatchTSTPredictor
from src.data.create_dataset_v7 import compute_rsi, load_raw_csv


def load_stock(csv_path):
    return load_raw_csv(csv_path)


def prepare_features(df):
    """Compute all features needed for prediction."""
    close = df["Close"].values.astype(float)
    volume = df["Volume"].values.astype(float)

    # Returns
    returns = np.diff(close) / close[:-1]
    returns = np.concatenate([[0], returns])

    # Volatility (20-day rolling std)
    vol_series = pd.Series(returns).rolling(20, min_periods=5).std().values

    # Volume change
    vol_ma20 = pd.Series(volume).rolling(20, min_periods=5).mean().values
    volume_change = np.where(vol_ma20 > 0, volume / vol_ma20, 1.0)

    # RSI
    rsi = compute_rsi(close, period=14)
    rsi = rsi / 100.0

    # Log volume (normalized)
    log_vol = np.log1p(volume)

    return {
        "close": close,
        "volume": volume,
        "returns": returns,
        "volatility": vol_series,
        "volume_change": volume_change,
        "rsi": rsi,
        "log_vol": log_vol,
        "dates": df["Date"].values,
    }


def build_context(features, idx, context_length=60):
    """Build PatchTST context array at position idx (uses [idx-60 : idx])."""
    close = features["close"]
    log_vol = features["log_vol"]
    rsi = features["rsi"]

    current_price = close[idx - 1]
    if current_price <= 0:
        return None, 0

    ctx_close = close[idx - context_length: idx] / current_price
    ctx_lv = log_vol[idx - context_length: idx]
    lv_mean = np.nanmean(ctx_lv)
    lv_std = np.nanstd(ctx_lv)
    if lv_std > 0:
        ctx_lv = (ctx_lv - lv_mean) / lv_std
    else:
        ctx_lv = ctx_lv - lv_mean
    ctx_rsi = rsi[idx - context_length: idx]

    context = np.stack([ctx_close, ctx_lv, ctx_rsi], axis=-1)  # (60, 3)

    if np.any(np.isnan(context)):
        return None, current_price

    return context, current_price


def run_backtest(config_path=None, eval_start=None, eval_end=None,
                 use_patchtst=True, stride=5):
    """Run walk-forward backtest.

    Args:
        config_path: path to v7_config.yaml
        eval_start: start date for evaluation (default: val_start from config)
        eval_end: end date for evaluation (default: val_end from config)
        use_patchtst: whether to include PatchTST scores
        stride: prediction stride in days
    """
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "configs" / "v7_config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    raw_dir = Path(__file__).parent.parent.parent / data_cfg["raw_dir"]
    out_dir = Path(__file__).parent.parent.parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    context_length = data_cfg["context_length"]
    prediction_length = data_cfg["prediction_length"]
    surge_threshold = data_cfg["surge_threshold"]
    min_data_days = data_cfg["min_data_days"]

    if eval_start is None:
        eval_start = pd.Timestamp(data_cfg["val_start"])
    else:
        eval_start = pd.Timestamp(eval_start)
    if eval_end is None:
        eval_end = pd.Timestamp(data_cfg["val_end"])
    else:
        eval_end = pd.Timestamp(eval_end)

    predictor = SurgePredictor(config_path)

    patchtst = None
    if use_patchtst:
        patchtst = PatchTSTPredictor(device="auto")
        if not patchtst.model:
            print("PatchTST model not found. Running stat-only backtest.")
            use_patchtst = False

    csv_files = sorted(raw_dir.glob("*.csv"))
    print(f"Backtesting {len(csv_files)} stocks, period: {eval_start} ~ {eval_end}")

    results = []

    for csv_file in tqdm(csv_files, desc="Backtesting"):
        ticker = csv_file.stem
        df = load_stock(csv_file)

        if len(df) < min_data_days:
            continue

        features = prepare_features(df)
        dates = features["dates"]

        # Reset refit counter per stock so stats refit correctly on first point
        predictor._last_refit = -predictor.refit_interval

        for i in range(context_length, len(df) - prediction_length, stride):
            date = pd.Timestamp(dates[i])
            if date < eval_start or date > eval_end:
                continue

            # Fit stats if needed (strictly past)
            predictor.fit_stats(
                features["returns"], features["volatility"],
                features["volume_change"], current_idx=i
            )

            # Stat prediction
            if use_patchtst:
                context, current_price = build_context(features, i, context_length)
                if context is None:
                    continue

                pt_result = patchtst.predict(context, current_price, surge_threshold,
                                             n_samples=5)
                stat_result = predictor.predict_stat(
                    features["returns"], features["volatility"],
                    features["volume_change"], features["volume"],
                    current_idx=i
                )
                combined = predictor.combine_scores(pt_result["surge_prob"], stat_result)
            else:
                stat_result = predictor.predict_stat_only(
                    features["returns"], features["volatility"],
                    features["volume_change"], features["volume"],
                    current_idx=i
                )
                combined = stat_result

            # Actual outcome
            future_prices = features["close"][i: i + prediction_length]
            current_price = features["close"][i - 1]
            if current_price <= 0:
                continue
            actual_max_ret = (np.max(future_prices) - current_price) / current_price
            actual_min_ret = (np.min(future_prices) - current_price) / current_price
            is_surge = actual_max_ret >= surge_threshold

            results.append({
                "ticker": ticker,
                "date": str(date.date()),
                "final_score": combined["final_score"],
                "signal": combined["signal"],
                "stat_score": combined.get("stat_score", combined["final_score"]),
                "patchtst_prob": combined.get("patchtst_prob"),
                "evt_prob": combined.get("evt_prob", 0),
                "hawkes_score": combined.get("hawkes_score", 0),
                "gate": combined.get("gate", 1.0),
                "vol_filter": combined.get("vol_filter", 1.0),
                "regime": combined.get("regime", "unknown"),
                "cluster_density": combined.get("cluster_density", 0),
                "actual_max_return": actual_max_ret,
                "actual_min_return": actual_min_ret,
                "is_actual_surge": is_surge,
            })

    df_results = pd.DataFrame(results)

    mode = "hybrid" if use_patchtst else "stat_only"
    output_path = out_dir / f"backtest_v7_{mode}.csv"
    df_results.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")

    # Summary
    print(f"\n{'='*60}")
    print(f"BACKTEST SUMMARY ({mode})")
    print(f"{'='*60}")
    print(f"Total predictions: {len(df_results)}")
    print(f"Base surge rate: {df_results.is_actual_surge.mean():.2%}")

    for th_name, th_val in [("WATCH (0.2)", 0.2), ("BUY (0.4)", 0.4),
                             ("STRONG_BUY (0.6)", 0.6)]:
        sig = df_results[df_results.final_score >= th_val]
        if len(sig) > 0:
            prec = sig.is_actual_surge.mean()
            avg_ret = sig.actual_max_return.mean()
            print(f"\n  {th_name}:")
            print(f"    Signals: {len(sig)}")
            print(f"    Precision: {prec:.2%}")
            print(f"    Avg Max Return: {avg_ret:+.2%}")
        else:
            print(f"\n  {th_name}: No signals")

    return df_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["val", "test", "stat_only"],
                        default="stat_only")
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--stride", type=int, default=5)
    args = parser.parse_args()

    config_path = Path(__file__).parent.parent.parent / "configs" / "v7_config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.mode == "test":
        start = args.start or cfg["data"]["test_start"]
        end = args.end or "2026-03-07"
        use_pt = True
    elif args.mode == "val":
        start = args.start or cfg["data"]["val_start"]
        end = args.end or cfg["data"]["val_end"]
        use_pt = True
    else:
        start = args.start or cfg["data"]["val_start"]
        end = args.end or cfg["data"]["val_end"]
        use_pt = False

    run_backtest(config_path, eval_start=start, eval_end=end,
                 use_patchtst=use_pt, stride=args.stride)
