"""
V7 Dataset Creator — Temporal Split, No Shuffle
Creates PatchTST training data with strict time-based train/val/test split.
"""
import numpy as np
import pandas as pd
from pathlib import Path
import yaml
import json


def compute_rsi(prices, period=14):
    delta = np.diff(prices)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(period, min_periods=period).mean().values
    avg_loss = pd.Series(loss).rolling(period, min_periods=period).mean().values
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 100.0)
    rsi = 100 - (100 / (1 + rs))
    # Pad front with NaN to match original length
    return np.concatenate([[np.nan] * (period), rsi])


def load_raw_csv(csv_path):
    """Load yfinance-format CSV with Price->Date rename and ticker row skip."""
    df = pd.read_csv(csv_path)
    if 'Price' in df.columns and 'Date' not in df.columns:
        df.rename(columns={'Price': 'Date'}, inplace=True)
    # Drop ticker row
    if len(df) > 0:
        try:
            float(df.iloc[0]['Close'])
        except (ValueError, TypeError):
            df = df.iloc[1:]
    # Keep rows with valid dates
    df = df[df['Date'].astype(str).str.contains(r'\d{4}', na=False)]
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['Date', 'Close', 'Volume'])
    df = df.sort_values('Date').reset_index(drop=True)
    return df


def process_stock(csv_path, context_length, prediction_length, surge_threshold):
    """Process a single stock CSV into samples with labels."""
    df = load_raw_csv(csv_path)

    if len(df) < context_length + prediction_length + 20:
        return [], [], []

    close = df["Close"].values.astype(float)
    volume = df["Volume"].values.astype(float)
    dates = df["Date"].values

    # Log volume
    log_vol = np.log1p(volume)

    # RSI
    rsi = compute_rsi(close, period=14)
    rsi = rsi / 100.0  # normalize to [0, 1]

    samples = []
    labels = []
    sample_dates = []

    valid_start = context_length
    valid_end = len(df) - prediction_length

    for t in range(valid_start, valid_end):
        # Check RSI is valid (need 14 periods before context)
        if np.isnan(rsi[t - 1]):
            continue

        current_price = close[t - 1]
        if current_price <= 0:
            continue

        # Future prices for label
        future_prices = close[t: t + prediction_length]
        max_future = np.max(future_prices)
        max_return = (max_future - current_price) / current_price

        label = 1 if max_return >= surge_threshold else 0

        # Normalize close prices relative to current price
        ctx_close = close[t - context_length: t] / current_price
        ctx_log_vol = log_vol[t - context_length: t]
        ctx_rsi = rsi[t - context_length: t]

        # Normalize log_vol to zero-mean within context
        lv_mean = np.nanmean(ctx_log_vol)
        lv_std = np.nanstd(ctx_log_vol)
        if lv_std > 0:
            ctx_log_vol = (ctx_log_vol - lv_mean) / lv_std
        else:
            ctx_log_vol = ctx_log_vol - lv_mean

        sample = np.stack([ctx_close, ctx_log_vol, ctx_rsi], axis=-1)  # (60, 3)

        # Future close (normalized) for PatchTST target
        future_close_norm = close[t: t + prediction_length] / current_price
        future_log_vol = log_vol[t: t + prediction_length]
        if lv_std > 0:
            future_log_vol = (future_log_vol - lv_mean) / lv_std
        else:
            future_log_vol = future_log_vol - lv_mean
        future_rsi = rsi[t: t + prediction_length]
        future = np.stack([future_close_norm, future_log_vol, future_rsi], axis=-1)  # (5, 3)

        full_sample = np.concatenate([sample, future], axis=0)  # (65, 3)

        # Skip samples with NaN (e.g. RSI warmup period)
        if np.isnan(full_sample).any():
            continue

        samples.append(full_sample)
        labels.append(label)
        sample_dates.append(pd.Timestamp(dates[t]))

    return samples, labels, sample_dates


def create_dataset(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    raw_dir = Path(__file__).parent.parent.parent / data_cfg["raw_dir"]
    out_dir = Path(__file__).parent.parent.parent / data_cfg["processed_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    context_length = data_cfg["context_length"]
    prediction_length = data_cfg["prediction_length"]
    surge_threshold = data_cfg["surge_threshold"]

    train_end = pd.Timestamp(data_cfg["train_end"])
    val_start = pd.Timestamp(data_cfg["val_start"])
    val_end = pd.Timestamp(data_cfg["val_end"])
    test_start = pd.Timestamp(data_cfg["test_start"])

    csv_files = sorted(raw_dir.glob("*.csv"))
    print(f"Found {len(csv_files)} CSV files in {raw_dir}")

    train_samples, train_labels = [], []
    val_samples, val_labels = [], []
    test_samples, test_labels = [], []

    for csv_file in csv_files:
        samples, labels, dates = process_stock(
            csv_file, context_length, prediction_length, surge_threshold
        )
        for s, l, d in zip(samples, labels, dates):
            if d <= train_end:
                train_samples.append(s)
                train_labels.append(l)
            elif val_start <= d <= val_end:
                val_samples.append(s)
                val_labels.append(l)
            elif d >= test_start:
                test_samples.append(s)
                test_labels.append(l)

    # Balance training set (undersample majority class)
    train_samples = np.array(train_samples)
    train_labels = np.array(train_labels)
    surge_idx = np.where(train_labels == 1)[0]
    nonsurge_idx = np.where(train_labels == 0)[0]

    print(f"\nTrain raw: {len(train_labels)} (surge: {len(surge_idx)}, non-surge: {len(nonsurge_idx)})")

    min_count = min(len(surge_idx), len(nonsurge_idx))
    if min_count == 0:
        print("ERROR: No surge or non-surge samples in train set!")
        return

    rng = np.random.RandomState(42)
    surge_sel = rng.choice(surge_idx, min_count, replace=False)
    nonsurge_sel = rng.choice(nonsurge_idx, min_count, replace=False)
    balanced_idx = np.concatenate([surge_sel, nonsurge_sel])
    rng.shuffle(balanced_idx)

    train_samples = train_samples[balanced_idx]
    train_labels = train_labels[balanced_idx]

    val_samples = np.array(val_samples)
    val_labels = np.array(val_labels)
    test_samples = np.array(test_samples)
    test_labels = np.array(test_labels)

    # Save
    np.savez_compressed(
        out_dir / "train.npz", samples=train_samples, labels=train_labels
    )
    np.savez_compressed(
        out_dir / "val.npz", samples=val_samples, labels=val_labels
    )
    np.savez_compressed(
        out_dir / "test.npz", samples=test_samples, labels=test_labels
    )

    stats = {
        "train": {"total": len(train_labels), "surge": int(train_labels.sum()),
                   "surge_rate": float(train_labels.mean())},
        "val": {"total": len(val_labels), "surge": int(val_labels.sum()),
                 "surge_rate": float(val_labels.mean()) if len(val_labels) > 0 else 0},
        "test": {"total": len(test_labels), "surge": int(test_labels.sum()),
                  "surge_rate": float(test_labels.mean()) if len(test_labels) > 0 else 0},
    }

    with open(out_dir / "split_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nTrain (balanced): {stats['train']['total']} (surge rate: {stats['train']['surge_rate']:.2%})")
    print(f"Val: {stats['val']['total']} (surge rate: {stats['val']['surge_rate']:.2%})")
    print(f"Test: {stats['test']['total']} (surge rate: {stats['test']['surge_rate']:.2%})")
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    config_path = Path(__file__).parent.parent.parent / "configs" / "v7_config.yaml"
    create_dataset(config_path)
