# -*- coding: utf-8 -*-
"""
V8 Dataset Creator — Regression target + Sector ID + Market Context + Technical Indicators

Input channels (8):
  0: close_norm      — close / current_price
  1: log_vol_norm    — z-scored log volume
  2: rsi             — RSI(14) / 100
  3: bollinger_pos   — (close - BB_lower) / (BB_upper - BB_lower)
  4: macd_norm       — MACD / close (normalized)
  5: kospi_return    — KOSPI daily return
  6: kosdaq_return   — KOSDAQ daily return
  7: market_vol      — KOSPI 20-day rolling volatility

Extra per-sample metadata:
  - sector_id: int (for nn.Embedding)
  - target: float (max return within 5 days, regression target)
  - surge_label: int (1 if target >= 0.15, for evaluation)
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings("ignore", category=RuntimeWarning)


def compute_rsi(prices, period=14):
    delta = np.diff(prices)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(period, min_periods=period).mean().values
    avg_loss = pd.Series(loss).rolling(period, min_periods=period).mean().values
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 100.0)
    rsi = 100 - (100 / (1 + rs))
    return np.concatenate([[np.nan] * period, rsi])


def compute_bollinger(prices, window=20):
    """Returns bollinger position: (price - lower) / (upper - lower), clipped to [0, 1]."""
    s = pd.Series(prices)
    ma = s.rolling(window, min_periods=window).mean()
    std = s.rolling(window, min_periods=window).std()
    upper = ma + 2 * std
    lower = ma - 2 * std
    width = upper - lower
    pos = np.where(width > 0, (s - lower) / width, 0.5)
    pos = np.clip(pos, 0, 1)
    result = np.full(len(prices), np.nan)
    result[window - 1:] = pos[window - 1:]
    return result


def compute_macd(prices, fast=12, slow=26, signal=9):
    """Returns MACD normalized by price."""
    s = pd.Series(prices)
    ema_fast = s.ewm(span=fast, adjust=False).mean()
    ema_slow = s.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    # Normalize by price to make cross-stock comparable
    macd_norm = macd / s
    result = macd_norm.values
    # First slow-1 values are warming up
    result[:slow] = np.nan
    return result


def load_raw_csv(csv_path):
    """Load yfinance-format CSV with Price->Date rename and ticker row skip."""
    df = pd.read_csv(csv_path)
    if 'Price' in df.columns and 'Date' not in df.columns:
        df.rename(columns={'Price': 'Date'}, inplace=True)
    if len(df) > 0:
        try:
            float(df.iloc[0]['Close'])
        except (ValueError, TypeError):
            df = df.iloc[1:]
    df = df[df['Date'].astype(str).str.contains(r'\d{4}', na=False)]
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['Date', 'Close', 'Volume'])
    df = df.sort_values('Date').reset_index(drop=True)
    return df


def process_stock(csv_path, market_df, sector_id, context_length, prediction_length, surge_threshold):
    """Process a single stock CSV into samples.

    Returns:
        samples: list of np.array (context_length, 8)
        targets: list of float (max return in prediction window)
        sector_ids: list of int
        sample_dates: list of pd.Timestamp
    """
    df = load_raw_csv(str(csv_path))

    min_warmup = max(context_length, 26)  # MACD needs 26
    if len(df) < min_warmup + prediction_length + 20:
        return [], [], [], []

    close = df["Close"].values.astype(float)
    volume = df["Volume"].values.astype(float)
    dates = pd.to_datetime(df["Date"].values)

    # Per-stock features
    log_vol = np.log1p(volume)
    rsi = compute_rsi(close, period=14) / 100.0
    boll_pos = compute_bollinger(close, window=20)
    macd_norm = compute_macd(close)

    # Build date-indexed market data lookup
    market_lookup = {}
    for _, row in market_df.iterrows():
        d = row['Date']
        market_lookup[d] = {
            'kospi_return': row['kospi_return'],
            'kosdaq_return': row['kosdaq_return'],
            'market_vol': row['market_vol'],
        }

    # Align market data to stock dates
    kospi_ret = np.zeros(len(df))
    kosdaq_ret = np.zeros(len(df))
    market_vol = np.zeros(len(df))

    for i, d in enumerate(dates):
        d_key = pd.Timestamp(d).normalize()
        if d_key in market_lookup:
            m = market_lookup[d_key]
            kospi_ret[i] = m['kospi_return']
            kosdaq_ret[i] = m['kosdaq_return']
            market_vol[i] = m['market_vol']

    samples = []
    targets = []
    sector_ids = []
    sample_dates = []

    valid_start = context_length
    valid_end = len(df) - prediction_length

    for t in range(valid_start, valid_end):
        # Skip if any feature has NaN in context window
        if np.isnan(rsi[t - 1]) or np.isnan(boll_pos[t - 1]) or np.isnan(macd_norm[t - 1]):
            continue

        current_price = close[t - 1]
        if current_price <= 0:
            continue

        # --- Target: max return within prediction window ---
        future_prices = close[t: t + prediction_length]
        max_future = np.max(future_prices)
        max_return = (max_future - current_price) / current_price

        # --- Build 8-channel context ---
        ctx_close = close[t - context_length: t] / current_price

        ctx_log_vol = log_vol[t - context_length: t]
        lv_mean = np.nanmean(ctx_log_vol)
        lv_std = np.nanstd(ctx_log_vol)
        if lv_std > 0:
            ctx_log_vol = (ctx_log_vol - lv_mean) / lv_std
        else:
            ctx_log_vol = ctx_log_vol - lv_mean

        ctx_rsi = rsi[t - context_length: t]
        ctx_boll = boll_pos[t - context_length: t]
        ctx_macd = macd_norm[t - context_length: t]
        ctx_kospi = kospi_ret[t - context_length: t]
        ctx_kosdaq = kosdaq_ret[t - context_length: t]
        ctx_mvol = market_vol[t - context_length: t]

        sample = np.stack([
            ctx_close,      # 0: close_norm
            ctx_log_vol,    # 1: log_vol_norm
            ctx_rsi,        # 2: RSI
            ctx_boll,       # 3: bollinger_pos
            ctx_macd,       # 4: macd_norm
            ctx_kospi,      # 5: kospi_return
            ctx_kosdaq,     # 6: kosdaq_return
            ctx_mvol,       # 7: market_vol
        ], axis=-1)  # (60, 8)

        if np.isnan(sample).any():
            continue

        samples.append(sample)
        targets.append(max_return)
        sector_ids.append(sector_id)
        sample_dates.append(pd.Timestamp(dates[t]))

    return samples, targets, sector_ids, sample_dates


def create_dataset(config_path=None):
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "configs" / "v8_config.yaml"

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    root = Path(__file__).parent.parent.parent
    raw_dir = root.parent / "stock_prediction" / "data" / "raw"
    out_dir = root / data_cfg["processed_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    context_length = data_cfg["context_length"]
    prediction_length = data_cfg["prediction_length"]
    surge_threshold = data_cfg["surge_threshold"]

    train_end = pd.Timestamp(data_cfg["train_end"])
    val_start = pd.Timestamp(data_cfg["val_start"])
    val_end = pd.Timestamp(data_cfg["val_end"])
    test_start = pd.Timestamp(data_cfg["test_start"])

    # Load market index
    market_path = root / data_cfg["market_index_path"]
    market_df = pd.read_csv(market_path)
    market_df['Date'] = pd.to_datetime(market_df['Date'])
    print(f"Market index loaded: {len(market_df)} rows", flush=True)

    # Load sector map
    sector_path = root / data_cfg["sector_map_path"]
    with open(sector_path, 'r', encoding='utf-8') as f:
        sector_data = json.load(f)
    stock_sectors = sector_data["stocks"]
    default_sector_id = sector_data["num_sectors"]  # unknown sector
    print(f"Sector map loaded: {sector_data['num_sectors']} sectors", flush=True)

    csv_files = sorted(raw_dir.glob("*.csv"))
    print(f"Found {len(csv_files)} CSV files", flush=True)

    train_samples, train_targets, train_sectors = [], [], []
    val_samples, val_targets, val_sectors = [], [], []
    test_samples, test_targets, test_sectors = [], [], []

    for i, csv_file in enumerate(csv_files):
        code = csv_file.stem

        # Find sector_id
        sector_id = default_sector_id
        for suffix in ['.KS', '.KQ']:
            key = code + suffix
            if key in stock_sectors:
                sector_id = stock_sectors[key]["sector_id"]
                break

        samples, targets, sids, dates = process_stock(
            csv_file, market_df, sector_id,
            context_length, prediction_length, surge_threshold
        )

        for s, t, sid, d in zip(samples, targets, sids, dates):
            if d <= train_end:
                train_samples.append(s)
                train_targets.append(t)
                train_sectors.append(sid)
            elif val_start <= d <= val_end:
                val_samples.append(s)
                val_targets.append(t)
                val_sectors.append(sid)
            elif d >= test_start:
                test_samples.append(s)
                test_targets.append(t)
                test_sectors.append(sid)

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(csv_files)}] processed... "
                  f"(train: {len(train_samples)}, val: {len(val_samples)}, test: {len(test_samples)})",
                  flush=True)

    print(f"\n  Total processed: {len(csv_files)} files", flush=True)

    # Convert to arrays
    train_samples = np.array(train_samples, dtype=np.float32)
    train_targets = np.array(train_targets, dtype=np.float32)
    train_sectors = np.array(train_sectors, dtype=np.int32)

    val_samples = np.array(val_samples, dtype=np.float32)
    val_targets = np.array(val_targets, dtype=np.float32)
    val_sectors = np.array(val_sectors, dtype=np.int32)

    test_samples = np.array(test_samples, dtype=np.float32)
    test_targets = np.array(test_targets, dtype=np.float32)
    test_sectors = np.array(test_sectors, dtype=np.int32)

    # Save (NO balancing for regression — use sample weights instead)
    np.savez_compressed(out_dir / "train.npz",
                        samples=train_samples, targets=train_targets, sectors=train_sectors)
    np.savez_compressed(out_dir / "val.npz",
                        samples=val_samples, targets=val_targets, sectors=val_sectors)
    np.savez_compressed(out_dir / "test.npz",
                        samples=test_samples, targets=test_targets, sectors=test_sectors)

    # Stats
    def surge_rate(t):
        return float((t >= surge_threshold).mean()) if len(t) > 0 else 0

    stats = {
        "train": {"total": len(train_targets),
                  "surge_count": int((train_targets >= surge_threshold).sum()),
                  "surge_rate": surge_rate(train_targets),
                  "mean_return": float(train_targets.mean()) if len(train_targets) > 0 else 0,
                  "std_return": float(train_targets.std()) if len(train_targets) > 0 else 0},
        "val": {"total": len(val_targets),
                "surge_count": int((val_targets >= surge_threshold).sum()),
                "surge_rate": surge_rate(val_targets),
                "mean_return": float(val_targets.mean()) if len(val_targets) > 0 else 0},
        "test": {"total": len(test_targets),
                 "surge_count": int((test_targets >= surge_threshold).sum()),
                 "surge_rate": surge_rate(test_targets),
                 "mean_return": float(test_targets.mean()) if len(test_targets) > 0 else 0},
        "channels": ["close_norm", "log_vol_norm", "rsi", "bollinger_pos",
                      "macd_norm", "kospi_return", "kosdaq_return", "market_vol"],
    }

    with open(out_dir / "split_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"\n{'='*60}", flush=True)
    print(f"  Train: {stats['train']['total']:,} samples "
          f"(surge: {stats['train']['surge_count']:,}, rate: {stats['train']['surge_rate']:.2%}, "
          f"mean_ret: {stats['train']['mean_return']:.4f})", flush=True)
    print(f"  Val:   {stats['val']['total']:,} samples "
          f"(surge: {stats['val']['surge_count']:,}, rate: {stats['val']['surge_rate']:.2%})", flush=True)
    print(f"  Test:  {stats['test']['total']:,} samples "
          f"(surge: {stats['test']['surge_count']:,}, rate: {stats['test']['surge_rate']:.2%})", flush=True)
    print(f"  Saved to {out_dir}", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    create_dataset()
