# -*- coding: utf-8 -*-
"""
Daily incremental update for all stock CSVs and market index.

- Reads last date from each CSV and downloads only missing rows
- Updates market_index.csv
- Fast: skips stocks already up-to-date

Usage:
  cd c:/Users/user/source/repos/jumproo
  python -X utf8 stock_prediction_v8/scripts/update_daily.py

Schedule (Windows Task Scheduler or cron):
  - Run after 18:00 KST on trading days (Mon~Fri)
"""
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

sys.stdout.reconfigure(encoding='utf-8')

import json

RAW_DIR = Path(__file__).parent.parent.parent / "stock_prediction" / "data" / "raw"
MARKET_INDEX_PATH = Path(__file__).parent.parent / "data" / "market_index.csv"
NAMES_PATH = Path(__file__).parent.parent.parent / "stock_prediction_v7" / "configs" / "ticker_names.json"

# Load ticker suffix map once: code -> yf_ticker
def load_ticker_map():
    if not NAMES_PATH.exists():
        return {}
    with open(NAMES_PATH, 'r', encoding='utf-8') as f:
        names = json.load(f)
    # names keys are like "005930.KS" — build code -> yf_ticker map
    ticker_map = {}
    for yf_ticker in names:
        code = yf_ticker.rsplit('.', 1)[0]
        ticker_map[code] = yf_ticker
    return ticker_map


def get_last_date(csv_path):
    """Return the last date in a CSV as pd.Timestamp."""
    try:
        # Read only last few rows efficiently
        df = pd.read_csv(csv_path, usecols=[0])
        col = df.columns[0]
        # Skip ticker rows
        dates = pd.to_datetime(df[col], errors='coerce').dropna()
        if len(dates) == 0:
            return None
        return dates.iloc[-1]
    except Exception:
        return None


def update_csv(csv_path, yf_ticker, last_date, end_date):
    """Append new rows to existing CSV. Returns number of new rows added."""
    start = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
    end = end_date

    if start >= end:
        return 0

    try:
        new_df = yf.download(yf_ticker, start=start, end=end,
                             progress=False, auto_adjust=False)
        if new_df is None or len(new_df) == 0:
            return 0

        new_df = new_df.reset_index()
        if hasattr(new_df.columns, 'levels'):
            new_df.columns = [c[0] if isinstance(c, tuple) else c for c in new_df.columns]

        # Append to existing CSV (no header)
        new_df.to_csv(csv_path, mode='a', header=False, index=False)
        return len(new_df)

    except Exception:
        return -1  # error


def update_market_index(end_date):
    """Update market_index.csv with latest KOSPI/KOSDAQ data."""
    if not MARKET_INDEX_PATH.exists():
        print("  market_index.csv not found, skipping.", flush=True)
        return

    existing = pd.read_csv(MARKET_INDEX_PATH)
    existing['Date'] = pd.to_datetime(existing['Date'])
    last_date = existing['Date'].max()
    start = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")

    if start >= end_date:
        print(f"  market_index already up-to-date ({last_date.date()})", flush=True)
        return

    print(f"  Updating market_index: {start} ~ {end_date}", flush=True)
    kospi = yf.download("^KS11", start=start, end=end_date, progress=False, auto_adjust=False)
    kosdaq = yf.download("^KQ11", start=start, end=end_date, progress=False, auto_adjust=False)

    if len(kospi) == 0:
        print("  No new market data.", flush=True)
        return

    kospi = kospi.reset_index()[['Date', 'Close']].rename(columns={'Close': 'kospi_close'})
    kosdaq = kosdaq.reset_index()[['Date', 'Close']].rename(columns={'Close': 'kosdaq_close'})
    if hasattr(kospi.columns, 'levels'):
        kospi.columns = [c[0] if isinstance(c, tuple) else c for c in kospi.columns]
    if hasattr(kosdaq.columns, 'levels'):
        kosdaq.columns = [c[0] if isinstance(c, tuple) else c for c in kosdaq.columns]

    new_rows = pd.merge(kospi, kosdaq, on='Date', how='outer').sort_values('Date')
    new_rows['kospi_return'] = new_rows['kospi_close'].pct_change().fillna(0)
    new_rows['kosdaq_return'] = new_rows['kosdaq_close'].pct_change().fillna(0)
    new_rows['market_vol'] = new_rows['kospi_return'].rolling(20, min_periods=1).std().fillna(0)
    new_rows['kospi_mom5'] = new_rows['kospi_close'].pct_change(5).fillna(0)
    new_rows['kosdaq_mom5'] = new_rows['kosdaq_close'].pct_change(5).fillna(0)

    # Append
    new_rows.to_csv(MARKET_INDEX_PATH, mode='a', header=False, index=False)
    print(f"  market_index: +{len(new_rows)} rows", flush=True)


def main():
    today = datetime.now()
    end_date = today.strftime("%Y-%m-%d")

    # If today is weekend, use Friday
    if today.weekday() >= 5:  # Sat=5, Sun=6
        days_back = today.weekday() - 4
        end_date = (today + timedelta(days=1)).strftime("%Y-%m-%d")  # yfinance end is exclusive

    print(f"{'='*60}", flush=True)
    print(f"  Daily Update — target end: {end_date}", flush=True)
    print(f"{'='*60}\n", flush=True)

    # Update market index first
    update_market_index(end_date)

    # Load ticker map
    ticker_map = load_ticker_map()

    # Update stock CSVs
    csvs = sorted(RAW_DIR.glob("*.csv"))
    total = len(csvs)
    updated = 0
    skipped = 0
    errors = 0

    print(f"\n종목 업데이트 중 ({total}개)...", flush=True)

    for i, csv_path in enumerate(csvs):
        code = csv_path.stem
        last_date = get_last_date(csv_path)

        if last_date is None:
            errors += 1
            continue

        # Already up to date (last date is yesterday or today)
        yesterday = datetime.now() - timedelta(days=1)
        if last_date.date() >= yesterday.date():
            skipped += 1
            if (i + 1) % 100 == 0:
                print(f"  [{i+1}/{total}] 진행 중 (업데이트: {updated}, 스킵: {skipped})", flush=True)
            continue

        # Find yfinance ticker from names map, fallback to .KS
        yf_ticker = ticker_map.get(code, code + '.KS')

        n = update_csv(csv_path, yf_ticker, last_date, end_date)

        if n > 0:
            updated += 1
            if (i + 1) % 50 == 0 or n > 0:
                print(f"  [{i+1}/{total}] {code}: +{n}행 ({last_date.date()} → {end_date})", flush=True)
        elif n == 0:
            skipped += 1
        else:
            errors += 1

        time.sleep(0.2)  # rate limit

    print(f"\n{'='*60}", flush=True)
    print(f"  완료: 업데이트 {updated}개 / 스킵 {skipped}개 / 오류 {errors}개", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
