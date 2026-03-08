# -*- coding: utf-8 -*-
"""
Download KOSPI (^KS11) and KOSDAQ (^KQ11) daily index data from yfinance.
Saves market_index.csv with columns: Date, kospi_close, kospi_return, kosdaq_close, kosdaq_return, market_vol

Usage:
  cd c:/Users/user/source/repos/jumproo
  python -X utf8 stock_prediction_v8/scripts/download_market_index.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

sys.stdout.reconfigure(encoding='utf-8')

OUT_PATH = Path(__file__).parent.parent / "data" / "market_index.csv"
START_DATE = "2020-06-01"  # extra buffer before 2021-01-01


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("KOSPI (^KS11) 다운로드 중...", flush=True)
    kospi = yf.download("^KS11", start=START_DATE, progress=False, auto_adjust=False)
    kospi = kospi.reset_index()
    if hasattr(kospi.columns, 'levels'):
        kospi.columns = [c[0] if isinstance(c, tuple) else c for c in kospi.columns]
    print(f"  KOSPI: {len(kospi)}행", flush=True)

    print("KOSDAQ (^KQ11) 다운로드 중...", flush=True)
    kosdaq = yf.download("^KQ11", start=START_DATE, progress=False, auto_adjust=False)
    kosdaq = kosdaq.reset_index()
    if hasattr(kosdaq.columns, 'levels'):
        kosdaq.columns = [c[0] if isinstance(c, tuple) else c for c in kosdaq.columns]
    print(f"  KOSDAQ: {len(kosdaq)}행", flush=True)

    # Merge on Date
    kospi = kospi[['Date', 'Close']].rename(columns={'Close': 'kospi_close'})
    kosdaq = kosdaq[['Date', 'Close']].rename(columns={'Close': 'kosdaq_close'})

    merged = pd.merge(kospi, kosdaq, on='Date', how='outer').sort_values('Date').reset_index(drop=True)
    merged['Date'] = pd.to_datetime(merged['Date'])

    # Forward fill any missing
    merged['kospi_close'] = merged['kospi_close'].ffill()
    merged['kosdaq_close'] = merged['kosdaq_close'].ffill()

    # Returns
    merged['kospi_return'] = merged['kospi_close'].pct_change().fillna(0)
    merged['kosdaq_return'] = merged['kosdaq_close'].pct_change().fillna(0)

    # Market volatility (KOSPI 20-day rolling std of returns)
    merged['market_vol'] = merged['kospi_return'].rolling(20, min_periods=5).std().fillna(0)

    # KOSPI 5-day momentum
    merged['kospi_mom5'] = merged['kospi_close'].pct_change(5).fillna(0)

    # KOSDAQ 5-day momentum
    merged['kosdaq_mom5'] = merged['kosdaq_close'].pct_change(5).fillna(0)

    merged.to_csv(OUT_PATH, index=False)
    print(f"\n저장 완료: {OUT_PATH} ({len(merged)}행)", flush=True)
    print(f"  기간: {merged['Date'].min()} ~ {merged['Date'].max()}", flush=True)
    print(f"  컬럼: {list(merged.columns)}", flush=True)


if __name__ == "__main__":
    main()
