# -*- coding: utf-8 -*-
"""
Crawl KOSPI + KOSDAQ stocks with market cap >= 5000억원.
Step 1: Get full listing from KIND
Step 2: Filter by market cap via yfinance (takes ~25 min for ~2400 stocks)
Step 3: Download price history 2021-01-01 ~ today for new tickers

Usage:
  cd c:/Users/user/source/repos/jumproo
  python -X utf8 stock_prediction_v7/scripts/crawl_large_caps.py
"""
import io
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

sys.stdout.reconfigure(encoding='utf-8')

RAW_DIR = Path(__file__).parent.parent.parent / "stock_prediction" / "data" / "raw"
NAMES_PATH = Path(__file__).parent.parent / "configs" / "ticker_names.json"
START_DATE = "2021-01-01"
MIN_CAP_KRW = 5_000 * 1e8   # 5000억원 in KRW
MIN_ROWS = 500               # at least ~2 years of trading days


def fetch_kind_listing():
    """Fetch full KOSPI+KOSDAQ listing from KIND."""
    print("KIND에서 전체 상장 종목 목록 다운로드 중...", flush=True)
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    url = 'https://kind.krx.co.kr/corpgeneral/corpList.do?method=download&searchType=13'
    r = requests.get(url, headers=headers, timeout=30)
    html = r.content.decode('cp949', errors='replace')
    df = pd.read_html(io.StringIO(html))[0]

    # Normalize code to 6 digits, keep numeric only (exclude ETF/SPAC codes like 0001A0)
    df['종목코드'] = df['종목코드'].astype(str).str.zfill(6)
    df = df[df['종목코드'].str.match(r'^\d{6}$')].copy()

    # Exclude SPAC (기업인수합병), 코넥스
    df = df[df['시장구분'] != '코넥스']
    spac_mask = df['주요제품'].astype(str).str.contains('기업인수합병', na=False)
    df = df[~spac_mask]

    market_map = {'유가': '.KS', '코스닥': '.KQ'}
    df['suffix'] = df['시장구분'].map(market_map).fillna('.KQ')
    df['yf_ticker'] = df['종목코드'] + df['suffix']

    print(f"  유효 종목: {len(df)}개 (SPAC/코넥스 제외)", flush=True)
    return df


def check_market_cap(yf_ticker):
    """Return market cap in KRW, 0 if unavailable.
    Note: yfinance returns marketCap in the stock's native currency (KRW for .KS/.KQ).
    """
    try:
        info = yf.Ticker(yf_ticker).info
        cap_krw = info.get('marketCap') or 0
        if cap_krw > 0:
            return cap_krw
        # Try alternate suffix
        alt = yf_ticker.replace('.KQ', '.KS') if '.KQ' in yf_ticker else yf_ticker.replace('.KS', '.KQ')
        info2 = yf.Ticker(alt).info
        cap_krw2 = info2.get('marketCap') or 0
        return cap_krw2
    except Exception:
        return 0


def download_history(yf_ticker, start_date, end_date):
    """Download OHLCV history, return DataFrame or None."""
    for ticker in [yf_ticker,
                   yf_ticker.replace('.KQ', '.KS'),
                   yf_ticker.replace('.KS', '.KQ')]:
        try:
            df = yf.download(ticker, start=start_date, end=end_date,
                             progress=False, auto_adjust=False)
            if df is not None and len(df) >= MIN_ROWS:
                df = df.reset_index()
                if hasattr(df.columns, 'levels'):
                    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
                return df
        except Exception:
            continue
    return None


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Existing tickers
    existing = {f.stem for f in RAW_DIR.glob("*.csv")}
    print(f"기존 CSV: {len(existing)}개", flush=True)

    # Load existing names
    names = {}
    if NAMES_PATH.exists():
        with open(NAMES_PATH, 'r', encoding='utf-8') as f:
            names = json.load(f)

    # Step 1: KIND listing
    df_all = fetch_kind_listing()

    # Step 2: Filter by market cap
    # Note: yfinance returns marketCap in KRW for .KS/.KQ tickers (no USD conversion needed)
    print(f"\n시총 조회 중 ({len(df_all)}개, 예상 {len(df_all)*0.7/60:.0f}~{len(df_all)*1.2/60:.0f}분)...", flush=True)
    large_caps = []
    skipped_existing = 0

    for i, row in df_all.iterrows():
        yf_ticker = row['yf_ticker']
        code = row['종목코드']
        name = row['회사명']

        # Already have data for this ticker
        if code in existing:
            skipped_existing += 1
            # Still add to names
            names[yf_ticker] = name
            large_caps.append({'code': code, 'name': name, 'yf_ticker': yf_ticker,
                                'market': row['시장구분'], 'new': False})
            continue

        cap_krw = check_market_cap(yf_ticker)
        cap_eok = cap_krw / 1e8

        if cap_krw >= MIN_CAP_KRW:
            large_caps.append({'code': code, 'name': name, 'yf_ticker': yf_ticker,
                                'market': row['시장구분'], 'cap_eok': cap_eok, 'new': True})
            names[yf_ticker] = name
            print(f"  [{i+1}/{len(df_all)}] {name} ({yf_ticker}): {cap_eok:,.0f}억 ✓", flush=True)
        else:
            if (i + 1) % 100 == 0:
                new_found = sum(1 for t in large_caps if t.get('new'))
                print(f"  [{i+1}/{len(df_all)}] 조회 중... 신규 시총5천억+ {new_found}개 발견", flush=True)

        # Rate limit
        time.sleep(0.3)

    new_tickers = [t for t in large_caps if t.get('new')]
    print(f"\n기존 CSV 포함: {skipped_existing}개", flush=True)
    print(f"신규 시총 5000억+ 종목: {len(new_tickers)}개", flush=True)

    # Save updated names
    with open(NAMES_PATH, 'w', encoding='utf-8') as f:
        json.dump(names, f, ensure_ascii=False, indent=2)
    print(f"ticker_names.json 업데이트 완료: {len(names)}개", flush=True)

    # Step 4: Download price history for new tickers
    if not new_tickers:
        print("다운로드할 신규 종목 없음.", flush=True)
        return

    end_date = datetime.now().strftime("%Y-%m-%d")
    success, failed = 0, []

    print(f"\n가격 데이터 다운로드 중 ({len(new_tickers)}개, 2021-01-01 ~ {end_date})...", flush=True)
    for i, t in enumerate(new_tickers):
        print(f"[{i+1}/{len(new_tickers)}] {t['name']} ({t['yf_ticker']}) ... ", end='', flush=True)
        df = download_history(t['yf_ticker'], START_DATE, end_date)
        if df is not None:
            csv_path = RAW_DIR / f"{t['code']}.csv"
            df.to_csv(csv_path, index=False)
            print(f"OK ({len(df)}행)", flush=True)
            success += 1
        else:
            print("SKIP (데이터 부족)", flush=True)
            failed.append(t)
        time.sleep(0.5)

    print(f"\n{'='*60}", flush=True)
    print(f"  완료: {success}개 다운로드 성공", flush=True)
    print(f"  실패/부족: {len(failed)}개", flush=True)
    print(f"  전체 CSV: {len(list(RAW_DIR.glob('*.csv')))}개", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
