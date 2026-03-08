# -*- coding: utf-8 -*-
"""
Crawl sector (업종) mapping from KIND for all stocks in raw data.
Creates sector_map.json: { "005930.KS": {"sector": "전기전자", "sector_id": 0}, ... }

Usage:
  cd c:/Users/user/source/repos/jumproo
  python -X utf8 stock_prediction_v8/scripts/crawl_sectors.py
"""
import io
import json
import sys
from pathlib import Path

import pandas as pd
import requests

sys.stdout.reconfigure(encoding='utf-8')

RAW_DIR = Path(__file__).parent.parent.parent / "stock_prediction" / "data" / "raw"
NAMES_PATH = Path(__file__).parent.parent.parent / "stock_prediction_v7" / "configs" / "ticker_names.json"
OUT_PATH = Path(__file__).parent.parent / "configs" / "sector_map.json"


def fetch_kind_listing():
    """Fetch full KOSPI+KOSDAQ listing from KIND with sector info."""
    print("KIND에서 전체 상장 종목 + 업종 다운로드 중...", flush=True)
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    url = 'https://kind.krx.co.kr/corpgeneral/corpList.do?method=download&searchType=13'
    r = requests.get(url, headers=headers, timeout=30)
    html = r.content.decode('cp949', errors='replace')
    df = pd.read_html(io.StringIO(html))[0]

    df['종목코드'] = df['종목코드'].astype(str).str.zfill(6)
    df = df[df['종목코드'].str.match(r'^\d{6}$')].copy()

    market_map = {'유가': '.KS', '코스닥': '.KQ'}
    df['suffix'] = df['시장구분'].map(market_map).fillna('.KQ')
    df['yf_ticker'] = df['종목코드'] + df['suffix']

    print(f"  KIND 전체: {len(df)}개", flush=True)
    return df


def main():
    df = fetch_kind_listing()

    # Get list of tickers we have data for
    existing_codes = {f.stem for f in RAW_DIR.glob("*.csv")}
    print(f"  기존 CSV: {len(existing_codes)}개", flush=True)

    # Load ticker_names for suffix mapping
    names = {}
    if NAMES_PATH.exists():
        with open(NAMES_PATH, 'r', encoding='utf-8') as f:
            names = json.load(f)

    # Extract sector info - KIND provides '업종코드' or '주요제품' but sector is in '업종'
    # The KIND download has: 회사명, 종목코드, 업종, 주요제품, 상장일, 결산월, 대표자명, ...
    sector_col = None
    for col in ['업종', '산업분류', '업종분류']:
        if col in df.columns:
            sector_col = col
            break

    if sector_col is None:
        print(f"  Available columns: {list(df.columns)}", flush=True)
        # KIND typically has: 회사명, 종목코드, 업종, 주요제품, 상장일, 결산월, 대표자명, 홈페이지, 지역
        # Try the 3rd column (index 2) which is usually 업종
        if len(df.columns) > 2:
            sector_col = df.columns[2]
            print(f"  Using column '{sector_col}' as sector", flush=True)
        else:
            print("ERROR: Cannot find sector column!", flush=True)
            return

    # Build sector map
    sectors = sorted(df[sector_col].dropna().unique())
    sector_to_id = {s: i for i, s in enumerate(sectors)}
    print(f"  업종 수: {len(sectors)}개", flush=True)

    sector_map = {}
    matched = 0

    for _, row in df.iterrows():
        code = row['종목코드']
        yf_ticker = row['yf_ticker']
        sector = row[sector_col] if pd.notna(row[sector_col]) else "기타"

        # Match to our existing CSVs
        if code in existing_codes:
            sector_map[yf_ticker] = {
                "sector": sector,
                "sector_id": sector_to_id.get(sector, len(sector_to_id)),
                "name": row.get('회사명', code),
            }
            matched += 1

    print(f"  매칭된 종목: {matched}개 / {len(existing_codes)}개", flush=True)

    # Save
    output = {
        "sector_list": sectors,
        "num_sectors": len(sectors),
        "stocks": sector_map,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n저장 완료: {OUT_PATH}", flush=True)
    print(f"  업종 목록: {sectors[:10]} ...", flush=True)

    # Print sector distribution
    from collections import Counter
    sector_counts = Counter(v["sector"] for v in sector_map.values())
    print(f"\n업종 분포 (상위 15개):", flush=True)
    for sector, count in sector_counts.most_common(15):
        print(f"  {sector}: {count}개", flush=True)


if __name__ == "__main__":
    main()
