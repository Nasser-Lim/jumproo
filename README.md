# Jumproo — Stock Surge Prediction v8

5일 이내 15% 이상 급등할 주식을 예측하는 파이프라인.
**PatchTST Regression + EVT + Hawkes Process + HMM** 기반 하이브리드 모델.

---

## 빠른 시작

### 대시보드 실행 (로컬)
```bash
# V8 (최신 — PatchTST 회귀 + 섹터 임베딩)
streamlit run stock_prediction_v8/src/app/app_v8.py

# V7 (구버전)
streamlit run stock_prediction_v7/src/app/app_v7.py
```

### 일일 데이터 업데이트
```bash
cd jumproo
python -X utf8 stock_prediction_v8/scripts/update_daily.py
```
> 매일 18:30 이후 실행 권장 (장 마감 후 yfinance 반영). 이미 최신인 종목은 자동 스킵.

---

## OBS 11 (RTX 4060) 학습 프로세스

### 최초 세팅
```bash
git clone https://github.com/Nasser-Lim/jumproo.git
cd jumproo

# 의존성
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install numpy pandas scipy scikit-learn pyyaml plotly streamlit tqdm hmmlearn yfinance requests

# raw 데이터 복사 (stock_prediction/data/raw/ — 647개 CSV, gitignore 대상)
# 로컬에서 압축 후 옮기거나 crawl_large_caps.py 재실행
python -X utf8 stock_prediction_v7/scripts/crawl_large_caps.py  # ~30~50분
```

### V8 학습 실행 (원클릭)
```bash
python -X utf8 stock_prediction_v8/scripts/run_training.py
# Step 1: 데이터셋 생성 (~10분)
# Step 2: PatchTST 학습 (~2~3시간, RTX 4060 기준)
```

### 학습 후 모델 push
```bash
git add stock_prediction_v8/models/patchtst/best_model.pt
git add stock_prediction_v8/models/patchtst/train_history.json
git commit -m "v8: GPU trained PatchTST regressor"
git push
```

### 로컬에서 pull 후 대시보드
```bash
git pull
streamlit run stock_prediction_v7/src/app/app_v7.py
```

---

## 아키텍처

### V8 전체 파이프라인

```
[입력] 647개 종목 × 60일 OHLCV
         │
         ├─ [PatchTST Regressor] ─────────────────────────────
         │    입력: 8채널 시계열 (60일)                        │
         │      0: close_norm    (현재가 대비 종가)            │
         │      1: log_vol_norm  (정규화 거래량)               │
         │      2: RSI(14)                                     │
         │      3: bollinger_pos (BB 내 위치)                  │
         │      4: MACD_norm     (MACD / 현재가)               │
         │      5: kospi_return  (KOSPI 일간 수익률)           │
         │      6: kosdaq_return (KOSDAQ 일간 수익률)          │
         │      7: market_vol    (시장 20일 변동성)            │
         │    + 섹터 임베딩: nn.Embedding(170, 16)             │
         │    출력: 예측 최대 수익률 (회귀)              (30%) │
         │                                                     │
         └─ [통계 파이프라인] ──────────────────────────────  │
              EVT/GPD   : 꼬리 확률 (40%)              (70%) │
              Hawkes    : 이벤트 군집도 (60%)                │
              HMM       : 시장 국면 soft gate               │
              거래량필터 : 단기/장기 비율                    │
                                                             │
         └─────────── Final Score (가중 합산) ───────────────┘
                  STRONG_BUY (≥60%) / BUY (≥40%) / WATCH (≥20%) / NEUTRAL
```

### V7 → V8 핵심 변경

| 항목 | V7 | V8 |
|---|---|---|
| **PatchTST 목적** | BCE 이진 분류 | **Huber 회귀** (max 5일 수익률) |
| **입력 채널** | 3 (close, vol, RSI) | **8** (+bollinger, MACD, KOSPI, KOSDAQ, market_vol) |
| **섹터 정보** | 없음 | **nn.Embedding(170, 16)** — 162개 업종 |
| **학습 데이터** | ~80K (balanced) | **468K** (가중 샘플링) |
| **데이터 종목** | 389개 → 159개 (시총 필터) | **647개** (5000억+ 재크롤) |
| **모델 파라미터** | 326K | **940K** (4 layers) |
| **클래스 불균형** | pos_weight 보정 → collapse | **3x surge 가중치 + WeightedSampler** |
| **PatchTST 가중치** | 0% (disabled) | **30%** (stats 70%와 조합) |

### 데이터 분할 (2026.03 기준)

| 구간 | 기간 | 샘플 수 | 용도 |
|---|---|---|---|
| Train | 2021.01 ~ 2024.06 | 468,790 | 모델 학습 |
| Val | 2024.07 ~ 2025.06 | 155,259 | threshold 튜닝 |
| Test | 2025.07 ~ 현재 | 103,595 | 최종 평가 (한 번만) |

---

## V7 검증된 OOS 성능 (Val set, stat-only)

| Threshold | 신호 수 | Precision | 평균 수익률 | Base Rate |
|---|---|---|---|---|
| WATCH (≥0.2) | 128 | 22.66% | +8.36% | 5.24% |
| BUY (≥0.4) | 80 | 22.50% | +8.05% | 5.24% |
| STRONG_BUY (≥0.6) | 6 | 50.00% | +21.38% | 5.24% |

> V8 학습 완료 후 val set 재평가 예정.

---

## 매수 전략 가이드

| 시그널 | 등급 | 조건 | 진입 비중 |
|---|---|---|---|
| STRONG_BUY | — | 극단적 변동성 국면 | 최대 3% |
| BUY | **A** | 5일 조정 후 반등 + 클러스터 ≥1.0 | 5% |
| BUY | **B** | 당일 급등 or 이미 5일 상승 | 3% (눌림목 대기) |
| BUY | **C** | 클러스터 <0.7 | 0% (관망) |
| WATCH | — | 신호 약함 | 0% |

**공통 리스크 관리**
- 손절: 진입가 -5%
- 1차 익절: +7% (포지션 50% 정리)
- 2차 익절: +10~15% (잔여 전량)
- 시간 손절: 5거래일 내 +5% 미달 시 전량

---

## 디렉토리 구조

```
jumproo/
├── stock_prediction_v8/              # 현재 버전 (v8)
│   ├── configs/
│   │   ├── v8_config.yaml            # 전체 설정
│   │   └── sector_map.json           # 162개 업종 매핑 (KIND)
│   ├── data/
│   │   ├── market_index.csv          # KOSPI/KOSDAQ 일간 지수
│   │   └── processed/                # 학습 데이터셋 npz (gitignore)
│   ├── models/patchtst/              # 학습된 모델 체크포인트 (gitignore)
│   ├── scripts/
│   │   ├── crawl_sectors.py          # KIND 업종 크롤러
│   │   ├── download_market_index.py  # KOSPI/KOSDAQ 다운로더
│   │   ├── update_daily.py           # 일일 증분 업데이트
│   │   └── run_training.py           # OBS 11 학습 런처
│   └── src/
│       ├── data/create_dataset_v8.py  # 8채널 데이터셋 생성
│       ├── model/patchtst_inference.py
│       └── train/train_v8.py          # PatchTSTRegressor
│
├── stock_prediction_v7/              # V7 (현재 대시보드 사용)
│   ├── configs/
│   │   ├── v7_config.yaml
│   │   └── ticker_names.json         # 647개 종목명 매핑
│   ├── scripts/crawl_large_caps.py   # 시총 5000억+ 종목 크롤
│   └── src/app/app_v7.py             # Streamlit 대시보드
│
└── stock_prediction/data/raw/        # 647개 원본 CSV (gitignore)
```

---

## 버전 히스토리

| 버전 | 모델 | Precision | 비고 |
|---|---|---|---|
| v1.0 | Chronos (T5) | 11.9% | In-sample |
| v2.0 | PatchTST + 거래량 | 13.9% | In-sample |
| v3.0 | PatchTST + MC Dropout | 12.1% | In-sample |
| v4.0 | 앙상블 5모델 | 20.5% | In-sample |
| v6.0 | EVT + Hawkes + HMM | 24.0% | In-sample (과대 추정) |
| v7.0 | PatchTST + EVT + Hawkes | 22.66% | **OOS (Val set)** |
| **v8.0** | **회귀 PatchTST + 섹터 임베딩** | **TBD** | **학습 예정 (OBS 11)** |

> v1~v6의 수치는 in-sample 과대 추정. v7+는 완전 분리된 Val set OOS 결과.
