# Stock Surge Prediction v7

과대포장 없는 주가 급등 예측 파이프라인.

## v7 핵심 변경사항 (vs v6)

| 문제 (v1~v6) | 해결 (v7) |
|---|---|
| 랜덤 셔플 분할 | 시간 기반 Train/Val/Test 분리 |
| 백테스트 = 훈련 기간 | 완전 분리된 Holdout Test |
| EVT/Hawkes expanding window | Rolling window (504일) |
| 평가 시점 데이터 포함 | Strictly past (t-1까지만) |
| Recency weighting 편향 | 균일 가중 |
| GPU 필수 | CPU/GPU 자동 선택 |

## 데이터 분할 (2026.03.07 기준)

```
Train:  2021.01 ~ 2024.06  (모델 학습)
Val:    2024.07 ~ 2025.06  (가중치/threshold 튜닝)
Test:   2025.07 ~ 2026.03  (최종 평가, 한 번만)
```

## 아키텍처

```
PatchTST (60일 context → 5일 예측)
    + 통계 필터 (EVT + Hawkes + HMM gate + 거래량 필터)
    = Final Score → STRONG_BUY / BUY / WATCH / NEUTRAL
```

## 실행 가이드

### 1. 데이터셋 생성 (로컬)
```bash
python stock_prediction_v7/src/data/create_dataset_v7.py
```

### 2. PatchTST 학습 (GPU 머신 또는 로컬 CPU)
```bash
# GPU 머신에서:
git pull
python stock_prediction_v7/src/train/train_v7.py

# 학습 완료 후:
git add stock_prediction_v7/models/
git commit -m "v7: trained PatchTST model"
git push
```

### 3. 백테스트 (로컬)
```bash
# 통계 파이프라인만 (PatchTST 없이)
python stock_prediction_v7/src/backtest/backtest_v7.py --mode stat_only

# PatchTST 포함 (모델 pull 후)
git pull
python stock_prediction_v7/src/backtest/backtest_v7.py --mode val

# 최종 Holdout 테스트 (한 번만!)
python stock_prediction_v7/src/backtest/backtest_v7.py --mode test
```

### 4. 대시보드 (로컬)
```bash
streamlit run stock_prediction_v7/src/app/app_v7.py
```

## 디렉토리 구조

```
stock_prediction_v7/
├── configs/v7_config.yaml     # 모든 설정
├── data/processed/            # 시간 분할된 데이터셋
├── models/patchtst/           # 학습된 모델 (git으로 동기화)
├── outputs/                   # 백테스트 결과
└── src/
    ├── data/create_dataset_v7.py      # 데이터 생성
    ├── model/
    │   ├── predictor_v7.py            # 통합 예측기
    │   ├── evt_gpd.py                 # EVT/GPD (rolling, uniform)
    │   ├── hawkes_timing.py           # Hawkes (strictly past)
    │   ├── hmm_regime.py              # HMM regime gate
    │   └── patchtst_inference.py      # PatchTST 추론
    ├── train/train_v7.py              # PatchTST 학습
    ├── backtest/backtest_v7.py        # Walk-forward 백테스트
    └── app/app_v7.py                  # Streamlit 대시보드
```

## 의존성

```bash
pip install torch numpy pandas scipy scikit-learn pyyaml plotly streamlit tqdm hmmlearn
```
