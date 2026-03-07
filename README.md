# Jumproo — Stock Surge Prediction v7

5일 이내 15% 이상 급등할 주식을 예측하는 파이프라인.
**PatchTST + EVT + Hawkes Process + HMM** 기반 하이브리드 모델.

---

## 빠른 시작

### 대시보드 실행
```bash
streamlit run stock_prediction_v7/src/app/app_v7.py
```

### 전체 실행 순서 (OBS11 GPU 머신)
```bash
git clone https://github.com/Nasser-Lim/jumproo.git
cd jumproo

# 의존성
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install numpy pandas scipy scikit-learn pyyaml plotly streamlit tqdm hmmlearn

# raw 데이터 복사 후 (stock_prediction/data/raw/ — 389개 CSV)
python stock_prediction_v7/src/data/create_dataset_v7.py   # 데이터셋 생성
python stock_prediction_v7/src/train/train_v7.py            # GPU 학습

# 학습 완료 후 push
git add stock_prediction_v7/models/patchtst/best_model.pt
git add stock_prediction_v7/models/patchtst/train_history.json
git commit -m "v7: GPU trained PatchTST"
git push
```

### 로컬에서 백테스트 + 대시보드
```bash
git pull
python stock_prediction_v7/src/backtest/backtest_v7.py --mode val   # 검증
python stock_prediction_v7/src/backtest/backtest_v7.py --mode test  # 최종 (한 번만)
streamlit run stock_prediction_v7/src/app/app_v7.py
```

---

## 아키텍처

```
[PatchTST]  60일 context → 5일 예측 (급등 확률)
     +
[통계 필터]  EVT(꼬리확률) + Hawkes(군집도) + HMM(국면) + 거래량 필터
     ↓
 Final Score → STRONG_BUY / BUY / WATCH / NEUTRAL
```

### 데이터 분할 (2026.03 기준)

| 구간 | 기간 | 용도 |
|---|---|---|
| Train | 2021.02 ~ 2024.06 | 모델 학습 |
| Val | 2024.07 ~ 2025.06 | 가중치/threshold 튜닝 |
| Test | 2025.07 ~ 2026.02 | 최종 평가 (한 번만) |

---

## v7 핵심 개선 (vs v1~v6)

| 문제 | 해결 |
|---|---|
| 랜덤 셔플 분할 → 데이터 유출 | 시간 기반 Train/Val/Test 분리 |
| 백테스트 = 훈련 기간 | 완전 분리된 Holdout Test |
| Expanding window (평가 데이터 포함 피팅) | Rolling window 504일, strictly past |
| Recency weighting 편향 | 균일 가중 |
| GPU 필수 | CPU/GPU 자동 선택 |

### 검증된 OOS 성능 (Val set 2024.07~2025.06, stat-only)

| Threshold | 신호 수 | Precision | 평균 수익률 | Base Rate |
|---|---|---|---|---|
| WATCH (≥0.2) | 128 | 22.66% | +8.36% | 5.24% |
| BUY (≥0.4) | 80 | 22.50% | +8.05% | 5.24% |
| STRONG_BUY (≥0.6) | 6 | 50.00% | +21.38% | 5.24% |

> v1~v6의 보고 수치는 in-sample 과대 추정임. 위 수치가 진정한 OOS 결과.

---

## 버전 히스토리 (참고)

| 버전 | 모델 | Precision | 비고 |
|---|---|---|---|
| v1.0 | Chronos (T5) | 11.9% | In-sample |
| v2.0 | PatchTST + 거래량 | 13.9% | In-sample |
| v3.0 | PatchTST + MC Dropout | 12.1% | In-sample, 과도한 보수성 |
| v4.0 | 앙상블 5모델 | 20.5% | In-sample |
| v6.0 | EVT + Hawkes + HMM | 24.0% | In-sample (과대 추정) |
| **v7.0** | **PatchTST + EVT + Hawkes** | **22.66%** | **OOS (Val set)** |

---

## 디렉토리 구조

```
jumproo/
├── stock_prediction_v7/          # 현재 버전 (v7)
│   ├── configs/v7_config.yaml    # 전체 설정
│   ├── data/processed/           # 시간 분할 데이터셋 (npz)
│   ├── models/patchtst/          # 학습된 모델 체크포인트
│   ├── outputs/                  # 백테스트 결과 CSV
│   └── src/
│       ├── data/create_dataset_v7.py
│       ├── model/{evt_gpd, hawkes_timing, hmm_regime, predictor_v7}.py
│       ├── train/train_v7.py
│       ├── backtest/backtest_v7.py
│       └── app/app_v7.py
└── stock_prediction/data/raw/    # 원본 CSV 389개 (gitignore)
```
