# 📈 Stock Surge Prediction Project (Chronos Fine-tuning)

## 1. 프로젝트 개요 (Project Overview)
이 프로젝트는 Amazon의 시계열 파운데이션 모델인 **Chronos (T5-based)**를 파인튜닝하여, **"향후 5일 이내에 15% 이상 급등할 가능성이 높은 주식"**을 예측하는 AI 모델을 구축하는 것을 목표로 합니다.

### 🎯 핵심 성과 (Key Achievements)
- **파인튜닝 데이터셋 구축**: 84개 주요 종목에서 20,000개 이상의 급등 패턴과 20,000개의 일반 패턴을 추출하여 **1:1 균형 데이터셋 (Balanced Dataset)**을 생성했습니다.
- **성능 개선**: 초기 모델의 과적합(Overfitting) 문제를 해결하고, 백테스트 결과 **평균 수익률 +8.23% (Threshold 0.6 기준)**를 달성했습니다.
- **실전 어플리케이션**: Streamlit 기반의 웹 대시보드를 통해 실시간 종목 분석이 가능합니다.

---

## 2. 프로젝트 구조 (Directory Structure)

```
jumproo/
├── stock_prediction/
│   ├── configs/                # 학습 설정 파일
│   │   └── train_config.yaml   # Chronos LoRA 파인튜닝 하이퍼파라미터
│   ├── data/                   # 데이터 저장소
│   │   ├── raw/                # 수집된 원본 CSV 파일 (yfinance)
│   │   └── processed/          # 학습용 JSONL 데이터셋 (balanced_finetune.jsonl)
│   ├── models/                 # 모델 저장소
│   │   └── finetuned/          # 파인튜닝된 체크포인트 (run-1/run-0/checkpoint-final)
│   ├── outputs/                # 백테스트 결과 및 시각화 이미지
│   └── src/                    # 소스 코드
│       ├── app/                # 웹 어플리케이션
│       │   └── app.py          # Streamlit 대시보드 메인 스크립트
│       ├── backtest/           # 백테스팅 모듈
│       │   ├── backtester.py   # 시뮬레이션 엔진 (배치 처리 최적화됨)
│       │   ├── run_comparison.py # 모델 비교 및 전체 백테스트 실행 스크립트
│       │   └── visualize_backtest.py # 결과 시각화 도구
│       ├── data/               # 데이터 파이프라인
│       │   ├── create_finetune_data.py # 학습 데이터 생성기 (Surge/Non-Surge 추출)
│       │   └── preprocessor.py # 전처리 유틸리티
│       └── model/              # 모델 로직
│           └── predictor.py    # Chronos 추론 클래스 (Surge Probability 계산)
```

---

## 3. 설치 및 실행 가이드 (How to Run)

### 🛠️ 사전 준비 (Requirements)
Python 3.10 이상이 필요합니다.
```bash
pip install torch transformers pandas yfinance plotly streamlit
pip install git+https://github.com/amazon-science/chronos-forecasting.git
```

### 🚀 1. 웹 대시보드 실행 (Web Dashboard)
가장 추천하는 사용 방법입니다. 직관적인 UI에서 종목을 분석할 수 있습니다.
```bash
# ✅ 최신 버전 (v6.0 권장) — HMM + EVT + Hawkes 5단계 통계 파이프라인
streamlit run stock_prediction_v6/src/app/app_v6.py

# (구버전) v1.0 Chronos 기반
streamlit run stock_prediction/src/app/app.py
```

### 📊 2. 백테스트 실행 (Backtesting)
전체 종목에 대해 모델 성능을 검증하고 싶을 때 사용합니다.
```bash
# Python 경로 설정 (Windows Powershell)
$env:PYTHONPATH += ";C:\Users\USER\repos\jumproo\chronos_repo\src"

# 실행
python stock_prediction/src/backtest/run_comparison.py
```

---

## 4. 모델 성능 요약 (Performance)
**389개 종목 대상 3년치 백테스트 결과 (Threshold 0.6 적용 시)**
- **정밀도 (Precision)**: 약 20.6% (5번 추천 중 1번은 15% 이상 폭등)
- **평균 수익률 (Avg Return)**: **+8.23%** (5일 보유 기준)
- **신호 빈도**: 약 3~4일에 1회 발생 (엄선된 종목 추천)

> **Conclusion**: 이 모델은 **"잃지 않는 매매"**를 지향하며, 확률 60% 이상일 때 진입하면 높은 기대 수익률을 제공합니다.
