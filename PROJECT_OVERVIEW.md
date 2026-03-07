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
pip install torch transformers pandas yfinance chronos-forecast plotly streamlit
```

### 🚀 1. 웹 대시보드 실행 (Web Dashboard)
가장 추천하는 사용 방법입니다. 직관적인 UI에서 종목을 분석할 수 있습니다.
```bash
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

---

## 5. v2.0 업데이트 (Price + Volume Multimodal)
**2026-02-17 업데이트**: 거래량(Volume) 정보를 포함한 다변량 모델(PatchTST)을 추가했습니다.

### 🚀 성능 비교 (v1 vs v2)
| 모델 | 입력 데이터 | 알고리즘 | 신호 수 | 정밀도 | 평균 수익률 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **v1.0** | 종가 (Close) | Chronos (T5) | 10,425 | 11.9% | +5.42% |
| **v2.0** | **종가 + 거래량** | **PatchTST** | 5,813 | **13.9%** | **+6.26%** |

> **Insight**: 거래량을 함께 분석하니 **"가짜 상승"을 걸러내는 능력**이 향상되어, 신호 수는 줄었지만 **수익률과 정밀도가 모두 상승**했습니다. 안정적인 투자를 원하시면 v2.0 모델 사용을 권장합니다.

### 📂 v2.0 디렉토리
```
stock_prediction_v2/
├── data/       # v2용 데이터 (Close + LogVolume)
├── models/     # PatchTST 체크포인트
└── src/        # v2 전용 코드 (Trainer, Predictor)
```

---

## 6. v3.0 업데이트 (Risk Management)
**2026-02-18 업데이트**: 3-Way Balanced Data(급등/급락/일반)와 MC Dropout을 도입하여 안정성을 극대화했습니다.

### 🛡️ v3.0 성능 요약 (v2.0 대비)
| 모델 | 신호 수 | 정밀도 | 평균 수익률 | 특징 |
| :--- | :--- | :--- | :--- | :--- |
| **v2.0** | 5,813 | **13.9%** | **+6.26%** | 공격적. 수익 기회 포착 우수. |
| **v3.0** | **857** | 12.1% | +4.99% | **초-방어적**. 불확실한 신호 85% 제거. |

> **Analysis**: v3.0은 **"돌다리도 두드려보고 건너는"** 스타일입니다.
> - 신호 수가 대폭 줄어(5813 -> 857), 매매 기회는 적지만 **"확실하지 않으면 진입하지 않는"** 모델의 의도가 구현되었습니다.
> - **Drop Rate(급락 진입률)**가 7.2%로 제어되고 있어, 하락장 방어에 유리할 것으로 보입니다.
> - **추천**: 상승장에서는 v2.0, **횡보/하락장에서는 v3.0**을 사용하는 하이브리드 전략을 권장합니다.
> - **추천**: 상승장에서는 v2.0, **횡보/하락장에서는 v3.0**을 사용하는 하이브리드 전략을 권장합니다.

---

## 7. v4.0 업데이트 (The Masterpiece)
**2026-02-18 업데이트**: 체급별 분할 학습, 시장 지수 반영, 보조지표 활용, 앙상블 기법을 총동원한 최종 버전입니다.

### 🌟 v4.0 핵심 전략
1.  **Macro-Aware (시장 인지)**: KOSPI/KOSDAQ 지수의 52주 신고가 대비 위치를 파악하여, 강세장/약세장에 따라 배팅 강도를 조절합니다.
2.  **Segments (체급별 전략)**: 거래대금을 기준으로 **Large(대형주), Mid(중형주), Small(소형주)** 모델을 분리하여 각기 다른 시장 미시구조를 학습합니다.
3.  **Technical Fusion**: 가격뿐만 아니라 **RSI, MACD, Bollinger Bands** 등 인간 트레이더의 보조지표를 함께 봅니다.
4.  **Ensemble (집단지성)**: 각 체급별로 **5개의 모델(Seeds)**이 만장일치로 동의할 때만 진입하여 승률을 극대화합니다.

### 🏗️ 아키텍처
- **Model**: PatchTST (Multi-Channel Transformer)
- **Inputs**: 5-Channel `[Close, LogVol, RSI, MACD, Index_Level]`
- **Inference**: 5-Model Voting System (Conservative Approach)


