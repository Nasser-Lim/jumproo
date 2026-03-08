"""
V8 Streamlit Dashboard — Stock Surge Prediction

Usage:
  streamlit run stock_prediction_v8/src/app/app_v8.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import yaml
import json
import sys
import warnings
import logging

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

warnings.filterwarnings("ignore", category=RuntimeWarning)
logging.getLogger("hmmlearn").setLevel(logging.ERROR)

from stock_prediction_v7.src.model.predictor_v7 import SurgePredictor
from stock_prediction_v7.src.backtest.backtest_v7 import prepare_features
from stock_prediction_v7.src.data.create_dataset_v7 import load_raw_csv
from stock_prediction_v8.src.model.patchtst_inference import PatchTSTPredictor
from stock_prediction_v8.src.data.create_dataset_v8 import (
    compute_rsi, compute_bollinger, compute_macd
)

# --- Page Config ---
st.set_page_config(
    page_title="Surge Predictor v8",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS ---
st.markdown("""
<style>
    .signal-strong-buy {
        background: linear-gradient(135deg, #ff4444, #ff6b6b);
        color: white; padding: 20px; border-radius: 12px;
        text-align: center; font-size: 28px; font-weight: bold;
        box-shadow: 0 4px 15px rgba(255,68,68,0.4);
    }
    .signal-buy {
        background: linear-gradient(135deg, #ff8c00, #ffa940);
        color: white; padding: 20px; border-radius: 12px;
        text-align: center; font-size: 28px; font-weight: bold;
        box-shadow: 0 4px 15px rgba(255,140,0,0.3);
    }
    .signal-watch {
        background: linear-gradient(135deg, #1890ff, #40a9ff);
        color: white; padding: 20px; border-radius: 12px;
        text-align: center; font-size: 28px; font-weight: bold;
    }
    .signal-neutral {
        background: linear-gradient(135deg, #8c8c8c, #bfbfbf);
        color: white; padding: 20px; border-radius: 12px;
        text-align: center; font-size: 28px; font-weight: bold;
    }
    .metric-card {
        background: #f8f9fa; padding: 15px; border-radius: 10px;
        border-left: 4px solid #1890ff; margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

V8_ROOT = Path(__file__).parent.parent.parent
V7_ROOT = V8_ROOT.parent / "stock_prediction_v7"


@st.cache_resource
def load_predictors():
    v7_config = V7_ROOT / "configs" / "v7_config.yaml"
    v8_model_path = V8_ROOT / "models" / "patchtst" / "best_model.pt"
    stat_predictor = SurgePredictor(v7_config)
    patchtst = PatchTSTPredictor(model_path=v8_model_path, device="cpu")
    return stat_predictor, patchtst, v7_config


@st.cache_data
def load_sector_map():
    path = V8_ROOT / "configs" / "sector_map.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"stocks": {}}


@st.cache_data
def load_market_index():
    path = V8_ROOT / "data" / "market_index.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    return df


@st.cache_data
def load_ticker_names():
    path = V7_ROOT / "configs" / "ticker_names.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


@st.cache_data
def get_stock_list():
    v7_config_path = V7_ROOT / "configs" / "v7_config.yaml"
    with open(v7_config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    raw_dir = V7_ROOT / cfg["data"]["raw_dir"]
    files = sorted(raw_dir.glob("*.csv"))
    ticker_names = load_ticker_names()
    stock_map = {}
    for f in files:
        ticker = f.stem
        # ticker_names keys like "005930.KS"
        name = ticker_names.get(ticker + ".KS") or ticker_names.get(ticker + ".KQ") or ticker
        label = f"{name} ({ticker})" if name != ticker else ticker
        stock_map[label] = {"path": str(f), "ticker": ticker, "name": name}
    return stock_map


def build_8channel_context(df, market_df, context_length=60):
    """Build 8-channel context array from raw df and market_index.

    Channels:
      0: close_norm (close / current_price)
      1: log_vol_norm (z-scored log volume)
      2: RSI(14)
      3: bollinger_pos
      4: macd_norm
      5: kospi_return
      6: kosdaq_return
      7: market_vol
    """
    if len(df) < context_length + 30:
        return None, None

    close = df["Close"].values.astype(float)
    volume = df["Volume"].values.astype(float)
    dates = pd.to_datetime(df["Date"].values)

    log_vol = np.log1p(volume)
    rsi = compute_rsi(close, period=14)
    boll_pos = compute_bollinger(close, window=20)
    macd_norm = compute_macd(close)

    # Market channels — align by date
    if market_df is not None:
        kospi_ret = np.zeros(len(close))
        kosdaq_ret = np.zeros(len(close))
        mkt_vol = np.zeros(len(close))
        for i, d in enumerate(dates):
            if d in market_df.index:
                row = market_df.loc[d]
                kospi_ret[i] = float(row.get("kospi_return", 0) or 0)
                kosdaq_ret[i] = float(row.get("kosdaq_return", 0) or 0)
                mkt_vol[i] = float(row.get("market_vol", 0) or 0)
    else:
        kospi_ret = np.zeros(len(close))
        kosdaq_ret = np.zeros(len(close))
        mkt_vol = np.zeros(len(close))

    t = len(close)  # use latest window
    current_price = close[t - 1]
    if current_price <= 0:
        return None, None

    ctx_close = close[t - context_length: t] / current_price

    ctx_log_vol = log_vol[t - context_length: t]
    lv_mean = np.nanmean(ctx_log_vol)
    lv_std = np.nanstd(ctx_log_vol)
    ctx_log_vol = (ctx_log_vol - lv_mean) / lv_std if lv_std > 0 else ctx_log_vol - lv_mean

    ctx_rsi = rsi[t - context_length: t]
    ctx_boll = boll_pos[t - context_length: t]
    ctx_macd = macd_norm[t - context_length: t]
    ctx_kospi = kospi_ret[t - context_length: t]
    ctx_kosdaq = kosdaq_ret[t - context_length: t]
    ctx_mvol = mkt_vol[t - context_length: t]

    sample = np.stack([
        ctx_close, ctx_log_vol, ctx_rsi, ctx_boll,
        ctx_macd, ctx_kospi, ctx_kosdaq, ctx_mvol,
    ], axis=-1)  # (60, 8)

    sample = np.nan_to_num(sample, nan=0.0, posinf=5.0, neginf=-5.0)
    sample = np.clip(sample, -10.0, 10.0)

    return sample, current_price


def combine_scores_v8(stat_result, pt_predicted_return, surge_threshold=0.15,
                      stat_weight=0.7, pt_weight=0.3):
    """Combine stat score + PatchTST predicted_return into final score."""
    stat_score = stat_result.get("stat_score", 0.0)

    # Map predicted_return to [0, 1]: return / (surge_threshold * 2), capped at 1
    pt_score = min(max(pt_predicted_return / (surge_threshold * 2), 0.0), 1.0)

    final_score = stat_weight * stat_score + pt_weight * pt_score

    # Classify signal
    if final_score >= 0.6:
        signal = "STRONG_BUY"
    elif final_score >= 0.4:
        signal = "BUY"
    elif final_score >= 0.2:
        signal = "WATCH"
    else:
        signal = "NEUTRAL"

    return {
        **stat_result,
        "patchtst_return": pt_predicted_return,
        "patchtst_score": pt_score,
        "stat_score": stat_score,
        "final_score": final_score,
        "signal": signal,
    }


def render_signal_badge(signal, score):
    css_class = f"signal-{signal.lower().replace('_', '-')}"
    emoji = {"STRONG_BUY": "🔥", "BUY": "⚡", "WATCH": "👀", "NEUTRAL": "😐"}
    st.markdown(
        f'<div class="{css_class}">{emoji.get(signal, "")} {signal} '
        f'<br><span style="font-size:18px">Score: {score:.1%}</span></div>',
        unsafe_allow_html=True
    )


def render_component_breakdown(result, pt_result=None):
    """Show score components side by side."""
    evt = result.get("evt_prob", 0)
    hawkes = result.get("hawkes_score", 0)
    gate = result.get("gate", 1.0)
    vol_f = result.get("vol_filter", 1.0)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 통계 파이프라인 (70%)")
        data = {
            "EVT (꼬리확률)": evt,
            "Hawkes (군집도)": hawkes,
            "HMM 게이트": gate,
            "거래량 필터": vol_f,
        }
        for name, val in data.items():
            pct = min(val * 100, 100)
            color = "#52c41a" if val >= 0.5 else ("#faad14" if val >= 0.2 else "#ff4d4f")
            st.markdown(
                f'**{name}**: `{val:.3f}` '
                f'<div style="background:#f0f0f0;border-radius:4px;height:20px;width:100%">'
                f'<div style="background:{color};height:20px;border-radius:4px;'
                f'width:{pct}%;text-align:center;color:white;font-size:11px;'
                f'line-height:20px">{val:.2f}</div></div>',
                unsafe_allow_html=True
            )

    with col2:
        st.markdown("#### V8 PatchTST 회귀 (30%)")
        if pt_result is not None:
            pred_ret = pt_result.get("predicted_return", 0)
            conf_lo = pt_result.get("confidence_low", 0)
            conf_hi = pt_result.get("confidence_high", 0)
            pt_score = result.get("patchtst_score", 0)
            st.metric("예측 최대 수익률 (5일)", f"{pred_ret:.2%}")
            st.metric("신뢰구간 (10~90%)", f"{conf_lo:.2%} ~ {conf_hi:.2%}")
            st.metric("PatchTST 점수 (→ 스코어)", f"{pt_score:.3f}")
        else:
            st.info("PatchTST 모델 없음")

        st.markdown("---")
        st.metric("통계 점수", f"{result.get('stat_score', 0):.3f}")
        st.metric("최종 스코어", f"{result.get('final_score', 0):.3f}")
        st.metric("국면", result.get("regime", "unknown"))
        st.metric("클러스터 밀도", f"{result.get('cluster_density', 0):.2f}")


def render_price_chart(df, ticker, result):
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.7, 0.3], vertical_spacing=0.05,
    )
    fig.add_trace(go.Candlestick(
        x=df["Date"], open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        name="Price", increasing_line_color="#ff4444",
        decreasing_line_color="#1890ff",
    ), row=1, col=1)
    ma20 = df["Close"].rolling(20).mean()
    fig.add_trace(go.Scatter(
        x=df["Date"], y=ma20, name="MA20",
        line=dict(color="#faad14", width=1.5, dash="dot"),
    ), row=1, col=1)
    colors = ["#ff4444" if c >= o else "#1890ff"
              for c, o in zip(df["Close"], df["Open"])]
    fig.add_trace(go.Bar(
        x=df["Date"], y=df["Volume"], name="Volume",
        marker_color=colors, opacity=0.6,
    ), row=2, col=1)

    signal = result.get("signal", "NEUTRAL")
    score = result.get("final_score", 0)
    title_color = {"STRONG_BUY": "red", "BUY": "orange",
                   "WATCH": "blue", "NEUTRAL": "gray"}.get(signal, "gray")
    fig.update_layout(
        title=dict(text=f"{ticker}  |  {signal} ({score:.1%})",
                   font=dict(size=20, color=title_color)),
        height=500, xaxis_rangeslider_visible=False,
        template="plotly_white", legend=dict(orientation="h", y=1.02),
        margin=dict(l=50, r=20, t=60, b=20),
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    return fig


def generate_strategy_guide(signal, score, chg_1d, chg_5d, cluster, current_price,
                             evt, hawkes, gate, regime, pt_return=None):
    stop_loss = current_price * 0.95
    tp1 = current_price * 1.07
    tp2 = current_price * 1.10
    tp3 = current_price * 1.15

    signal_desc = {
        "STRONG_BUY": ("🔴 STRONG_BUY", "이미 급등이 진행 중. 추격매수 자제, 기보유 시 분할매도 준비."),
        "BUY": ("🟠 BUY", "주력 진입 대상. 단, 5일 수익률로 이미 올랐는지 반드시 확인."),
        "WATCH": ("🔵 WATCH", "매수 금지. 워치리스트에만 등록."),
        "NEUTRAL": ("⚪ NEUTRAL", "시그널 없음. 관망."),
    }
    header, principle = signal_desc.get(signal, ("⚪ NEUTRAL", "시그널 없음"))

    lines = [f"### 📋 매수 전략 가이드", f"**{header}** (스코어 {score:.1%})", f"> {principle}", ""]
    lines += ["| 항목 | 값 |", "|------|-----|",
              f"| 당일 등락 | {chg_1d:+.2%} |",
              f"| 5일 등락 | {chg_5d:+.2%} |",
              f"| 클러스터 밀도 | {cluster:.2f} |",
              f"| EVT 꼬리확률 | {evt:.4f} |",
              f"| Hawkes 군집도 | {hawkes:.4f} |",
              f"| HMM 국면 | {regime} (게이트 {gate:.3f}) |"]
    if pt_return is not None:
        lines.append(f"| PatchTST 예측 수익률 | {pt_return:.2%} |")
    lines.append("")

    if signal == "STRONG_BUY":
        lines += ["#### ⚠️ 판단"]
        if chg_5d > 0.30:
            lines.append(f"- 5일간 **+{chg_5d:.1%} 급등** 후 — **추격매수 금지**")
        if chg_1d < -0.10:
            lines.append(f"- 당일 **{chg_1d:+.1%} 급락 중** — 나이프 잡기 위험, 관망")
        elif chg_1d > 0.10:
            lines.append(f"- 당일 **{chg_1d:+.1%} 급등** — 보유 중이면 익절 타이밍")
        if chg_5d <= 0.30 and abs(chg_1d) <= 0.10:
            lines.append("- 극단적 변동성 국면. 소량만 진입 (비중 3%)")
        lines += ["", "#### 🎯 포지션 가이드",
                  "- **진입 비중**: 최대 3% (고위험)",
                  f"- **손절가**: ₩{stop_loss:,.0f} (-5%)",
                  f"- **1차 익절**: ₩{tp1:,.0f} (+7%) — 포지션 50% 정리",
                  f"- **2차 익절**: ₩{tp3:,.0f} (+15%) — 잔여 전량 정리",
                  "- **시간 손절**: 5거래일 내 +5% 미도달 시 전량 정리"]

    elif signal == "BUY":
        lines += ["#### 📊 판단"]
        if chg_1d > 0.10:
            lines += [f"- **B등급**: 당일 **{chg_1d:+.1%} 급등** — 내일 눌림목 확인 후 진입", "- 오늘 진입 금지"]
            weight = 3
        elif chg_5d > 0.25:
            lines += [f"- **B등급**: 5일 **+{chg_5d:.1%}** 이미 급등 — 추격 자제"]
            weight = 3
        elif cluster < 0.7:
            lines += [f"- **C등급**: 클러스터 밀도 {cluster:.2f} 낮음 — 신호 약함. 관망"]
            weight = 0
        elif chg_5d < -0.05 and cluster >= 1.0:
            lines += [f"- **A등급**: 조정 후 반등 시점 (5일 {chg_5d:+.1%}, 클러스터 {cluster:.2f})"]
            weight = 5
        else:
            lines += ["- **B등급**: 조건부 진입"]
            weight = 3
        lines += ["", "#### 🎯 포지션 가이드"]
        if weight > 0:
            lines += [f"- **진입 비중**: {weight}%",
                      f"- **손절가**: ₩{stop_loss:,.0f} (-5%)",
                      f"- **1차 익절**: ₩{tp1:,.0f} (+7%) — 포지션 50% 정리",
                      f"- **2차 익절**: ₩{tp2:,.0f} (+10%) — 잔여 전량 정리",
                      "- **시간 손절**: 5거래일 내 +5% 미도달 시 전량 정리"]
        else:
            lines += ["- **진입 비중**: 0% (관망)", "- 클러스터 밀도 1.0 이상 + BUY 유지 시 재검토"]

    elif signal == "WATCH":
        lines += ["#### 📊 판단", "- **매수 금지**. 워치리스트에만 등록",
                  "", "#### 🎯 포지션 가이드", "- **진입 비중**: 0% (관망)", "- BUY 전환 시 재검토"]
    else:
        lines += ["#### 📊 판단", "- 시그널 없음. 관망",
                  "", "#### 🎯 포지션 가이드", "- **진입 비중**: 0%"]

    lines += ["", "---", "#### ⚡ 리스크 관리 원칙",
              "| 규칙 | 기준 |", "|------|------|",
              "| 손절 | 진입가 -5% |",
              "| 1차 익절 | +7~10% (포지션 50% 정리) |",
              "| 2차 익절 | +15% (잔여 전량 정리) |",
              "| 시간 손절 | 5거래일 내 +5% 미도달 시 전량 정리 |",
              "| 1종목 최대 비중 | 총 자산의 5~10% |",
              "| 동시 보유 | 최대 3~5종목 |"]

    return "\n".join(lines)


def generate_comment(row):
    signal = row["시그널"]
    chg_1d = row["등락률"]
    chg_5d = row["5일등락"]
    cluster = row["클러스터밀도"]
    parts = []
    if signal == "STRONG_BUY":
        if chg_5d > 0.30:
            parts.append(f"5일 +{chg_5d:.0%} 급등 후. 추격매수 금지")
        if chg_1d < -0.10:
            parts.append(f"당일 {chg_1d:+.0%} 급락 중. 나이프 잡기 위험")
        elif chg_1d > 0.10:
            parts.append(f"당일 {chg_1d:+.0%} 급등. 보유 시 익절 검토")
        if not parts:
            parts.append("극단적 변동성. 소량만 진입 (비중 3%)")
        parts.append("손절 -5%")
    elif signal == "BUY":
        if chg_1d > 0.10:
            parts.append(f"당일 {chg_1d:+.0%} 급등. 내일 눌림목 확인 후 진입")
        elif chg_5d > 0.25:
            parts.append(f"5일 +{chg_5d:.0%} 이미 급등. 추격 자제")
        elif chg_5d < -0.05 and cluster >= 1.0:
            parts.append("조정 후 반등 시점. 1차 진입 추천 (비중 5%)")
        elif cluster < 0.7:
            parts.append("클러스터 밀도 낮음. 신호 약함. 관망")
        else:
            parts.append("조건부 진입 (비중 3%)")
        parts.append("손절 -5%, 익절 +10%")
    elif signal == "WATCH":
        parts.append("매수 금지. 워치리스트만")
    else:
        parts.append("신호 없음")
    return " / ".join(parts)


def scan_all_stocks(stocks, v7_config_path, patchtst, sector_map, market_df, use_pt):
    """Scan all stocks and return sorted DataFrame."""
    rows = []
    progress = st.progress(0, text="전체 종목 스캔 중...")
    total = len(stocks)

    for i, (label, info) in enumerate(stocks.items()):
        progress.progress((i + 1) / total, text=f"스캔 중: {info['name']} ({i+1}/{total})")
        try:
            df = load_raw_csv(info["path"])
            if len(df) < 120:
                continue
            features = prepare_features(df)
            idx = len(df) - 1

            predictor = SurgePredictor(v7_config_path)
            predictor.fit_stats(
                features["returns"], features["volatility"],
                features["volume_change"], current_idx=idx
            )
            stat_result = predictor.predict_stat_only(
                features["returns"], features["volatility"],
                features["volume_change"], features["volume"],
                current_idx=idx
            )

            pt_return = 0.0
            if use_pt and patchtst.model is not None:
                ticker = info["ticker"]
                sector_id = sector_map.get("stocks", {}).get(ticker, {}).get("sector_id", 0)
                context, _ = build_8channel_context(df, market_df)
                if context is not None:
                    pt_res = patchtst.predict(context, sector_id=sector_id, n_samples=5)
                    pt_return = pt_res.get("predicted_return", 0.0)

            result = combine_scores_v8(stat_result, pt_return)

            close_arr = features["close"]
            current_price = close_arr[idx]
            prev_price = close_arr[idx - 1] if idx > 0 else current_price
            price_change = (current_price - prev_price) / prev_price if prev_price > 0 else 0
            ret_5d = (current_price - close_arr[idx - 5]) / close_arr[idx - 5] if idx >= 5 else 0

            rows.append({
                "종목명": info["name"],
                "코드": info["ticker"],
                "현재가": current_price,
                "등락률": price_change,
                "5일등락": ret_5d,
                "최종스코어": result["final_score"],
                "시그널": result["signal"],
                "EVT": result.get("evt_prob", 0),
                "Hawkes": result.get("hawkes_score", 0),
                "HMM게이트": result.get("gate", 0),
                "거래량필터": result.get("vol_filter", 0),
                "국면": result.get("regime", ""),
                "클러스터밀도": result.get("cluster_density", 0),
                "PT예측수익": pt_return,
                "분석일": str(pd.Timestamp(features["dates"][idx]).date()),
            })
        except Exception:
            continue

    progress.empty()
    if not rows:
        return pd.DataFrame()

    result_df = pd.DataFrame(rows).sort_values("최종스코어", ascending=False).reset_index(drop=True)
    result_df.index += 1
    return result_df


def render_screening_page(patchtst, sector_map, market_df):
    st.title("🔎 전체 종목 스크리닝 (V8)")

    v7_config_path = V7_ROOT / "configs" / "v7_config.yaml"
    stocks = get_stock_list()

    col_f1, col_f2, col_f3 = st.columns([1, 1, 2])
    with col_f1:
        signal_filter = st.multiselect(
            "시그널 필터",
            ["STRONG_BUY", "BUY", "WATCH", "NEUTRAL"],
            default=["STRONG_BUY", "BUY", "WATCH"],
        )
    with col_f2:
        use_pt_scan = st.checkbox(
            "PatchTST 포함 (느림)",
            value=False,
            help="체크 시 종목당 MC Dropout 5회 → 전체 약 5분 추가 소요"
        )

    if "scan_result" not in st.session_state:
        st.session_state.scan_result = None

    if st.button("전체 종목 스캔 시작", type="primary"):
        st.session_state.scan_result = scan_all_stocks(
            stocks, v7_config_path, patchtst, sector_map, market_df, use_pt_scan
        )

    result_df = st.session_state.scan_result
    if result_df is None or result_df.empty:
        st.info("'전체 종목 스캔 시작' 버튼을 눌러 종목을 분석하세요.")
        return

    filtered = result_df[result_df["시그널"].isin(signal_filter)].copy()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("전체 종목", f"{len(result_df)}")
    c2.metric("STRONG_BUY", f"{(result_df['시그널'] == 'STRONG_BUY').sum()}")
    c3.metric("BUY", f"{(result_df['시그널'] == 'BUY').sum()}")
    c4.metric("WATCH", f"{(result_df['시그널'] == 'WATCH').sum()}")

    st.markdown(f"**필터 결과: {len(filtered)}개 종목**")
    if filtered.empty:
        st.warning("선택한 시그널에 해당하는 종목이 없습니다.")
        return

    page_size = 20
    total_pages = max(1, (len(filtered) + page_size - 1) // page_size)
    page_num = st.number_input("페이지", min_value=1, max_value=total_pages, value=1, step=1)
    start = (page_num - 1) * page_size
    end = min(start + page_size, len(filtered))
    page_df = filtered.iloc[start:end].copy()

    page_df["판단"] = page_df.apply(generate_comment, axis=1)

    display_df = page_df.copy()
    display_df["현재가"] = display_df["현재가"].apply(lambda x: f"₩{x:,.0f}")
    display_df["등락률"] = display_df["등락률"].apply(lambda x: f"{x:+.2%}")
    display_df["5일등락"] = display_df["5일등락"].apply(lambda x: f"{x:+.2%}")
    display_df["최종스코어"] = display_df["최종스코어"].apply(lambda x: f"{x:.1%}")
    display_df["PT예측수익"] = display_df["PT예측수익"].apply(lambda x: f"{x:.2%}")
    display_df["EVT"] = display_df["EVT"].apply(lambda x: f"{x:.3f}")
    display_df["Hawkes"] = display_df["Hawkes"].apply(lambda x: f"{x:.3f}")
    display_df["HMM게이트"] = display_df["HMM게이트"].apply(lambda x: f"{x:.3f}")
    display_df["클러스터밀도"] = display_df["클러스터밀도"].apply(lambda x: f"{x:.2f}")

    col_order = ["종목명", "코드", "현재가", "등락률", "5일등락", "최종스코어", "시그널", "판단",
                 "PT예측수익", "EVT", "Hawkes", "HMM게이트", "국면", "클러스터밀도", "분석일"]
    display_df = display_df[[c for c in col_order if c in display_df.columns]]

    def color_signal(val):
        colors = {
            "STRONG_BUY": "background-color: #ff4444; color: white",
            "BUY": "background-color: #ff8c00; color: white",
            "WATCH": "background-color: #1890ff; color: white",
            "NEUTRAL": "background-color: #bfbfbf; color: white",
        }
        return colors.get(val, "")

    st.dataframe(
        display_df.style.map(color_signal, subset=["시그널"]),
        use_container_width=True,
        height=min(len(page_df) * 38 + 40, 800),
    )
    st.caption(f"페이지 {page_num}/{total_pages} (전체 {len(filtered)}개 중 {start+1}~{end})")


def render_backtest_page():
    st.title("📊 V8 백테스트 결과")
    bt_path = V8_ROOT / "data" / "processed" / "backtest_v8_patchtst.json"
    if not bt_path.exists():
        st.info("백테스트 결과가 없습니다. backtest_v8.py를 먼저 실행하세요.")
        return

    with open(bt_path, "r") as f:
        data = json.load(f)

    for split_name, key in [("Val Set (2024.07~2025.06)", "val_results"),
                             ("Test Set (2025.07~)", "test_results")]:
        rows = data.get(key, [])
        if not rows:
            continue
        st.markdown(f"### {split_name}")
        df = pd.DataFrame(rows)
        df["threshold"] = df["threshold"].apply(lambda x: f"{x:.3f}")
        df["signals"] = df["signals"].apply(lambda x: f"{x:,}")
        df["precision"] = df["precision"].apply(lambda x: f"{x:.2%}")
        df["recall"] = df["recall"].apply(lambda x: f"{x:.2%}")
        df["avg_ret_sig"] = df["avg_ret_sig"].apply(lambda x: f"{x:.4f}")
        df.columns = ["Threshold", "신호수", "Precision", "Recall", "AvgRet(신호)"]
        st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("""
    **해석 가이드**
    - Val set base rate: **4.62%** (surge ≥15% 비율)
    - Test set base rate: **7.80%**
    - Threshold 0.125에서 Val **14.05%**, Test **27.31%** — base rate 대비 3~4x lift
    - Stats 70% + PatchTST 30% 조합으로 실제 대시보드 스코어는 추가 필터링됨
    """)


# ==================== MAIN ====================
def main():
    stat_predictor, patchtst, v7_config_path = load_predictors()
    sector_map = load_sector_map()
    market_df = load_market_index()
    stocks = get_stock_list()

    st.sidebar.title("📈 Surge Predictor v8")
    st.sidebar.markdown("---")

    model_status = "✅ 로드됨" if patchtst.model is not None else "❌ 없음 (git pull 필요)"
    st.sidebar.markdown(f"**V8 PatchTST**: {model_status}")
    if patchtst.model is None:
        st.sidebar.warning("best_model.pt 없음.\ngit pull 후 재시작.")
    st.sidebar.markdown("---")

    page = st.sidebar.radio("페이지", ["🔍 종목 분석", "🔎 전체 스크리닝", "📊 백테스트 결과"])

    if page == "🔍 종목 분석":
        st.sidebar.markdown("### 종목 선택")
        stock_labels = list(stocks.keys())
        selected_label = st.sidebar.selectbox("종목 검색 (한글/코드)", stock_labels, index=0)
        stock_info = stocks[selected_label]
        ticker = stock_info["ticker"]
        stock_name = stock_info["name"]

        use_pt = st.sidebar.checkbox(
            "PatchTST 포함 (stats 70% + PT 30%)",
            value=patchtst.model is not None,
            disabled=patchtst.model is None,
        )

        st.sidebar.markdown("---")
        st.sidebar.markdown(
            "**V8 핵심 개선**\n"
            "- 회귀 PatchTST (Huber loss)\n"
            "- 8채널 입력 (+bollinger, MACD, KOSPI/KOSDAQ, market_vol)\n"
            "- 섹터 임베딩 nn.Embedding(170, 16)\n"
            "- Stats 70% + PatchTST 30%"
        )

        df = load_raw_csv(stock_info["path"])
        features = prepare_features(df)
        idx = len(df) - 1

        stat_predictor.fit_stats(
            features["returns"], features["volatility"],
            features["volume_change"], current_idx=idx
        )
        stat_result = stat_predictor.predict_stat_only(
            features["returns"], features["volatility"],
            features["volume_change"], features["volume"],
            current_idx=idx
        )

        pt_result = None
        pt_return = 0.0
        if use_pt and patchtst.model is not None:
            sector_id = sector_map.get("stocks", {}).get(ticker, {}).get("sector_id", 0)
            context, _ = build_8channel_context(df, market_df)
            if context is not None:
                pt_result = patchtst.predict(context, sector_id=sector_id, n_samples=30)
                pt_return = pt_result.get("predicted_return", 0.0)

        result = combine_scores_v8(stat_result, pt_return)

        # Layout
        top_left, top_right = st.columns([2, 1])
        with top_left:
            recent_df = df.tail(120).copy()
            fig = render_price_chart(recent_df, f"{stock_name} ({ticker})", result)
            st.plotly_chart(fig, use_container_width=True)

        with top_right:
            render_signal_badge(result["signal"], result["final_score"])
            st.markdown("")

            close_arr = features["close"]
            current_price = close_arr[idx]
            prev_price = close_arr[idx - 1] if idx > 0 else current_price
            price_change = (current_price - prev_price) / prev_price if prev_price > 0 else 0

            c1, c2 = st.columns(2)
            c1.metric("현재가", f"₩{current_price:,.0f}", delta=f"{price_change:+.2%}")
            c2.metric("분석 일자", str(pd.Timestamp(features["dates"][idx]).date()))

        st.markdown("---")
        render_component_breakdown(result, pt_result)

        st.markdown("---")
        chg_1d = (close_arr[idx] - close_arr[idx - 1]) / close_arr[idx - 1] if idx > 0 else 0
        chg_5d = (close_arr[idx] - close_arr[idx - 5]) / close_arr[idx - 5] if idx >= 5 else 0

        guide_text = generate_strategy_guide(
            signal=result["signal"], score=result["final_score"],
            chg_1d=chg_1d, chg_5d=chg_5d,
            cluster=result.get("cluster_density", 0),
            current_price=close_arr[idx],
            evt=result.get("evt_prob", 0),
            hawkes=result.get("hawkes_score", 0),
            gate=result.get("gate", 1.0),
            regime=result.get("regime", "unknown"),
            pt_return=pt_return if use_pt else None,
        )
        st.markdown(guide_text)

    elif page == "🔎 전체 스크리닝":
        render_screening_page(patchtst, sector_map, market_df)

    elif page == "📊 백테스트 결과":
        render_backtest_page()


if __name__ == "__main__":
    main()
