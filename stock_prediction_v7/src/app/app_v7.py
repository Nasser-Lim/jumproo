"""
V7 Streamlit Dashboard — Stock Surge Prediction

Usage:
  streamlit run stock_prediction_v7/src/app/app_v7.py
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

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Suppress noisy warnings during scanning
warnings.filterwarnings("ignore", category=RuntimeWarning)
logging.getLogger("hmmlearn").setLevel(logging.ERROR)

from src.model.predictor_v7 import SurgePredictor
from src.model.patchtst_inference import PatchTSTPredictor
from src.backtest.backtest_v7 import prepare_features, build_context
from src.data.create_dataset_v7 import compute_rsi, load_raw_csv

# --- Page Config ---
st.set_page_config(
    page_title="Surge Predictor v7",
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
    .component-bar {
        height: 24px; border-radius: 4px; margin: 2px 0;
        display: inline-block; text-align: center; color: white;
        font-size: 12px; line-height: 24px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_predictor():
    config_path = Path(__file__).parent.parent.parent / "configs" / "v7_config.yaml"
    predictor = SurgePredictor(config_path)
    patchtst = PatchTSTPredictor(device="cpu")
    return predictor, patchtst, config_path


@st.cache_data
def load_ticker_names():
    names_path = Path(__file__).parent.parent.parent / "configs" / "ticker_names.json"
    if names_path.exists():
        with open(names_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


@st.cache_data
def get_stock_list():
    config_path = Path(__file__).parent.parent.parent / "configs" / "v7_config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    raw_dir = Path(__file__).parent.parent.parent / cfg["data"]["raw_dir"]
    files = sorted(raw_dir.glob("*.csv"))
    ticker_names = load_ticker_names()
    # {display_label: csv_path}
    stock_map = {}
    for f in files:
        ticker = f.stem
        name = ticker_names.get(ticker, ticker)
        label = f"{name} ({ticker})" if name != ticker else ticker
        stock_map[label] = {"path": str(f), "ticker": ticker, "name": name}
    return stock_map


def render_signal_badge(signal, score):
    css_class = f"signal-{signal.lower().replace('_', '-')}"
    emoji = {"STRONG_BUY": "🔥", "BUY": "⚡", "WATCH": "👀", "NEUTRAL": "😐"}
    st.markdown(
        f'<div class="{css_class}">{emoji.get(signal, "")} {signal} '
        f'<br><span style="font-size:18px">Score: {score:.1%}</span></div>',
        unsafe_allow_html=True
    )


def render_component_breakdown(result):
    """Show horizontal stacked bar of score components."""
    evt = result.get("evt_prob", 0)
    hawkes = result.get("hawkes_score", 0)
    gate = result.get("gate", 1.0)
    vol_f = result.get("vol_filter", 1.0)
    pt = result.get("patchtst_prob")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 통계 파이프라인")
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
        st.markdown("#### 최종 스코어 구성")
        if pt is not None:
            st.metric("PatchTST 급등 확률 (참고용, 미반영)", f"{pt:.1%}")
        st.metric("통계 점수", f"{result.get('stat_score', 0):.3f}")
        st.metric("국면", result.get("regime", "unknown"))
        st.metric("클러스터 밀도", f"{result.get('cluster_density', 0):.2f}")


def render_price_chart(df, ticker, result):
    """Interactive price chart with volume."""
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.05,
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df["Date"], open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        name="Price", increasing_line_color="#ff4444",
        decreasing_line_color="#1890ff",
    ), row=1, col=1)

    # 20-day MA
    ma20 = df["Close"].rolling(20).mean()
    fig.add_trace(go.Scatter(
        x=df["Date"], y=ma20, name="MA20",
        line=dict(color="#faad14", width=1.5, dash="dot"),
    ), row=1, col=1)

    # Volume
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
        title=dict(
            text=f"{ticker}  |  {signal} ({score:.1%})",
            font=dict(size=20, color=title_color),
        ),
        height=500,
        xaxis_rangeslider_visible=False,
        template="plotly_white",
        legend=dict(orientation="h", y=1.02),
        margin=dict(l=50, r=20, t=60, b=20),
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)

    return fig


def scan_all_stocks(stocks, config_path):
    """Scan all stocks and return sorted DataFrame of scores."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

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

            predictor = SurgePredictor(config_path)
            predictor.fit_stats(
                features["returns"], features["volatility"],
                features["volume_change"], current_idx=idx
            )
            result = predictor.predict_stat_only(
                features["returns"], features["volatility"],
                features["volume_change"], features["volume"],
                current_idx=idx
            )

            current_price = features["close"][idx]
            prev_price = features["close"][idx - 1] if idx > 0 else current_price
            price_change = (current_price - prev_price) / prev_price if prev_price > 0 else 0

            # 5-day return
            if idx >= 5:
                p5 = features["close"][idx - 5]
                ret_5d = (current_price - p5) / p5 if p5 > 0 else 0
            else:
                ret_5d = 0

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
                "분석일": str(pd.Timestamp(features["dates"][idx]).date()),
            })
        except Exception:
            continue

    progress.empty()

    if not rows:
        return pd.DataFrame()

    result_df = pd.DataFrame(rows)
    result_df = result_df.sort_values("최종스코어", ascending=False).reset_index(drop=True)
    result_df.index += 1  # 1-based ranking
    return result_df


def generate_strategy_guide(signal, score, chg_1d, chg_5d, cluster, current_price,
                            evt, hawkes, gate, regime):
    """Generate detailed buy strategy guide for individual stock analysis."""
    stop_loss = current_price * 0.95
    tp1 = current_price * 1.07
    tp2 = current_price * 1.10
    tp3 = current_price * 1.15

    # Signal header
    signal_desc = {
        "STRONG_BUY": ("🔴 STRONG_BUY", "이미 급등이 진행 중. 추격매수 자제, 기보유 시 분할매도 준비."),
        "BUY": ("🟠 BUY", "주력 진입 대상. 단, 5일 수익률로 이미 올랐는지 반드시 확인."),
        "WATCH": ("🔵 WATCH", "매수 금지. 워치리스트에만 등록."),
        "NEUTRAL": ("⚪ NEUTRAL", "시그널 없음. 관망."),
    }
    header, principle = signal_desc.get(signal, ("⚪ NEUTRAL", "시그널 없음"))

    lines = []
    lines.append(f"### 📋 매수 전략 가이드")
    lines.append(f"**{header}** (스코어 {score:.1%})")
    lines.append(f"> {principle}")
    lines.append("")

    # Momentum summary
    lines.append("| 항목 | 값 |")
    lines.append("|------|-----|")
    lines.append(f"| 당일 등락 | {chg_1d:+.2%} |")
    lines.append(f"| 5일 등락 | {chg_5d:+.2%} |")
    lines.append(f"| 클러스터 밀도 | {cluster:.2f} |")
    lines.append(f"| EVT 꼬리확률 | {evt:.4f} |")
    lines.append(f"| Hawkes 군집도 | {hawkes:.4f} |")
    lines.append(f"| HMM 국면 | {regime} (게이트 {gate:.3f}) |")
    lines.append("")

    if signal == "STRONG_BUY":
        # Detailed STRONG_BUY analysis
        lines.append("#### ⚠️ 판단")
        if chg_5d > 0.30:
            lines.append(f"- 5일간 **+{chg_5d:.1%} 급등** 후 — **추격매수 금지**")
        if chg_1d < -0.10:
            lines.append(f"- 당일 **{chg_1d:+.1%} 급락 중** — 나이프 잡기 위험, 관망")
        elif chg_1d > 0.10:
            lines.append(f"- 당일 **{chg_1d:+.1%} 급등** — 보유 중이면 익절 타이밍")
        if chg_5d <= 0.30 and abs(chg_1d) <= 0.10:
            lines.append(f"- 극단적 변동성 국면. 소량만 진입 (비중 3%)")
        lines.append("")
        lines.append("#### 🎯 포지션 가이드")
        lines.append(f"- **진입 비중**: 최대 3% (고위험)")
        lines.append(f"- **손절가**: ₩{stop_loss:,.0f} (-5%)")
        lines.append(f"- **1차 익절**: ₩{tp1:,.0f} (+7%) — 포지션 50% 정리")
        lines.append(f"- **2차 익절**: ₩{tp3:,.0f} (+15%) — 잔여 전량 정리")
        lines.append(f"- **시간 손절**: 5거래일 내 +5% 미도달 시 전량 정리")

    elif signal == "BUY":
        lines.append("#### 📊 판단")

        # Grade classification
        if chg_1d > 0.10:
            grade = "B"
            lines.append(f"- **B등급** (조건부 진입): 당일 **{chg_1d:+.1%} 급등**")
            lines.append(f"- 내일 시초 눌림목 (-2~3% 약세 출발) 확인 후 진입")
            lines.append(f"- 오늘 진입 금지")
            weight = 3
        elif chg_5d > 0.25:
            grade = "B"
            lines.append(f"- **B등급** (조건부): 5일 **+{chg_5d:.1%}** 이미 급등")
            lines.append(f"- 추격매수 자제. 눌림목 대기")
            weight = 3
        elif cluster < 0.7:
            grade = "C"
            lines.append(f"- **C등급** (약한 신호): 클러스터 밀도 {cluster:.2f} 낮음")
            lines.append(f"- 신호 약함. 관망 추천")
            weight = 0
        elif chg_5d < -0.05 and cluster >= 1.0:
            grade = "A"
            lines.append(f"- **A등급** (진입 추천): 조정 후 반등 시점")
            lines.append(f"- 5일 {chg_5d:+.1%} 조정 + 클러스터 {cluster:.2f} 높음")
            weight = 5
        elif abs(chg_5d) <= 0.10 and cluster >= 1.0:
            grade = "A"
            lines.append(f"- **A등급** (진입 추천): 아직 초기 단계")
            lines.append(f"- 5일 등락 {chg_5d:+.1%}로 소폭, 클러스터 {cluster:.2f} 높음")
            weight = 5
        else:
            grade = "B"
            lines.append(f"- **B등급** (조건부 진입)")
            weight = 3

        lines.append("")
        lines.append("#### 🎯 포지션 가이드")
        if weight > 0:
            lines.append(f"- **진입 비중**: {weight}%")
            lines.append(f"- **손절가**: ₩{stop_loss:,.0f} (-5%)")
            lines.append(f"- **1차 익절**: ₩{tp1:,.0f} (+7%) — 포지션 50% 정리")
            lines.append(f"- **2차 익절**: ₩{tp2:,.0f} (+10%) — 잔여 전량 정리")
            lines.append(f"- **시간 손절**: 5거래일 내 +5% 미도달 시 전량 정리")
        else:
            lines.append(f"- **진입 비중**: 0% (관망)")
            lines.append(f"- 클러스터 밀도 1.0 이상 + BUY 유지 시 재검토")

    elif signal == "WATCH":
        lines.append("#### 📊 판단")
        lines.append("- **매수 금지**. 워치리스트에만 등록")
        if chg_5d < -0.15:
            lines.append(f"- 5일 **{chg_5d:+.1%} 급락** 중. BUY 전환 시 재검토")
        else:
            lines.append("- 신호 약함. 추가 모멘텀 확인 필요")
        lines.append("")
        lines.append("#### 🎯 포지션 가이드")
        lines.append("- **진입 비중**: 0% (관망)")
        lines.append("- BUY 전환 시 재검토")

    else:  # NEUTRAL
        lines.append("#### 📊 판단")
        lines.append("- 시그널 없음. 관망")
        lines.append("")
        lines.append("#### 🎯 포지션 가이드")
        lines.append("- **진입 비중**: 0%")

    # Risk management footer (always shown)
    lines.append("")
    lines.append("---")
    lines.append("#### ⚡ 리스크 관리 원칙")
    lines.append("| 규칙 | 기준 |")
    lines.append("|------|------|")
    lines.append("| 손절 | 진입가 -5% |")
    lines.append("| 1차 익절 | +7~10% (포지션 50% 정리) |")
    lines.append("| 2차 익절 | +15% (잔여 전량 정리) |")
    lines.append("| 시간 손절 | 5거래일 내 +5% 미도달 시 전량 정리 |")
    lines.append("| 1종목 최대 비중 | 총 자산의 5~10% |")
    lines.append("| 동시 보유 | 최대 3~5종목 |")

    return "\n".join(lines)


def generate_comment(row):
    """Generate actionable comment based on signal, price momentum, and indicators."""
    signal = row["시그널"]
    chg_1d = row["등락률"]
    chg_5d = row["5일등락"]
    cluster = row["클러스터밀도"]
    score = row["최종스코어"]

    parts = []

    # STRONG_BUY specific warnings
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

    # BUY
    elif signal == "BUY":
        if chg_1d > 0.10:
            parts.append(f"당일 {chg_1d:+.0%} 급등. 내일 눌림목 확인 후 진입")
        elif chg_5d > 0.25:
            parts.append(f"5일 +{chg_5d:.0%} 이미 급등. 추격 자제")
        elif chg_5d < -0.05 and cluster >= 1.0:
            parts.append("조정 후 반등 시점. 1차 진입 추천 (비중 5%)")
        elif abs(chg_5d) <= 0.10 and cluster >= 1.0:
            parts.append("아직 초기 단계. 1차 진입 추천 (비중 5%)")
        elif cluster < 0.7:
            parts.append("클러스터 밀도 낮음. 신호 약함. 관망")
        else:
            parts.append("조건부 진입 (비중 3%)")
        parts.append("손절 -5%, 익절 +10%")

    # WATCH
    elif signal == "WATCH":
        parts.append("매수 금지. 워치리스트만")
        if chg_5d < -0.15:
            parts.append(f"5일 {chg_5d:+.0%} 급락 중. BUY 전환 시 재검토")
        else:
            parts.append("신호 약함. 관망")

    # NEUTRAL
    else:
        parts.append("신호 없음")

    return " / ".join(parts)


def render_screening_page():
    """Full stock screening with pagination."""
    st.title("🔎 전체 종목 스크리닝")

    config_path = Path(__file__).parent.parent.parent / "configs" / "v7_config.yaml"
    stocks = get_stock_list()

    # Signal filter
    col_f1, col_f2 = st.columns([1, 3])
    with col_f1:
        signal_filter = st.multiselect(
            "시그널 필터",
            ["STRONG_BUY", "BUY", "WATCH", "NEUTRAL"],
            default=["STRONG_BUY", "BUY", "WATCH"],
        )

    # Scan button with session state caching
    if "scan_result" not in st.session_state:
        st.session_state.scan_result = None

    if st.button("전체 종목 스캔 시작", type="primary"):
        st.session_state.scan_result = scan_all_stocks(stocks, config_path)

    result_df = st.session_state.scan_result
    if result_df is None or result_df.empty:
        st.info("'전체 종목 스캔 시작' 버튼을 눌러 389개 종목을 분석하세요.")
        return

    # Apply signal filter
    filtered = result_df[result_df["시그널"].isin(signal_filter)].copy()

    # Summary metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("전체 종목", f"{len(result_df)}")
    c2.metric("STRONG_BUY", f"{(result_df['시그널'] == 'STRONG_BUY').sum()}")
    c3.metric("BUY", f"{(result_df['시그널'] == 'BUY').sum()}")
    c4.metric("WATCH", f"{(result_df['시그널'] == 'WATCH').sum()}")

    st.markdown(f"**필터 결과: {len(filtered)}개 종목**")

    if filtered.empty:
        st.warning("선택한 시그널에 해당하는 종목이 없습니다.")
        return

    # Pagination
    page_size = 20
    total_pages = max(1, (len(filtered) + page_size - 1) // page_size)
    page_num = st.number_input("페이지", min_value=1, max_value=total_pages, value=1, step=1)

    start = (page_num - 1) * page_size
    end = min(start + page_size, len(filtered))
    page_df = filtered.iloc[start:end].copy()

    # Generate comments
    page_df["판단"] = page_df.apply(generate_comment, axis=1)

    # Format for display
    display_df = page_df.copy()
    display_df["현재가"] = display_df["현재가"].apply(lambda x: f"₩{x:,.0f}")
    display_df["등락률"] = display_df["등락률"].apply(lambda x: f"{x:+.2%}")
    display_df["5일등락"] = display_df["5일등락"].apply(lambda x: f"{x:+.2%}")
    display_df["최종스코어"] = display_df["최종스코어"].apply(lambda x: f"{x:.1%}")
    display_df["EVT"] = display_df["EVT"].apply(lambda x: f"{x:.3f}")
    display_df["Hawkes"] = display_df["Hawkes"].apply(lambda x: f"{x:.3f}")
    display_df["HMM게이트"] = display_df["HMM게이트"].apply(lambda x: f"{x:.3f}")
    display_df["거래량필터"] = display_df["거래량필터"].apply(lambda x: f"{x:.2f}")
    display_df["클러스터밀도"] = display_df["클러스터밀도"].apply(lambda x: f"{x:.2f}")

    # Reorder columns to put 판단 after 시그널
    col_order = ["종목명", "코드", "현재가", "등락률", "5일등락", "최종스코어", "시그널", "판단",
                 "EVT", "Hawkes", "HMM게이트", "거래량필터", "국면", "클러스터밀도", "분석일"]
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


def render_backtest_summary():
    """Show backtest results if available."""
    out_dir = Path(__file__).parent.parent.parent / "outputs"
    files = list(out_dir.glob("backtest_v7_*.csv"))
    if not files:
        st.info("백테스트 결과가 없습니다. backtest_v7.py를 먼저 실행하세요.")
        return

    for f in sorted(files):
        st.markdown(f"### 📊 {f.stem}")
        df = pd.read_csv(f)

        col1, col2, col3 = st.columns(3)
        col1.metric("총 예측 수", f"{len(df):,}")
        col2.metric("기본 급등률", f"{df.is_actual_surge.mean():.1%}")
        col3.metric("데이터 기간", f"{df.date.min()} ~ {df.date.max()}")

        # Threshold analysis
        rows = []
        for th in [0.2, 0.3, 0.4, 0.5, 0.6]:
            sig = df[df.final_score >= th]
            if len(sig) > 0:
                rows.append({
                    "Threshold": f"≥ {th:.1f}",
                    "신호 수": len(sig),
                    "Precision": f"{sig.is_actual_surge.mean():.1%}",
                    "평균 수익률": f"{sig.actual_max_return.mean():+.2%}",
                    "중앙 수익률": f"{sig.actual_max_return.median():+.2%}",
                })
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Score distribution
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=df[df.is_actual_surge == True].final_score,
            name="실제 급등", marker_color="#ff4444", opacity=0.7, nbinsx=50,
        ))
        fig.add_trace(go.Histogram(
            x=df[df.is_actual_surge == False].final_score,
            name="비급등", marker_color="#1890ff", opacity=0.5, nbinsx=50,
        ))
        fig.update_layout(
            title="Score Distribution: 급등 vs 비급등",
            barmode="overlay", height=350, template="plotly_white",
            xaxis_title="Final Score", yaxis_title="Count",
        )
        st.plotly_chart(fig, use_container_width=True)


# ==================== MAIN ====================
def main():
    predictor, patchtst, config_path = load_predictor()
    stocks = get_stock_list()

    # Sidebar
    st.sidebar.title("📈 Surge Predictor v7")
    st.sidebar.markdown("---")

    page = st.sidebar.radio("페이지", ["🔍 종목 분석", "🔎 전체 스크리닝", "📊 백테스트 결과"])

    if page == "🔍 종목 분석":
        st.sidebar.markdown("### 종목 선택")
        stock_labels = list(stocks.keys())
        selected_label = st.sidebar.selectbox("종목 검색 (한글/코드)", stock_labels,
                                               index=0)
        stock_info = stocks[selected_label]
        ticker = stock_info["ticker"]
        stock_name = stock_info["name"]

        use_pt = st.sidebar.checkbox("PatchTST 포함",
                                      value=patchtst.model is not None,
                                      disabled=patchtst.model is None)

        if patchtst.model is None:
            st.sidebar.warning("PatchTST 모델 없음.\n학습 후 git pull 필요.")

        st.sidebar.markdown("---")
        st.sidebar.markdown(
            "**v7 핵심 개선사항**\n"
            "- 시간 기반 Train/Val/Test 분리\n"
            "- Rolling window (504일)\n"
            "- Strictly past data only\n"
            "- CPU/GPU 자동 선택"
        )

        # Main content
        csv_path = stock_info["path"]
        df = load_raw_csv(csv_path)
        features = prepare_features(df)

        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        context_length = cfg["data"]["context_length"]
        surge_threshold = cfg["data"]["surge_threshold"]

        # Use latest data point for live analysis (not backtest offset)
        idx = len(df) - 1

        # Fit & predict
        predictor.fit_stats(
            features["returns"], features["volatility"],
            features["volume_change"], current_idx=idx
        )

        if use_pt and patchtst.model is not None:
            context, current_price = build_context(features, idx, context_length)
            if context is not None:
                pt_result = patchtst.predict(context, current_price, surge_threshold)
                stat_result = predictor.predict_stat(
                    features["returns"], features["volatility"],
                    features["volume_change"], features["volume"],
                    current_idx=idx
                )
                result = predictor.combine_scores(pt_result["surge_prob"], stat_result)
                result["pt_confidence_low"] = pt_result.get("confidence_low")
                result["pt_confidence_high"] = pt_result.get("confidence_high")
            else:
                result = predictor.predict_stat_only(
                    features["returns"], features["volatility"],
                    features["volume_change"], features["volume"],
                    current_idx=idx
                )
        else:
            result = predictor.predict_stat_only(
                features["returns"], features["volatility"],
                features["volume_change"], features["volume"],
                current_idx=idx
            )

        # Layout
        top_left, top_right = st.columns([2, 1])

        with top_left:
            recent_df = df.tail(120).copy()
            chart_title = f"{stock_name} ({ticker})"
            fig = render_price_chart(recent_df, chart_title, result)
            st.plotly_chart(fig, use_container_width=True)

        with top_right:
            render_signal_badge(result["signal"], result["final_score"])
            st.markdown("")

            current_price = features["close"][idx - 1]
            price_change = (features["close"][idx - 1] - features["close"][idx - 2]) / features["close"][idx - 2]

            c1, c2 = st.columns(2)
            c1.metric("현재가", f"₩{current_price:,.0f}",
                      delta=f"{price_change:+.2%}")
            c2.metric("분석 일자", str(pd.Timestamp(features["dates"][idx]).date()))

            if result.get("pt_confidence_low") is not None:
                st.caption("⚠️ PatchTST — 참고용 (미반영)")
                c3, c4 = st.columns(2)
                c3.metric("PT 급등확률", f"{result.get('patchtst_prob', 0):.1%}")
                c4.metric("PT 신뢰구간", f"{result['pt_confidence_low']:.1%} ~ {result['pt_confidence_high']:.1%}")

        st.markdown("---")
        render_component_breakdown(result)

        # Buy strategy guide
        st.markdown("---")
        current_price_for_guide = features["close"][idx]
        chg_1d = (features["close"][idx] - features["close"][idx - 1]) / features["close"][idx - 1] if idx > 0 else 0
        chg_5d = (features["close"][idx] - features["close"][idx - 5]) / features["close"][idx - 5] if idx >= 5 else 0

        guide_text = generate_strategy_guide(
            signal=result["signal"],
            score=result["final_score"],
            chg_1d=chg_1d,
            chg_5d=chg_5d,
            cluster=result.get("cluster_density", 0),
            current_price=current_price_for_guide,
            evt=result.get("evt_prob", 0),
            hawkes=result.get("hawkes_score", 0),
            gate=result.get("gate", 1.0),
            regime=result.get("regime", "unknown"),
        )
        st.markdown(guide_text)

    elif page == "🔎 전체 스크리닝":
        render_screening_page()

    elif page == "📊 백테스트 결과":
        st.title("📊 V7 백테스트 결과")
        render_backtest_summary()


if __name__ == "__main__":
    main()
