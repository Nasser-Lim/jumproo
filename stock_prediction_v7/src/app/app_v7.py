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
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

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
def get_stock_list():
    config_path = Path(__file__).parent.parent.parent / "configs" / "v7_config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    raw_dir = Path(__file__).parent.parent.parent / cfg["data"]["raw_dir"]
    files = sorted(raw_dir.glob("*.csv"))
    return {f.stem: str(f) for f in files}


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
            st.metric("PatchTST 급등 확률", f"{pt:.1%}")
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

    page = st.sidebar.radio("페이지", ["🔍 종목 분석", "📊 백테스트 결과"])

    if page == "🔍 종목 분석":
        st.sidebar.markdown("### 종목 선택")
        ticker = st.sidebar.selectbox("종목 코드", list(stocks.keys()),
                                       index=0)
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
        csv_path = stocks[ticker]
        df = load_raw_csv(csv_path)
        features = prepare_features(df)

        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        context_length = cfg["data"]["context_length"]
        surge_threshold = cfg["data"]["surge_threshold"]

        idx = len(df) - cfg["data"]["prediction_length"] - 1

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
                result["forecast_5d"] = pt_result.get("forecast_5d")
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
            fig = render_price_chart(recent_df, ticker, result)
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

            if result.get("forecast_5d"):
                st.markdown("**5일 예측 가격**")
                forecast = result["forecast_5d"]
                cols = st.columns(5)
                for j, (col, p) in enumerate(zip(cols, forecast)):
                    col.metric(f"D+{j+1}", f"₩{p:,.0f}")

        st.markdown("---")
        render_component_breakdown(result)

    elif page == "📊 백테스트 결과":
        st.title("📊 V7 백테스트 결과")
        render_backtest_summary()


if __name__ == "__main__":
    main()
