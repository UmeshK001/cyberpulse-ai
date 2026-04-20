"""
app/dashboard.py
─────────────────
CyberPulse AI – Streamlit interactive dashboard.

Run with:
    streamlit run app/dashboard.py
"""

from __future__ import annotations

import sys
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from models import lstm_model, prophet_model, risk_scorer
from pipeline.trainer import load_metrics

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title=config.DASHBOARD_TITLE,
    page_icon=config.DASHBOARD_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

log = logging.getLogger(__name__)

# ── Colour palette per attack type ────────────────────────────
PALETTE = {
    "ransomware":    "#e63946",
    "phishing":      "#f4a261",
    "ddos":          "#2a9d8f",
    "sql_injection": "#e9c46a",
    "xss":           "#a8dadc",
    "rce":           "#9b2226",
    "privilege_esc": "#8338ec",
    "data_breach":   "#3a86ff",
    "zero_day":      "#fb5607",
    "supply_chain":  "#6a994e",
}

ATTACK_LABELS = {k: k.replace("_", " ").title() for k in config.ATTACK_KEYWORDS}


# ── Data loaders (cached) ─────────────────────────────────────

@st.cache_data(show_spinner="Loading historical data …")
def load_monthly() -> pd.DataFrame:
    path = config.DATA_PROC_DIR / "monthly_counts.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(show_spinner="Loading CVE features …")
def load_features() -> pd.DataFrame:
    path = config.DATA_PROC_DIR / "cve_features.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["published"])
    return risk_scorer.compute_risk_scores(df)


@st.cache_data(show_spinner="Generating LSTM forecast …")
def get_lstm_forecast(attack_type: str, steps: int) -> np.ndarray | None:
    df = load_monthly()
    if df.empty or attack_type not in df.columns:
        return None
    try:
        series = df[attack_type].values.astype(float)
        return lstm_model.forecast(series, attack_type, steps=steps)
    except FileNotFoundError:
        return None


@st.cache_data(show_spinner="Generating Prophet forecast …")
def get_prophet_forecast(attack_type: str, steps: int) -> pd.DataFrame | None:
    try:
        return prophet_model.forecast(attack_type, steps=steps)
    except FileNotFoundError:
        return None


def forecast_months_labels(df_monthly: pd.DataFrame, steps: int) -> list[str]:
    last_month = pd.Period(df_monthly["year_month"].iloc[-1], freq="M")
    return [(last_month + i + 1).strftime("%b %Y") for i in range(steps)]


# ── Sidebar ────────────────────────────────────────────────────

def sidebar() -> tuple[str, int, str]:
    st.sidebar.image(
        "https://img.icons8.com/color/96/security-checked.png", width=80
    )
    st.sidebar.title("CyberPulse AI")
    st.sidebar.markdown("*Proactive Cyber Threat Forecasting*")
    st.sidebar.divider()

    attack_type = st.sidebar.selectbox(
        "Attack Type",
        options=list(config.ATTACK_KEYWORDS.keys()),
        format_func=lambda k: ATTACK_LABELS[k],
    )

    steps = st.sidebar.slider("Forecast horizon (months)", min_value=1, max_value=12, value=6)

    industry = st.sidebar.selectbox(
        "Industry profile",
        options=list(risk_scorer.INDUSTRY_PROFILE.keys()),
    )

    st.sidebar.divider()
    st.sidebar.caption(
        "Data source: [NIST NVD](https://nvd.nist.gov/)  \n"
        "Models: Bidirectional LSTM + Prophet  \n"
        f"Forecast steps: {steps} months"
    )
    return attack_type, steps, industry


# ── Plot helpers ───────────────────────────────────────────────

def _hex_to_rgba(hex_colour: str, alpha: float = 0.15) -> str:
    """Convert a #RRGGBB hex string to an rgba() string for Plotly."""
    h = hex_colour.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def plot_historical_trend(df_monthly: pd.DataFrame, attack_type: str) -> go.Figure:
    colour = PALETTE.get(attack_type, "#636EFA")
    label  = ATTACK_LABELS[attack_type]
    if colour.startswith("#"):
        fill_colour = _hex_to_rgba(colour, 0.15)
    else:
        fill_colour = colour.replace(")", ", 0.15)").replace("rgb", "rgba")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_monthly["year_month"],
        y=df_monthly[attack_type],
        mode="lines+markers",
        name=label,
        line=dict(color=colour, width=2),
        marker=dict(size=4),
        fill="tozeroy",
        fillcolor=fill_colour,
    ))
    fig.update_layout(
        title=f"Historical Monthly CVE Count – {label}",
        xaxis_title="Month",
        yaxis_title="CVE Count",
        template="plotly_dark",
        hovermode="x unified",
        height=380,
    )
    return fig


def plot_forecast(
    df_monthly: pd.DataFrame,
    lstm_preds: np.ndarray | None,
    prophet_fc: pd.DataFrame | None,
    attack_type: str,
    steps: int,
) -> go.Figure:
    colour = PALETTE.get(attack_type, "#636EFA")
    label  = ATTACK_LABELS[attack_type]
    months = forecast_months_labels(df_monthly, steps)

    fig = go.Figure()

    # Historical context (last 18 months)
    hist = df_monthly.tail(18)
    fig.add_trace(go.Scatter(
        x=hist["year_month"], y=hist[attack_type],
        mode="lines", name="Historical",
        line=dict(color="#888", width=1.5, dash="dot"),
    ))

    if lstm_preds is not None:
        fig.add_trace(go.Scatter(
            x=months, y=lstm_preds,
            mode="lines+markers", name="LSTM Forecast",
            line=dict(color=colour, width=2.5),
            marker=dict(size=6, symbol="diamond"),
        ))

    if prophet_fc is not None:
        fig.add_trace(go.Scatter(
            x=months, y=prophet_fc["yhat"].values,
            mode="lines+markers", name="Prophet Forecast",
            line=dict(color="#ffd166", width=2, dash="dash"),
            marker=dict(size=6, symbol="circle"),
        ))
        # Confidence band
        fig.add_trace(go.Scatter(
            x=months + months[::-1],
            y=list(prophet_fc["yhat_upper"]) + list(prophet_fc["yhat_lower"])[::-1],
            fill="toself",
            fillcolor="rgba(255,209,102,0.15)",
            line=dict(color="rgba(255,209,102,0)"),
            name="95 % CI",
            showlegend=True,
        ))

    fig.update_layout(
        title=f"{label} – {steps}-Month Forecast",
        xaxis_title="Month",
        yaxis_title="Predicted CVE Count",
        template="plotly_dark",
        hovermode="x unified",
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def plot_attack_heatmap(df_monthly: pd.DataFrame) -> go.Figure:
    attack_cols = list(config.ATTACK_KEYWORDS.keys())
    available   = [c for c in attack_cols if c in df_monthly.columns]
    tail        = df_monthly.tail(24)

    z = tail[available].T.values
    x = tail["year_month"].tolist()
    y = [ATTACK_LABELS[c] for c in available]

    fig = go.Figure(go.Heatmap(
        z=z, x=x, y=y,
        colorscale="YlOrRd",
        hoverongaps=False,
    ))
    fig.update_layout(
        title="Attack-Type Intensity Heatmap (Last 24 months)",
        template="plotly_dark",
        height=400,
        xaxis_title="Month",
    )
    return fig


def plot_risk_distribution(df_feat: pd.DataFrame) -> go.Figure:
    counts = df_feat["severity_label"].value_counts().reindex(
        ["Low", "Medium", "High", "Critical", "Extreme"]
    ).fillna(0)
    colours = ["#2a9d8f", "#e9c46a", "#f4a261", "#e63946", "#9b2226"]

    fig = px.bar(
        x=counts.index, y=counts.values,
        labels={"x": "Severity", "y": "CVE Count"},
        color=counts.index,
        color_discrete_sequence=colours,
        title="CVE Severity Distribution (Risk Score model)",
        template="plotly_dark",
        height=350,
    )
    fig.update_layout(showlegend=False)
    return fig


def plot_top_risky(df_feat: pd.DataFrame) -> go.Figure:
    top = df_feat.nlargest(20, "risk_score")[["cve_id", "risk_score", "cvss_score"]].reset_index(drop=True)

    fig = go.Figure(go.Bar(
        x=top["risk_score"],
        y=top["cve_id"],
        orientation="h",
        marker=dict(
            color=top["risk_score"],
            colorscale="YlOrRd",
            showscale=True,
            colorbar=dict(title="Risk Score"),
        ),
        text=top["cvss_score"].apply(lambda s: f"CVSS {s:.1f}"),
        textposition="outside",
    ))
    fig.update_layout(
        title="Top 20 Highest-Risk CVEs",
        xaxis_title="Risk Score (0–100)",
        yaxis=dict(autorange="reversed"),
        template="plotly_dark",
        height=520,
    )
    return fig


# ── Main ───────────────────────────────────────────────────────

def main():
    attack_type, steps, industry = sidebar()

    st.title(f"{config.DASHBOARD_ICON} {config.DASHBOARD_TITLE}")
    st.caption(
        "Forecasts upcoming cyber-attack trends using Bidirectional LSTM + Prophet "
        "trained on NIST NVD data. All predictions are probabilistic estimates."
    )

    df_monthly = load_monthly()
    df_feat    = load_features()

    # ── Data not available ────────────────────────────────────
    if df_monthly.empty:
        st.error(
            "No processed data found. Please run the pipeline first:\n\n"
            "```\npython main.py --fetch --preprocess\n```"
        )
        st.stop()

    # ── KPI strip ─────────────────────────────────────────────
    total_cves   = len(df_feat) if not df_feat.empty else "–"
    months_avail = len(df_monthly)
    lstm_fc      = get_lstm_forecast(attack_type, steps)
    prophet_fc   = get_prophet_forecast(attack_type, steps)

    avg_forecast = (
        round(float(np.mean(lstm_fc)), 1) if lstm_fc is not None else
        round(float(prophet_fc["yhat"].mean()), 1) if prophet_fc is not None else "–"
    )

    recent_avg = round(float(df_monthly[attack_type].tail(6).mean()), 1) if attack_type in df_monthly else "–"
    trend_pct  = (
        round(((avg_forecast - recent_avg) / max(recent_avg, 1)) * 100, 1)
        if isinstance(avg_forecast, float) and isinstance(recent_avg, float) else "–"
    )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total CVEs analysed", f"{total_cves:,}" if isinstance(total_cves, int) else total_cves)
    k2.metric("Months of data", months_avail)
    k3.metric(f"Avg forecast – {ATTACK_LABELS[attack_type]}", avg_forecast)
    k4.metric("Trend vs last 6 months", f"{trend_pct:+}%" if isinstance(trend_pct, float) else trend_pct)

    st.divider()

    # ── Tab layout ────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Forecast", "🗺️ Heatmap", "⚠️ Risk Scores", "🏭 Industry Report"
    ])

    with tab1:
        col_hist, col_fc = st.columns([1, 1])
        with col_hist:
            st.plotly_chart(
                plot_historical_trend(df_monthly, attack_type),
                use_container_width=True,
            )
        with col_fc:
            if lstm_fc is None and prophet_fc is None:
                st.warning(
                    f"No trained models found for **{ATTACK_LABELS[attack_type]}**.\n\n"
                    "Run: `python main.py --train`"
                )
            else:
                st.plotly_chart(
                    plot_forecast(df_monthly, lstm_fc, prophet_fc, attack_type, steps),
                    use_container_width=True,
                )

        # Model metrics
        metrics = load_metrics()
        if attack_type in metrics and metrics[attack_type]:
            with st.expander("Model Evaluation Metrics"):
                m = metrics[attack_type]
                cols = st.columns(4)
                for i, (k, v) in enumerate(m.items()):
                    cols[i % 4].metric(k.replace("_", " ").upper(), f"{v:.2f}")

    with tab2:
        st.plotly_chart(plot_attack_heatmap(df_monthly), use_container_width=True)

        st.markdown("#### Attack-Type Correlation Matrix")
        attack_cols = [c for c in config.ATTACK_KEYWORDS if c in df_monthly.columns]
        corr = df_monthly[attack_cols].corr()
        fig_corr = px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            title="Monthly CVE Count Correlations",
            template="plotly_dark",
            height=460,
        )
        st.plotly_chart(fig_corr, use_container_width=True)

    with tab3:
        if df_feat.empty:
            st.warning("CVE features not found. Run `python main.py --preprocess`.")
        else:
            c1, c2 = st.columns([1, 1.6])
            with c1:
                st.plotly_chart(plot_risk_distribution(df_feat), use_container_width=True)
            with c2:
                st.plotly_chart(plot_top_risky(df_feat), use_container_width=True)

            st.markdown("#### Recent High-Risk CVEs")
            cols_show = ["cve_id", "published", "cvss_score", "risk_score", "severity_label"] + [
                c for c in config.ATTACK_KEYWORDS if c in df_feat.columns
            ]
            recent_high = (
                df_feat[df_feat["risk_score"] >= 70]
                .sort_values("published", ascending=False)
                .head(50)[cols_show]
            )
            st.dataframe(
                recent_high.style.background_gradient(
                    subset=["risk_score"], cmap="YlOrRd"
                ),
                use_container_width=True,
                height=400,
            )

    with tab4:
        st.subheader(f"Industry Risk Report – {industry}")
        st.caption(
            "Compares the forecasted threat level against recent historical averages "
            "for attack types most relevant to this sector."
        )

        # Build a combined forecast df
        rows = []
        for atk in config.ATTACK_KEYWORDS:
            fc = get_lstm_forecast(atk, steps)
            if fc is not None:
                labels = forecast_months_labels(df_monthly, steps)
                for month, count in zip(labels, fc):
                    rows.append({"attack_type": atk, "month": month, "predicted_count": count})

        if rows:
            fc_df   = pd.DataFrame(rows)
            report  = risk_scorer.industry_risk_report(fc_df, df_monthly, industry)

            st.dataframe(
                report.style.background_gradient(subset=["pct_change"], cmap="RdYlGn_r"),
                use_container_width=True,
            )

            st.plotly_chart(
                px.bar(
                    report,
                    x="attack_type", y="pct_change",
                    color="pct_change",
                    color_continuous_scale="RdYlGn_r",
                    title=f"Forecasted Change vs Recent Average – {industry}",
                    labels={"attack_type": "Attack Type", "pct_change": "% Change"},
                    template="plotly_dark",
                    height=380,
                ),
                use_container_width=True,
            )
        else:
            st.info("Train models first to generate the industry report: `python main.py --train`")

    st.divider()
    st.caption(
        "CyberPulse AI | Final Year Project | "
        "Data: NIST NVD API | Models: Bi-LSTM + Prophet"
    )


if __name__ == "__main__":
    main()
