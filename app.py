import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from model import load_data, preprocess, train_and_forecast, compute_risk_scores, get_country_list

st.set_page_config(
    page_title="EpiWatch — Epidemic Spread Predictor",
    page_icon="🌍",
    layout="wide"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title { font-size: 2.2rem; font-weight: 700; color: #1D9E75; margin-bottom: 0; }
    .sub-title  { font-size: 1rem; color: #888; margin-bottom: 1.5rem; }
    .metric-card {
        background: #f8f9fa; border-radius: 12px;
        padding: 16px 20px; text-align: center;
        border: 1px solid #e9ecef;
    }
    .metric-val  { font-size: 1.6rem; font-weight: 700; color: #1D9E75; }
    .metric-label{ font-size: 0.8rem; color: #888; margin-top: 4px; }
    .section-title { font-size: 1.2rem; font-weight: 600; margin: 1.5rem 0 0.5rem 0; }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">EpiWatch</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">AI-Powered Epidemic Spread Predictor · CodeCure 2026 · Track C</div>', unsafe_allow_html=True)

# ── Load data (cached so it only loads once) ───────────────────────────────────
@st.cache_data(show_spinner=False)
def get_data():
    raw = load_data()
    return preprocess(raw)

with st.spinner("Loading JHU COVID-19 dataset..."):
    df_long = get_data()

countries = get_country_list(df_long)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/virus.png", width=60)
    st.title("Controls")

    selected_country = st.selectbox(
        "Country for detailed forecast",
        options=countries,
        index=countries.index("India") if "India" in countries else 0
    )

    forecast_days = st.slider("Forecast horizon (days)", 7, 60, 30)
    top_n = st.slider("Countries to rank", 5, 30, 15)

    st.markdown("---")
    st.caption("Data: JHU CSSE COVID-19 Dataset")
    st.caption("Model: Facebook Prophet")
    st.caption("Built with Streamlit + Plotly")

# ── Compute risk scores ────────────────────────────────────────────────────────
with st.spinner("Computing global risk scores..."):
    risk_df = compute_risk_scores(df_long, top_n=top_n)

# ── KPI Cards ──────────────────────────────────────────────────────────────────
latest_date   = df_long["date"].max().strftime("%d %b %Y")
total_countries = df_long["country"].nunique()
highest_risk  = risk_df.iloc[0]["country"] if len(risk_df) > 0 else "N/A"
avg_growth    = risk_df["growth_rate"].mean() if len(risk_df) > 0 else 0

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f'''<div class="metric-card">
        <div class="metric-val">{total_countries}</div>
        <div class="metric-label">Countries tracked</div>
    </div>''', unsafe_allow_html=True)
with c2:
    st.markdown(f'''<div class="metric-card">
        <div class="metric-val">{latest_date}</div>
        <div class="metric-label">Data last updated</div>
    </div>''', unsafe_allow_html=True)
with c3:
    st.markdown(f'''<div class="metric-card">
        <div class="metric-val">{highest_risk}</div>
        <div class="metric-label">Highest risk country</div>
    </div>''', unsafe_allow_html=True)
with c4:
    st.markdown(f'''<div class="metric-card">
        <div class="metric-val">{avg_growth:.1f}%</div>
        <div class="metric-label">Avg 14-day growth rate</div>
    </div>''', unsafe_allow_html=True)

st.markdown("---")

# ── World Risk Map ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">🌍 Global Risk Map</div>', unsafe_allow_html=True)
st.caption("Color intensity = projected outbreak risk score based on case growth rate × volume")

fig_map = px.choropleth(
    risk_df,
    locations="country",
    locationmode="country names",
    color="risk_score",
    hover_name="country",
    hover_data={
        "risk_score": ":.1f",
        "growth_rate": ":.1f",
        "avg_daily_cases": ":,.0f"
    },
    color_continuous_scale="YlOrRd",
    labels={
        "risk_score": "Risk score",
        "growth_rate": "Growth rate (%)",
        "avg_daily_cases": "Avg daily cases"
    }
)
fig_map.update_layout(
    margin=dict(l=0, r=0, t=10, b=0),
    geo=dict(showframe=False, showcoastlines=True, bgcolor="rgba(0,0,0,0)"),
    paper_bgcolor="rgba(0,0,0,0)",
    coloraxis_colorbar=dict(title="Risk score")
)
st.plotly_chart(fig_map, use_container_width=True)

st.markdown("---")

# ── Forecast + Risk Table ──────────────────────────────────────────────────────
left_col, right_col = st.columns([2, 1])

with left_col:
    st.markdown(f'<div class="section-title">📈 {forecast_days}-Day Forecast — {selected_country}</div>', unsafe_allow_html=True)

    with st.spinner(f"Training forecast model for {selected_country}..."):
        forecast, actual = train_and_forecast(df_long, selected_country, forecast_days)

    if forecast is not None and actual is not None:
        fig = go.Figure()

        # Actual historical cases
        fig.add_trace(go.Scatter(
            x=actual["ds"], y=actual["y"],
            name="Actual cases",
            line=dict(color="#1D9E75", width=2.5)
        ))

        # Future forecast only
        future_mask = forecast["ds"] > actual["ds"].max()
        future = forecast[future_mask]

        fig.add_trace(go.Scatter(
            x=future["ds"], y=future["yhat"],
            name="Forecast",
            line=dict(color="#D85A30", width=2.5, dash="dash")
        ))

        # Confidence interval
        fig.add_trace(go.Scatter(
            x=pd.concat([future["ds"], future["ds"][::-1]]),
            y=pd.concat([future["yhat_upper"], future["yhat_lower"][::-1]]),
            fill="toself",
            fillcolor="rgba(216, 90, 48, 0.12)",
            line=dict(color="rgba(255,255,255,0)"),
            name="80% confidence interval"
        ))

        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Daily new cases (7-day avg)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            margin=dict(l=0, r=0, t=30, b=0),
            hovermode="x unified",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        fig.update_xaxes(showgrid=True, gridcolor="#f0f0f0")
        fig.update_yaxes(showgrid=True, gridcolor="#f0f0f0")

        st.plotly_chart(fig, use_container_width=True)

        # Peak info
        if len(future) > 0:
            peak_val  = int(future["yhat"].max())
            peak_date = future.loc[future["yhat"].idxmax(), "ds"].strftime("%d %b %Y")
            st.info(f"Projected peak: **{peak_val:,} cases/day** around **{peak_date}**")
    else:
        st.warning(f"Not enough historical data to forecast {selected_country}.")

with right_col:
    st.markdown('<div class="section-title">🚨 Top At-Risk Countries</div>', unsafe_allow_html=True)

    display = risk_df[["country", "risk_score", "growth_rate", "avg_daily_cases"]].copy()
    display.columns = ["Country", "Risk score", "Growth %", "Avg daily cases"]
    display["Risk score"]      = display["Risk score"].apply(lambda x: f"{x:.1f}")
    display["Growth %"]        = display["Growth %"].apply(lambda x: f"{x:.1f}%")
    display["Avg daily cases"] = display["Avg daily cases"].apply(lambda x: f"{x:,.0f}")
    display.index = range(1, len(display) + 1)

    st.dataframe(display, use_container_width=True, height=400)

st.markdown("---")

# ── Historical Trends ──────────────────────────────────────────────────────────
st.markdown('<div class="section-title">📊 Historical Case Trends — Top 5 At-Risk Countries</div>', unsafe_allow_html=True)

top5 = risk_df["country"].head(5).tolist()
trend = df_long[
    (df_long["country"].isin(top5)) &
    (df_long["date"] >= "2021-01-01")
].copy()

fig_trend = px.line(
    trend,
    x="date", y="cases_smooth", color="country",
    labels={
        "cases_smooth": "7-day avg daily cases",
        "date": "Date",
        "country": "Country"
    },
    color_discrete_sequence=px.colors.qualitative.Set2
)
fig_trend.update_layout(
    margin=dict(l=0, r=0, t=10, b=0),
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)"
)
fig_trend.update_xaxes(showgrid=True, gridcolor="#f0f0f0")
fig_trend.update_yaxes(showgrid=True, gridcolor="#f0f0f0")
st.plotly_chart(fig_trend, use_container_width=True)

# ── Wave Detection ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-title">🔍 Wave Detection — ' + selected_country + '</div>', unsafe_allow_html=True)
st.caption("Identifying distinct outbreak waves based on 7-day rolling average peaks")

country_hist = df_long[df_long["country"] == selected_country].copy()
country_hist["peak"] = (
    (country_hist["cases_smooth"] > country_hist["cases_smooth"].shift(1)) &
    (country_hist["cases_smooth"] > country_hist["cases_smooth"].shift(-1)) &
    (country_hist["cases_smooth"] > country_hist["cases_smooth"].quantile(0.75))
)

fig_wave = go.Figure()
fig_wave.add_trace(go.Scatter(
    x=country_hist["date"], y=country_hist["cases_smooth"],
    name="Daily cases (smoothed)",
    fill="tozeroy",
    fillcolor="rgba(29, 158, 117, 0.1)",
    line=dict(color="#1D9E75", width=2)
))

peaks = country_hist[country_hist["peak"]]
fig_wave.add_trace(go.Scatter(
    x=peaks["date"], y=peaks["cases_smooth"],
    mode="markers",
    name="Wave peaks",
    marker=dict(color="#D85A30", size=8, symbol="triangle-up")
))

fig_wave.update_layout(
    xaxis_title="Date",
    yaxis_title="7-day avg daily cases",
    margin=dict(l=0, r=0, t=10, b=0),
    hovermode="x unified",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)"
)
fig_wave.update_xaxes(showgrid=True, gridcolor="#f0f0f0")
fig_wave.update_yaxes(showgrid=True, gridcolor="#f0f0f0")
st.plotly_chart(fig_wave, use_container_width=True)

st.markdown("---")
st.caption("EpiWatch · CodeCure 2026 · Data: JHU CSSE · Model: Facebook Prophet · Stack: Python, Streamlit, Plotly")