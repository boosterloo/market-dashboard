# pages/Yield_US_vs_EU.py
# 🇺🇸 vs 🇪🇺 — kernvergelijking + 90d μ±1σ-band & Z-score toggle
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from google.api_core.exceptions import NotFound, BadRequest
from utils.bq import run_query, bq_ping

st.set_page_config(page_title="🇺🇸 vs 🇪🇺 Yield Curve", layout="wide")
st.title("🇺🇸 vs 🇪🇺 Yield Curve Vergelijking")

# ---------- Health ----------
try:
    if not bq_ping():
        st.error("Geen BigQuery-verbinding.")
        st.stop()
except Exception as e:
    st.error("Geen BigQuery-verbinding.")
    st.caption(f"Details: {e}")
    st.stop()

# ---------- Tables/Views uit secrets (met sane defaults) ----------
US_VIEW = st.secrets.get("tables", {}).get("us_yield_view", "nth-pier-468314-p7.marketdata.us_yield_v")
EU_VIEW = st.secrets.get("tables", {}).get("eu_yield_view", "nth-pier-468314-p7.marketdata.eu_yield_v")
WIDE_VIEW = st.secrets.get("tables", {}).get("yield_wide_view", "nth-pier-468314-p7.marketdata.yield_curve_analysis_wide")

with st.expander("🔎 Debug: gebruikte bronnen"):
    st.write({"US_VIEW": US_VIEW, "EU_VIEW": EU_VIEW, "WIDE_VIEW": WIDE_VIEW})

# ---------- Data loaders ----------
def load_from_two_views():
    q = f"""
    WITH us AS (
      SELECT date, y_2y AS us_2y, y_10y AS us_10y, (y_10y - y_2y) AS us_spread
      FROM `{US_VIEW}`
    ),
    eu AS (
      SELECT date, y_2y AS eu_2y, y_10y AS eu_10y, (y_10y - y_2y) AS eu_spread
      FROM `{EU_VIEW}`
    )
    SELECT
      us.date,
      us.us_2y, us.us_10y, us.us_spread,
      eu.eu_2y, eu.eu_10y, eu.eu_spread,
      (us.us_2y - eu.eu_2y)           AS diff_2y,
      (us.us_10y - eu.eu_10y)         AS diff_10y,
      (us.us_spread - eu.eu_spread)   AS diff_spread
    FROM us
    JOIN eu USING(date)
    ORDER BY date
    """
    return run_query(q)

def load_from_wide_view():
    # Verwacht schema: date, region ('US'/'EU'), y_2y, y_10y
    q = f"""
    WITH base AS (
      SELECT date, region, y_2y, y_10y
      FROM `{WIDE_VIEW}`
      WHERE region IN ('US','EU')
    ),
    pivot AS (
      SELECT
        date,
        MAX(IF(region='US', y_2y, NULL))  AS us_2y,
        MAX(IF(region='US', y_10y, NULL)) AS us_10y,
        MAX(IF(region='EU', y_2y, NULL))  AS eu_2y,
        MAX(IF(region='EU', y_10y, NULL)) AS eu_10y
      FROM base
      GROUP BY date
    )
    SELECT
      date,
      us_2y, us_10y,
      (us_10y - us_2y) AS us_spread,
      eu_2y, eu_10y,
      (eu_10y - eu_2y) AS eu_spread,
      (us_2y - eu_2y)  AS diff_2y,
      (us_10y - eu_10y) AS diff_10y,
      ((us_10y - us_2y) - (eu_10y - eu_2y)) AS diff_spread
    FROM pivot
    WHERE us_2y IS NOT NULL AND eu_2y IS NOT NULL
    ORDER BY date
    """
    return run_query(q)

@st.cache_data(ttl=1800, show_spinner=False)
def load_data_resilient():
    # 1) Probeer de twee losse views
    try:
        return load_from_two_views(), "two_views"
    except (NotFound, BadRequest) as e1:
        st.info("Kon US/EU losse views niet laden — probeer fallback via brede view.")
        with st.expander("Technische foutmelding (losse views)"):
            st.code(str(e1))
        # 2) Fallback naar brede view
        try:
            return load_from_wide_view(), "wide_view"
        except (NotFound, BadRequest) as e2:
            with st.expander("Technische foutmelding (brede view)"):
                st.code(str(e2))
            raise

# ---------- Load ----------
with st.spinner("Data laden…"):
    try:
        df, mode = load_data_resilient()
    except Exception as e:
        st.error("Kon geen yield-data laden uit BigQuery.")
        st.caption("Controleer of de opgegeven views/tables bestaan en of de kolommen y_2y en y_10y aanwezig zijn.")
        st.stop()

if df.empty:
    st.warning("Geen data gevonden (join op datum kan leeg zijn).")
    st.stop()

st.caption(f"Bronmodus: **{mode}**")

# ---------- Helpers ----------
def add_roll_stats(frame: pd.DataFrame, col: str, window: int = 90, minp: int = 30):
    mu = frame[col].rolling(window, min_periods=minp).mean()
    sd = frame[col].rolling(window, min_periods=minp).std()
    frame[f"{col}_mu"] = mu
    frame[f"{col}_sd"] = sd
    frame[f"{col}_z"]  = (frame[col] - mu) / sd
    return frame

for c in ["diff_spread", "diff_2y", "diff_10y"]:
    df = add_roll_stats(df, c)

# ---------- Term structure overlay ----------
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=df["date"], y=df["us_2y"],  mode="lines", name="US 2Y"))
fig1.add_trace(go.Scatter(x=df["date"], y=df["us_10y"], mode="lines", name="US 10Y"))
fig1.add_trace(go.Scatter(x=df["date"], y=df["eu_2y"],  mode="lines", name="EU 2Y", yaxis="y2"))
fig1.add_trace(go.Scatter(x=df["date"], y=df["eu_10y"], mode="lines", name="EU 10Y", yaxis="y2"))
fig1.update_layout(
    title="Yield Curves US vs EU (2Y & 10Y)",
    xaxis_title="Date",
    yaxis=dict(title="US Yield (%)"),
    yaxis2=dict(title="EU Yield (%)", overlaying="y", side="right"),
    legend=dict(orientation="h")
)
st.plotly_chart(fig1, use_container_width=True)

# ---------- Spreads ----------
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=df["date"], y=df["us_spread"], mode="lines", name="US 10Y–2Y"))
fig2.add_trace(go.Scatter(x=df["date"], y=df["eu_spread"], mode="lines", name="EU 10Y–2Y"))
fig2.add_trace(go.Bar(x=df["date"], y=df["diff_spread"], name="US–EU diff (bp)", opacity=0.35))
fig2.update_layout(
    title="10Y–2Y Spreads (US, EU) + US–EU differential",
    xaxis_title="Date",
    yaxis_title="Basis points",
    legend=dict(orientation="h")
)
st.plotly_chart(fig2, use_container_width=True)

# ---------- Rolling band & Z-scores ----------
st.subheader("US–EU differentials — 90d μ ± 1σ & Z-scores")
show_band = st.checkbox("Toon 90d μ ± 1σ band", value=True)
as_z = st.checkbox("Toon als Z-scores (rolling, 90d)", value=False)
tabs = st.tabs(["Spread (10Y–2Y)", "2Y", "10Y"])

def plot_diff(tab, df: pd.DataFrame, base_col: str, title_txt: str, unit: str):
    with tab:
        series = df[f"{base_col}_z"] if as_z else df[base_col]
        mu = df[f"{base_col}_mu"]
        sd = df[f"{base_col}_sd"]

        if as_z:
            y_title = "Z-score"
            upper = (mu + sd - mu) / sd  # +1
            lower = (mu - sd - mu) / sd  # -1
            mu_plot = (mu - mu) / sd     # 0
        else:
            y_title = unit
            upper = mu + sd
            lower = mu - sd
            mu_plot = mu

        fig = go.Figure()
        if show_band:
            fig.add_trace(go.Scatter(x=df["date"], y=upper, name="μ+1σ", mode="lines", line=dict(width=0.5), showlegend=False))
            fig.add_trace(go.Scatter(x=df["date"], y=lower, name="μ-1σ", mode="lines", fill="tonexty", line=dict(width=0.5), opacity=0.20, showlegend=False))
        fig.add_trace(go.Scatter(x=df["date"], y=mu_plot, name="μ (90d)", mode="lines", line=dict(width=1, dash="dot")))
        fig.add_trace(go.Scatter(x=df["date"], y=series, name="US–EU differential", mode="lines"))
        fig.update_layout(title=title_txt + (" — Z-scores" if as_z else ""), xaxis_title="Date", yaxis_title=y_title, legend=dict(orientation="h"))
        st.plotly_chart(fig, use_container_width=True)

        last = df.dropna(subset=[base_col, f"{base_col}_z"]).iloc[-1]
        c1, c2 = st.columns(2)
        c1.metric("Laatste differential", f"{last[base_col]:.1f} {unit}")
        c2.metric("Laatste Z-score (90d)", f"{last[f'{base_col}_z']:.2f}")

plot_diff(tabs[0], df, "diff_spread", "US–EU differential: (10Y–2Y)", "bp")
plot_diff(tabs[1], df, "diff_2y",     "US–EU differential: 2Y", "bp")
plot_diff(tabs[2], df, "diff_10y",    "US–EU differential: 10Y", "bp")
