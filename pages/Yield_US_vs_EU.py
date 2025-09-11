# pages/Yield_US_EU_Compare.py

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
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

# ---------- Data ----------
US_VIEW = st.secrets.get("tables", {}).get("us_yield_view", "nth-pier-468314-p7.marketdata.us_yield_v")
EU_VIEW = st.secrets.get("tables", {}).get("eu_yield_view", "nth-pier-468314-p7.marketdata.eu_yield_v")

@st.cache_data(ttl=1800, show_spinner=False)
def load_data():
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
      (us.us_2y - eu.eu_2y) AS diff_2y,
      (us.us_10y - eu.eu_10y) AS diff_10y,
      (us.us_spread - eu.eu_spread) AS diff_spread
    FROM us
    JOIN eu USING(date)
    ORDER BY date
    """
    return run_query(q)

with st.spinner("Data laden…"):
    df = load_data()

if df.empty:
    st.warning("Geen data gevonden.")
    st.stop()

# ---------- Term structure overlay ----------
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=df["date"], y=df["us_2y"], mode="lines", name="US 2Y"))
fig1.add_trace(go.Scatter(x=df["date"], y=df["us_10y"], mode="lines", name="US 10Y"))
fig1.add_trace(go.Scatter(x=df["date"], y=df["eu_2y"], mode="lines", name="EU 2Y", yaxis="y2"))
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
fig2.add_trace(go.Bar(x=df["date"], y=df["diff_spread"], name="US–EU diff (bp)", opacity=0.3))

fig2.update_layout(
    title="10Y–2Y Spreads & Differentials",
    xaxis_title="Date",
    yaxis_title="Spread (bp)",
    legend=dict(orientation="h")
)

st.plotly_chart(fig2, use_container_width=True)

# ---------- KPI’s ----------
latest = df.dropna().iloc[-1]
st.subheader("Laatste waarden")
cols = st.columns(3)
cols[0].metric("🇺🇸 US 10Y–2Y", f"{latest.us_spread:.1f} bp")
cols[1].metric("🇪🇺 EU 10Y–2Y", f"{latest.eu_spread:.1f} bp")
cols[2].metric("Δ(US–EU)", f"{latest.diff_spread:.1f} bp")
