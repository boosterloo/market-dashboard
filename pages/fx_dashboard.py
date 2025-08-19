# pages/fx_dashboard.py
import streamlit as st
import pandas as pd
import plotly.express as px
from utils.bq import run_query, bq_ping

st.set_page_config(page_title="🌍 FX Dashboard", layout="wide")
st.title("🌍 FX Dashboard")

# ---- View in BigQuery ----
FX_VIEW = "nth-pier-468314-p7.marketdata.fx_rates_dashboard_v"

# ---- Verbindingscheck ----
try:
    with st.spinner("BigQuery check…"):
        bq_ping()
except Exception as e:
    st.error("Geen BigQuery-verbinding. Controleer Secrets ([gcp_service_account]).")
    st.caption(f"Details: {e}")
    st.stop()

# ---- Data ophalen ----
@st.cache_data(ttl=3600)
def load_data():
    return run_query(f"SELECT * FROM `{FX_VIEW}` ORDER BY date DESC")

with st.spinner("FX data laden…"):
    df = load_data()

if df.empty:
    st.warning("Geen data ontvangen uit view. Check of de view bestaat en gevuld is.")
    st.stop()

# ---- Periode selectie ----
min_date, max_date = df["date"].min(), df["date"].max()
default_start = max_date - pd.Timedelta(days=180)
date_range = st.slider(
    "📅 Periode",
    min_value=min_date,
    max_value=max_date,
    value=(default_start, max_date),
    format="YYYY-MM-DD"
)

df = df[(df["date"] >= date_range[0]) & (df["date"] <= date_range[1])].sort_values("date")

# ---- Beschikbare pairs afleiden ----
def infer_pairs(df_cols):
    pairs = []
    for c in df_cols:
        if c.endswith("_close"):
            pairs.append(c.replace("_close", ""))
    return sorted(pairs)

pairs = infer_pairs(df.columns)
default_pairs = [p for p in ["eurusd", "gbpusd", "usdjpy"] if p in pairs]
sel_pairs = st.multiselect("Valutaparen (grafieken)", options=pairs, default=default_pairs)

# ---- Tabs ----
tab1, tab2, tab3 = st.tabs(["📊 Overzicht (Close/Delta)", "📈 Trend (MA50/200)", "🎢 Volatiliteit (RV20 & ATR)"])

with tab1:
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Dagelijkse bewegingen (close)")
        y_cols = [f"{p}_close" for p in sel_pairs if f"{p}_close" in df.columns]
        if y_cols:
            fig = px.line(df, x="date", y=y_cols, title="FX closes")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Selecteer minimaal één pair met *_close in de dataset.")

    with col2:
        st.subheader("Delta% heatmap")
        heat_cols = [c for c in df.columns if c.endswith("_delta_pct")]
        if heat_cols:
            df_heat = df[["date"] + heat_cols].set_index("date").sort_index(ascending=False)
            st.dataframe(df_heat.style.format("{:.2%}"))
        else:
            st.caption("Geen *_delta_pct kolommen gevonden.")

    st.divider()
    st.subheader("Delta (abs/%): snelle tabel")
    show_cols = ["date"]
    for p in sel_pairs:
        for suffix in ["_delta_abs", "_delta_pct"]:
            col = f"{p}{suffix}"
            if col in df.columns:
                show_cols.append(col)
    st.dataframe(df[show_cols].sort_values("date", ascending=False), use_container_width=True)

with tab2:
    st.subheader("MA50/MA200 per pair")
    for p in sel_pairs:
        cols_needed = [f"{p}_close", f"{p}_ma50", f"{p}_ma200"]
        if all(c in df.columns for c in cols_needed):
            fig = px.line(
                df,
                x="date",
                y=cols_needed,
                title=f"{p.upper()} – close vs MA50/MA200"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption(f"{p.upper()}: ontbrekende kolommen (verwacht: close, ma50, ma200).")

with tab3:
    st.subheader("Volatiliteit")
    rv_cols = [f"{p}_rv20" for p in sel_pairs if f"{p}_rv20" in df.columns]
    if rv_cols:
        fig_rv = px.line(df, x="date", y=rv_cols, title="20D Realized Vol (annualized)")
        st.plotly_chart(fig_rv, use_container_width=True)
    else:
        st.caption("Geen *_rv20 kolommen gevonden.")

    atr_cols = [f"{p}_atr14" for p in sel_pairs if f"{p}_atr14" in df.columns]
    if atr_cols:
        fig_atr = px.line(df, x="date", y=atr_cols, title="ATR14")
        st.plotly_chart(fig_atr, use_container_width=True)
    else:
        st.caption("Geen *_atr14 kolommen gevonden.")
