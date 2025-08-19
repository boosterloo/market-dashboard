# pages/fx_dashboard.py
import streamlit as st
import pandas as pd
import plotly.express as px
from utils.bq import run_query

st.set_page_config(page_title="🌍 FX Dashboard", layout="wide")
st.title("🌍 FX Dashboard")

# --- Views ---
FX_WIDE = "nth-pier-468314-p7.marketdata.fx_rates_wide_v"
FX_ENRICHED_WIDE = "nth-pier-468314-p7.marketdata.fx_rates_enriched_wide_v"

# --- Data ophalen ---
@st.cache_data(ttl=3600)
def load_data():
    df_overview = run_query(f"SELECT * FROM `{FX_WIDE}` ORDER BY date DESC")
    df_enriched = run_query(f"SELECT * FROM `{FX_ENRICHED_WIDE}` ORDER BY date DESC")
    return df_overview, df_enriched

df_overview, df_enriched = load_data()

# --- Datum selectie ---
min_date, max_date = df_overview["date"].min(), df_overview["date"].max()
date_range = st.slider(
    "📅 Periode",
    min_value=min_date,
    max_value=max_date,
    value=(max_date - pd.Timedelta(days=180), max_date),
    format="YYYY-MM-DD"
)
mask = (df_overview["date"] >= date_range[0]) & (df_overview["date"] <= date_range[1])
df_overview = df_overview.loc[mask]
df_enriched = df_enriched.loc[
    (df_enriched["date"] >= date_range[0]) & (df_enriched["date"] <= date_range[1])
]

# --- Tabs ---
tab1, tab2 = st.tabs(["📊 Overzicht (Close/Delta)", "📈 Indicators"])

with tab1:
    st.subheader("Dagelijkse bewegingen")
    # Heatmap delta%
    df_heat = df_overview.set_index("date")[[c for c in df_overview.columns if "delta_pct" in c]]
    st.dataframe(df_heat.style.format("{:.2%}").background_gradient(cmap="RdYlGn"))
    
    # Lijnchart EURUSD / GBPUSD
    fig = px.line(df_overview, x="date", y=["eurusd_close","gbpusd_close"], title="EURUSD & GBPUSD")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Trend & Volatiliteit")
    # EURUSD met MA50/200
    fig = px.line(df_enriched, x="date", y=["eurusd_close","eurusd_ma50","eurusd_ma200"],
                  title="EURUSD met MA50/200")
    st.plotly_chart(fig, use_container_width=True)
    
    # Volatiliteit
    fig2 = px.line(df_enriched, x="date", y=["eurusd_rv20","gbpusd_rv20","audusd_rv20"],
                   title="20D Realized Vol (Annualized)")
    st.plotly_chart(fig2, use_container_width=True)
