
from datetime import date
import streamlit as st
import pandas as pd

from utils.bq import load_sp500, load_vix
from utils.helpers import ensure_datetime, heikin_ashi
from utils.charts import plot_candles, plot_line

st.set_page_config(page_title="S&P 500 + VIX", layout="wide")

st.sidebar.subheader("📚 Pagina's")
st.sidebar.page_link("app.py", label="🏠 Dashboard instellingen")
st.sidebar.page_link("pages/1_SP500.py", label="📈 S&P 500 + VIX")
st.sidebar.page_link("pages/2_VIX.py", label="📉 VIX (alleen)")
st.sidebar.page_link("pages/3_SPX_Options.py", label="🧮 SPX Opties")

st.title("📈 S&P 500 + 📉 VIX")

view_start = st.session_state.get("view_start")
view_end   = st.session_state.get("view_end")
if not (view_start and view_end):
    today = date.today()
    view_start, view_end = today.replace(day=1), today

df_spx = load_sp500(view_start, view_end)
df_vix = load_vix(view_start, view_end)

if df_spx.empty:
    st.warning("Geen S&P 500 data in de gekozen periode.")
if df_vix.empty:
    st.warning("Geen VIX data in de gekozen periode.")

st.subheader("S&P 500")
col1, col2, col3 = st.columns([1,1,1])
with col1:
    mode = st.selectbox(
        "Weergave",
        ["Heikin-Ashi (candlestick)", "Line + MA"],
        index=0,
    )
with col2:
    ma_len = st.number_input("Moving Average lengte", min_value=1, value=20, step=1)
with col3:
    show_volume = st.toggle("Volume tonen (alleen bij line)", value=False)

if not df_spx.empty:
    df_spx = ensure_datetime(df_spx, date_col="date")
    if mode.startswith("Heikin"):
        df_ha = heikin_ashi(df_spx.rename(columns={
            "open":"Open","high":"High","low":"Low","close":"Close"
        }))
        fig_spx = plot_candles(
            df=df_ha,
            date_col="date",
            open_col="HA_Open",
            high_col="HA_High",
            low_col="HA_Low",
            close_col="HA_Close",
            title="S&P 500 — Heikin-Ashi"
        )
    else:
        df_line = df_spx.sort_values("date").copy()
        df_line["ma"] = df_line["close"].rolling(int(ma_len)).mean()
        fig_spx = plot_line(
            df=df_line,
            x="date",
            y_cols=["close","ma"],
            names=["Close", f"MA({ma_len})"],
            title="S&P 500 — Line + MA",
            show_volume=show_volume,
            volume_col="volume"
        )
    st.plotly_chart(fig_spx, use_container_width=True)

    with st.expander("🔎 Tabel S&P 500 (download)"):
        st.dataframe(df_spx)
        st.download_button(
            "Download CSV (S&P 500)",
            data=df_spx.to_csv(index=False).encode("utf-8"),
            file_name="sp500.csv",
            mime="text/csv",
        )

st.subheader("VIX (zelfde periode)")
if not df_vix.empty:
    df_vix = ensure_datetime(df_vix, date_col="date")
    fig_vix = plot_line(
        df=df_vix,
        x="date",
        y_cols=["close"],
        names=["VIX Close"],
        title="VIX — Close",
        show_volume=False
    )
    st.plotly_chart(fig_vix, use_container_width=True)

    with st.expander("🔎 Tabel VIX (download)"):
        st.dataframe(df_vix)
        st.download_button(
            "Download CSV (VIX)",
            data=df_vix.to_csv(index=False).encode("utf-8"),
            file_name="vix.csv",
            mime="text/csv",
        )
