
from datetime import date
import streamlit as st
import pandas as pd
import numpy as np

from utils.bq import load_spx_options
from utils.helpers import ensure_datetime
from utils.charts import plot_line
import plotly.express as px

st.set_page_config(page_title="SPX Opties", layout="wide")

st.sidebar.subheader("📚 Pagina's")
st.sidebar.page_link("app.py", label="🏠 Dashboard instellingen")
st.sidebar.page_link("pages/1_SP500.py", label="📈 S&P 500 + VIX")
st.sidebar.page_link("pages/2_VIX.py", label="📉 VIX (alleen)")
st.sidebar.page_link("pages/3_SPX_Options.py", label="🧮 SPX Opties")

st.title("🧮 SPX Opties — Overzicht")

view_start = st.session_state.get("view_start")
view_end   = st.session_state.get("view_end")
if not (view_start and view_end):
    view_end = date.today()
    from datetime import timedelta
    view_start = view_end - timedelta(days=14)

df = load_spx_options(view_start, view_end)

if df.empty:
    st.warning("Geen SPX optiedata gevonden voor de gekozen periode.")
    st.stop()

# Ensure types
num_cols = ["strike","last_price","bid","ask","implied_volatility","open_interest","volume","underlying_price","days_to_exp","ppd","vix"]
for c in num_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

df = ensure_datetime(df, "snapshot_date")
df = df.sort_values(["snapshot_date","expiration","strike"])

with st.expander("🔎 Ruwe data (download)"):
    st.dataframe(df)
    st.download_button(
        "Download CSV (SPX opties)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="spx_options.csv",
        mime="text/csv",
    )

# ---- Filters (zonder datasource selectie) ----
left, mid, right = st.columns([1,1,1])

with left:
    opt_type = st.radio("Type", ["Both","Call","Put"], horizontal=True)
with mid:
    # Strike range
    smin = float(np.nanmin(df["strike"])) if "strike" in df else 0.0
    smax = float(np.nanmax(df["strike"])) if "strike" in df else 10000.0
    strike_range = st.slider("Strike range", min_value=float(smin), max_value=float(smax), value=(float(smin), float(smax)))
with right:
    dmin = int(np.nanmin(df["days_to_exp"])) if "days_to_exp" in df else 0
    dmax = int(np.nanmax(df["days_to_exp"])) if "days_to_exp" in df else 365
    dte_range = st.slider("Days to Expiration", min_value=int(dmin), max_value=int(dmax), value=(int(dmin), int(min(dmax, 60))))

mask = (df["strike"].between(strike_range[0], strike_range[1])) & (df["days_to_exp"].between(dte_range[0], dte_range[1]))
if opt_type != "Both" and "type" in df.columns:
    mask &= (df["type"].str.upper().str[0] == opt_type[0].upper())

df_f = df[mask].copy()

st.subheader("Samenvattingen")
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Records", len(df_f))
with c2:
    st.metric("Gem. IV", f"{df_f['implied_volatility'].mean():.2%}" if 'implied_volatility' in df_f else "n/a")
with c3:
    st.metric("Gem. OI", int(df_f['open_interest'].mean()) if 'open_interest' in df_f else 0)
with c4:
    st.metric("Gem. PPD", f"{df_f['ppd'].mean():.2f}" if 'ppd' in df_f else "n/a")

st.subheader("IV vs DTE (gemiddeld)")
if "implied_volatility" in df_f and "days_to_exp" in df_f:
    iv_dte = df_f.groupby("days_to_exp", as_index=False)["implied_volatility"].mean().dropna()
    fig = px.line(iv_dte, x="days_to_exp", y="implied_volatility", markers=True, title="Term Structure (IV vs DTE)")
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Open Interest per Strike (laatste snapshot)")
if "open_interest" in df_f and "strike" in df_f:
    last_snap = df_f["snapshot_date"].max()
    snap_df = df_f[df_f["snapshot_date"] == last_snap]
    fig = px.bar(snap_df.groupby("strike", as_index=False)["open_interest"].sum(), x="strike", y="open_interest", title=f"OI by Strike — {last_snap.date()}")
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Mid vs Strike (dichtstbijzijnde expiratie)")
if "expiration" in df_f and "bid" in df_f and "ask" in df_f:
    df_f["mid"] = (df_f["bid"] + df_f["ask"]) / 2.0
    # neem expiratie met kleinste DTE binnen filter
    dte_min = df_f["days_to_exp"].min()
    near = df_f[df_f["days_to_exp"] == dte_min]
    if not near.empty:
        fig = px.scatter(near, x="strike", y="mid", color="type", title=f"Mid Price vs Strike — DTE {dte_min}")
        st.plotly_chart(fig, use_container_width=True)
