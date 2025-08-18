# pages/1_SP500.py
import streamlit as st
import pandas as pd
from datetime import date, timedelta
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from utils.bq import run_query, get_bq_client, bq_ping

st.set_page_config(page_title="S&P 500", layout="wide")
st.title("📈 S&P 500")

# ---- Verbindingscheck ----
try:
    with st.spinner("BigQuery check…"):
        bq_ping()
except Exception as e:
    st.error("Geen BigQuery-verbinding. Controleer Secrets ([gcp_service_account]).")
    st.caption(f"Details: {e}")
    st.stop()

# ---- Config uit Secrets (view vastzetten) ----
SPX_VIEW = st.secrets.get("tables", {}).get(
    "spx_view",
    "nth-pier-468314-p7.marketdata.spx_with_vix_v"  # fallback
)

with st.sidebar:
    st.subheader("⚙️ Instellingen")
    days = st.slider("Periode (dagen terug)", 30, 1095, 180, step=30)
    end_d = date.today()
    start_d = end_d - timedelta(days=days)
    fast_mode = st.checkbox("Snelle modus (alleen Close)", value=True)
    show_ma   = st.checkbox("Toon MA50/MA200 (op Close)", value=True)
    show_vix  = st.checkbox("Toon VIX overlay (2e as)", value=True)
    st.caption(f"Databron: `{SPX_VIEW}`")

status = st.status("Data laden…", expanded=False)

def fetch_close_only():
    sql = f"""
    SELECT DATE(date) AS date,
           CAST(close AS FLOAT64)     AS close,
           CAST(vix_close AS FLOAT64) AS vix_close
    FROM `{SPX_VIEW}`
    WHERE DATE(date) BETWEEN @start AND @end
    ORDER BY date
    """
    return run_query(sql, params={"start": start_d, "end": end_d}, timeout=25)

def fetch_ohlc():
    sql = f"""
    SELECT DATE(date) AS date,
           CAST(open  AS FLOAT64)     AS open,
           CAST(high  AS FLOAT64)     AS high,
           CAST(low   AS FLOAT64)     AS low,
           CAST(close AS FLOAT64)     AS close,
           CAST(volume AS INT64)      AS volume,
           CAST(vix_close AS FLOAT64) AS vix_close
    FROM `{SPX_VIEW}`
    WHERE DATE(date) BETWEEN @start AND @end
    ORDER BY date
    """
    return run_query(sql, params={"start": start_d, "end": end_d}, timeout=25)

# ---- Laden met fallback ----
try:
    df = fetch_close_only() if fast_mode else fetch_ohlc()
    if df.empty:
        raise RuntimeError("Lege dataset voor de gekozen periode/view.")
    status.update(label="✅ Data geladen", state="complete", expanded=False)
except Exception as e:
    status.update(label="⚠️ Trage/mislukte query — fallback naar 90 dagen close-only", state="error", expanded=True)
    st.caption(f"Details: {e}")
    try:
        fallback_days = 90
        df = run_query(
            f"""
            SELECT DATE(date) AS date,
                   CAST(close AS FLOAT64)     AS close,
                   CAST(vix_close AS FLOAT64) AS vix_close
            FROM `{SPX_VIEW}`
            WHERE DATE(date) BETWEEN @start AND @end
            ORDER BY date
            """,
            params={"start": end_d - timedelta(days=fallback_days), "end": end_d},
            timeout=20
        )
        if df.empty:
            st.error("Nog steeds geen data. Controleer de view-naam en inhoud in BigQuery.")
            st.stop()
        fast_mode = True
    except Exception as e2:
        st.error(f"Fallback faalde ook: {e2}")
        st.stop()

df = df.sort_values("date").reset_index(drop=True)

# ---- MA's op close ----
if show_ma and "close" in df.columns:
    df["ma50"]  = df["close"].rolling(50).mean()
    df["ma200"] = df["close"].rolling(200).mean()

# ---- Chart (VIX op 2e as) ----
fig = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]])
if fast_mode:
    fig.add_trace(go.Scatter(x=df["date"], y=df["close"], mode="lines", name="SPX Close"),
                  row=1, col=1, secondary_y=False)
else:
    fig.add_trace(go.Candlestick(
        x=df["date"], open=df["open"], high=df["high"], low=df["low"], close=df["close"], name="SPX OHLC"
    ), row=1, col=1, secondary_y=False)

if show_ma and fast_mode:
    if df["ma50"].notna().any():
        fig.add_trace(go.Scatter(x=df["date"], y=df["ma50"], mode="lines", name="MA50"),
                      row=1, col=1, secondary_y=False)
    if df["ma200"].notna().any():
        fig.add_trace(go.Scatter(x=df["date"], y=df["ma200"], mode="lines", name="MA200"),
                      row=1, col=1, secondary_y=False)

if show_vix and "vix_close" in df.columns and df["vix_close"].notna().any():
    fig.add_trace(go.Scatter(x=df["date"], y=df["vix_close"], mode="lines", name="VIX"),
                  row=1, col=1, secondary_y=True)

fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=540, legend_orientation="h")
fig.update_yaxes(title_text="SPX", secondary_y=False)
if show_vix:
    fig.update_yaxes(title_text="VIX", secondary_y=True)
fig.update_xaxes(showspikes=True, spikemode="across")
st.plotly_chart(fig, use_container_width=True)

st.caption(
    f"Periode: {start_d} → {end_d} • Rijen: {len(df):,} • Modus: {'Close-only' if fast_mode else 'OHLC'} • "
    f"VIX overlay: {'aan' if show_vix else 'uit'}"
)
