# pages/1_SP500.py
import streamlit as st
import pandas as pd
from datetime import date, timedelta
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from utils.bq import run_query, get_bq_client

st.set_page_config(page_title="S&P 500", layout="wide")
st.title("📈 S&P 500")

with st.sidebar:
    st.subheader("⚙️ Instellingen")
    client = get_bq_client()
    PROJECT = client.project
    DATASET = "marketdata"

    tbl_spx = st.text_input(
        "BigQuery tabel",
        value=f"{PROJECT}.{DATASET}.sp500_prices",
        help="Volledige tabelnaam"
    )

    days = st.slider("Periode (dagen terug)", 30, 1095, 180, step=30)
    end_d = date.today()
    start_d = end_d - timedelta(days=days)

    fast_mode = st.checkbox("Snelle modus (alleen Close)", value=True,
                            help="Supersnel: alleen date+close. Uitschakelen = OHLC/candlestick laden.")
    show_ma = st.checkbox("Toon MA50/MA200 (op Close)", value=True)

status = st.status("Data laden…", expanded=False)

def fetch_close_only():
    sql = f"""
    SELECT DATE(date) AS date, CAST(close AS FLOAT64) AS close
    FROM `{tbl_spx}`
    WHERE DATE(date) BETWEEN @start AND @end
    ORDER BY date
    """
    return run_query(sql, params={"start": start_d, "end": end_d}, timeout=25)

def fetch_ohlc():
    sql = f"""
    SELECT DATE(date) AS date,
           CAST(open AS FLOAT64)  AS open,
           CAST(high AS FLOAT64)  AS high,
           CAST(low  AS FLOAT64)  AS low,
           CAST(close AS FLOAT64) AS close,
           CAST(volume AS INT64)  AS volume
    FROM `{tbl_spx}`
    WHERE DATE(date) BETWEEN @start AND @end
    ORDER BY date
    """
    return run_query(sql, params={"start": start_d, "end": end_d}, timeout=25)

# ---- Probeer snel pad; val terug op close-only 90d bij problemen ----
try:
    df = fetch_close_only() if fast_mode else fetch_ohlc()
    if df.empty:
        raise RuntimeError("Lege dataset voor de gekozen periode/tabel.")
    status.update(label="✅ Data geladen", state="complete", expanded=False)
except Exception as e:
    status.update(label="⚠️ Trage of mislukte query — val terug op 90 dagen close-only", state="error", expanded=True)
    st.caption(f"Details: {e}")
    try:
        fallback_days = 90
        df = run_query(
            f"""
            SELECT DATE(date) AS date, CAST(close AS FLOAT64) AS close
            FROM `{tbl_spx}`
            WHERE DATE(date) BETWEEN @start AND @end
            ORDER BY date
            """,
            params={"start": end_d - timedelta(days=fallback_days), "end": end_d},
            timeout=20
        )
        if df.empty:
            st.error("Nog steeds geen data. Controleer de tabelnaam/gegevens in BigQuery.")
            st.stop()
        fast_mode = True  # we zitten in fallback
    except Exception as e2:
        st.error(f"Fallback faalde ook: {e2}")
        st.stop()

df = df.sort_values("date").reset_index(drop=True)

# ---- MA's op close (lichtgewicht) ----
if show_ma and "close" in df.columns:
    df["ma50"] = df["close"].rolling(50).mean()
    df["ma200"] = df["close"].rolling(200).mean()

# ---- Chart ----
fig = make_subplots(rows=1, cols=1)

if fast_mode:
    fig.add_trace(go.Scatter(x=df["date"], y=df["close"], mode="lines", name="SPX Close"))
else:
    fig.add_trace(go.Candlestick(
        x=df["date"], open=df["open"], high=df["high"], low=df["low"], close=df["close"], name="SPX OHLC"
    ))

if show_ma and fast_mode:
    if df["ma50"].notna().any():
        fig.add_trace(go.Scatter(x=df["date"], y=df["ma50"], mode="lines", name="MA50"))
    if df["ma200"].notna().any():
        fig.add_trace(go.Scatter(x=df["date"], y=df["ma200"], mode="lines", name="MA200"))

fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=520, legend_orientation="h")
fig.update_xaxes(showspikes=True, spikemode="across")
st.plotly_chart(fig, use_container_width=True)

# ---- Info ----
rows = len(df)
st.caption(f"Periode: {start_d} → {end_d} • Rijen: {rows:,} • Modus: {'Close-only' if fast_mode else 'OHLC'}")
