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

# ---- Bepaal dynamisch het maximum bereik ----
try:
    df_min = run_query(f"SELECT MIN(DATE(date)) AS d FROM `{SPX_VIEW}`", timeout=10)
    min_date = pd.to_datetime(df_min["d"].iloc[0]).date()
    today = date.today()
    dyn_max_days = max(30, (today - min_date).days)
    dyn_max_days = min(dyn_max_days, 3650)  # optionele cap (±10 jaar)
except Exception:
    today = date.today()
    dyn_max_days = 3650

# ---- Sidebar ----
with st.sidebar:
    st.subheader("⚙️ Instellingen")
    days = st.slider("Periode (dagen terug)", 30, int(dyn_max_days), min(180, int(dyn_max_days)), step=30)
    end_d = today
    start_d = end_d - timedelta(days=days)

    fast_mode      = st.checkbox("Snelle modus (alleen Close)", value=True)
    show_ma        = st.checkbox("Toon MA50/MA200 (op Close)", value=True)
    show_vix       = st.checkbox("Toon VIX overlay (2e as)", value=True)
    show_ha_chart  = st.checkbox("Toon Heikin Ashi grafiek (OHLC nodig)", value=True)

    st.caption(f"Databron: `{SPX_VIEW}`")

status = st.status("Data laden…", expanded=False)

# ---- Queries via de VIEW ----
def fetch_close_only():
    sql = f"""
    SELECT DATE(date) AS date,
           CAST(close     AS FLOAT64) AS close,
           CAST(vix_close AS FLOAT64) AS vix_close
    FROM `{SPX_VIEW}`
    WHERE DATE(date) BETWEEN @start AND @end
    ORDER BY date
    """
    return run_query(sql, params={"start": start_d, "end": end_d}, timeout=25)

def fetch_ohlc():
    sql = f"""
    SELECT DATE(date) AS date,
           CAST(open  AS FLOAT64) AS open,
           CAST(high  AS FLOAT64) AS high,
           CAST(low   AS FLOAT64) AS low,
           CAST(close AS FLOAT64) AS close,
           CAST(volume AS INT64)  AS volume,
           CAST(vix_close AS FLOAT64) AS vix_close
    FROM `{SPX_VIEW}`
    WHERE DATE(date) BETWEEN @start AND @end
    ORDER BY date
    """
    return run_query(sql, params={"start": start_d, "end": end_d}, timeout=25)

# ---- Heikin Ashi helper ----
def heikin_ashi(ohlc: pd.DataFrame) -> pd.DataFrame:
    df = ohlc[["date", "open", "high", "low", "close"]].copy().reset_index(drop=True)
    ha_close = (df["open"] + df["high"] + df["low"] + df["close"]) / 4.0
    ha_open = []
    for i in range(len(df)):
        if i == 0:
            ha_open.append((df.loc[i, "open"] + df.loc[i, "close"]) / 2.0)
        else:
            ha_open.append((ha_open[-1] + ha_close.iloc[i - 1]) / 2.0)
    ha_open = pd.Series(ha_open, index=df.index, name="ha_open")
    ha_high = pd.concat([df["high"], ha_open, ha_close], axis=1).max(axis=1).rename("ha_high")
    ha_low  = pd.concat([df["low"],  ha_open, ha_close], axis=1).min(axis=1).rename("ha_low")
    out = pd.DataFrame({
        "date": df["date"],
        "ha_open": ha_open,
        "ha_high": ha_high,
        "ha_low":  ha_low,
        "ha_close": ha_close,
    })
    return out

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
                   CAST(close     AS FLOAT64) AS close,
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

# ---- Hoofdgrafiek (VIX op 2e as) ----
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

# ---- Heikin Ashi grafiek ----
if show_ha_chart:
    st.subheader("🕯️ Heikin Ashi")
    try:
        # Zorg dat we OHLC hebben (fast_mode = extra query)
        if fast_mode:
            with st.spinner("OHLC ophalen voor Heikin Ashi…"):
                df_ohlc = fetch_ohlc()
        else:
            df_ohlc = df

        needed = {"open", "high", "low", "close"}
        if df_ohlc.empty or not needed.issubset(df_ohlc.columns):
            st.info("Geen OHLC-data beschikbaar om Heikin Ashi te berekenen.")
        else:
            ha = heikin_ashi(df_ohlc).dropna().reset_index(drop=True)

            ha_fig = make_subplots(rows=1, cols=1)
            ha_fig.add_trace(go.Candlestick(
                x=ha["date"],
                open=ha["ha_open"],
                high=ha["ha_high"],
                low=ha["ha_low"],
                close=ha["ha_close"],
                name="Heikin Ashi"
            ), row=1, col=1)
            ha_fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), height=420, showlegend=False)
            ha_fig.update_xaxes(showspikes=True, spikemode="across")
            st.plotly_chart(ha_fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Heikin Ashi kon niet worden getekend: {e}")

# ---- Footer info ----
st.caption(
    f"Periode: {start_d} → {end_d} • Rijen: {len(df):,} • Modus: {'Close-only' if fast_mode else 'OHLC'} • "
    f"VIX overlay: {'aan' if show_vix else 'uit'}"
)
