import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date, timedelta

from utils.bq import run_query, get_bq_client

st.set_page_config(page_title="S&P 500 + VIX", layout="wide")

st.title("📈 S&P 500 + 📉 VIX")

# ---- Instellingen (links) ----
with st.sidebar:
    st.subheader("⚙️ Dashboard instellingen")

    # Tabellen (pas aan indien jouw namen anders zijn)
    client = get_bq_client()
    PROJECT = client.project
    DATASET_DEFAULT = "marketdata"

    tbl_spx = st.text_input(
        "Tabel S&P 500",
        value=f"{PROJECT}.{DATASET_DEFAULT}.sp500_prices",
        help="Volledige tabelnaam in BigQuery"
    )
    tbl_vix = st.text_input(
        "Tabel VIX",
        value=f"{PROJECT}.{DATASET_DEFAULT}.vix_prices",
        help="Volledige tabelnaam in BigQuery"
    )

    # Snel en veilig defaulten (snelle cold start)
    default_days = st.slider("🔎 Periode (dagen terug)", min_value=30, max_value=1095, value=180, step=30)
    end_date = date.today()
    start_date = end_date - timedelta(days=default_days)

    use_ha = st.checkbox("Toon Heikin Ashi (SPX)", value=False)
    show_ma = st.checkbox("Toon 50/200 MA (SPX)", value=True)

# ---- Helpers ----
def heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    """Maak HA-OHLC op basis van gewone OHLC."""
    out = df.copy()
    out["ha_close"] = (out["open"] + out["high"] + out["low"] + out["close"]) / 4.0
    ha_open = []
    for i, row in out.iterrows():
        if i == 0:
            ha_open.append((row["open"] + row["close"]) / 2.0)
        else:
            ha_open.append((ha_open[-1] + out.loc[i-1, "ha_close"]) / 2.0)
    out["ha_open"] = ha_open
    out["ha_high"] = out[["high", "ha_open", "ha_close"]].max(axis=1)
    out["ha_low"]  = out[["low", "ha_open", "ha_close"]].min(axis=1)
    return out

# ---- Data laden met duidelijke status ----
status = st.status("Data laden…", expanded=False)
try:
    sql_spx = f"""
    SELECT
      DATE(date) AS date, open, high, low, close, volume
    FROM `{tbl_spx}`
    WHERE DATE(date) BETWEEN @start AND @end
    ORDER BY date
    """
    df_spx = run_query(sql_spx, params={"start": start_date, "end": end_date})

    sql_vix = f"""
    SELECT
      DATE(date) AS date, close
    FROM `{tbl_vix}`
    WHERE DATE(date) BETWEEN @start AND @end
    ORDER BY date
    """
    df_vix = run_query(sql_vix, params={"start": start_date, "end": end_date})

    status.update(label="✅ Data geladen", state="complete", expanded=False)
except Exception as e:
    status.update(label="❌ Laden mislukt", state="error", expanded=True)
    st.exception(e)
    st.stop()

# ---- Validatie ----
if df_spx.empty:
    st.warning("Geen S&P 500-gegevens voor de gekozen periode/tabel.")
    st.stop()
if df_vix.empty:
    st.warning("Geen VIX-gegevens voor de gekozen periode/tabel.")
    st.stop()

# ---- Berekeningen ----
df_spx = df_spx.sort_values("date").reset_index(drop=True)
if show_ma:
    df_spx["ma50"] = df_spx["close"].rolling(50).mean()
    df_spx["ma200"] = df_spx["close"].rolling(200).mean()

if use_ha:
    df_ha = heikin_ashi(df_spx)

# ---- Plotten (2 rijen, gedeelde x) ----
fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                    vertical_spacing=0.08,
                    row_heights=[0.65, 0.35],
                    subplot_titles=("S&P 500", "VIX"))

# SPX
if use_ha:
    fig.add_trace(
        go.Candlestick(x=df_ha["date"],
                       open=df_ha["ha_open"], high=df_ha["ha_high"],
                       low=df_ha["ha_low"], close=df_ha["ha_close"],
                       name="SPX (Heikin Ashi)"),
        row=1, col=1
    )
else:
    fig.add_trace(
        go.Scatter(x=df_spx["date"], y=df_spx["close"], mode="lines", name="SPX Close"),
        row=1, col=1
    )

if show_ma and not use_ha:
    if df_spx["ma50"].notna().any():
        fig.add_trace(go.Scatter(x=df_spx["date"], y=df_spx["ma50"], mode="lines", name="MA50"), row=1, col=1)
    if df_spx["ma200"].notna().any():
        fig.add_trace(go.Scatter(x=df_spx["date"], y=df_spx["ma200"], mode="lines", name="MA200"), row=1, col=1)

# VIX
fig.add_trace(
    go.Scatter(x=df_vix["date"], y=df_vix["close"], mode="lines", name="VIX Close"),
    row=2, col=1
)

fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=700, legend_orientation="h")
fig.update_xaxes(showspikes=True, spikesnap="cursor", spikemode="across")
fig.update_yaxes(tickformat=",", separatethousands=True)

st.plotly_chart(fig, use_container_width=True)

# ---- Extra: laatste update info ----
st.caption(
    f"Periode: {start_date} → {end_date} • Rijen SPX: {len(df_spx):,} • Rijen VIX: {len(df_vix):,}"
)
