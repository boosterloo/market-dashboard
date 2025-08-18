# pages/1_SP500.py
import streamlit as st
import pandas as pd
import numpy as np
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
    dyn_max_days = min(dyn_max_days, 3650)  # cap ±10 jaar
except Exception:
    today = date.today()
    dyn_max_days = 3650

# ---- Sidebar ----
with st.sidebar:
    st.subheader("⚙️ Instellingen")
    days = st.slider("Periode (dagen terug)", 30, int(dyn_max_days), min(180, int(dyn_max_days)), step=30)
    end_d = today
    start_d = end_d - timedelta(days=days)

    fast_mode     = st.checkbox("Snelle modus (alleen Close)", value=True)
    show_ma       = st.checkbox("Toon MA50/MA200 (op Close)", value=True)
    show_vix      = st.checkbox("Toon VIX overlay (2e as)", value=True)
    show_ha_chart = st.checkbox("Toon Heikin Ashi + SuperTrend + Donchian", value=True)

    with st.expander("🔧 Indicator-instellingen (optioneel)"):
        st_len  = st.number_input("SuperTrend length", min_value=1, max_value=200, value=10, step=1)
        st_mult = st.number_input("SuperTrend multiplier", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
        dc_len  = st.number_input("Donchian Channel length", min_value=1, max_value=400, value=20, step=1)
        dc_fill = st.checkbox("Vul Donchian-band", value=False)
        dc_opacity = st.slider("Donchian opacity", 0.0, 0.4, 0.08, 0.01, disabled=not dc_fill)

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

# ---- ATR (RMA) + SuperTrend op HA ----
def atr_rma(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low  - prev_close).abs()
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/length, adjust=False).mean()
    return atr

def supertrend_on_ha(ha: pd.DataFrame, length: int = 10, multiplier: float = 1.0) -> pd.DataFrame:
    high, low, close = ha["ha_high"], ha["ha_low"], ha["ha_close"]
    atr = atr_rma(high, low, close, length)
    hl2 = (high + low) / 2.0

    upper_basic = hl2 + multiplier * atr
    lower_basic = hl2 - multiplier * atr

    final_upper = np.zeros(len(ha))
    final_lower = np.zeros(len(ha))
    trend = np.ones(len(ha))  # +1 up, -1 down
    final_upper[:] = np.nan
    final_lower[:] = np.nan

    for i in range(len(ha)):
        if i == 0:
            final_upper[i] = upper_basic.iloc[i]
            final_lower[i] = lower_basic.iloc[i]
            trend[i] = 1
            continue

        # Final upper band
        if (upper_basic.iloc[i] < final_upper[i-1]) or (close.iloc[i-1] > final_upper[i-1]):
            final_upper[i] = upper_basic.iloc[i]
        else:
            final_upper[i] = final_upper[i-1]

        # Final lower band
        if (lower_basic.iloc[i] > final_lower[i-1]) or (close.iloc[i-1] < final_lower[i-1]):
            final_lower[i] = lower_basic.iloc[i]
        else:
            final_lower[i] = final_lower[i-1]

        # Trend switch
        if close.iloc[i] > final_upper[i-1]:
            trend[i] = 1
        elif close.iloc[i] < final_lower[i-1]:
            trend[i] = -1
        else:
            trend[i] = trend[i-1]

    st_line = pd.Series(np.where(trend == 1, final_lower, final_upper), index=ha.index, name="st_line")
    trend_s = pd.Series(trend, index=ha.index, name="trend")
    return pd.DataFrame({"date": ha["date"], "st_line": st_line, "trend": trend_s})

# ---- Donchian op HA ----
def donchian_on_ha(ha: pd.DataFrame, length: int = 20) -> pd.DataFrame:
    upper = ha["ha_high"].rolling(length).max().rename("dc_upper")
    lower = ha["ha_low"].rolling(length).min().rename("dc_lower")
    mid = ((upper + lower) / 2.0).rename("dc_mid")
    return pd.concat([ha["date"], upper, lower, mid], axis=1)

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

fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=520, legend_orientation="h")
fig.update_yaxes(title_text="SPX", secondary_y=False)
if show_vix:
    fig.update_yaxes(title_text="VIX", secondary_y=True)
fig.update_xaxes(showspikes=True, spikemode="across", rangeslider_visible=False)  # rangeslider uit
st.plotly_chart(fig, use_container_width=True)

# ---- Heikin Ashi + SuperTrend + Donchian (tweede grafiek) ----
if show_ha_chart:
    st.subheader("🕯️ Heikin Ashi + SuperTrend + Donchian")

    # Zorg dat we OHLC hebben
    df_ohlc = fetch_ohlc() if fast_mode else df
    needed = {"open", "high", "low", "close"}
    if df_ohlc.empty or not needed.issubset(df_ohlc.columns):
        st.info("Geen OHLC-data beschikbaar om Heikin Ashi te berekenen.")
    else:
        ha = heikin_ashi(df_ohlc).dropna().reset_index(drop=True)
        st_df = supertrend_on_ha(ha, length=int(st_len), multiplier=float(st_mult))
        dc = donchian_on_ha(ha, length=int(dc_len))

        ha_fig = make_subplots(rows=1, cols=1)

        # 1) Heikin Ashi candles
        ha_fig.add_trace(go.Candlestick(
            x=ha["date"], open=ha["ha_open"], high=ha["ha_high"], low=ha["ha_low"], close=ha["ha_close"],
            name="Heikin Ashi"
        ))

        # 2) Donchian Channel (transparant; optionele vulling)
        #   - altijd de lijnen tonen
        ha_fig.add_trace(go.Scatter(x=dc["date"], y=dc["dc_upper"], mode="lines", name="DC Upper", line=dict(width=1)))
        ha_fig.add_trace(go.Scatter(x=dc["date"], y=dc["dc_lower"], mode="lines", name="DC Lower", line=dict(width=1)))
        #   - optioneel: zachte bandvulling tussen lower en upper
        if dc_fill:
            ha_fig.add_trace(go.Scatter(
                x=dc["date"], y=dc["dc_upper"], mode="lines", showlegend=False, line=dict(width=0)
            ))
            ha_fig.add_trace(go.Scatter(
                x=dc["date"], y=dc["dc_lower"], mode="lines", showlegend=False,
                fill="tonexty", fillcolor=f"rgba(150,150,220,{dc_opacity})", line=dict(width=0)
            ))

        # 3) SuperTrend (groen/rood)
        st_up = st_df.where(st_df["trend"] == 1)["st_line"]
        st_dn = st_df.where(st_df["trend"] == -1)["st_line"]
        if st_up.notna().any():
            ha_fig.add_trace(go.Scatter(x=st_df["date"], y=st_up, mode="lines", name="SuperTrend Up",
                                        line=dict(width=2, color="green")))
        if st_dn.notna().any():
            ha_fig.add_trace(go.Scatter(x=st_df["date"], y=st_dn, mode="lines", name="SuperTrend Down",
                                        line=dict(width=2, color="red")))

        ha_fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), height=460, legend_orientation="h")
        ha_fig.update_xaxes(showspikes=True, spikemode="across", rangeslider_visible=False)  # rangeslider uit
        st.plotly_chart(ha_fig, use_container_width=True)

# ---- Footer info ----
st.caption(
    f"Periode: {start_d} → {end_d} • Rijen: {len(df):,} • Modus: {'Close-only' if fast_mode else 'OHLC'} • "
    f"VIX overlay: {'aan' if show_vix else 'uit'} • ST({int(st_len)},{float(st_mult)}) DC({int(dc_len)})"
)
