# pages/2_AEX.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.bq import run_query, bq_ping

st.set_page_config(page_title="📊 AEX", layout="wide")
st.title("📊 AEX Dashboard")

# ---- View config (uit secrets of fallback) ----
AEX_VIEW = st.secrets.get("tables", {}).get(
    "aex_view",
    "nth-pier-468314-p7.marketdata.aex_with_vix_v"
)

# ---- Health check ----
try:
    with st.spinner("BigQuery check…"):
        ok = bq_ping()
        if not ok:
            st.error("Geen BigQuery-verbinding (bq_ping==False).")
            st.stop()
except Exception as e:
    st.error("Geen BigQuery-verbinding.")
    st.caption(f"Details: {e}")
    st.stop()

# ---- Data ophalen ----
@st.cache_data(ttl=1800, show_spinner=False)
def load_aex():
    sql = f"SELECT * FROM `{AEX_VIEW}` ORDER BY date"
    return run_query(sql)

with st.spinner("AEX data laden…"):
    df = load_aex()

if df.empty:
    st.warning("Geen data in view ontvangen.")
    st.stop()

# ---- Datatypes & schoonmaak ----
df["date"] = pd.to_datetime(df["date"]).dt.date
for col in ["open","high","low","close","vix_close","ma50","ma200","delta_pct","delta_abs"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# =======================
# Indicator helpers
# =======================
def ema(series: pd.Series, span: int):
    return series.ewm(span=span, adjust=False).mean()

def heikin_ashi(src: pd.DataFrame):
    ha = src[["open","high","low","close"]].copy()
    ha_open = [ha["open"].iloc[0]]
    ha_close = (ha["open"] + ha["high"] + ha["low"] + ha["close"]) / 4.0
    for i in range(1, len(ha)):
        ha_open.append((ha_open[i-1] + ha_close.iloc[i-1]) / 2.0)
    ha_high = pd.concat([ha["high"], pd.Series(ha_open, index=ha.index), ha_close], axis=1).max(axis=1)
    ha_low  = pd.concat([ha["low"],  pd.Series(ha_open, index=ha.index), ha_close], axis=1).min(axis=1)
    out = pd.DataFrame({
        "ha_open": ha_open,
        "ha_high": ha_high,
        "ha_low":  ha_low,
        "ha_close": ha_close
    }, index=src.index)
    return out

def true_range(df_):
    high_low = df_["high"] - df_["low"]
    high_close = (df_["high"] - df_["close"].shift()).abs()
    low_close  = (df_["low"]  - df_["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr

def atr(df_, period=10):
    tr = true_range(df_)
    return tr.rolling(period).mean()

def supertrend(df_, period=10, multiplier=3.0):
    # gebaseerd op klassieke supertrend
    hl2 = (df_["high"] + df_["low"]) / 2.0
    _atr = atr(df_, period)
    upperband = hl2 + multiplier * _atr
    lowerband = hl2 - multiplier * _atr

    st_trend = pd.Series(index=df_.index, dtype=float)
    st_dir   = pd.Series(index=df_.index, dtype=int)  # 1=up, -1=down

    st_trend.iloc[0] = upperband.iloc[0]
    st_dir.iloc[0]   = 1

    for i in range(1, len(df_)):
        curr_up = upperband.iloc[i]
        curr_lo = lowerband.iloc[i]
        prev_st = st_trend.iloc[i-1]
        prev_dir = st_dir.iloc[i-1]
        close_i = df_["close"].iloc[i]

        # trailing logic
        if prev_dir == 1:
            curr_up = min(curr_up, prev_st)
        else:
            curr_lo = max(curr_lo, prev_st)

        if close_i > curr_up:
            st_dir.iloc[i] = 1
            st_trend.iloc[i] = curr_lo
        elif close_i < curr_lo:
            st_dir.iloc[i] = -1
            st_trend.iloc[i] = curr_up
        else:
            st_dir.iloc[i] = prev_dir
            st_trend.iloc[i] = curr_lo if prev_dir == 1 else curr_up

    return st_trend, st_dir

def donchian(df_, length=20):
    dc_high = df_["high"].rolling(length).max()
    dc_low  = df_["low"].rolling(length).min()
    return dc_high, dc_low

def rsi(series, length=14):
    delta = series.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=series.index).rolling(length).mean()
    roll_dn = pd.Series(down, index=series.index).rolling(length).mean()
    rs = roll_up / roll_dn
    return 100 - (100 / (1 + rs))

# =======================
# UI – filters
# =======================
colL, colR = st.columns([3,1])
with colL:
    min_d, max_d = df["date"].min(), df["date"].max()
    default_start = max_d - timedelta(days=365)
    start_date, end_date = st.slider(
        "Periode", min_value=min_d, max_value=max_d,
        value=(max(default_start, min_d), max_d), format="YYYY-MM-DD"
    )
with colR:
    use_heikin = st.checkbox("Heikin‑Ashi", value=True)
    show_supertrend = st.checkbox("Supertrend", value=True)
    show_donchian = st.checkbox("Donchian Channel", value=True)

mask = (df["date"] >= start_date) & (df["date"] <= end_date)
d = df.loc[mask].reset_index(drop=True).copy()

# Indicators (client‑side)
d["ema20"]  = ema(d["close"], 20)
d["ema50"]  = ema(d["close"], 50)
d["ema200"] = ema(d["close"], 200)
d["rsi14"]  = rsi(d["close"], 14)

dc_high, dc_low = donchian(d, 20)
d["dc_high"], d["dc_low"] = dc_high, dc_low

st_line, st_dir = supertrend(d, period=10, multiplier=3.0)
d["supertrend"] = st_line
d["st_dir"] = st_dir

# Heikin Ashi voor weergave (optioneel)
if use_heikin:
    ha = heikin_ashi(d)
    d = pd.concat([d, ha], axis=1)

# =======================
# KPI’s / regime
# =======================
last = d.iloc[-1]
regime = "Bullish" if (last["close"] > last["ema200"]) and (last["ema50"] > last["ema200"]) else \
         "Bearish" if (last["close"] < last["ema200"]) and (last["ema50"] < last["ema200"]) else "Neutraal"

k1, k2, k3, k4 = st.columns(4)
k1.metric("Laatste close", f"{last['close']:.2f}")
k2.metric("Δ % (dag)", f"{last['delta_pct']*100:.2f}%" if pd.notnull(last["delta_pct"]) else "—")
k3.metric("VIX (close)", f"{last['vix_close']:.2f}" if pd.notnull(last.get("vix_close")) else "—")
k4.metric("Regime (EMA200)", regime)

# =======================
# Grafieken
# =======================
fig = make_subplots(
    rows=3, cols=1, shared_xaxes=True,
    row_heights=[0.56, 0.24, 0.20],
    vertical_spacing=0.02,
    specs=[[{"type": "xy"}],
           [{"type": "xy"}],
           [{"type": "xy"}]]
)

# 1) Price panel
if use_heikin:
    fig.add_trace(go.Candlestick(
        x=d["date"], open=d["ha_open"], high=d["ha_high"],
        low=d["ha_low"], close=d["ha_close"],
        name="AEX (HA)", showlegend=True
    ), row=1, col=1)
else:
    fig.add_trace(go.Candlestick(
        x=d["date"], open=d["open"], high=d["high"],
        low=d["low"], close=d["close"],
        name="AEX", showlegend=True
    ), row=1, col=1)

# MA/EMA
fig.add_trace(go.Scatter(x=d["date"], y=d["ema20"],  mode="lines", name="EMA20"),  row=1, col=1)
fig.add_trace(go.Scatter(x=d["date"], y=d["ema50"],  mode="lines", name="EMA50"),  row=1, col=1)
fig.add_trace(go.Scatter(x=d["date"], y=d["ema200"], mode="lines", name="EMA200"), row=1, col=1)

# Donchian
if show_donchian:
    fig.add_trace(go.Scatter(x=d["date"], y=d["dc_high"], mode="lines", name="DC High",
                             line=dict(dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=d["date"], y=d["dc_low"],  mode="lines", name="DC Low",
                             line=dict(dash="dot"), fill="tonexty", opacity=0.15,
                             hoverinfo="skip", showlegend=True), row=1, col=1)

# Supertrend
if show_supertrend:
    fig.add_trace(go.Scatter(
        x=d["date"], y=d["supertrend"], mode="lines",
        name="Supertrend",
    ), row=1, col=1)

# 2) VIX panel
if "vix_close" in d.columns:
    fig.add_trace(go.Scatter(x=d["date"], y=d["vix_close"], mode="lines", name="VIX"),
                  row=2, col=1)

# 3) RSI + histogram daily Δ%
fig.add_trace(go.Scatter(x=d["date"], y=d["rsi14"], mode="lines", name="RSI(14)"),
              row=3, col=1)
fig.add_hline(y=70, line_dash="dot", row=3, col=1)
fig.add_hline(y=30, line_dash="dot", row=3, col=1)

fig.update_layout(
    height=900, margin=dict(l=20, r=20, t=30, b=20),
    xaxis3_rangeslider_visible=False
)
fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=True, zeroline=False)

st.plotly_chart(fig, use_container_width=True)

# Histogram daily returns
st.subheader("Histogram dagrendementen (%)")
hist_df = d.dropna(subset=["delta_pct"]).copy()
hist_df["delta_pct"] = hist_df["delta_pct"] * 100.0
st.caption("Verdeling van dagelijkse procentuele veranderingen; helpt om volatiliteit/regimes te herkennen.")
hist = go.Figure()
hist.add_trace(go.Histogram(x=hist_df["delta_pct"], nbinsx=60, name="Δ%"))
hist.update_layout(height=300, bargap=0.02, margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(hist, use_container_width=True)

# Extra toelichting (optioneel)
with st.expander("Uitleg & trend-signalen"):
    st.markdown("""
- **Regime**: Bullish als close > EMA200 en EMA50 > EMA200; Bearish als beide eronder.  
- **EMA‑ribbon (20/50/200)**: kruisingen (EMA20 ↔ EMA50) geven kortetermijn momentum; positie t.o.v. EMA200 geeft langetermijn trend.  
- **Donchian(20)**: uitbraak boven DC‑High = momentum‑breakout; onder DC‑Low = breakdown.  
- **Supertrend(10,3)**: dynamische trailing‑trend; boven prijs = bearish druk, onder prijs = bullish steun.  
- **RSI(14)**: >70 overbought, <30 oversold – in sterke trends blijven waarden lang ‘extreem’.  
- **Histogram Δ%**: scheefheid en staartdikte geven regime/volatiliteit aan.
""")
