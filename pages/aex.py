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

AEX_VIEW = st.secrets.get("tables", {}).get(
    "aex_view", "nth-pier-468314-p7.marketdata.aex_with_vix_v"
)

# ---------- Health ----------
try:
    if not bq_ping():
        st.error("Geen BigQuery-verbinding."); st.stop()
except Exception as e:
    st.error("Geen BigQuery-verbinding."); st.caption(f"Details: {e}"); st.stop()

# ---------- Data ----------
@st.cache_data(ttl=1800, show_spinner=False)
def load_aex():
    return run_query(f"SELECT * FROM `{AEX_VIEW}` ORDER BY date")

with st.spinner("AEX data laden…"):
    df = load_aex()
if df.empty:
    st.warning("Geen data in view."); st.stop()

df["date"] = pd.to_datetime(df["date"]).dt.date
for col in ["open","high","low","close","vix_close","ma50","ma200","delta_pct","delta_abs"]:
    if col in df.columns: df[col] = pd.to_numeric(df[col], errors="coerce")

# ---------- helpers ----------
def ema(s: pd.Series, span: int): return s.ewm(span=span, adjust=False).mean()

def heikin_ashi(src: pd.DataFrame):
    ha_close = (src["open"] + src["high"] + src["low"] + src["close"]) / 4.0
    ha_open = [src["open"].iloc[0]]
    for i in range(1, len(src)):
        ha_open.append((ha_open[i-1] + ha_close.iloc[i-1]) / 2.0)
    ha_open = pd.Series(ha_open, index=src.index)
    ha_high = pd.concat([src["high"], ha_open, ha_close], axis=1).max(axis=1)
    ha_low  = pd.concat([src["low"],  ha_open, ha_close], axis=1).min(axis=1)
    return pd.DataFrame({"ha_open":ha_open,"ha_high":ha_high,"ha_low":ha_low,"ha_close":ha_close}, index=src.index)

def true_range(d):
    return pd.concat([d["high"]-d["low"],
                      (d["high"]-d["close"].shift()).abs(),
                      (d["low"] -d["close"].shift()).abs()], axis=1).max(axis=1)

def atr(d, n=10): return true_range(d).rolling(n).mean()

def supertrend(d, period=10, mult=3.0):
    hl2 = (d["high"]+d["low"])/2.0
    _atr = atr(d, period)
    upper = hl2 + mult*_atr
    lower = hl2 - mult*_atr
    st_line = pd.Series(index=d.index, dtype=float)
    st_dir  = pd.Series(index=d.index, dtype=int)
    st_line.iloc[0] = upper.iloc[0]; st_dir.iloc[0] = 1
    for i in range(1, len(d)):
        up, lo = upper.iloc[i], lower.iloc[i]
        prev = st_line.iloc[i-1]; prev_dir = st_dir.iloc[i-1]
        close = d["close"].iloc[i]
        if prev_dir==1: up = min(up, prev)
        else:           lo = max(lo, prev)
        if close>up:    st_dir.iloc[i]=1;  st_line.iloc[i]=lo
        elif close<lo:  st_dir.iloc[i]=-1; st_line.iloc[i]=up
        else:           st_dir.iloc[i]=prev_dir; st_line.iloc[i]=lo if prev_dir==1 else up
    return st_line, st_dir

def donchian(d, n=20): return d["high"].rolling(n).max(), d["low"].rolling(n).min()

def rsi(s, n=14):
    delta = s.diff()
    up = pd.Series(np.where(delta>0, delta, 0.0), index=s.index).rolling(n).mean()
    dn = pd.Series(np.where(delta<0, -delta,0.0), index=s.index).rolling(n).mean()
    rs = up/dn
    return 100 - (100/(1+rs))

def cci(df, n=20):
    tp = (df["high"]+df["low"]+df["close"])/3
    sma = tp.rolling(n).mean()
    mad = (tp - sma).abs().rolling(n).mean()
    return (tp - sma) / (0.015*mad)

# YTD/PYTD from FULL dataset (not filtered)
def ytd_return_full(full_df: pd.DataFrame):
    max_d = full_df["date"].max()
    start = date(max_d.year, 1, 1)
    sub = full_df[full_df["date"] >= start]
    if len(sub) >= 2: return (sub["close"].iloc[-1]/sub["close"].iloc[0]-1)*100
    return None

def pytd_return_full(full_df: pd.DataFrame):
    max_d = full_df["date"].max()
    prev_year = max_d.year - 1
    start = date(prev_year, 1, 1)
    # same calendar day last year (fallback 31-12 if not exists)
    try:
        end = max_d.replace(year=prev_year)
    except ValueError:
        end = date(prev_year, 12, 31)
    sub = full_df[(full_df["date"] >= start) & (full_df["date"] <= end)]
    if len(sub) >= 2: return (sub["close"].iloc[-1]/sub["close"].iloc[0]-1)*100
    return None

# ---------- UI (single slider only) ----------
min_d, max_d = df["date"].min(), df["date"].max()
default_start = max(max_d - timedelta(days=365), min_d)
start_date, end_date = st.slider(
    "Periode", min_value=min_d, max_value=max_d,
    value=(default_start, max_d), format="YYYY-MM-DD"
)

show_supertrend = st.checkbox("Supertrend", value=True)
show_donchian  = st.checkbox("Donchian Channel", value=True)
show_rsi       = st.checkbox("Toon RSI", value=True)
show_cci       = st.checkbox("Toon CCI", value=False)
show_vix       = st.checkbox("Toon VIX‑paneel", value=False)

# filtered view for charts
mask = (df["date"] >= start_date) & (df["date"] <= end_date)
d = df.loc[mask].reset_index(drop=True).copy()

# ---------- indicators ----------
d["ema20"], d["ema50"], d["ema200"] = ema(d["close"],20), ema(d["close"],50), ema(d["close"],200)
d["rsi14"] = rsi(d["close"],14)
d["cci20"] = cci(d,20)
dc_high, dc_low = donchian(d,20)
d["dc_high"], d["dc_low"] = dc_high, dc_low
st_line, st_dir = supertrend(d,10,3.0)
d["supertrend"], d["st_dir"] = st_line, st_dir
ha = heikin_ashi(d)
d = pd.concat([d, ha], axis=1)

# ---------- KPIs ----------
last = d.iloc[-1]
regime = ("Bullish" if (last["close"]>last["ema200"]) and (last["ema50"]>last["ema200"])
          else "Bearish" if (last["close"]<last["ema200"]) and (last["ema50"]<last["ema200"])
          else "Neutraal")

ytd_full  = ytd_return_full(df)   # from full dataset
pytd_full = pytd_return_full(df)  # from full dataset

k1,k2,k3,k4,k5,k6 = st.columns(6)
k1.metric("Laatste close", f"{last['close']:.2f}")
k2.metric("Δ % (dag)", f"{last['delta_pct']*100:.2f}%" if pd.notnull(last["delta_pct"]) else "—")
k3.metric("VIX (close)", f"{last['vix_close']:.2f}" if pd.notnull(last.get("vix_close")) else "—")
k4.metric("Regime", regime)
k5.metric("YTD Return",  f"{ytd_full:.2f}%"  if ytd_full  is not None else "—")
k6.metric("PYTD Return", f"{pytd_full:.2f}%" if pytd_full is not None else "—")

# ---------- Chart (no mini preview; rangeslider disabled) ----------
rows = 3 if show_vix else 2
row_heights = [0.62, 0.38] if rows==2 else [0.54, 0.23, 0.23]

fig = make_subplots(rows=rows, cols=1, shared_xaxes=True,
                    row_heights=row_heights, vertical_spacing=0.02)

# 1) Heikin‑Ashi + EMA ribbon
fig.add_trace(go.Candlestick(
    x=d["date"], open=d["ha_open"], high=d["ha_high"], low=d["ha_low"], close=d["ha_close"],
    name="AEX (Heikin‑Ashi)"
), row=1, col=1)

fig.add_trace(go.Scatter(x=d["date"], y=d["ema20"],  mode="lines", name="EMA20"),  row=1,col=1)
fig.add_trace(go.Scatter(x=d["date"], y=d["ema50"],  mode="lines", name="EMA50"),  row=1,col=1)
fig.add_trace(go.Scatter(x=d["date"], y=d["ema200"], mode="lines", name="EMA200"), row=1,col=1)

if show_donchian:
    fig.add_trace(go.Scatter(x=d["date"], y=d["dc_high"], mode="lines",
                             line=dict(dash="dot", width=2), name="DC High"), row=1,col=1)
    fig.add_trace(go.Scatter(x=d["date"], y=d["dc_low"], mode="lines",
                             line=dict(dash="dot", width=2), name="DC Low"),  row=1,col=1)

if show_supertrend:
    st_up = d["supertrend"].where(d["st_dir"]==1)
    st_dn = d["supertrend"].where(d["st_dir"]==-1)
    fig.add_trace(go.Scatter(x=d["date"], y=st_up, mode="lines",
                             line=dict(width=2, color="green"),
                             name="Supertrend (Bullish)"), row=1,col=1)
    fig.add_trace(go.Scatter(x=d["date"], y=st_dn, mode="lines",
                             line=dict(width=2, color="red"),
                             name="Supertrend (Bearish)"), row=1,col=1)

# 2) optional VIX
next_row = 2
if show_vix and "vix_close" in d.columns:
    fig.add_trace(go.Scatter(x=d["date"], y=d["vix_close"], mode="lines", name="VIX"),
                  row=2, col=1)
    next_row = 3

# RSI/CCI panel
if show_rsi:
    fig.add_trace(go.Scatter(x=d["date"], y=d["rsi14"], mode="lines", name="RSI(14)"),
                  row=next_row, col=1)
    fig.add_hline(y=70, line_dash="dot", row=next_row, col=1)
    fig.add_hline(y=30, line_dash="dot", row=next_row, col=1)
if show_cci:
    fig.add_trace(go.Scatter(x=d["date"], y=d["cci20"], mode="lines", name="CCI(20)"),
                  row=next_row, col=1)
    fig.add_hline(y=100, line_dash="dot", row=next_row, col=1)
    fig.add_hline(y=-100, line_dash="dot", row=next_row, col=1)

# kill the mini preview slider (this was your “second chart”)
fig.update_layout(xaxis_rangeslider_visible=False)
fig.update_xaxes(rangeslider_visible=False)

fig.update_layout(height=860, margin=dict(l=20,r=20,t=30,b=20))
st.plotly_chart(fig, use_container_width=True)

# ---------- Histograms ----------
st.subheader("Histogram dagrendementen")
c1, c2 = st.columns(2)
hist_df = d.dropna(subset=["delta_abs","delta_pct"]).copy()
with c1:
    fig_abs = go.Figure()
    fig_abs.add_trace(go.Histogram(x=hist_df["delta_abs"], nbinsx=60))
    fig_abs.update_layout(height=300, bargap=0.02, margin=dict(l=10,r=10,t=10,b=10))
    st.plotly_chart(fig_abs, use_container_width=True)
with c2:
    fig_pct = go.Figure()
    fig_pct.add_trace(go.Histogram(x=hist_df["delta_pct"]*100.0, nbinsx=60))
    fig_pct.update_layout(height=300, bargap=0.02, margin=dict(l=10,r=10,t=10,b=10))
    st.plotly_chart(fig_pct, use_container_width=True)
