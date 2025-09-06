import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.bq import run_query, bq_ping

st.set_page_config(page_title="📈 S&P 500", layout="wide")
st.title("📈 S&P 500")

SPX_VIEW = st.secrets.get("tables", {}).get(
    "spx_view", "nth-pier-468314-p7.marketdata.spx_with_vix_v"
)

# ---- Health ----
try:
    if not bq_ping():
        st.error("Geen BigQuery-verbinding."); st.stop()
except Exception as e:
    st.error("Geen BigQuery-verbinding."); st.caption(f"Details: {e}"); st.stop()

# ---- Data ----
@st.cache_data(ttl=1800, show_spinner=False)
def load_spx():
    return run_query(f"SELECT * FROM `{SPX_VIEW}` ORDER BY date")

with st.spinner("SPX data laden…"):
    df = load_spx()
if df.empty:
    st.warning("Geen data in view."); st.stop()

df["date"] = pd.to_datetime(df["date"]).dt.date
for c in ["open","high","low","close","vix_close","delta_abs","delta_pct"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# ---- Helpers ----
def ema(s: pd.Series, span: int):
    return s.ewm(span=span, adjust=False).mean()

def heikin_ashi(src: pd.DataFrame):
    ha_close = (src["open"] + src["high"] + src["low"] + src["close"]) / 4.0
    ha_open = [src["open"].iloc[0]]
    for i in range(1, len(src)):
        ha_open.append((ha_open[i-1] + ha_close.iloc[i-1]) / 2.0)
    ha_open = pd.Series(ha_open, index=src.index)
    ha_high = pd.concat([src["high"], ha_open, ha_close], axis=1).max(axis=1)
    ha_low  = pd.concat([src["low"],  ha_open, ha_close], axis=1).min(axis=1)
    return pd.DataFrame({"ha_open":ha_open,"ha_high":ha_high,"ha_low":ha_low,"ha_close":ha_close}, index=src.index)

def atr_rma(high, low, close, length: int):
    prev_close = close.shift(1)
    tr = pd.concat([(high-low).abs(), (high-prev_close).abs(), (low-prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/length, adjust=False).mean()

def supertrend_on_ha(ha: pd.DataFrame, length: int = 10, multiplier: float = 1.0):
    high, low, close = ha["ha_high"], ha["ha_low"], ha["ha_close"]
    atr = atr_rma(high, low, close, length)
    hl2 = (high + low) / 2.0
    upper_basic = hl2 + multiplier * atr
    lower_basic = hl2 - multiplier * atr

    final_upper = np.full(len(ha), np.nan)
    final_lower = np.full(len(ha), np.nan)
    trend = np.ones(len(ha), dtype=int)

    for i in range(len(ha)):
        if i == 0:
            final_upper[i] = upper_basic.iloc[i]
            final_lower[i] = lower_basic.iloc[i]
            trend[i] = 1
            continue

        final_upper[i] = upper_basic.iloc[i] if (upper_basic.iloc[i] < final_upper[i-1]) or (close.iloc[i-1] > final_upper[i-1]) else final_upper[i-1]
        final_lower[i] = lower_basic.iloc[i] if (lower_basic.iloc[i] > final_lower[i-1]) or (close.iloc[i-1] < final_lower[i-1]) else final_lower[i-1]

        if close.iloc[i] > final_upper[i-1]:
            trend[i] = 1
        elif close.iloc[i] < final_lower[i-1]:
            trend[i] = -1
        else:
            trend[i] = trend[i-1]

    st_line = pd.Series(np.where(trend == 1, final_lower, final_upper), index=ha.index, name="st_line")
    trend_s = pd.Series(trend, index=ha.index, name="trend")
    return pd.DataFrame({"st_line": st_line, "trend": trend_s}, index=ha.index)

def donchian(d, n=20): return d["high"].rolling(n).max(), d["low"].rolling(n).min()
def rsi(s, n=14):
    delta = s.diff()
    up = pd.Series(np.where(delta>0, delta, 0.0), index=s.index).rolling(n).mean()
    dn = pd.Series(np.where(delta<0, -delta,0.0), index=s.index).rolling(n).mean()
    rs = up/dn
    return 100 - (100/(1+rs))
def cci(df_, n=20):
    tp = (df_["high"]+df_["low"]+df_["close"])/3
    sma = tp.rolling(n).mean()
    mad = (tp - sma).abs().rolling(n).mean()
    return (tp - sma) / (0.015*mad)

def ytd_return_full(full_df: pd.DataFrame):
    max_d = full_df["date"].max(); start = date(max_d.year, 1, 1)
    sub = full_df[full_df["date"] >= start]
    return (sub["close"].iloc[-1]/sub["close"].iloc[0]-1)*100 if len(sub)>=2 else None
def pytd_return_full(full_df: pd.DataFrame):
    max_d = full_df["date"].max(); prev_year = max_d.year-1
    start = date(prev_year,1,1)
    try: end = max_d.replace(year=prev_year)
    except ValueError: end = date(prev_year,12,31)
    sub = full_df[(full_df["date"]>=start)&(full_df["date"]<=end)]
    return (sub["close"].iloc[-1]/sub["close"].iloc[0]-1)*100 if len(sub)>=2 else None

# ---- UI / filter ----
min_d, max_d = df["date"].min(), df["date"].max()
default_start = max(max_d - timedelta(days=365), min_d)
start_date, end_date = st.slider("Periode", min_value=min_d, max_value=max_d,
                                 value=(default_start, max_d), format="YYYY-MM-DD")
d = df[(df["date"]>=start_date)&(df["date"]<=end_date)].reset_index(drop=True).copy()
if "delta_abs" not in d or d["delta_abs"].isna().all(): d["delta_abs"]=d["close"].diff()
if "delta_pct" not in d or d["delta_pct"].isna().all(): d["delta_pct"]=d["close"].pct_change()*100.0

# ✨ Nieuw: keuzemenu voor delta in paneel 3
delta_mode = st.selectbox("Paneel 3: kies delta-metric", ["Δ punten", "Δ %"], index=0)
if delta_mode == "Δ %":
    delta_series = d["delta_pct"]
    delta_title = "Δ dag (%)"
    delta_legend = "Δ dag (%)"
else:
    delta_series = d["delta_abs"]
    delta_title = "Δ dag (punten)"
    delta_legend = "Δ dag (punten)"

# ---- Indicators ----
d["ema20"], d["ema50"], d["ema200"] = ema(d["close"],20), ema(d["close"],50), ema(d["close"],200)
d["rsi14"] = rsi(d["close"],14); d["cci20"] = cci(d,20)
dc_high, dc_low = donchian(d,20); d["dc_high"], d["dc_low"] = dc_high, dc_low
ha = heikin_ashi(d); st_df = supertrend_on_ha(ha, length=10, multiplier=1.0)
d[["ha_open","ha_high","ha_low","ha_close"]] = ha[["ha_open","ha_high","ha_low","ha_close"]]
d["st_line"], d["st_trend"] = st_df["st_line"], st_df["trend"]

# ---- KPI ----
last = d.iloc[-1]
regime = ("Bullish" if (last["close"]>d["ema200"].iloc[-1]) and (d["ema50"].iloc[-1]>d["ema200"].iloc[-1])
          else "Bearish" if (last["close"]<d["ema200"].iloc[-1]) and (d["ema50"].iloc[-1]<d["ema200"].iloc[-1])
          else "Neutraal")
ytd_full, pytd_full = ytd_return_full(df), pytd_return_full(df)

c1,c2,c3,c4,c5,c6 = st.columns(6)
c1.metric("Laatste close", f"{last['close']:.2f}")
c2.metric("Δ % (dag)", f"{(last['close']/d['close'].shift(1).iloc[-1]-1)*100:.2f}%" if len(d)>1 else "—")
c3.metric("VIX (close)", f"{last.get('vix_close'):.2f}" if pd.notnull(last.get("vix_close")) else "—")
c4.metric("Regime", regime)
c5.metric("YTD Return",  f"{ytd_full:.2f}%"  if ytd_full  is not None else "—")
c6.metric("PYTD Return", f"{pytd_full:.2f}%" if pytd_full is not None else "—")

# ---- Chart (5 panelen) ----
fig = make_subplots(
    rows=5, cols=1, shared_xaxes=True,
    specs=[
        [{}],
        [{"secondary_y": True}],
        [{}],
        [{}],
        [{}],
    ],
    subplot_titles=[
        "SP500 Heikin-Ashi + Supertrend (10,1) + Donchian",
        "Close + EMA(20/50/200) + VIX (2e y-as)",
        delta_title,
        "RSI(14)",
        "CCI(20)"
    ],
    row_heights=[0.38, 0.28, 0.14, 0.10, 0.10],
    vertical_spacing=0.06
)

# (1) HA + DC + ST
fig.add_trace(go.Candlestick(
    x=d["date"], open=d["ha_open"], high=d["ha_high"], low=d["ha_low"], close=d["ha_close"],
    name="SPX (Heikin-Ashi)"),
    row=1, col=1
)
fig.add_trace(go.Scatter(x=d["date"], y=d["dc_high"], mode="lines",
                         line=dict(dash="dot", width=2), name="DC High"), row=1, col=1)
fig.add_trace(go.Scatter(x=d["date"], y=d["dc_low"], mode="lines",
                         line=dict(dash="dot", width=2), name="DC Low"), row=1, col=1)
st_up = d["st_line"].where(d["st_trend"]==1); st_dn = d["st_line"].where(d["st_trend"]==-1)
fig.add_trace(go.Scatter(x=d["date"], y=st_up, mode="lines",
                         line=dict(width=2, color="green"), name="Supertrend ↑ (10,1)"), row=1,col=1)
fig.add_trace(go.Scatter(x=d["date"], y=st_dn, mode="lines",
                         line=dict(width=2, color="red"),   name="Supertrend ↓ (10,1)"), row=1,col=1)

# (2) Close + EMA (primair) & VIX (secundair)
fig.add_trace(go.Scatter(x=d["date"], y=d["close"], mode="lines", name="Close"),
              row=2, col=1, secondary_y=False)
fig.add_trace(go.Scatter(x=d["date"], y=d["ema20"], mode="lines", name="EMA20"),
              row=2, col=1, secondary_y=False)
fig.add_trace(go.Scatter(x=d["date"], y=d["ema50"], mode="lines", name="EMA50"),
              row=2, col=1, secondary_y=False)
fig.add_trace(go.Scatter(x=d["date"], y=d["ema200"], mode="lines", name="EMA200"),
              row=2, col=1, secondary_y=False)

if "vix_close" in d.columns and d["vix_close"].notna().any():
    fig.add_trace(go.Scatter(x=d["date"], y=d["vix_close"], mode="lines",
                             name="VIX (sec. y)"),
                  row=2, col=1, secondary_y=True)

# (3) Delta-bars (punten of procenten)
delta_colors = np.where(delta_series >= 0, "rgba(16,150,24,0.7)", "rgba(219,64,82,0.7)")
fig.add_trace(go.Bar(x=d["date"], y=delta_series, name=delta_legend,
                     marker=dict(color=delta_colors), opacity=0.9),
              row=3, col=1)

# (4) RSI
fig.add_trace(go.Scatter(x=d["date"], y=d["rsi14"], mode="lines", name="RSI(14)"), row=4,col=1)
fig.add_hline(y=70, line_dash="dot", row=4, col=1); fig.add_hline(y=30, line_dash="dot", row=4, col=1)

# (5) CCI
fig.add_trace(go.Scatter(x=d["date"], y=d["cci20"], mode="lines", name="CCI(20)"), row=5,col=1)
fig.add_hline(y=100, line_dash="dot", row=5, col=1); fig.add_hline(y=-100, line_dash="dot", row=5, col=1)

# Layout & assen
fig.update_layout(
    height=1400, margin=dict(l=20,r=20,t=60,b=20),
    legend_orientation="h", legend_yanchor="top", legend_y=1.08, legend_x=0
)
fig.update_layout(xaxis_rangeslider_visible=False); fig.update_xaxes(rangeslider_visible=False)

fig.update_yaxes(title_text="Index (HA)", row=1,col=1)
fig.update_yaxes(title_text="Close/EMA", row=2,col=1, secondary_y=False)
fig.update_yaxes(title_text="VIX", row=2,col=1, secondary_y=True)
fig.update_yaxes(title_text=delta_title, row=3,col=1)
fig.update_yaxes(title_text="RSI", row=4,col=1)
fig.update_yaxes(title_text="CCI", row=5,col=1)

st.plotly_chart(fig, use_container_width=True)

# ---- Histogrammen ----
st.subheader("Histogram dagrendementen")
bins = st.slider("Aantal bins", 10, 120, 60, 5)
c1, c2 = st.columns(2)
hist_df = d.dropna(subset=["delta_abs","delta_pct"]).copy()
with c1:
    fig_abs = go.Figure([go.Histogram(x=hist_df["delta_abs"], nbinsx=int(bins))])
    fig_abs.update_layout(title="Δ abs (punten)", height=320, bargap=0.02, margin=dict(l=10,r=10,t=40,b=10))
    st.plotly_chart(fig_abs, use_container_width=True)
with c2:
    # FIX: delta_pct is al in %, dus geen extra *100
    fig_pct = go.Figure([go.Histogram(x=hist_df["delta_pct"], nbinsx=int(bins))])
    fig_pct.update_layout(title="Δ %", height=320, bargap=0.02, margin=dict(l=10,r=10,t=40,b=10))
    st.plotly_chart(fig_pct, use_container_width=True)
