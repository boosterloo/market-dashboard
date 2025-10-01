import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.bq import run_query, bq_ping

st.set_page_config(page_title="📊 AEX", layout="wide")
st.title("📊 AEX")

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
for c in ["open","high","low","close","vix_close","delta_abs","delta_pct"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# ---------- Helpers ----------
def ema(s: pd.Series, span: int): 
    return s.ewm(span=span, adjust=False, min_periods=span).mean()

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

        if close.iloc[i] > final_upper[i-1]:   trend[i] = 1
        elif close.iloc[i] < final_lower[i-1]: trend[i] = -1
        else:                                  trend[i] = trend[i-1]

    st_line = pd.Series(np.where(trend == 1, final_lower, final_upper), index=ha.index, name="st_line")
    trend_s = pd.Series(trend, index=ha.index, name="trend")
    return pd.DataFrame({"st_line": st_line, "trend": trend_s}, index=ha.index)

def donchian(d, n=20): 
    return d["high"].rolling(n, min_periods=n).max(), d["low"].rolling(n, min_periods=n).min()

# --- MACD (trendmomentum) ---
def macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = series.ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

# --- ADX (trendsterkte) ---
def adx(df_: pd.DataFrame, length: int = 14):
    high, low, close = df_["high"], df_["low"], df_["close"]
    up_move   = high.diff()
    down_move = -low.diff()

    plus_dm  = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr1 = (high - low).abs()
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low  - close.shift(1)).abs()
    tr  = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1/length, adjust=False).mean()

    plus_di  = 100 * pd.Series(plus_dm, index=df_.index).ewm(alpha=1/length, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=df_.index).ewm(alpha=1/length, adjust=False).mean() / atr

    dx  = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.ewm(alpha=1/length, adjust=False).mean()
    return plus_di, minus_di, adx

def ytd_return_full(full_df: pd.DataFrame):
    max_d = full_df["date"].max(); start = date(max_d.year, 1, 1)
    sub = full_df[full_df["date"] >= start]
    return (sub["close"].iloc[-1]/sub["close"].iloc[0]-1)*100 if len(sub)>=2 else None

def pytd_return_full(full_df: pd.DataFrame):
    max_d = full_df["date"].max(); prev_year = max_d.year-1
    start = date(prev_year,1,1)
    try: 
        end = max_d.replace(year=prev_year)
    except ValueError: 
        end = date(prev_year,12,31)
    sub = full_df[(full_df["date"]>=start)&(full_df["date"]<=end)]
    return (sub["close"].iloc[-1]/sub["close"].iloc[0]-1)*100 if len(sub)>=2 else None

# ---------- Periode ----------
min_d, max_d = df["date"].min(), df["date"].max()
default_start = max(max_d - timedelta(days=365), min_d)
start_date, end_date = st.slider("Periode", min_value=min_d, max_value=max_d,
                                 value=(default_start, max_d), format="YYYY-MM-DD")

# ---------- Top-UI ----------
c_top1, c_top2, c_top3 = st.columns([1.1,1.2,1.2])
with c_top1:
    show_vix = st.checkbox("Toon VIX (paneel 1)", value=True,
                           help="VIX op 2e y-as in de bovenste grafiek.")
with c_top2:
    corr_vs = st.radio("Rolling correlatie vs VIX", ["% change", "level"], index=0, horizontal=True,
                       help="Corr(Δ% AEX, Δ% VIX) of Corr(Δ% AEX, VIX-level).")
with c_top3:
    corr_win = st.slider("Correlatie-window", 5, 90, 20, step=1,
                         help="Aantal dagen voor de rolling correlatie (paneel 5).")

# ---------- Filter & basis ----------
d = df[(df["date"]>=start_date)&(df["date"]<=end_date)].reset_index(drop=True).copy()
if "delta_abs" not in d or d["delta_abs"].isna().all(): d["delta_abs"]=d["close"].diff()
if "delta_pct" not in d or d["delta_pct"].isna().all(): d["delta_pct"]=d["close"].pct_change()*100.0

# MAs
d["ema20"], d["ema50"], d["ema200"] = ema(d["close"],20), ema(d["close"],50), ema(d["close"],200)
# Donchian + HA + Supertrend
dc_high, dc_low = donchian(d,20); d["dc_high"], d["dc_low"] = dc_high, dc_low
ha = heikin_ashi(d); st_df = supertrend_on_ha(ha, length=10, multiplier=1.0)
d[["ha_open","ha_high","ha_low","ha_close"]] = ha[["ha_open","ha_high","ha_low","ha_close"]]
d["st_line"], d["st_trend"] = st_df["st_line"], st_df["trend"]
# MACD & ADX
d["macd_line"], d["macd_signal"], d["macd_hist"] = macd(d["close"], fast=12, slow=26, signal=9)
d["di_plus"], d["di_minus"], d["adx14"] = adx(d, length=14)

# Regime & signals
up_regime = (d["close"] > d["ema200"]) & (d["ema50"] > d["ema200"])
down_regime = (d["close"] < d["ema200"]) & (d["ema50"] < d["ema200"])

# Buy: uptrend + MACD hist cross boven 0
d["buy_sig"] = up_regime & (d["macd_hist"].shift(1) <= 0) & (d["macd_hist"] > 0)
# Sell: close < EMA20 of MACD hist cross onder 0
d["sell_sig"] = (d["close"] < d["ema20"]) | ((d["macd_hist"].shift(1) >= 0) & (d["macd_hist"] < 0))

# ---------- KPI ----------
last = d.iloc[-1]
regime = ("Bullish" if up_regime.iloc[-1]
          else "Bearish" if down_regime.iloc[-1]
          else "Neutraal")
ytd_full, pytd_full = ytd_return_full(df), pytd_return_full(df)
k1,k2,k3,k4,k5,k6 = st.columns(6)
k1.metric("Laatste close", f"{last['close']:.2f}")
k2.metric("Δ % (dag)", f"{(last['close']/d['close'].shift(1).iloc[-1]-1)*100:.2f}%" if len(d)>1 else "—")
k3.metric("VIX (close)", f"{last.get('vix_close'):.2f}" if pd.notnull(last.get("vix_close")) else "—")
k4.metric("Regime", regime)
k5.metric("YTD Return",  f"{ytd_full:.2f}%"  if ytd_full  is not None else "—")
k6.metric("PYTD Return", f"{pytd_full:.2f}%" if pytd_full is not None else "—")

# =======================
# Paneel 1 – HA + ST + Donchian (+ VIX)
# =======================
fig1 = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]],
                     subplot_titles=["AEX Heikin-Ashi + Supertrend (10,1) + Donchian" + (" + VIX (2e y-as)" if show_vix else "")])
fig1.add_trace(go.Candlestick(x=d["date"], open=d["ha_open"], high=d["ha_high"], low=d["ha_low"], close=d["ha_close"],
                              name="AEX (Heikin-Ashi)"), row=1, col=1, secondary_y=False)
fig1.add_trace(go.Scatter(x=d["date"], y=d["dc_high"], mode="lines", line=dict(dash="dot", width=2), name="DC High"),
              row=1, col=1, secondary_y=False)
fig1.add_trace(go.Scatter(x=d["date"], y=d["dc_low"], mode="lines", line=dict(dash="dot", width=2), name="DC Low"),
              row=1, col=1, secondary_y=False)
st_up = d["st_line"].where(d["st_trend"]==1); st_dn = d["st_line"].where(d["st_trend"]==-1)
fig1.add_trace(go.Scatter(x=d["date"], y=st_up, mode="lines", line=dict(width=2, color="green"), name="Supertrend ↑ (10,1)"),
              row=1, col=1, secondary_y=False)
fig1.add_trace(go.Scatter(x=d["date"], y=st_dn, mode="lines", line=dict(width=2, color="red"), name="Supertrend ↓ (10,1)"),
              row=1, col=1, secondary_y=False)
if show_vix and ("vix_close" in d.columns and d["vix_close"].notna().any()):
    fig1.add_trace(go.Scatter(x=d["date"], y=d["vix_close"], mode="lines", name="VIX (sec. y)"),
                  row=1, col=1, secondary_y=True)

fig1.update_layout(height=520, margin=dict(l=20,r=20,t=60,b=10),
                   legend_orientation="h", legend_yanchor="top", legend_y=1.08, legend_x=0)
fig1.update_xaxes(rangeslider_visible=False)
fig1.update_yaxes(title_text="Index (HA)", row=1, col=1, secondary_y=False)
fig1.update_yaxes(title_text="VIX", row=1, col=1, secondary_y=True)
st.plotly_chart(fig1, use_container_width=True)

# ---------------- Δ-instellingen ----------------
st.markdown("#### Δ-instellingen")
d1, d2, d3, d4 = st.columns([1.0, 1.0, 0.9, 1.2])
with d1:
    delta_mode = st.radio("Weergave", ["Δ punten", "Δ %"], index=0, horizontal=True,
                          help="Wat tonen we in het Δ-paneel direct hieronder?")
with d2:
    agg_mode = st.selectbox("Aggregatie", ["Dagelijks", "Wekelijks", "Maandelijks"], index=0,
                            help="Δ punten = som per periode; Δ % = gecomponeerd per periode.")
with d3:
    smooth_on = st.checkbox("Smoothing (MA)", value=False)
with d4:
    ma_window = st.slider("MA-window", 2, 60, 5, step=1, disabled=not smooth_on)

def aggregate_delta(_df: pd.DataFrame, mode: str, how: str) -> pd.Series:
    t = _df.copy()
    t["date_dt"] = pd.to_datetime(t["date"]); t = t.set_index("date_dt")
    if how == "Dagelijks":
        return t["delta_pct"] if mode == "Δ %" else t["delta_abs"]
    rule = "W-FRI" if how == "Wekelijks" else "M"
    if mode == "Δ %":
        return t["delta_pct"].groupby(pd.Grouper(freq=rule)).apply(
            lambda g: (np.prod((g.dropna()/100.0 + 1.0)) - 1.0) * 100.0 if len(g.dropna()) else np.nan
        )
    else:
        return t["delta_abs"].groupby(pd.Grouper(freq=rule)).sum(min_count=1)

delta_series = aggregate_delta(d, delta_mode, agg_mode)
if smooth_on:
    delta_series = delta_series.rolling(ma_window, min_periods=1).mean()
delta_x = delta_series.index if isinstance(delta_series.index, pd.DatetimeIndex) else pd.to_datetime(d["date"])
delta_x = delta_x.date
delta_legend = "Δ (%)" if delta_mode=="Δ %" else "Δ (punten)"
if smooth_on: delta_legend += f" — MA{ma_window}"
delta_colors = np.where(delta_series.values >= 0, "rgba(16,150,24,0.7)", "rgba(219,64,82,0.7)")

# =======================
# Paneel 2–5
# =======================
fig2 = make_subplots(
    rows=4, cols=1, shared_xaxes=True,
    subplot_titles=[
        f"{'Δ (%)' if delta_mode=='Δ %' else 'Δ (punten)'} — {agg_mode.lower()}{' — MA'+str(ma_window) if smooth_on else ''}",
        "Close + EMA(20/50/200) — regime, buy/sell",
        "MACD(12,26,9) — lijn/signal/hist",
        f"ADX(14) + DI± — corr window={corr_win} t.b.v. paneel 5 hieronder"
    ],
    row_heights=[0.22, 0.36, 0.21, 0.21],
    vertical_spacing=0.06
)

# (1) Δ bars
fig2.add_trace(go.Bar(x=delta_x, y=delta_series.values, name=delta_legend,
                      marker=dict(color=delta_colors), opacity=0.9), row=1, col=1)

# (2) Close + EMA + regime shading + signals
fig2.add_trace(go.Scatter(x=d["date"], y=d["close"], mode="lines", name="Close"), row=2, col=1)
fig2.add_trace(go.Scatter(x=d["date"], y=d["ema20"], mode="lines", name="EMA20"), row=2, col=1)
fig2.add_trace(go.Scatter(x=d["date"], y=d["ema50"], mode="lines", name="EMA50"), row=2, col=1)
fig2.add_trace(go.Scatter(x=d["date"], y=d["ema200"], mode="lines", name="EMA200"), row=2, col=1)

buys  = d.loc[d["buy_sig"]]
sells = d.loc[d["sell_sig"]]
fig2.add_trace(go.Scatter(
    x=buys["date"], y=buys["close"], mode="markers", name="Buy",
    marker=dict(symbol="triangle-up", size=11, color="#00A65A", line=dict(width=1, color="black")),
    hovertemplate="Buy<br>%{x}<br>Close=%{y:.2f}<extra></extra>"
), row=2, col=1)
fig2.add_trace(go.Scatter(
    x=sells["date"], y=sells["close"], mode="markers", name="Sell",
    marker=dict(symbol="triangle-down", size=11, color="#D55E00", line=dict(width=1, color="black")),
    hovertemplate="Sell<br>%{x}<br>Close=%{y:.2f}<extra></extra>"
), row=2, col=1)

# (3) MACD
fig2.add_trace(go.Scatter(x=d["date"], y=d["macd_line"],   mode="lines", name="MACD"),   row=3, col=1)
fig2.add_trace(go.Scatter(x=d["date"], y=d["macd_signal"], mode="lines", name="Signal"), row=3, col=1)
fig2.add_trace(go.Bar(x=d["date"], y=d["macd_hist"], name="Hist", 
                      marker_color=np.where(d["macd_hist"]>=0, "rgba(16,150,24,0.6)", "rgba(219,64,82,0.6)")),
               row=3, col=1)
fig2.add_hline(y=0, line_dash="dot", row=3, col=1)

# (4) ADX + DI± (trendkwaliteit)
fig2.add_trace(go.Scatter(x=d["date"], y=d["adx14"],   mode="lines", name="ADX(14)"), row=4, col=1)
fig2.add_trace(go.Scatter(x=d["date"], y=d["di_plus"], mode="lines", name="+DI"),     row=4, col=1)
fig2.add_trace(go.Scatter(x=d["date"], y=d["di_minus"],mode="lines", name="−DI"),     row=4, col=1)
fig2.add_hline(y=20, line_dash="dot", row=4, col=1)  # >20 vaak 'zinvolle' trend

fig2.update_layout(height=980, margin=dict(l=20,r=20,t=50,b=20),
                   legend_orientation="h", legend_yanchor="top", legend_y=1.06, legend_x=0)
fig2.update_xaxes(rangeslider_visible=False)
fig2.update_yaxes(title_text="Δ", row=1, col=1)
fig2.update_yaxes(title_text="Close/EMA", row=2, col=1)
fig2.update_yaxes(title_text="MACD", row=3, col=1)
fig2.update_yaxes(title_text="ADX / DI", row=4, col=1)

st.plotly_chart(fig2, use_container_width=True)

# ---------- Rolling correlatie met VIX ----------
st.markdown("#### Rolling correlatie met VIX")
corr_df = d.copy(); corr_df["date_dt"] = pd.to_datetime(corr_df["date"]); corr_df = corr_df.set_index("date_dt")
aex_ret = corr_df["delta_pct"]  # %
vix_series = (corr_df["vix_close"].pct_change()*100.0) if corr_vs=="% change" else corr_df["vix_close"]
corr_join = pd.concat([aex_ret.rename("aex"), vix_series.rename("vix")], axis=1).dropna()
rolling_corr = corr_join["aex"].rolling(corr_win).corr(corr_join["vix"])

fig_corr = go.Figure()
fig_corr.add_trace(go.Scatter(x=rolling_corr.index.date, y=rolling_corr.values, mode="lines", name="Rolling corr"))
fig_corr.add_hline(y=0.0, line_dash="dot")
fig_corr.add_hrect(y0=-1, y1=-0.5, fillcolor="rgba(255,0,0,0.06)", line_width=0)
fig_corr.add_hrect(y0=0.5, y1=1,   fillcolor="rgba(0,128,0,0.06)", line_width=0)
fig_corr.update_layout(height=300, margin=dict(l=20,r=20,t=30,b=20), yaxis=dict(range=[-1,1], title="corr"), xaxis_title="")
st.plotly_chart(fig_corr, use_container_width=True)

# ---------- Heatmap (maand/jaar) ----------
st.subheader("Maand/jaar-heatmap van Δ")
heat_on = st.checkbox("Toon heatmap", value=True,
                      help="Geaggregeerde Δ per maand: punten = som, % = gecomponeerd.")
if heat_on:
    t = d.copy()
    t["date_dt"] = pd.to_datetime(t["date"]); t = t.set_index("date_dt")
    if delta_mode == "Δ %":
        monthly = t["delta_pct"].groupby(pd.Grouper(freq="M")).apply(
            lambda g: (np.prod((g.dropna()/100.0 + 1.0)) - 1.0) * 100.0 if len(g.dropna()) else np.nan
        )
        value_title = "Δ% (maand, compounded)"
    else:
        monthly = t["delta_abs"].groupby(pd.Grouper(freq="M")).sum(min_count=1)
        value_title = "Δ punten (maand, som)"

    hm = pd.DataFrame({"year": monthly.index.year, "month": monthly.index.month, "value": monthly.values}).dropna()
    month_names = {1:"jan",2:"feb",3:"mrt",4:"apr",5:"mei",6:"jun",7:"jul",8:"aug",9:"sep",10:"okt",11:"nov",12:"dec"}
    hm["mname"] = hm["month"].map(month_names)
    pivot = hm.pivot_table(index="year", columns="mname", values="value", aggfunc="first")
    pivot = pivot.reindex(columns=[month_names[m] for m in range(1,13)])

    z = pivot.values
    heat = go.Figure(data=go.Heatmap(z=z, x=pivot.columns, y=pivot.index,
                                     coloraxis="coloraxis",
                                     hovertemplate="Jaar %{y} — %{x}: %{z:.2f}<extra></extra>"))
    heat.update_layout(height=380, margin=dict(l=20, r=20, t=30, b=20),
                       coloraxis=dict(colorscale="RdBu", cauto=True, colorbar_title=value_title),
                       xaxis_title="Maand", yaxis_title="Jaar", title=f"Heatmap — {value_title}")
    st.plotly_chart(heat, use_container_width=True)

# ---------- Histogrammen ----------
st.subheader("Histogram dagrendementen")
bins = st.slider("Aantal bins", 10, 120, 60, 5)
c1, c2 = st.columns(2)
hist_df = d.dropna(subset=["delta_abs","delta_pct"]).copy()
with c1:
    fig_abs = go.Figure([go.Histogram(x=hist_df["delta_abs"], nbinsx=int(bins))])
    fig_abs.update_layout(title="Δ abs (punten)", height=320, bargap=0.02, margin=dict(l=10,r=10,t=40,b=10))
    st.plotly_chart(fig_abs, use_container_width=True)
with c2:
    fig_pct = go.Figure([go.Histogram(x=hist_df["delta_pct"], nbinsx=int(bins))])
    fig_pct.update_layout(title="Δ %", height=320, bargap=0.02, margin=dict(l=10,r=10,t=40,b=10))
    st.plotly_chart(fig_pct, use_container_width=True)
