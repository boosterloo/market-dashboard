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

# ── Bedieningskolom links
left, right = st.columns([1,4])
with left:
    # Delta-panel controls
    delta_mode = st.radio("Δ onder prijs", ["Δ punten", "Δ %"], index=0, horizontal=False)
    agg_mode = st.selectbox("Aggregatie", ["Dagelijks", "Wekelijks", "Maandelijks"], index=0,
                            help="Wekelijks = eind vrijdag; Maandelijks = eind maand.")
    smooth_on = st.checkbox("Gladmaken (MA)", value=False)
    ma_window = st.slider("MA-window", min_value=2, max_value=60, value=5, step=1, disabled=not smooth_on)

    # VIX toggle
    show_vix = st.checkbox("Toon VIX (paneel 1)", value=True)

    # Correlatie-controls
    st.markdown("---")
    st.caption("Rolling correlatie (paneel 6)")
    corr_vs = st.radio("Met VIX-…", ["% change", "level"], index=0, horizontal=True)
    corr_win = st.slider("Window", min_value=5, max_value=90, value=20, step=1)

# Delta-mode tekst
if delta_mode == "Δ %":
    delta_title = "Δ (geaggregeerd, %)"
    delta_legend = "Δ (%)"
else:
    delta_title = "Δ (geaggregeerd, punten)"
    delta_legend = "Δ (punten)"

# ▶︎ Aggregatie helper
def aggregate_delta(_df: pd.DataFrame, mode: str, how: str) -> pd.Series:
    t = _df.copy()
    t["date_dt"] = pd.to_datetime(t["date"])
    t = t.set_index("date_dt")

    if how == "Dagelijks":
        return t["delta_pct"] if mode == "Δ %" else t["delta_abs"]

    rule = "W-FRI" if how == "Wekelijks" else ("M" if how == "Maandelijks" else "D")

    if mode == "Δ %":
        comp = t["delta_pct"].groupby(pd.Grouper(freq=rule)).apply(
            lambda g: (np.prod((g.dropna()/100.0 + 1.0)) - 1.0) * 100.0 if len(g.dropna()) else np.nan
        )
        return comp
    else:
        summ = t["delta_abs"].groupby(pd.Grouper(freq=rule)).sum(min_count=1)
        return summ

# Bereken geaggregeerde delta
delta_series = aggregate_delta(d, delta_mode, agg_mode)

# Smoothing (MA) optioneel
if smooth_on:
    delta_series = delta_series.rolling(ma_window, min_periods=1).mean()
    delta_legend = f"{delta_legend} — MA{ma_window}"

# X-as labels en kleuren
delta_x = delta_series.index if isinstance(delta_series.index, pd.DatetimeIndex) else pd.to_datetime(d["date"])
delta_x = delta_x.date
delta_colors = np.where(delta_series.values >= 0, "rgba(16,150,24,0.7)", "rgba(219,64,82,0.7)")

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

# ── Rolling correlatie (dagelijks, Δ% vs VIX)
corr_df = d.copy()
corr_df["date_dt"] = pd.to_datetime(corr_df["date"])
corr_df = corr_df.set_index("date_dt")
spx_ret = corr_df["delta_pct"]  # al in %

if corr_vs == "% change":
    vix_series = corr_df["vix_close"].pct_change() * 100.0
    corr_label = f"corr(Δ% SPX, Δ% VIX) — window={corr_win}"
else:
    vix_series = corr_df["vix_close"]
    corr_label = f"corr(Δ% SPX, VIX level) — window={corr_win}"

corr_join = pd.concat([spx_ret.rename("spx"), vix_series.rename("vix")], axis=1).dropna()
rolling_corr = corr_join["spx"].rolling(corr_win).corr(corr_join["vix"])

# ---- Chart indeling (6 panelen) ----
# 1) Heikin-Ashi + DC + Supertrend + (optioneel) VIX (sec. y)
# 2) Δ bars (punten of %, met aggregatie en optionele smoothing)
# 3) Close + EMA
# 4) RSI
# 5) CCI
# 6) Rolling correlatie (dagelijks)
fig = make_subplots(
    rows=6, cols=1, shared_xaxes=True,
    specs=[
        [{"secondary_y": True}],
        [{}],
        [{}],
        [{}],
        [{}],
        [{}],
    ],
    subplot_titles=[
        "SP500 Heikin-Ashi + Supertrend (10,1) + Donchian" + (" + VIX (2e y-as)" if show_vix else ""),
        f"{'Δ (%)' if delta_mode=='Δ %' else 'Δ (punten)'} — {agg_mode.lower()}{' — MA'+str(ma_window) if smooth_on else ''}",
        "Close + EMA(20/50/200)",
        "RSI(14)",
        "CCI(20)",
        corr_label
    ],
    row_heights=[0.34, 0.16, 0.24, 0.09, 0.09, 0.08],
    vertical_spacing=0.055
)

# (1) HA + DC + ST (+ VIX optioneel)
fig.add_trace(go.Candlestick(
    x=d["date"], open=d["ha_open"], high=d["ha_high"], low=d["ha_low"], close=d["ha_close"],
    name="SPX (Heikin-Ashi)"),
    row=1, col=1, secondary_y=False
)
fig.add_trace(go.Scatter(x=d["date"], y=d["dc_high"], mode="lines",
                         line=dict(dash="dot", width=2), name="DC High"), row=1, col=1, secondary_y=False)
fig.add_trace(go.Scatter(x=d["date"], y=d["dc_low"], mode="lines",
                         line=dict(dash="dot", width=2), name="DC Low"), row=1, col=1, secondary_y=False)
st_up = d["st_line"].where(d["st_trend"]==1); st_dn = d["st_line"].where(d["st_trend"]==-1)
fig.add_trace(go.Scatter(x=d["date"], y=st_up, mode="lines",
                         line=dict(width=2, color="green"), name="Supertrend ↑ (10,1)"),
              row=1, col=1, secondary_y=False)
fig.add_trace(go.Scatter(x=d["date"], y=st_dn, mode="lines",
                         line=dict(width=2, color="red"),   name="Supertrend ↓ (10,1)"),
              row=1, col=1, secondary_y=False)

if show_vix and ("vix_close" in d.columns and d["vix_close"].notna().any()):
    fig.add_trace(go.Scatter(x=d["date"], y=d["vix_close"], mode="lines",
                             name="VIX (sec. y)"),
                  row=1, col=1, secondary_y=True)

# (2) Δ bars
fig.add_trace(go.Bar(x=delta_x, y=delta_series.values, name=delta_legend,
                     marker=dict(color=delta_colors), opacity=0.9),
              row=2, col=1)

# (3) Close + EMA
fig.add_trace(go.Scatter(x=d["date"], y=d["close"], mode="lines", name="Close"),
              row=3, col=1)
fig.add_trace(go.Scatter(x=d["date"], y=d["ema20"], mode="lines", name="EMA20"),
              row=3, col=1)
fig.add_trace(go.Scatter(x=d["date"], y=d["ema50"], mode="lines", name="EMA50"),
              row=3, col=1)
fig.add_trace(go.Scatter(x=d["date"], y=d["ema200"], mode="lines", name="EMA200"),
              row=3, col=1)

# (4) RSI
fig.add_trace(go.Scatter(x=d["date"], y=d["rsi14"], mode="lines", name="RSI(14)"), row=4,col=1)
fig.add_hline(y=70, line_dash="dot", row=4, col=1); fig.add_hline(y=30, line_dash="dot", row=4, col=1)

# (5) CCI
fig.add_trace(go.Scatter(x=d["date"], y=d["cci20"], mode="lines", name="CCI(20)"), row=5,col=1)
fig.add_hline(y=100, line_dash="dot", row=5, col=1); fig.add_hline(y=-100, line_dash="dot", row=5, col=1)

# (6) Rolling correlatie
fig.add_trace(go.Scatter(x=rolling_corr.index.date, y=rolling_corr.values, mode="lines",
                         name="Rolling corr"),
              row=6, col=1)
fig.add_hline(y=0.0, line_dash="dot", row=6, col=1)

# Layout & assen
fig.update_layout(
    height=1650, margin=dict(l=20,r=20,t=60,b=20),
    legend_orientation="h", legend_yanchor="top", legend_y=1.06, legend_x=0
)
fig.update_layout(xaxis_rangeslider_visible=False); fig.update_xaxes(rangeslider_visible=False)

fig.update_yaxes(title_text="Index (HA)", row=1,col=1, secondary_y=False)
fig.update_yaxes(title_text="VIX", row=1,col=1, secondary_y=True)
fig.update_yaxes(title_text=f"{'Δ (%)' if delta_mode=='Δ %' else 'Δ (punten)'}", row=2,col=1)
fig.update_yaxes(title_text="Close/EMA", row=3,col=1)
fig.update_yaxes(title_text="RSI", row=4,col=1)
fig.update_yaxes(title_text="CCI", row=5,col=1)
fig.update_yaxes(title_text="corr", row=6,col=1, range=[-1,1])

st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────── Heatmap (maand/jaar) ───────────────────────────
st.subheader("Maand/jaar-heatmap van Δ")
heat_on = st.checkbox("Toon heatmap", value=True,
                      help="Gebruik geaggregeerde Δ per maand: punten = som, % = compounded.")
if heat_on:
    t = d.copy()
    t["date_dt"] = pd.to_datetime(t["date"])
    t = t.set_index("date_dt")
    if delta_mode == "Δ %":
        monthly = t["delta_pct"].groupby(pd.Grouper(freq="M")).apply(
            lambda g: (np.prod((g.dropna()/100.0 + 1.0)) - 1.0) * 100.0 if len(g.dropna()) else np.nan
        )
        value_title = "Δ% (maand, compounded)"
    else:
        monthly = t["delta_abs"].groupby(pd.Grouper(freq="M")).sum(min_count=1)
        value_title = "Δ punten (maand, som)"

    hm = pd.DataFrame({
        "year": monthly.index.year,
        "month": monthly.index.month,
        "value": monthly.values
    }).dropna(subset=["value"])

    month_names = {1:"jan",2:"feb",3:"mrt",4:"apr",5:"mei",6:"jun",7:"jul",8:"aug",9:"sep",10:"okt",11:"nov",12:"dec"}
    hm["mname"] = hm["month"].map(month_names)
    pivot = hm.pivot_table(index="year", columns="mname", values="value", aggfunc="first")
    pivot = pivot.reindex(columns=[month_names[m] for m in range(1,13)])

    z = pivot.values
    heat = go.Figure(data=go.Heatmap(
        z=z, x=pivot.columns, y=pivot.index,
        coloraxis="coloraxis",
        hovertemplate="Jaar %{y} — %{x}: %{z:.2f}<extra></extra>"
    ))
    heat.update_layout(
        height=380, margin=dict(l=20, r=20, t=30, b=20),
        coloraxis=dict(colorscale="RdBu", cmin=np.nanmin(z), cmax=np.nanmax(z), cauto=True, colorbar_title=value_title),
        xaxis_title="Maand", yaxis_title="Jaar", title=f"Heatmap — {value_title}"
    )
    st.plotly_chart(heat, use_container_width=True)

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
    fig_pct = go.Figure([go.Histogram(x=hist_df["delta_pct"], nbinsx=int(bins))])  # delta_pct is al in %
    fig_pct.update_layout(title="Δ %", height=320, bargap=0.02, margin=dict(l=10,r=10,t=40,b=10))
    st.plotly_chart(fig_pct, use_container_width=True)
