# pages/1_SP500.py
# 📈 S&P 500 — Signals & Backtest (ATR/TP + EMA toggle)
# Complete versie met RSI (Wilder/RMA), dynamische zones en agressiviteit-presets.

import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta, datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------- BigQuery helpers ----------
try:
    from utils.bq import run_query, bq_ping
except Exception:
    from google.cloud import bigquery
    from google.oauth2 import service_account

    @st.cache_resource(show_spinner=False)
    def _bq_client_from_secrets():
        creds = st.secrets["gcp_service_account"]
        credentials = service_account.Credentials.from_service_account_info(creds)
        return bigquery.Client(credentials=credentials, project=creds["project_id"])

    def run_query(sql: str) -> pd.DataFrame:
        client = _bq_client_from_secrets()
        return client.query(sql).to_dataframe()

    def bq_ping() -> bool:
        try:
            _ = run_query("SELECT 1 AS ok")
            return True
        except Exception:
            return False

# =========================
# App setup
# =========================
st.set_page_config(page_title="📈 S&P 500 — Signals & Backtest", layout="wide")
st.title("📈 S&P 500 — Signals & Backtest (ATR/TP + EMA toggle)")

SPX_VIEW = st.secrets.get("tables", {}).get(
    "spx_view", "nth-pier-468314-p7.marketdata.spx_with_vix_v"
)

if not bq_ping():
    st.error("Geen BigQuery-verbinding."); st.stop()

# =========================
# Data
# =========================
@st.cache_data(ttl=1800)
def load_spx():
    return run_query(f"SELECT * FROM `{SPX_VIEW}` ORDER BY date")

df = load_spx()
if df.empty:
    st.warning("Geen data in view."); st.stop()
df["date"] = pd.to_datetime(df["date"])
df = df[df["date"].dt.weekday < 5]
for c in ["open","high","low","close","vix_close"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# =========================
# Constants & helpers
# =========================
DEFAULTS = {
    "ema_spans": [20, 50, 200],
    "rsi_period": 14,
    "adx_length": 14,
    "adx_threshold": 20,
    "rsi_ob": 70, "rsi_os": 30,
    "rsi_dyn_win": 252
}

def ema(s, span): return s.ewm(span=span, adjust=False).mean()
def rma(x, n): return x.ewm(alpha=1/n, adjust=False).mean()

def rsi_wilder(close, length=14):
    d = close.diff()
    up = d.clip(lower=0)
    down = -d.clip(upper=0)
    rs = rma(up, length) / rma(down, length)
    return 100 - 100/(1+rs)

def rolling_percentile(s, q, win=252):
    return s.rolling(win, min_periods=int(win*0.6)).quantile(q)

def crossed_up(s, level=0):   return (s.shift(1) <= level) & (s > level)
def crossed_down(s, level=0): return (s.shift(1) >= level) & (s < level)

# --- ADX ---
def adx(df_, n=14):
    h,l,c = df_["high"], df_["low"], df_["close"]
    up, dn = h.diff(), -l.diff()
    plus_dm  = np.where((up>dn)&(up>0), up, 0.0)
    minus_dm = np.where((dn>up)&(dn>0), dn, 0.0)
    tr = pd.concat([(h-l).abs(), (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/n, adjust=False).mean()
    plus_di  = 100 * pd.Series(plus_dm).ewm(alpha=1/n, adjust=False).mean()/atr
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/n, adjust=False).mean()/atr
    dx = 100*(plus_di-minus_di).abs()/(plus_di+minus_di)
    return plus_di, minus_di, dx.ewm(alpha=1/n, adjust=False).mean()

# =========================
# Indicators
# =========================
@st.cache_data(ttl=1800)
def compute_indicators(df):
    df = df.copy()
    for s in DEFAULTS["ema_spans"]:
        df[f"ema{s}"] = ema(df["close"], s)
    df["rsi14"] = rsi_wilder(df["close"], DEFAULTS["rsi_period"])
    df["rsi14_s"] = df["rsi14"].ewm(span=5, adjust=False).mean()
    df["rsi_dyn_hi"] = rolling_percentile(df["rsi14"], 0.80)
    df["rsi_dyn_lo"] = rolling_percentile(df["rsi14"], 0.20)
    _,_,df["adx14"] = adx(df, DEFAULTS["adx_length"])
    return df

df = compute_indicators(df)

# =========================
# Periode
# =========================
min_d, max_d = df["date"].min().date(), df["date"].max().date()
default_start = max_d - timedelta(days=365)
start_date, end_date = st.slider("Periode", min_value=min_d, max_value=max_d,
                                 value=(default_start, max_d), format="YYYY-MM-DD")
d = df[(df["date"].dt.date>=start_date)&(df["date"].dt.date<=end_date)].reset_index(drop=True)

# =========================
# Sidebar presets
# =========================
with st.sidebar:
    st.header("⚙️ Instellingen")
    SIG_MODE = st.radio("Signaalset", ["Advanced","EMA crossover"], index=0)
    SIG_PRESET = st.radio("🎯 Signaal-intensiteit",
                          ["Conservatief","Gebalanceerd","Aggressief"], index=1)

def signal_params(preset, base=DEFAULTS["adx_threshold"]):
    if preset=="Conservatief":
        return base+5, True, True, 5
    if preset=="Aggressief":
        return base-5, False, False, 3
    return base, False, True, 5

# =========================
# Signalen
# =========================
if SIG_MODE=="Advanced":
    ADX_TH, REQ_MACD, REQ_PRICE, RSI_S_SPAN = signal_params(SIG_PRESET)
    up = (d["close"]>d["ema200"])&(d["ema50"]>d["ema200"])
    dn = (d["close"]<d["ema200"])&(d["ema50"]<d["ema200"])
    strong = d["adx14"]>ADX_TH

    rsi_s = d["rsi14"].ewm(span=RSI_S_SPAN, adjust=False).mean()
    rsi_up, rsi_dn = crossed_up(rsi_s,50), crossed_down(rsi_s,50)
    ex_long, ex_short = rsi_s>=d["rsi_dyn_hi"], rsi_s<=d["rsi_dyn_lo"]

    price_long = (d["close"]>d["ema20"]) if REQ_PRICE else True
    price_short= (d["close"]<d["ema20"]) if REQ_PRICE else True

    d["buy_sig"]   = up & strong & rsi_up & price_long
    d["short_sig"] = dn & strong & rsi_dn & price_short
    d["sell_sig"]  = (d["buy_sig"].shift(1)&(rsi_dn|ex_long)) | \
                     (d["short_sig"].shift(1)&(rsi_up|ex_short))
else:
    f,s = 20,50
    d["ema_f"], d["ema_s"] = ema(d["close"],f), ema(d["close"],s)
    cross_up=(d["ema_f"].shift(1)<=d["ema_s"].shift(1))&(d["ema_f"]>d["ema_s"])
    cross_dn=(d["ema_f"].shift(1)>=d["ema_s"].shift(1))&(d["ema_f"]<d["ema_s"])
    d["buy_sig"], d["sell_sig"], d["short_sig"]=cross_up,cross_dn,cross_dn

for c in ["buy_sig","sell_sig","short_sig"]:
    d[c]=d[c].fillna(False)

st.caption(f"Preset **{SIG_PRESET}** · ADX≥{ADX_TH} · "
           f"{'Prijsfilter aan' if REQ_PRICE else 'Geen prijsfilter'}")

# =========================
# Alerts
# =========================
alerts=[]
last=d.iloc[-1]
if last["adx14"]>ADX_TH: alerts.append(("green","Trend sterk",f"ADX {last['adx14']:.1f}"))
elif last["adx14"]<=ADX_TH: alerts.append(("gray","Trend zwak",f"ADX {last['adx14']:.1f}"))

if crossed_up(d["rsi14_s"],50).iloc[-1]:
    alerts.append(("green","RSI ↑50","Momentum positief"))
elif crossed_down(d["rsi14_s"],50).iloc[-1]:
    alerts.append(("red","RSI ↓50","Momentum negatief"))

def badge(c,t):
    m={"green":"#00A65A","red":"#D55E00","gray":"#6c757d"}
    return f"<span style='background:{m[c]};color:white;padding:2px 8px;border-radius:10px'>{t}</span>"

st.subheader("⚠️ Alerts")
if not alerts: st.success("Geen alerts.")
else:
    for c,t,m in alerts:
        st.markdown(f"{badge(c,t)}  {m}", unsafe_allow_html=True)

# =========================
# Plot RSI + signals
# =========================
fig = make_subplots(rows=2,cols=1,shared_xaxes=True,
                    subplot_titles=["Close + EMA(20/50/200)","RSI Wilder (dynamisch)"],
                    row_heights=[0.6,0.4])
fig.add_trace(go.Scatter(x=d["date"],y=d["close"],mode="lines",name="Close"),row=1,col=1)
for s in [20,50,200]:
    fig.add_trace(go.Scatter(x=d["date"],y=d[f"ema{s}"],mode="lines",name=f"EMA{s}"),row=1,col=1)

b=d.loc[d["buy_sig"]]; s=d.loc[d["sell_sig"]]
fig.add_trace(go.Scatter(x=b["date"],y=b["close"],mode="markers",name="Buy",
                         marker=dict(symbol="triangle-up",size=10,color="green")),row=1,col=1)
fig.add_trace(go.Scatter(x=s["date"],y=s["close"],mode="markers",name="Sell",
                         marker=dict(symbol="triangle-down",size=10,color="red")),row=1,col=1)

fig.add_trace(go.Scatter(x=d["date"],y=d["rsi14_s"],mode="lines",name="RSI smoothed"),row=2,col=1)
fig.add_trace(go.Scatter(x=d["date"],y=d["rsi_dyn_hi"],mode="lines",name="dyn-high",line=dict(dash="dot")),row=2,col=1)
fig.add_trace(go.Scatter(x=d["date"],y=d["rsi_dyn_lo"],mode="lines",name="dyn-low",line=dict(dash="dot")),row=2,col=1)
fig.add_hline(y=50,row=2,col=1,line_dash="dot")
fig.update_yaxes(range=[0,100],row=2,col=1)
fig.update_layout(height=700,legend_orientation="h")
st.plotly_chart(fig,use_container_width=True)
