import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.bq import run_query, bq_ping

# -------------------- Setup --------------------
st.set_page_config(page_title="📈 S&P 500 Strategy", layout="wide")
st.title("📈 S&P 500 Strategy (Long + Short, ATR stops, TP-partials)")

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

df["date"] = pd.to_datetime(df["date"])
df = df[df["date"].dt.weekday < 5]  # skip weekends
for c in ["open","high","low","close","vix_close","delta_abs","delta_pct"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# ---------- Constants ----------
DEFAULTS = {
    "ema_spans": [20, 50, 200],
    "macd": (12, 26, 9),
    "rsi_period": 14,
    "adx_length": 14,
    "adx_threshold": 20,
}

# ---- Helpers ----
def ema(s: pd.Series, span: int):
    return s.ewm(span=span, adjust=False, min_periods=span).mean()

def macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = series.ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

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
    return plus_di, minus_di, adx, atr

def rsi(series: pd.Series, period: int = 14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def crossed_up(s, level=0):   return (s.shift(1) <= level) & (s > level)
def crossed_down(s, level=0): return (s.shift(1) >= level) & (s < level)

def max_drawdown(series):
    roll_max = series.cummax()
    dd = series/roll_max - 1.0
    return dd.min(), dd

def sharpe_ratio(ret_series, periods_per_year=252):
    mu = ret_series.mean()
    sd = ret_series.std(ddof=0)
    if sd == 0 or np.isnan(sd): return np.nan
    return (mu * periods_per_year) / (sd * np.sqrt(periods_per_year))

# ---- Indicators ----
@st.cache_data(ttl=1800)
def compute_indicators(full_df):
    for span in DEFAULTS["ema_spans"]:
        full_df[f"ema{span}"] = ema(full_df["close"], span)
    full_df["macd_line"], full_df["macd_signal"], full_df["macd_hist"] = macd(full_df["close"], *DEFAULTS["macd"])
    full_df["di_plus"], full_df["di_minus"], full_df["adx14"], full_df["atr14"] = adx(full_df, DEFAULTS["adx_length"])
    full_df["rsi14"] = rsi(full_df["close"], DEFAULTS["rsi_period"])
    return full_df

df = compute_indicators(df)

# ---- Periode ----
min_d = df["date"].min().date()
max_d = df["date"].max().date()
default_start = max((df["date"].max() - timedelta(days=365)).date(), min_d)
start_date, end_date = st.slider("Periode", min_value=min_d, max_value=max_d,
                                 value=(default_start, max_d), format="YYYY-MM-DD")

d = df[(df["date"].dt.date >= start_date) & (df["date"].dt.date <= end_date)].reset_index(drop=True).copy()

# ---------- Sidebar ----------
st.sidebar.markdown("### Risk/Backtest Settings")
ATR_MULT   = st.sidebar.slider("ATR-stop multiplier", 1.0, 5.0, 2.0, 0.5)
START_CAP  = st.sidebar.number_input("Startkapitaal (€)", 1000.0, 1_000_000.0, 10_000.0, step=500.0)
FEE_BPS    = st.sidebar.slider("Kosten (bps per trade)", 0.0, 10.0, 1.0, 0.5)
SLIP_BPS   = st.sidebar.slider("Slippage (bps per trade)", 0.0, 10.0, 1.0, 0.5)
RISK_PCT   = st.sidebar.slider("Position sizing: % equity risico", 0.0, 5.0, 1.0, 0.5)
ALLOW_SHORT = st.sidebar.checkbox("Shorts toestaan", value=True)

# ---------- Signal Logic ----------
up_regime   = (d["close"] > d["ema200"]) & (d["ema50"] > d["ema200"])
down_regime = (d["close"] < d["ema200"]) & (d["ema50"] < d["ema200"])
strong_trend = d["adx14"] > DEFAULTS["adx_threshold"]

d["long_entry"]  = up_regime & strong_trend & crossed_up(d["macd_hist"]) & (d["close"] > d["ema20"]) & (d["rsi14"] > 45)
d["long_exit"]   = crossed_down(d["macd_hist"]) | (d["close"] < d["ema20"])

d["short_entry"] = down_regime & strong_trend & crossed_down(d["macd_hist"]) & (d["close"] < d["ema20"]) & (d["rsi14"] < 55)
d["short_exit"]  = crossed_up(d["macd_hist"]) | (d["close"] > d["ema20"])

# ---------- Backtest Engine ----------
def run_backtest(df, start_capital, atr_mult, fee_bps, slip_bps, risk_pct, allow_short):
    df = df.copy().reset_index(drop=True)
    cash = start_capital
    pos = 0   # 0=flat, 1=long, -1=short
    shares = 0.0
    entry_px, entry_idx = None, None
    stop = np.nan
    equity_curve = []
    trades = []

    fee = fee_bps/1e4
    slip = slip_bps/1e4

    for i in range(len(df)):
        close = df.loc[i,"close"]
        equity_curve.append(cash + shares*close*pos if pos!=0 else cash)

        if i == len(df)-1: break
        nxt_open = df.loc[i+1,"open"]
        atr_next = df.loc[i+1,"atr14"]

        # Trailing stop update
        if pos == 1:
            stop = max(stop, close - atr_mult*atr_next)
        elif pos == -1:
            stop = min(stop, close + atr_mult*atr_next)

        # Exit
        exit_flag = False
        reason = None
        if pos == 1:
            if df.loc[i,"long_exit"]: exit_flag, reason = True,"signal"
            elif nxt_open < stop:     exit_flag, reason = True,"stop"
        elif pos == -1:
            if df.loc[i,"short_exit"]: exit_flag, reason = True,"signal"
            elif nxt_open > stop:     exit_flag, reason = True,"stop"

        if exit_flag:
            exec_px = nxt_open*(1-slip) if pos==1 else nxt_open*(1+slip)
            cash += shares*exec_px*pos
            cash -= abs(shares*exec_px)*fee
            trades.append({
                "side":"LONG" if pos==1 else "SHORT",
                "entry_date":df.loc[entry_idx,"date"],"entry_px":entry_px,
                "exit_date":df.loc[i+1,"date"],"exit_px":exec_px,
                "ret_pct": (exec_px/entry_px-1)*100 if pos==1 else (1-exec_px/entry_px)*100,
                "reason":reason
            })
            pos,shares,stop = 0,0.0,np.nan

        # Entry
        if pos==0:
            if df.loc[i,"long_entry"]:
                buy_px = nxt_open*(1+slip)
                stop = buy_px - atr_mult*atr_next
                risk_amt = cash*(risk_pct/100) if risk_pct>0 else cash
                per_share_risk = max(buy_px-stop,1e-9)
                shares = np.floor(risk_amt/per_share_risk) if risk_pct>0 else cash/buy_px
                cost = shares*buy_px
                cash -= cost+cost*fee
                entry_px,entry_idx,pos = buy_px,i+1,1
            elif allow_short and df.loc[i,"short_entry"]:
                sell_px = nxt_open*(1-slip)
                stop = sell_px + atr_mult*atr_next
                risk_amt = cash*(risk_pct/100) if risk_pct>0 else cash
                per_share_risk = max(stop-sell_px,1e-9)
                shares = np.floor(risk_amt/per_share_risk) if risk_pct>0 else cash/sell_px
                cash += shares*sell_px - abs(shares*sell_px)*fee
                entry_px,entry_idx,pos = sell_px,i+1,-1

    return pd.Series(equity_curve,index=df["date"],name="equity"), pd.DataFrame(trades)

equity, trades_df = run_backtest(d, START_CAP, ATR_MULT, FEE_BPS, SLIP_BPS, RISK_PCT, ALLOW_SHORT)

# Buy & Hold
bh_equity = (START_CAP/ d["open"].iloc[1])*d["close"] if len(d)>1 else pd.Series([START_CAP],index=d["date"])

# Curves
d["strategy_cum_pct"] = (equity/START_CAP - 1)*100
d["buyhold_cum_pct"]  = (bh_equity/START_CAP - 1)*100

# ---------- Metrics ----------
mdd_val,_=max_drawdown(equity)
bh_mdd,_=max_drawdown(bh_equity)
stats = {
    "Trades":len(trades_df),
    "Winrate":(trades_df["ret_pct"]>0).mean()*100 if len(trades_df) else 0,
    "Avg win":trades_df.loc[trades_df["ret_pct"]>0,"ret_pct"].mean(),
    "Avg loss":trades_df.loc[trades_df["ret_pct"]<=0,"ret_pct"].mean(),
    "Sharpe strat":sharpe_ratio(equity.pct_change().fillna(0)),
    "Sharpe B&H":sharpe_ratio(bh_equity.pct_change().fillna(0)),
    "MaxDD strat":mdd_val*100,"MaxDD B&H":bh_mdd*100
}

# ---------- Outputs ----------
st.subheader("Performance curves")
fig = go.Figure()
fig.add_trace(go.Scatter(x=d["date"], y=d["buyhold_cum_pct"], name="Buy & Hold"))
fig.add_trace(go.Scatter(x=d["date"], y=d["strategy_cum_pct"], name="Strategy (L+S)"))
st.plotly_chart(fig,use_container_width=True)

st.subheader("Trades")
if len(trades_df):
    st.dataframe(trades_df,use_container_width=True)
else:
    st.info("Geen trades in periode")

st.subheader("Metrics")
st.json(stats)
