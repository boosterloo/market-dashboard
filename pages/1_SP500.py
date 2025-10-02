import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.bq import run_query, bq_ping

# =========================================
# App setup
# =========================================
st.set_page_config(page_title="📈 S&P 500 Strategy (L/S + ATR + TP)", layout="wide")
st.title("📈 S&P 500 – Long & Short met ATR-stop, Trailing & TP1/TP2")

SPX_VIEW = st.secrets.get("tables", {}).get(
    "spx_view", "nth-pier-468314-p7.marketdata.spx_with_vix_v"
)

# ---- Health ----
try:
    if not bq_ping():
        st.error("Geen BigQuery-verbinding."); st.stop()
except Exception as e:
    st.error("Geen BigQuery-verbinding."); st.caption(f"Details: {e}"); st.stop()

# =========================================
# Data
# =========================================
@st.cache_data(ttl=1800, show_spinner=False)
def load_spx():
    return run_query(f"SELECT * FROM `{SPX_VIEW}` ORDER BY date")

with st.spinner("SPX data laden…"):
    df = load_spx()
if df.empty:
    st.warning("Geen data in view."); st.stop()

df["date"] = pd.to_datetime(df["date"])
df = df[df["date"].dt.weekday < 5]  # alleen handelsdagen
for c in ["open","high","low","close","vix_close"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# Zorg voor delta_pct voor KPI’s
if "delta_pct" not in df.columns or df["delta_pct"].isna().all():
    df["delta_pct"] = df["close"].pct_change()*100.0

# =========================================
# Indicator helpers
# =========================================
DEFAULTS = {
    "ema_spans": [20, 50, 200],
    "macd": (12, 26, 9),
    "rsi_period": 14,
    "adx_length": 14,
    "adx_threshold": 20,
    "vix_high": 25,
    "vix_low": 15
}

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

def ytd_return_full(full_df: pd.DataFrame):
    max_d = full_df["date"].max(); start = pd.Timestamp(date(max_d.year, 1, 1))
    sub = full_df[full_df["date"] >= start].dropna(subset=["close"])
    return (sub["close"].iloc[-1]/sub["close"].iloc[0]-1)*100 if len(sub)>=2 else None

def pytd_return_full(full_df: pd.DataFrame):
    max_d = full_df["date"].max(); prev_year = max_d.year-1
    start = pd.Timestamp(date(prev_year,1,1))
    try:
        end = max_d.replace(year=prev_year)
    except ValueError:
        end = pd.Timestamp(date(prev_year,12,31))
    sub = full_df[(full_df["date"]>=start)&(full_df["date"]<=end)].dropna(subset=["close"])
    return (sub["close"].iloc[-1]/sub["close"].iloc[0]-1)*100 if len(sub)>=2 else None

# =========================================
# Indicator calc
# =========================================
@st.cache_data(ttl=1800)
def compute_indicators(full_df):
    for span in DEFAULTS["ema_spans"]:
        full_df[f"ema{span}"] = ema(full_df["close"], span)
    full_df["macd_line"], full_df["macd_signal"], full_df["macd_hist"] = macd(full_df["close"], *DEFAULTS["macd"])
    full_df["di_plus"], full_df["di_minus"], full_df["adx14"], full_df["atr14"] = adx(full_df, DEFAULTS["adx_length"])
    full_df["rsi14"] = rsi(full_df["close"], DEFAULTS["rsi_period"])
    return full_df

df = compute_indicators(df)

# =========================================
# UI – periode & sidebar
# =========================================
min_d = df["date"].min().date()
max_d = df["date"].max().date()
default_start = max((df["date"].max() - timedelta(days=365)).date(), min_d)
c1, c2, c3 = st.columns([0.2, 0.6, 0.2])
with c2:
    start_date, end_date = st.slider("Periode", min_value=min_d, max_value=max_d,
                                     value=(default_start, max_d), format="YYYY-MM-DD")

d = df[(df["date"].dt.date >= start_date) & (df["date"].dt.date <= end_date)].reset_index(drop=True).copy()

with st.sidebar:
    st.markdown("### ⚙️ Backtest & Risk")
    ATR_MULT   = st.slider("ATR-stop multiplier", 1.0, 5.0, 2.0, 0.5)
    START_CAP  = st.number_input("Startkapitaal (€)", 1000.0, 2_000_000.0, 10_000.0, step=500.0)
    FEE_BPS    = st.slider("Kosten (bps per trade)", 0.0, 20.0, 1.0, 0.5)
    SLIP_BPS   = st.slider("Slippage (bps per trade)", 0.0, 20.0, 1.0, 0.5)
    RISK_PCT   = st.slider("% equity risico per trade", 0.0, 5.0, 1.0, 0.5,
                           help="0% = all-in; >0% → sizing = (equity × %risk) / (entry − stop).")
    st.divider()
    st.markdown("### 🎯 Take Profit")
    TP1_R   = st.slider("TP1 (R-multiple)", 1.25, 3.0, 1.75, 0.25)
    TP2_R   = st.slider("TP2 (R-multiple)", 2.0, 5.0, 3.0, 0.25)
    TP1_PART = st.slider("TP1: % positie sluiten", 10, 80, 50, 5)
    MOVE_BE  = st.checkbox("Na TP1 stop → Break-Even", value=True)
    st.divider()
    st.markdown("### 🔎 Filters")
    ADX_TH     = st.slider("ADX drempel", 10, 40, DEFAULTS["adx_threshold"], 1)
    ALLOW_SHORT = st.checkbox("Shorts toestaan", value=True)
    GATE_BY_VIX = st.checkbox("Shorts alleen bij VIX > drempel", value=False)
    VIX_TH  = st.slider("VIX drempel (shorts)", 10, 40, 20, 1, disabled=not GATE_BY_VIX)

# =========================================
# Signalen (regime-aware)
# =========================================
up_regime   = (d["close"] > d["ema200"]) & (d["ema50"] > d["ema200"])
down_regime = (d["close"] < d["ema200"]) & (d["ema50"] < d["ema200"])
strong_trend = d["adx14"] > ADX_TH

d["long_entry"]  = up_regime & strong_trend & crossed_up(d["macd_hist"]) & (d["close"] > d["ema20"]) & (d["rsi14"] > 45)
d["long_exit"]   = crossed_down(d["macd_hist"]) | (d["close"] < d["ema20"])

if GATE_BY_VIX and "vix_close" in d.columns and d["vix_close"].notna().any():
    short_gate = d["vix_close"] > VIX_TH
else:
    short_gate = True

d["short_entry"] = down_regime & strong_trend & crossed_down(d["macd_hist"]) & (d["close"] < d["ema20"]) & (d["rsi14"] < 55) & short_gate
d["short_exit"]  = crossed_up(d["macd_hist"]) | (d["close"] > d["ema20"])

# =========================================
# Backtest engine – L/S, ATR-stop, Trailing, TP1/TP2 partials
# =========================================
def run_backtest(df, start_capital, atr_mult, fee_bps, slip_bps, risk_pct, allow_short, tp1_r, tp2_r, tp1_part, move_be):
    """
    Long & Short backtest:
      - Entry op EOD-signaal, uitvoering next open (slippage/kosten).
      - Initieel stop: long = entry - ATR*mult; short = entry + ATR*mult.
      - Trailing stop: long = max(stop, close - ATR*mult); short = min(stop, close + ATR*mult).
      - TP1: bij tp1_r * R → sluit tp1_part% van positie; optioneel stop → break-even.
      - TP2: bij tp2_r * R → sluit resterende positie.
      - Exit ook bij tegengesteld signaal of gap door stop (next open).
      - Sizing: %risico van equity / (entry − stop); 0% = all-in.
    Retourneert: total/long/short equity curves + trades (combined-run).
    """
    df = df.copy().reset_index(drop=True)

    fee = fee_bps / 1e4
    slip = slip_bps / 1e4

    cash = start_capital
    pos  = 0             # 0=flat, 1=long, -1=short
    shares = 0.0
    entry_px = None
    entry_idx = None
    stop = np.nan
    took_tp1 = False

    eq_total, eq_long, eq_short = [], [], []
    trades = []

    for i in range(len(df)):
        close = float(df.loc[i, "close"])

        # Mark-to-market voor drie curves
        if pos == 1:
            eq_val = cash + shares * close
            eq_total.append(eq_val); eq_long.append(eq_val); eq_short.append(cash)
        elif pos == -1:
            pnl = shares * (entry_px - close)  # short-PnL benadering
            eq_val = cash + pnl
            eq_total.append(eq_val); eq_long.append(cash); eq_short.append(eq_val)
        else:
            eq_total.append(cash); eq_long.append(cash); eq_short.append(cash)

        if i == len(df) - 1:
            break  # geen next open

        nxt_open = float(df.loc[i+1, "open"])
        atr_next = float(df.loc[i+1, "atr14"]) if pd.notna(df.loc[i+1,"atr14"]) else float(df.loc[i,"atr14"])

        # Trailing stop
        if pos == 1:
            stop = max(stop, close - atr_mult * atr_next)
        elif pos == -1:
            stop = min(stop, close + atr_mult * atr_next)

        # Huidige R (voor TP-bepaling)
        if pos == 1 and not np.isnan(stop):
            R_now = (close - entry_px) / max(entry_px - stop, 1e-9)
        elif pos == -1 and not np.isnan(stop):
            R_now = (entry_px - close) / max(stop - entry_px, 1e-9)
        else:
            R_now = 0.0

        # Helper: exit uitvoeren
        def do_full_exit(exec_px, reason):
            nonlocal cash, shares, pos, stop, entry_px, entry_idx, took_tp1
            if pos == 1:
                cash += shares * exec_px
                cash -= abs(shares * exec_px) * fee
                ret_pct = (exec_px / entry_px - 1.0) * 100.0
            else:  # short
                cash += shares * (entry_px - exec_px)
                cash -= abs(shares * exec_px) * fee
                ret_pct = (1.0 - exec_px / entry_px) * 100.0
            trades.append({
                "side": "LONG" if pos == 1 else "SHORT",
                "entry_date": df.loc[entry_idx, "date"],
                "entry_px": entry_px,
                "exit_date": df.loc[i+1, "date"],
                "exit_px": exec_px,
                "ret_pct": ret_pct,
                "reason": reason
            })
            shares = 0.0; pos = 0; stop = np.nan; entry_px = None; entry_idx = None; took_tp1 = False

        # ====== TP2 eerst (sterker signaal) ======
        if pos != 0 and R_now >= tp2_r:
            exec_px = nxt_open * (1 - slip) if pos == 1 else nxt_open * (1 + slip)
            do_full_exit(exec_px, "TP2")
            continue

        # ====== TP1 (partial) ======
        if pos != 0 and (not took_tp1) and R_now >= tp1_r:
            exec_px = nxt_open * (1 - slip) if pos == 1 else nxt_open * (1 + slip)
            part = tp1_part / 100.0
            close_shares = np.floor(shares * part)
            if close_shares >= 1:
                if pos == 1:
                    cash += close_shares * exec_px
                    cash -= abs(close_shares * exec_px) * fee
                else:
                    cash += close_shares * (entry_px - exec_px)
                    cash -= abs(close_shares * exec_px) * fee
                shares -= close_shares
            took_tp1 = True
            if move_be:
                stop = max(stop, entry_px) if pos == 1 else min(stop, entry_px)

        # ====== Exit op signaal of stop-gap ======
        exit_signal = (pos == 1 and bool(d.loc[i, "long_exit"])) or (pos == -1 and bool(d.loc[i, "short_exit"]))
        stop_gap = (pos == 1 and not np.isnan(stop) and nxt_open < stop) or (pos == -1 and not np.isnan(stop) and nxt_open > stop)

        if pos != 0 and (exit_signal or stop_gap):
            exec_px = nxt_open * (1 - slip) if pos == 1 else nxt_open * (1 + slip)
            do_full_exit(exec_px, "signal" if exit_signal and not stop_gap else "stop")
            continue

        # ====== Entry ======
        if pos == 0:
            # Long
            if bool(d.loc[i, "long_entry"]):
                buy_px = nxt_open * (1 + slip)
                init_stop = buy_px - atr_mult * atr_next
                if risk_pct > 0:
                    risk_amount = cash * (risk_pct/100.0)
                    per_share_risk = max(buy_px - init_stop, 1e-9)
                    shares = np.floor(risk_amount / per_share_risk)
                    shares = max(shares, 0)
                    cost = shares * buy_px
                    if cost > cash:
                        shares = np.floor(cash / buy_px)
                        cost = shares * buy_px
                else:
                    cost = cash
                    shares = cost / buy_px
                cash -= cost
                cash -= cost * fee
                pos = 1; entry_px = buy_px; entry_idx = i+1; stop = init_stop; took_tp1 = False

            # Short
            elif allow_short and bool(d.loc[i, "short_entry"]):
                sell_px = nxt_open * (1 - slip)
                init_stop = sell_px + atr_mult * atr_next
                if risk_pct > 0:
                    risk_amount = cash * (risk_pct/100.0)
                    per_share_risk = max(init_stop - sell_px, 1e-9)
                    shares = np.floor(risk_amount / per_share_risk)
                    shares = max(shares, 0)
                else:
                    shares = cash / sell_px
                cash += shares * sell_px
                cash -= abs(shares * sell_px) * fee
                pos = -1; entry_px = sell_px; entry_idx = i+1; stop = init_stop; took_tp1 = False

    # Series
    eq_total = pd.Series(eq_total, index=df["date"], name="equity_total")
    eq_long  = pd.Series(eq_long,  index=df["date"], name="equity_long")
    eq_short = pd.Series(eq_short, index=df["date"], name="equity_short")
    trades_df = pd.DataFrame(trades)
    return eq_total, eq_long, eq_short, trades_df

eq_total, eq_long, eq_short, trades_df = run_backtest(
    d,
    start_capital=START_CAP,
    atr_mult=ATR_MULT,
    fee_bps=FEE_BPS,
    slip_bps=SLIP_BPS,
    risk_pct=RISK_PCT,
    allow_short=ALLOW_SHORT,
    tp1_r=TP1_R,
    tp2_r=TP2_R,
    tp1_part=TP1_PART,
    move_be=MOVE_BE
)

# Buy & Hold (zelfde start, koop dag 1 open)
def buyhold_equity(df, start_capital):
    df = df.copy().reset_index(drop=True)
    if len(df) < 2:
        return pd.Series([start_capital], index=df["date"])
    buy_px = float(df.loc[1, "open"])
    shares = start_capital / buy_px
    return (shares * df["close"]).rename("bh_equity")

bh_equity = buyhold_equity(d, START_CAP)

# =========================================
# Metrics per curve
# =========================================
def max_drawdown(series):
    roll_max = series.cummax()
    dd = series/roll_max - 1.0
    return dd.min(), dd

def sharpe_ratio(ret_series, periods_per_year=252):
    mu = ret_series.mean()
    sd = ret_series.std(ddof=0)
    if sd == 0 or np.isnan(sd): return np.nan
    return (mu * periods_per_year) / (sd * np.sqrt(periods_per_year))

def curve_metrics(eq_series, name):
    rets = eq_series.pct_change().fillna(0.0)
    mdd_val, _ = max_drawdown(eq_series)
    cagr = ((eq_series.iloc[-1]/eq_series.iloc[0])**(252/len(eq_series)) - 1)*100 if len(eq_series)>30 else np.nan
    sharpe = sharpe_ratio(rets)
    return {
        "Curve": name,
        "CAGR %": round(cagr, 2),
        "MaxDD %": round(mdd_val*100, 2),
        "Sharpe": round(sharpe, 2)
    }

metrics_table = pd.DataFrame([
    curve_metrics(eq_total, "Strategy (Combined)"),
    curve_metrics(eq_long,  "Long-only (mark-to-market)"),
    curve_metrics(eq_short, "Short-only (mark-to-market)"),
    curve_metrics(bh_equity,"Buy & Hold"),
])

# =========================================
# KPI’s bovenaan
# =========================================
last = d.iloc[-1]
regime = ("Bullish" if (d["close"].iloc[-1] > d["ema200"].iloc[-1] and d["ema50"].iloc[-1] > d["ema200"].iloc[-1])
          else "Bearish" if (d["close"].iloc[-1] < d["ema200"].iloc[-1] and d["ema50"].iloc[-1] < d["ema200"].iloc[-1])
          else "Neutraal")
ytd_full, pytd_full = ytd_return_full(df), pytd_return_full(df)
volatility = d['delta_pct'].std()

k1,k2,k3,k4,k5,k6,k7,k8,k9 = st.columns(9)
k1.metric("Laatste close", f"{last['close']:.2f}")
k2.metric("Δ % (dag)", f"{(d['close'].pct_change().iloc[-1]*100):.2f}%")
k3.metric("VIX (close)", f"{last.get('vix_close', np.nan):.2f}")
k4.metric("Regime", regime)
k5.metric("YTD Return",  f"{ytd_full:.2f}%" if ytd_full is not None else "—")
k6.metric("PYTD Return", f"{pytd_full:.2f}%" if pytd_full is not None else "—")
k7.metric("Volatiliteit (std Δ%)", f"{volatility:.2f}%")
k8.metric("Trades (combined)", f"{len(trades_df)}")
winrate = (trades_df["ret_pct"] > 0).mean()*100 if len(trades_df) else 0.0
k9.metric("Winrate (combined)", f"{winrate:.1f}%")

# =========================================
# Charts
# =========================================
# 1) Price + EMA + entry/exit markers
st.subheader("Prijs + EMA + Entry/Exit markers")
fig_p = make_subplots(rows=1, cols=1, shared_xaxes=True)
fig_p.add_trace(go.Scatter(x=d["date"], y=d["close"], name="Close"))
fig_p.add_trace(go.Scatter(x=d["date"], y=d["ema20"], name="EMA20"))
fig_p.add_trace(go.Scatter(x=d["date"], y=d["ema50"], name="EMA50"))
fig_p.add_trace(go.Scatter(x=d["date"], y=d["ema200"], name="EMA200"))

# Markers uit trades_df
if len(trades_df):
    longs  = trades_df[trades_df["side"]=="LONG"]
    shorts = trades_df[trades_df["side"]=="SHORT"]
    if len(longs):
        fig_p.add_trace(go.Scatter(
            x=longs["entry_date"],
            y=[d.loc[d["date"]==ts, "close"].iloc[0] for ts in longs["entry_date"]],
            mode="markers", name="Long entry",
            marker=dict(symbol="triangle-up", size=11, color="#00A65A", line=dict(width=1, color="black"))
        ))
        fig_p.add_trace(go.Scatter(
            x=longs["exit_date"],
            y=[d.loc[d["date"]==ts, "close"].iloc[0] for ts in longs["exit_date"]],
            mode="markers", name="Long exit",
            marker=dict(symbol="x", size=10, color="#006400", line=dict(width=1, color="black"))
        ))
    if len(shorts):
        fig_p.add_trace(go.Scatter(
            x=shorts["entry_date"],
            y=[d.loc[d["date"]==ts, "close"].iloc[0] for ts in shorts["entry_date"]],
            mode="markers", name="Short entry",
            marker=dict(symbol="triangle-down", size=11, color="#D55E00", line=dict(width=1, color="black"))
        ))
        fig_p.add_trace(go.Scatter(
            x=shorts["exit_date"],
            y=[d.loc[d["date"]==ts, "close"].iloc[0] for ts in shorts["exit_date"]],
            mode="markers", name="Short exit",
            marker=dict(symbol="x", size=10, color="#8B0000", line=dict(width=1, color="black"))
        ))

fig_p.update_layout(height=420, legend_orientation="h")
st.plotly_chart(fig_p, use_container_width=True)

# 2) Equity curves: long-only, short-only, combined vs B&H
st.subheader("Equity curves t.o.v. Buy & Hold (start €{:,})".format(int(START_CAP)))
curves = pd.DataFrame({
    "date": d["date"],
    "Total %": (eq_total/START_CAP - 1.0)*100.0,
    "Long %":  (eq_long/START_CAP - 1.0)*100.0,
    "Short %": (eq_short/START_CAP - 1.0)*100.0,
    "B&H %":   (bh_equity/START_CAP - 1.0)*100.0,
}).set_index("date")

fig_eq = go.Figure()
for col, style in [
    ("B&H %", {"width":2}),
    ("Total %", {"width":2}),
    ("Long %", {"width":1}),
    ("Short %", {"width":1}),
]:
    fig_eq.add_trace(go.Scatter(x=curves.index, y=curves[col], name=col, line=style))
fig_eq.update_layout(height=420, legend_orientation="h", yaxis_title="% sinds start")
st.plotly_chart(fig_eq, use_container_width=True)

# 3) Metrics-tabel per curve
st.subheader("Metrics per curve")
st.dataframe(metrics_table, use_container_width=True)

# 4) Trades
st.subheader("Trades (combined run met redenen)")
if len(trades_df):
    show = trades_df.copy()
    show["ret_pct"] = show["ret_pct"].map(lambda x: f"{x:,.2f}%")
    st.dataframe(show[["side","entry_date","entry_px","exit_date","exit_px","ret_pct","reason"]], use_container_width=True)
else:
    st.info("Geen trades in de geselecteerde periode.")

# =========================================
# Uitleg / Advies
# =========================================
with st.expander("Uitleg & Advies bij ATR/TP"):
    st.markdown(f"""
**Stops & Trailing**  
- Initieel: `entry ± ATR × {ATR_MULT:.2f}` (long: min; short: plus).  
- Trailing: dagelijks schuift stop mee met `close ± ATR × {ATR_MULT:.2f}` (alleen richting winst).  

**Take Profits**  
- **TP1 = {TP1_R:.2f}R** → sluit **{TP1_PART}%** van de positie; {'zet stop op **break-even**' if MOVE_BE else 'stop blijft ongewijzigd'}.  
- **TP2 = {TP2_R:.2f}R** → sluit rest (volledige exit).  

**Sizing**  
- Zet **% equity risico** op bijv. **1%** → stukken = \\( \\frac{{equity × 1\\%}}{{entry − stop}} \\).  
- 0% = all-in (minder robuust bij whipsaws).

**Shorts**  
- Alleen **down-regime + ADX > {ADX_TH}**; optioneel VIX-gate (via sidebar).  
- Bij extreem hoge VIX liever spreads (opties) i.p.v. naked shorts.
""")
