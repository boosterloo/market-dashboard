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

df["date"] = pd.to_datetime(df["date"])
df = df[df["date"].dt.weekday < 5]  # skip weekends/holidays
for c in ["open","high","low","close","vix_close","delta_abs","delta_pct"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# ---------- Constants ----------
DEFAULTS = {
    "ema_spans": [20, 50, 200],
    "macd": (12, 26, 9),
    "rsi_period": 14,
    "adx_length": 14,
    "supertrend": {"length": 10, "multiplier": 1.0},
    "donchian_n": 20,
    "corr_win_default": 20,
    "adx_threshold": 20,
    "rsi_ob": 70,
    "rsi_os": 30,
    "vix_high": 25,
    "vix_low": 15
}

# ---- Helpers ----
def ema(s: pd.Series, span: int):
    return s.ewm(span=span, adjust=False, min_periods=span).mean()

def heikin_ashi(src: pd.DataFrame):
    ha_close = (src["open"] + src["high"] + src["low"] + src["close"]) / 4.0
    ha_open = pd.Series(index=src.index, dtype=float)
    ha_open.iloc[0] = src["open"].iloc[0]
    for i in range(1, len(src)):
        ha_open.iloc[i] = (ha_open.iloc[i-1] + ha_close.iloc[i-1]) / 2.0
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

    final_upper = pd.Series(np.full(len(ha), np.nan), index=ha.index)
    final_lower = pd.Series(np.full(len(ha), np.nan), index=ha.index)
    trend = pd.Series(np.ones(len(ha), dtype=int), index=ha.index)

    for i in range(len(ha)):
        if i == 0:
            final_upper.iloc[i] = upper_basic.iloc[i]
            final_lower.iloc[i] = lower_basic.iloc[i]
            trend.iloc[i] = 1
            continue

        final_upper.iloc[i] = upper_basic.iloc[i] if (upper_basic.iloc[i] < final_upper.iloc[i-1]) or (close.iloc[i-1] > final_upper.iloc[i-1]) else final_upper.iloc[i-1]
        final_lower.iloc[i] = lower_basic.iloc[i] if (lower_basic.iloc[i] > final_lower.iloc[i-1]) or (close.iloc[i-1] < final_lower.iloc[i-1]) else final_lower.iloc[i-1]

        if close.iloc[i] > final_upper.iloc[i-1]:   trend.iloc[i] = 1
        elif close.iloc[i] < final_lower.iloc[i-1]: trend.iloc[i] = -1
        else:                                       trend.iloc[i] = trend.iloc[i-1]

    st_line = pd.Series(np.where(trend == 1, final_lower, final_upper), index=ha.index, name="st_line")
    trend_s = trend
    return pd.DataFrame({"st_line": st_line, "trend": trend_s}, index=ha.index)

def donchian(d, n=20):
    return d["high"].rolling(n, min_periods=n).max(), d["low"].rolling(n, min_periods=n).min()

# --- MACD ---
def macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = series.ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

# --- ADX ---
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

# --- RSI ---
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

@st.cache_data(ttl=1800)
def compute_indicators(full_df):
    # EMAs
    for span in DEFAULTS["ema_spans"]:
        full_df[f"ema{span}"] = ema(full_df["close"], span)
    # Donchian
    dc_high, dc_low = donchian(full_df, DEFAULTS["donchian_n"])
    full_df["dc_high"], full_df["dc_low"] = dc_high, dc_low
    # HA + Supertrend
    ha = heikin_ashi(full_df)
    full_df[["ha_open","ha_high","ha_low","ha_close"]] = ha[["ha_open","ha_high","ha_low","ha_close"]]
    st_df = supertrend_on_ha(ha, **DEFAULTS["supertrend"])
    full_df["st_line"], full_df["st_trend"] = st_df["st_line"], st_df["trend"]
    # MACD & ADX
    full_df["macd_line"], full_df["macd_signal"], full_df["macd_hist"] = macd(full_df["close"], *DEFAULTS["macd"])
    full_df["di_plus"], full_df["di_minus"], full_df["adx14"] = adx(full_df, DEFAULTS["adx_length"])
    # RSI
    full_df["rsi14"] = rsi(full_df["close"], DEFAULTS["rsi_period"])
    # ATR voor stops (klassiek TR-RMA op reguliere OHLC)
    full_df["atr14"] = atr_rma(full_df["high"], full_df["low"], full_df["close"], 14)
    return full_df

df = compute_indicators(df)

# ---- Periode (bovenaan, midden) ----
min_d = df["date"].min().date()
max_d = df["date"].max().date()
default_start = max((df["date"].max() - timedelta(days=365)).date(), min_d)
c1, c2, c3 = st.columns([0.1, 0.8, 0.1])
with c2:
    start_date, end_date = st.slider("Periode", min_value=min_d, max_value=max_d,
                                     value=(default_start, max_d), format="YYYY-MM-DD")

# ---------- Sidebar (opgeruimd) ----------
with st.sidebar:
    st.markdown("### ⚙️ Instellingen")
    st.caption("Gegroepeerd zodat labels niet verticaal afbreken.")
    st.markdown("""
        <style>
        [data-testid="stSidebar"] {min-width: 340px;}
        </style>
    """, unsafe_allow_html=True)

    st.markdown("#### Δ-paneel")
    delta_mode = st.radio("Weergave", ["Δ punten", "Δ %"], index=0, horizontal=False)
    agg_mode   = st.selectbox("Aggregatie", ["Dagelijks", "Wekelijks", "Maandelijks"], index=0)
    smooth_on  = st.checkbox("Smoothing (MA) inschakelen", value=False)
    ma_window  = st.slider("MA-window", 2, 60, 5, step=1, disabled=not smooth_on)

    st.divider()
    st.markdown("#### Correlatie & VIX")
    show_vix  = st.toggle("Toon VIX in paneel 1", value=True, help="VIX op 2e y-as in de bovenste grafiek.")
    corr_vs   = st.radio("Rolling correlatie vs VIX", ["% change", "level"], index=0, horizontal=False)
    corr_win  = st.slider("Correlatie-window (dagen)", 5, 90, DEFAULTS["corr_win_default"], step=1)

    st.divider()
    st.markdown("#### Visuals")
    heat_on   = st.checkbox("Toon heatmap (maand/jaar)", value=True)
    hist_on   = st.checkbox("Toon histogrammen", value=True)
    bins      = st.slider("Aantal histogram-bins", 10, 120, 60, 5, disabled=not hist_on)

    st.divider()
    st.markdown("#### Backtest (geavanceerd)")
    START_CAP  = st.number_input("Startkapitaal (€)", 1000.0, 2_000_000.0, 10_000.0, step=500.0)
    ATR_MULT   = st.slider("ATR-stop multiplier", 1.0, 5.0, 2.0, 0.5)
    TP1_R      = st.slider("TP1 (R)", 1.25, 3.0, 1.75, 0.25)
    TP2_R      = st.slider("TP2 (R)", 2.0, 6.0, 3.0, 0.25)
    TP1_PART   = st.slider("TP1: % sluiten", 10, 80, 50, 5)
    MOVE_BE    = st.checkbox("Na TP1: stop → break-even", value=True)
    FEE_BPS    = st.slider("Kosten (bps per trade)", 0.0, 20.0, 1.0, 0.5)
    SLIP_BPS   = st.slider("Slippage (bps per trade)", 0.0, 20.0, 1.0, 0.5)
    RISK_PCT   = st.slider("% equity risico per trade", 0.0, 5.0, 1.0, 0.5,
                           help="0% = all-in; >0%: sizing = (equity×%risk)/(entry−stop).")
    ALLOW_SHORT= st.checkbox("Shorts toestaan", True)

# ---- Filter & basis ----
d = df[(df["date"].dt.date >= start_date) & (df["date"].dt.date <= end_date)].reset_index(drop=True).copy()
if "delta_abs" not in d or d["delta_abs"].isna().all():
    d["delta_abs"] = d["close"].diff().fillna(0)
if "delta_pct" not in d or d["delta_pct"].isna().all():
    d["delta_pct"] = d["close"].pct_change().fillna(0)*100.0

# --------- Regime & SIGNALS (verbeterd) ---------
up_regime = (d["close"] > d["ema200"]) & (d["ema50"] > d["ema200"])
down_regime = (d["close"] < d["ema200"]) & (d["ema50"] < d["ema200"])
strong_trend = d["adx14"] > DEFAULTS["adx_threshold"]

# Verbeterde BUY/SELL:
d["buy_sig"]  = up_regime & strong_trend & (d["macd_hist"].shift(1) <= 0) & (d["macd_hist"] > 0) & (d["close"] > d["ema20"]) & (d["rsi14"] > 45)
d["sell_sig"] = ((d["close"] < d["ema20"]) | ((d["macd_hist"].shift(1) >= 0) & (d["macd_hist"] < 0))) & strong_trend
d["short_sig"] = down_regime & strong_trend & (d["macd_hist"].shift(1) >= 0) & (d["macd_hist"] < 0) & (d["close"] < d["ema20"]) & (d["rsi14"] < 55)

d["buy_sig"]   = d["buy_sig"].fillna(False)
d["sell_sig"]  = d["sell_sig"].fillna(False)
d["short_sig"] = d["short_sig"].fillna(False)

# --------- Geavanceerde Backtest met ATR/TP ---------
def run_backtest(df_, allow_short=True):
    df_ = df_.copy().reset_index(drop=True)
    fee = FEE_BPS/1e4; slip = SLIP_BPS/1e4

    cash = START_CAP
    pos = 0              # 0=flat, 1=long, -1=short
    shares = 0.0
    entry_px = None; entry_idx = None
    stop = np.nan
    took_tp1 = False

    eq_total = []
    trades = []

    for i in range(len(df_)):
        close_i = float(df_.loc[i,"close"])
        # mark-to-market
        if pos == 1:
            eq_total.append(cash + shares*close_i)
        elif pos == -1:
            pnl = shares*(entry_px - close_i)
            eq_total.append(cash + pnl)
        else:
            eq_total.append(cash)

        if i == len(df_)-1:
            break

        nxt_open = float(df_.loc[i+1,"open"])
        atr_next = float(df_.loc[i+1,"atr14"]) if pd.notna(df_.loc[i+1,"atr14"]) else float(df_.loc[i,"atr14"])

        # trailing stop
        if pos == 1:
            stop = max(stop, close_i - ATR_MULT*atr_next)
        elif pos == -1:
            stop = min(stop, close_i + ATR_MULT*atr_next)

        # R-now (voor TP)
        if pos == 1 and not np.isnan(stop):
            R_now = (close_i - entry_px) / max(entry_px - stop, 1e-9)
        elif pos == -1 and not np.isnan(stop):
            R_now = (entry_px - close_i) / max(stop - entry_px, 1e-9)
        else:
            R_now = 0.0

        # Helper exit
        def do_exit(px, reason):
            nonlocal cash, shares, pos, stop, entry_px, entry_idx, took_tp1
            if pos == 1:
                cash += shares*px
                cash -= abs(shares*px)*fee
                ret = (px/entry_px - 1)*100
            else:
                cash += shares*(entry_px - px)
                cash -= abs(shares*px)*fee
                ret = (1 - px/entry_px)*100
            trades.append({
                "side": "LONG" if pos==1 else "SHORT",
                "entry_date": df_.loc[entry_idx,"date"],
                "entry_px": entry_px,
                "exit_date": df_.loc[i+1,"date"],
                "exit_px": px,
                "ret_pct": ret,
                "reason": reason
            })
            shares = 0.0; pos = 0; stop = np.nan; entry_px = None; entry_idx = None; took_tp1 = False

        # --- TP2 ---
        if pos != 0 and R_now >= TP2_R:
            px = nxt_open*(1 - slip) if pos==1 else nxt_open*(1 + slip)
            do_exit(px, "TP2"); continue

        # --- TP1 (partial) ---
        if pos != 0 and (not took_tp1) and R_now >= TP1_R:
            px = nxt_open*(1 - slip) if pos==1 else nxt_open*(1 + slip)
            close_sh = np.floor(shares * (TP1_PART/100.0))
            if close_sh >= 1:
                if pos == 1:
                    cash += close_sh*px
                    cash -= abs(close_sh*px)*fee
                else:
                    cash += close_sh*(entry_px - px)
                    cash -= abs(close_sh*px)*fee
                shares -= close_sh
            took_tp1 = True
            if MOVE_BE:
                stop = max(stop, entry_px) if pos==1 else min(stop, entry_px)

        # --- Exit signal of stop-gap ---
        exit_signal = (pos==1 and bool(df_.loc[i,"sell_sig"])) or (pos==-1 and bool(df_.loc[i,"buy_sig"]))
        stop_gap = (pos==1 and not np.isnan(stop) and nxt_open < stop) or (pos==-1 and not np.isnan(stop) and nxt_open > stop)
        if pos != 0 and (exit_signal or stop_gap):
            px = nxt_open*(1 - slip) if pos==1 else nxt_open*(1 + slip)
            do_exit(px, "signal" if exit_signal and not stop_gap else "stop")
            continue

        # --- Entries (next open) ---
        if pos == 0:
            if bool(df_.loc[i,"buy_sig"]):
                buy_px = nxt_open*(1 + slip)
                init_stop = buy_px - ATR_MULT*atr_next
                if RISK_PCT > 0:
                    risk_amt = cash*(RISK_PCT/100.0)
                    per_share = max(buy_px - init_stop, 1e-9)
                    shares = np.floor(risk_amt/per_share)
                    cost = shares*buy_px
                    if cost > cash:
                        shares = np.floor(cash/buy_px); cost = shares*buy_px
                else:
                    cost = cash; shares = cost/buy_px
                cash -= cost; cash -= cost*fee
                pos = 1; entry_px = buy_px; entry_idx = i+1; stop = init_stop; took_tp1 = False

            elif allow_short and bool(df_.loc[i,"short_sig"]):
                sell_px = nxt_open*(1 - slip)
                init_stop = sell_px + ATR_MULT*atr_next
                if RISK_PCT > 0:
                    risk_amt = cash*(RISK_PCT/100.0)
                    per_share = max(init_stop - sell_px, 1e-9)
                    shares = np.floor(risk_amt/per_share)
                else:
                    shares = cash/sell_px
                cash += shares*sell_px
                cash -= abs(shares*sell_px)*fee
                pos = -1; entry_px = sell_px; entry_idx = i+1; stop = init_stop; took_tp1 = False

    eq_series = pd.Series(eq_total, index=df_["date"], name="equity")
    trades_df = pd.DataFrame(trades)
    return eq_series, trades_df

eq_equity, trades_df = run_backtest(d, allow_short=ALLOW_SHORT)

# Buy & Hold (index met start=100 én equity-curve met startkapitaal)
def buyhold_index(df_):
    return (1 + df_["delta_pct"].fillna(0)/100).cumprod()*100

def buyhold_equity(df_, start_cap):
    if len(df_) < 2: return pd.Series([start_cap], index=df_["date"])
    buy_px = float(df_.loc[1,"open"])
    shares = start_cap / buy_px
    return shares*df_["close"]

bh_index = buyhold_index(d)
bh_equity = buyhold_equity(d, START_CAP)

# ---- KPI ----
last = d.iloc[-1]
regime = ("Bullish" if up_regime.iloc[-1]
          else "Bearish" if down_regime.iloc[-1]
          else "Neutraal")
ytd_full, pytd_full = ytd_return_full(df), pytd_return_full(df)
volatility = d['delta_pct'].std()
strategy_vs_bh = (eq_equity.iloc[-1]/eq_equity.iloc[0] - bh_equity.iloc[-1]/bh_equity.iloc[0]) * 100
num_buys, num_sells = d['buy_sig'].sum(), d['sell_sig'].sum()

k1,k2,k3,k4,k5,k6,k7,k8,k9 = st.columns(9)
k1.metric("Laatste close", f"{last['close']:.2f}")
k2.metric("Δ % (dag)", f"{last['delta_pct']:.2f}%")
k3.metric("VIX (close)", f"{last.get('vix_close', np.nan):.2f}")
k4.metric("Regime", regime)
k5.metric("YTD Return",  f"{ytd_full:.2f}%" if ytd_full is not None else "—")
k6.metric("PYTD Return", f"{pytd_full:.2f}%" if pytd_full is not None else "—")
k7.metric("Volatiliteit (std Δ%)", f"{volatility:.2f}%")
k8.metric("Strategy vs B&H", f"{strategy_vs_bh:.1f}%")
k9.metric("Signals (buy/sell)", f"{int(num_buys)} / {int(num_sells)}")

# ---------- Alerts (regime-aware) ----------
last_idx = d.index[-1]
alerts = []

# 1) VIX-regime
if "vix_close" in d.columns and pd.notna(d.loc[last_idx, "vix_close"]):
    if d.loc[last_idx, "vix_close"] > DEFAULTS["vix_high"]:
        alerts.append(("red", "VIX hoog", f"VIX {d.loc[last_idx,'vix_close']:.1f} > {DEFAULTS['vix_high']} — verhoogde volatiliteit."))
    elif d.loc[last_idx, "vix_close"] < DEFAULTS["vix_low"]:
        alerts.append(("blue", "VIX laag", f"VIX {d.loc[last_idx,'vix_close']:.1f} < {DEFAULTS['vix_low']} — ‘rustiger’ regime."))

# 2) Regime & ADX
if up_regime.iloc[-1] and (d.loc[last_idx, "adx14"] > DEFAULTS["adx_threshold"]):
    alerts.append(("green", "Trend ↑ (sterk)", f"UP-regime + ADX {d.loc[last_idx,'adx14']:.1f} > {DEFAULTS['adx_threshold']}."))
elif down_regime.iloc[-1] and (d.loc[last_idx, "adx14"] > DEFAULTS["adx_threshold"]):
    alerts.append(("red", "Trend ↓ (sterk)", f"DOWN-regime + ADX {d.loc[last_idx,'adx14']:.1f} > {DEFAULTS['adx_threshold']}."))
elif d.loc[last_idx, "adx14"] <= DEFAULTS["adx_threshold"]:
    alerts.append(("gray", "Trend zwak", f"ADX {d.loc[last_idx,'adx14']:.1f} ≤ {DEFAULTS['adx_threshold']} — range/ruis."))

# 3) Pullback/rally (RSI, regime-aware)
if up_regime.iloc[-1] and (d.loc[last_idx, "rsi14"] < 40):
    alerts.append(("orange", "Pullback in uptrend", f"RSI14 {d.loc[last_idx,'rsi14']:.1f} < 40 — mogelijke long-kans bij hervatting."))
if down_regime.iloc[-1] and (d.loc[last_idx, "rsi14"] > 60):
    alerts.append(("orange", "Rally in downtrend", f"RSI14 {d.loc[last_idx,'rsi14']:.1f} > 60 — mogelijke short-kans bij hervatting."))

# 4) MACD-histogram cross
if crossed_up(d["macd_hist"]).iloc[-1]:
    alerts.append(("green", "Momentum draait ↑", "MACD-hist cross ↑ 0."))
elif crossed_down(d["macd_hist"]).iloc[-1]:
    alerts.append(("red", "Momentum draait ↓", "MACD-hist cross ↓ 0."))

# 5) Korte trend: close vs EMA20
if up_regime.iloc[-1] and (d.loc[last_idx, "close"] < d.loc[last_idx, "ema20"]):
    alerts.append(("red", "Onder EMA20 (uptrend)", "Korte trend verzwakt — overweeg trims/strakkere stop."))
if down_regime.iloc[-1] and (d.loc[last_idx, "close"] > d.loc[last_idx, "ema20"]):
    alerts.append(("green", "Boven EMA20 (downtrend)", "Tegenbeweging binnen neertrend."))

# 6) Donchian breakout
if pd.notna(d.loc[last_idx, "dc_high"]) and (d.loc[last_idx, "close"] > d.loc[last_idx, "dc_high"]):
    alerts.append(("green", "Breakout ↑ (Donchian)", "Close boven hoogste 20-daagse."))
if pd.notna(d.loc[last_idx, "dc_low"]) and (d.loc[last_idx, "close"] < d.loc[last_idx, "dc_low"]):
    alerts.append(("red", "Breakout ↓ (Donchian)", "Close onder laagste 20-daagse."))

# 7) Supertrend flip
if len(d) >= 2 and (d["st_trend"].iloc[-1] != d["st_trend"].iloc[-2]):
    direction = "↑" if d["st_trend"].iloc[-1] == 1 else "↓"
    color = "green" if direction == "↑" else "red"
    alerts.append((color, f"Supertrend flip {direction}", "Trendwissel op HA-basis."))

def badge(color, text):
    colors = {"green":"#00A65A","red":"#D55E00","orange":"#E69F00","blue":"#1f77b4","gray":"#6c757d"}
    return f"""<span style="background:{colors[color]};color:white;padding:2px 8px;border-radius:12px;font-size:0.85rem;">{text}</span>"""

with st.expander("⚠️ Alerts & Signal status", expanded=True):
    if not alerts:
        st.success("Geen alerts op dit moment.")
    else:
        for color, title, msg in alerts:
            st.markdown(f"{badge(color, title)} &nbsp; {msg}", unsafe_allow_html=True)

# =======================
# Paneel 1 – HA + ST + Donchian (+ VIX) + VIX shading
# =======================
fig1 = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]],
                     subplot_titles=["S&P 500 Heikin-Ashi + Supertrend (10,1) + Donchian" + (" + VIX (2e y-as)" if show_vix else "")])
fig1.add_trace(go.Candlestick(x=d["date"], open=d["ha_open"], high=d["ha_high"], low=d["ha_low"], close=d["ha_close"],
                              name="SPX (Heikin-Ashi)"), row=1, col=1, secondary_y=False)
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
    if "vix_high" in d.columns:
        in_high = False
        for i in range(len(d)):
            if d["vix_high"].iloc[i] and not in_high:
                start_idx = i; in_high = True
            elif not d["vix_high"].iloc[i] and in_high:
                fig1.add_vrect(x0=d["date"].iloc[start_idx], x1=d["date"].iloc[i],
                               fillcolor="yellow", opacity=0.1, layer="below", row=1, col=1)
                in_high = False
        if in_high:
            fig1.add_vrect(x0=d["date"].iloc[start_idx], x1=d["date"].iloc[-1],
                           fillcolor="yellow", opacity=0.1, layer="below", row=1, col=1)

fig1.update_layout(height=520, margin=dict(l=20,r=20,t=60,b=10),
                   legend_orientation="h", legend_yanchor="top", legend_y=1.08, legend_x=0)
fig1.update_xaxes(rangeslider_visible=False)
fig1.update_yaxes(title_text="Index (HA)", row=1, col=1, secondary_y=False)
if show_vix:
    fig1.update_yaxes(title_text="VIX", row=1, col=1, secondary_y=True)
st.plotly_chart(fig1, use_container_width=True)

# ---------------- Δ-aggregatie ----------------
def aggregate_delta(_df: pd.DataFrame, mode: str, how: str) -> pd.Series:
    t = _df.copy().set_index("date")
    if how == "Dagelijks":
        series = t["delta_pct"].dropna() if mode == "Δ %" else t["delta_abs"].dropna()
        return series.reindex(t.index, fill_value=0)
    rule = "W-FRI" if how == "Wekelijks" else "M"
    if mode == "Δ %":
        res = t["delta_pct"].groupby(pd.Grouper(freq=rule)).apply(
            lambda g: (np.prod((g.dropna()/100.0 + 1.0)) - 1.0) * 100.0 if len(g.dropna()) else np.nan
        )
        return res.fillna(0)
    else:
        return t["delta_abs"].groupby(pd.Grouper(freq=rule)).sum(min_count=1).fillna(0)

delta_series = aggregate_delta(d, delta_mode, agg_mode)
if smooth_on:
    delta_series = delta_series.rolling(ma_window, min_periods=1).mean()
delta_x = delta_series.index
delta_legend = "Δ (%)" if delta_mode=="Δ %" else "Δ (punten)"
if smooth_on: delta_legend += f" — MA{ma_window}"
delta_colors = np.where(delta_series.values >= 0, "rgba(16,150,24,0.7)", "rgba(219,64,82,0.7)")

# =======================
# Paneel 2–6: Δ, Close+EMA, MACD, ADX, RSI
# =======================
fig2 = make_subplots(
    rows=5, cols=1, shared_xaxes=True,
    subplot_titles=[
        f"{'Δ (%)' if delta_mode=='Δ %' else 'Δ (punten)'} — {agg_mode.lower()}{' — MA'+str(ma_window) if smooth_on else ''}",
        "Close + EMA(20/50/200) — regime, buy/sell",
        "MACD(12,26,9) — lijn/signal/hist",
        f"ADX(14) + DI± — drempel={DEFAULTS['adx_threshold']}",
        "RSI(14) — >70 overbought, <30 oversold"
    ],
    row_heights=[0.18, 0.3, 0.18, 0.18, 0.16],
    vertical_spacing=0.06
)

# (1) Δ bars
fig2.add_trace(go.Bar(x=delta_x, y=delta_series.values, name=delta_legend,
                      marker=dict(color=delta_colors), opacity=0.9), row=1, col=1)

# (2) Close + EMA + signals
fig2.add_trace(go.Scatter(x=d["date"], y=d["close"], mode="lines", name="Close"), row=2, col=1)
fig2.add_trace(go.Scatter(x=d["date"], y=d["ema20"], mode="lines", name="EMA20"), row=2, col=1)
fig2.add_trace(go.Scatter(x=d["date"], y=d["ema50"], mode="lines", name="EMA50"), row=2, col=1)
fig2.add_trace(go.Scatter(x=d["date"], y=d["ema200"], mode="lines", name="EMA200"), row=2, col=1)

buys  = d.loc[d["buy_sig"]]
sells = d.loc[d["sell_sig"]]
fig2.add_trace(go.Scatter(
    x=buys["date"], y=buys["close"], mode="markers", name="Buy",
    marker=dict(symbol="triangle-up", size=11, color="#00A65A", line=dict(width=1, color="black")),
    hovertemplate="Buy<br>%{x|%Y-%m-%d}<br>Close=%{y:.2f}<extra></extra>"
), row=2, col=1)
fig2.add_trace(go.Scatter(
    x=sells["date"], y=sells["close"], mode="markers", name="Sell",
    marker=dict(symbol="triangle-down", size=11, color="#D55E00", line=dict(width=1, color="black")),
    hovertemplate="Sell<br>%{x|%Y-%m-%d}<br>Close=%{y:.2f}<extra></extra>"
), row=2, col=1)

# (3) MACD
fig2.add_trace(go.Scatter(x=d["date"], y=d["macd_line"],   mode="lines", name="MACD"),   row=3, col=1)
fig2.add_trace(go.Scatter(x=d["date"], y=d["macd_signal"], mode="lines", name="Signal"), row=3, col=1)
fig2.add_trace(go.Bar(x=d["date"], y=d["macd_hist"], name="Hist",
                      marker_color=np.where(d["macd_hist"]>=0, "rgba(16,150,24,0.6)", "rgba(219,64,82,0.6)")),
               row=3, col=1)
fig2.add_hline(y=0, line_dash="dot", row=3, col=1)

# (4) ADX + DI±
fig2.add_trace(go.Scatter(x=d["date"], y=d["adx14"],   mode="lines", name="ADX(14)"), row=4, col=1)
fig2.add_trace(go.Scatter(x=d["date"], y=d["di_plus"], mode="lines", name="+DI"),     row=4, col=1)
fig2.add_trace(go.Scatter(x=d["date"], y=d["di_minus"],mode="lines", name="−DI"),     row=4, col=1)
fig2.add_hline(y=DEFAULTS["adx_threshold"], line_dash="dot", row=4, col=1)

# (5) RSI
fig2.add_trace(go.Scatter(x=d["date"], y=d["rsi14"], mode="lines", name="RSI(14)"), row=5, col=1)
fig2.add_hline(y=DEFAULTS["rsi_ob"], line_dash="dash", line_color="red", row=5, col=1)
fig2.add_hline(y=DEFAULTS["rsi_os"], line_dash="dash", line_color="green", row=5, col=1)

fig2.update_layout(height=1100, margin=dict(l=20,r=20,t=50,b=20),
                   legend_orientation="h", legend_yanchor="top", legend_y=1.06, legend_x=0)
fig2.update_xaxes(rangeslider_visible=False)
fig2.update_yaxes(title_text="Δ", row=1, col=1)
fig2.update_yaxes(title_text="Close/EMA", row=2, col=1)
fig2.update_yaxes(title_text="MACD", row=3, col=1)
fig2.update_yaxes(title_text="ADX / DI", row=4, col=1)
fig2.update_yaxes(title_text="RSI", row=5, col=1, range=[0,100])

st.plotly_chart(fig2, use_container_width=True)

# ---------- Equity Curves (geavanceerd) ----------
st.markdown("#### Equity Curves (geavanceerde backtest) — % sinds start")
equity_pct = (eq_equity/eq_equity.iloc[0]-1)*100.0
bh_pct     = (bh_equity/bh_equity.iloc[0]-1)*100.0
fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=d["date"], y=bh_pct, name="Buy & Hold", line=dict(color="blue")))
fig3.add_trace(go.Scatter(x=d["date"], y=equity_pct, name="Signal Strategy (ATR/TP)", line=dict(color="orange")))
fig3.update_layout(yaxis_title="% sinds start (equity)", height=400)
st.plotly_chart(fig3, use_container_width=True)

# ---------- Rolling correlatie met VIX ----------
st.markdown("#### Rolling correlatie met VIX")
corr_df = d.copy().set_index("date")
spx_ret = corr_df["delta_pct"]
vix_series = (corr_df["vix_close"].pct_change()*100.0) if corr_vs=="% change" else corr_df["vix_close"]
corr_join = pd.concat([spx_ret.rename("spx"), vix_series.rename("vix")], axis=1).dropna()
rolling_corr = corr_join["spx"].rolling(corr_win).corr(corr_join["vix"])

fig_corr = go.Figure()
fig_corr.add_trace(go.Scatter(x=rolling_corr.index, y=rolling_corr.values, mode="lines", name="Rolling corr"))
fig_corr.add_hline(y=0.0, line_dash="dot")
fig_corr.add_hrect(y0=-1, y1=-0.5, fillcolor="rgba(255,0,0,0.06)", line_width=0)
fig_corr.add_hrect(y0=0.5, y1=1,   fillcolor="rgba(0,128,0,0.06)", line_width=0)
fig_corr.update_layout(height=300, margin=dict(l=20,r=20,t=30,b=20), yaxis=dict(range=[-1,1], title="corr"), xaxis_title="")
st.plotly_chart(fig_corr, use_container_width=True)

# ---------- Heatmap ----------
if heat_on:
    st.subheader("Maand/jaar-heatmap van Δ")
    t = d.copy().set_index("date")
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
if hist_on:
    st.subheader("Histogram dagrendementen")
    col_a, col_b = st.columns(2)
    hist_df = d.dropna(subset=["delta_abs","delta_pct"]).copy()
    with col_a:
        fig_abs = go.Figure([go.Histogram(x=hist_df["delta_abs"], nbinsx=int(bins))])
        fig_abs.update_layout(title="Δ abs (punten)", height=320, bargap=0.02, margin=dict(l=10,r=10,t=40,b=10))
        st.plotly_chart(fig_abs, use_container_width=True)
    with col_b:
        fig_pct = go.Figure([go.Histogram(x=hist_df["delta_pct"], nbinsx=int(bins))])
        fig_pct.update_layout(title="Δ %", height=320, bargap=0.02, margin=dict(l=10,r=10,t=40,b=10))
        st.plotly_chart(fig_pct, use_container_width=True)

# ---------- Trades & Uitleg ----------
st.subheader("Trades (geavanceerde backtest)")
if len(trades_df):
    show = trades_df.copy()
    show["ret_pct"] = show["ret_pct"].map(lambda x: f"{x:,.2f}%")
    st.dataframe(show[["side","entry_date","entry_px","exit_date","exit_px","ret_pct","reason"]], use_container_width=True)
else:
    st.info("Geen trades in de geselecteerde periode.")

with st.expander("Uitleg over performance-assen & backtest-logica"):
    st.markdown("""
- **Buy & Hold** en **Strategy** worden in de equity-plot getoond als **% sinds startkapitaal** (y-as).
- In de oude plot leek het ~1% omdat het een **index (start=100)** was op een korte periode; nu zie je **equity %** helder.
- Backtest: entry op **next open**, **ATR × multiplier** voor initieel en **trailing stop**, **TP1/TP2** partials.
- Signal-kwaliteit: **regime (EMA’s)** + **ADX > drempel** + **MACD-hist cross** + **EMA20/RSI-filters** om whipsaws te beperken.
""")
