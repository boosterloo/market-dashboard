import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.bq import run_query, bq_ping

# =========================
# App setup
# =========================
st.set_page_config(page_title="📈 S&P 500 — Signals & Backtest", layout="wide")
st.title("📈 S&P 500 — Signals & Backtest (ATR/TP + EMA toggle)")

SPX_VIEW = st.secrets.get("tables", {}).get(
    "spx_view", "nth-pier-468314-p7.marketdata.spx_with_vix_v"
)

# ---- Health ----
try:
    if not bq_ping():
        st.error("Geen BigQuery-verbinding."); st.stop()
except Exception as e:
    st.error("Geen BigQuery-verbinding."); st.caption(f"Details: {e}"); st.stop()

# =========================
# Data
# =========================
@st.cache_data(ttl=1800, show_spinner=False)
def load_spx():
    return run_query(f"SELECT * FROM `{SPX_VIEW}` ORDER BY date")

with st.spinner("SPX data laden…"):
    df = load_spx()
if df.empty:
    st.warning("Geen data in view."); st.stop()

df["date"] = pd.to_datetime(df["date"])
df = df[df["date"].dt.weekday < 5]  # handelsdagen
for c in ["open","high","low","close","vix_close","delta_abs","delta_pct"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# =========================
# Constants & helpers
# =========================
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
    "vix_low": 15,
}

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
    return pd.DataFrame({"st_line": st_line, "trend": trend}, index=ha.index)

def donchian(d, n=20):
    return d["high"].rolling(n, min_periods=n).max(), d["low"].rolling(n, min_periods=n).min()

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
    return plus_di, minus_di, adx

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

# =========================
# Indicators
# =========================
@st.cache_data(ttl=1800)
def compute_indicators(full_df):
    for span in DEFAULTS["ema_spans"]:
        full_df[f"ema{span}"] = ema(full_df["close"], span)
    dc_high, dc_low = donchian(full_df, DEFAULTS["donchian_n"])
    full_df["dc_high"], full_df["dc_low"] = dc_high, dc_low
    ha = heikin_ashi(full_df)
    full_df[["ha_open","ha_high","ha_low","ha_close"]] = ha[["ha_open","ha_high","ha_low","ha_close"]]
    st_df = supertrend_on_ha(ha, **DEFAULTS["supertrend"])
    full_df["st_line"], full_df["st_trend"] = st_df["st_line"], st_df["trend"]
    full_df["macd_line"], full_df["macd_signal"], full_df["macd_hist"] = macd(full_df["close"], *DEFAULTS["macd"])
    full_df["di_plus"], full_df["di_minus"], full_df["adx14"] = adx(full_df, DEFAULTS["adx_length"])
    full_df["rsi14"] = rsi(full_df["close"], DEFAULTS["rsi_period"])
    full_df["atr14"] = atr_rma(full_df["high"], full_df["low"], full_df["close"], 14)
    if "delta_abs" not in full_df or full_df["delta_abs"].isna().all():
        full_df["delta_abs"] = full_df["close"].diff().fillna(0)
    if "delta_pct" not in full_df or full_df["delta_pct"].isna().all():
        full_df["delta_pct"] = full_df["close"].pct_change().fillna(0)*100.0
    return full_df

df = compute_indicators(df)

# =========================
# Periode & Sidebar
# =========================
min_d = df["date"].min().date()
max_d = df["date"].max().date()
default_start = max((df["date"].max() - timedelta(days=365)).date(), min_d)
c1, c2, c3 = st.columns([0.1, 0.8, 0.1])
with c2:
    start_date, end_date = st.slider("Periode", min_value=min_d, max_value=max_d,
                                     value=(default_start, max_d), format="YYYY-MM-DD")

with st.sidebar:
    st.markdown("### ⚙️ Instellingen")
    st.markdown("""
        <style>
        [data-testid="stSidebar"] {min-width: 360px;}
        </style>
    """, unsafe_allow_html=True)

    st.markdown("#### Δ-paneel")
    delta_mode = st.radio("Weergave", ["Δ punten", "Δ %"], index=0)
    agg_mode   = st.selectbox("Aggregatie", ["Dagelijks", "Wekelijks", "Maandelijks"], index=0)
    smooth_on  = st.checkbox("Smoothing (MA)", value=False)
    ma_window  = st.slider("MA-window", 2, 60, 5, step=1, disabled=not smooth_on)

    st.divider()
    st.markdown("#### Correlatie & VIX")
    show_vix  = st.toggle("Toon VIX in paneel 1", value=True, help="VIX op 2e y-as in de bovenste grafiek.")
    corr_vs   = st.radio("Rolling correlatie vs VIX", ["% change", "level"], index=0)
    corr_win  = st.slider("Correlatie-window (dagen)", 5, 90, DEFAULTS["corr_win_default"], step=1)

    st.divider()
    st.markdown("#### 📐 Signaalset")
    SIG_MODE = st.radio("Welke signalen gebruiken?", ["Advanced", "EMA crossover"], index=0, horizontal=True)
    EMA_FAST = st.slider("EMA fast", 5, 100, 50, 1, disabled=(SIG_MODE!="EMA crossover"))
    EMA_SLOW = st.slider("EMA slow", 20, 300, 200, 5, disabled=(SIG_MODE!="EMA crossover"))
    ALLOW_SHORT_EMA = st.checkbox("Shorts toestaan (EMA-modus)", True, disabled=(SIG_MODE!="EMA crossover"))

    st.divider()
    st.markdown("#### 🧪 Backtest")
    START_CAP  = st.number_input("Startkapitaal (€)", 1000.0, 2_000_000.0, 10_000.0, step=500.0)
    ATR_MULT   = st.slider("ATR-stop multiplier", 1.0, 5.0, 2.0, 0.5)
    TP1_R      = st.slider("TP1 (R)", 1.25, 3.0, 1.75, 0.25)
    TP2_R      = st.slider("TP2 (R)", 2.0, 6.0, 3.0, 0.25)
    TP1_PART   = st.slider("TP1: % sluiten", 10, 80, 50, 5)
    MOVE_BE    = st.checkbox("Na TP1: stop → break-even", value=True)
    FEE_BPS    = st.slider("Kosten (bps)", 0.0, 20.0, 1.0, 0.5)
    SLIP_BPS   = st.slider("Slippage (bps)", 0.0, 20.0, 1.0, 0.5)
    RISK_PCT   = st.slider("% equity risico per trade", 0.0, 5.0, 1.0, 0.5,
                           help="0% = all-in; >0%: sizing = (equity×%risk)/(entry−stop).")
    ALLOW_SHORT= st.checkbox("Shorts toestaan (Advanced)", True)

    st.divider()
    st.markdown("#### 🟨 Event windows + Options-proxy")
    highlight_on = st.checkbox("Toon gele balken (event windows)", value=True)
    window_rule = st.selectbox("Window-regel", ["VIX spike", "Supertrend flip", "Donchian breakout", "UNION (alles)"], index=0)
    vix_z = st.slider("VIX spike: z-score drempel", 1.0, 4.0, 2.0, 0.1,
                      help="(VIX - MA20) / STD20 > drempel")
    flip_pad = st.slider("Padding rond flip/breakout (dagen)", 0, 10, 3, 1)
    min_len = st.slider("Min. window-lengte (dagen)", 1, 30, 3, 1)

    st.markdown("##### Options-proxy parameters (per dag)")
    straddle_cost_atr = st.slider("Long Straddle — cost (× ATR)", 0.2, 3.0, 1.0, 0.1)
    strangle_prem_atr = st.slider("Short Strangle — premium (× ATR)", 0.1, 3.0, 0.6, 0.1)
    strangle_width_atr = st.slider("Short Strangle — breedte (× ATR)", 0.5, 5.0, 1.5, 0.1)

# =========================
# Filter subset
# =========================
d = df[(df["date"].dt.date >= start_date) & (df["date"].dt.date <= end_date)].reset_index(drop=True).copy()

# =========================
# Signalen
# =========================
if SIG_MODE == "Advanced":
    up_regime   = (d["close"] > d["ema200"]) & (d["ema50"] > d["ema200"])
    down_regime = (d["close"] < d["ema200"]) & (d["ema50"] < d["ema200"])
    strong_trend = d["adx14"] > DEFAULTS["adx_threshold"]

    d["buy_sig"]   = up_regime & strong_trend & crossed_up(d["macd_hist"]) & (d["close"] > d["ema20"]) & (d["rsi14"] > 45)
    d["sell_sig"]  = ((d["close"] < d["ema20"]) | crossed_down(d["macd_hist"])) & strong_trend
    d["short_sig"] = down_regime & strong_trend & crossed_down(d["macd_hist"]) & (d["close"] < d["ema20"]) & (d["rsi14"] < 55)
else:
    d["ema_fast_custom"] = ema(d["close"], EMA_FAST)
    d["ema_slow_custom"] = ema(d["close"], EMA_SLOW)
    cross_up   = (d["ema_fast_custom"].shift(1) <= d["ema_slow_custom"].shift(1)) & (d["ema_fast_custom"] > d["ema_slow_custom"])
    cross_down = (d["ema_fast_custom"].shift(1) >= d["ema_slow_custom"].shift(1)) & (d["ema_fast_custom"] < d["ema_slow_custom"])
    d["buy_sig"]  = cross_up
    d["sell_sig"] = cross_down
    d["short_sig"]= cross_down if ALLOW_SHORT_EMA else False

for c in ["buy_sig","sell_sig","short_sig"]:
    d[c] = d[c].fillna(False)

# =========================
# Event windows (gele balken)
# =========================
def _windows_from_bool(mask: pd.Series, min_len=3, pad=0):
    mask = mask.fillna(False).astype(bool)
    if pad > 0:
        mask = mask.rolling(pad, min_periods=1).max().astype(bool) | mask | mask[::-1].rolling(pad, min_periods=1).max()[::-1].astype(bool)
    starts, ends = [], []
    in_win = False
    for i, v in enumerate(mask.values):
        if v and not in_win:
            starts.append(i); in_win = True
        if in_win and (not v or i == len(mask)-1):
            ends.append(i if not v else i)  # inclusive index
            in_win = False
    # filter minimum lengte
    windows = []
    for s_idx, e_idx in zip(starts, ends):
        if e_idx - s_idx + 1 >= min_len:
            windows.append((d.loc[s_idx, "date"], d.loc[e_idx, "date"]))
    return windows

# regels
vix_ma = d["vix_close"].rolling(20, min_periods=20).mean()
vix_sd = d["vix_close"].rolling(20, min_periods=20).std()
vix_zscore = (d["vix_close"] - vix_ma) / vix_sd
mask_vix = vix_zscore > vix_z

st_flip = d["st_trend"].fillna(method="ffill")
mask_flip = st_flip.ne(st_flip.shift(1)).fillna(False)

mask_dc_break = (d["close"] > d["dc_high"].shift(1)) | (d["close"] < d["dc_low"].shift(1))

if window_rule == "VIX spike":
    win_list = _windows_from_bool(mask_vix, min_len=min_len, pad=flip_pad)
elif window_rule == "Supertrend flip":
    win_list = _windows_from_bool(mask_flip, min_len=min_len, pad=flip_pad)
elif window_rule == "Donchian breakout":
    win_list = _windows_from_bool(mask_dc_break, min_len=min_len, pad=flip_pad)
else:
    union_mask = (mask_vix | mask_flip | mask_dc_break)
    win_list = _windows_from_bool(union_mask, min_len=min_len, pad=flip_pad)

# helper: snel check in-window
def in_any_window(ts):
    if not win_list: return pd.Series(False, index=d.index)
    m = pd.Series(False, index=d.index)
    for s, e in win_list:
        m |= (d["date"] >= s) & (d["date"] <= e)
    return m

in_window = in_any_window(d["date"])

# =========================
# Backtest engine (ATR/TP, sizing, fees/slip)
# =========================
def run_backtest(df_, start_capital, atr_mult, fee_bps, slip_bps, risk_pct,
                 allow_long=True, allow_short=True,
                 tp1_r=1.75, tp2_r=3.0, tp1_part=50, move_be=True):
    df_ = df_.copy().reset_index(drop=True)
    fee = fee_bps/1e4; slip = slip_bps/1e4

    cash = start_capital
    pos  = 0           # 0=flat, 1=long, -1=short
    shares = 0.0
    entry_px = None; entry_idx = None
    stop = np.nan
    took_tp1 = False

    equity_curve = []
    trades = []

    for i in range(len(df_)):
        close = float(df_.loc[i,"close"])
        # mark-to-market
        if pos == 1:
            equity_curve.append(cash + shares*close)
        elif pos == -1:
            equity_curve.append(cash + shares*(entry_px - close))  # short PnL benadering
        else:
            equity_curve.append(cash)

        if i == len(df_) - 1:
            break

        nxt_open = float(df_.loc[i+1,"open"])
        atr_next = float(df_.loc[i+1,"atr14"]) if pd.notna(df_.loc[i+1,"atr14"]) else float(df_.loc[i,"atr14"])

        # trailing stop
        if pos == 1:
            stop = max(stop, close - atr_mult*atr_next)
        elif pos == -1:
            stop = min(stop, close + atr_mult*atr_next)

        # R-now
        if pos == 1 and not np.isnan(stop):
            R_now = (close - entry_px) / max(entry_px - stop, 1e-9)
        elif pos == -1 and not np.isnan(stop):
            R_now = (entry_px - close) / max(stop - entry_px, 1e-9)
        else:
            R_now = 0.0

        def do_exit(exec_px, reason):
            nonlocal cash, shares, pos, stop, entry_px, entry_idx, took_tp1
            if pos == 1:
                cash += shares*exec_px
                cash -= abs(shares*exec_px)*fee
                ret = (exec_px/entry_px - 1.0)*100
            else:
                cash += shares*(entry_px - exec_px)
                cash -= abs(shares*exec_px)*fee
                ret = (1.0 - exec_px/entry_px)*100
            trades.append({
                "side": "LONG" if pos==1 else "SHORT",
                "entry_date": df_.loc[entry_idx,"date"],
                "entry_px": entry_px,
                "exit_date": df_.loc[i+1,"date"],
                "exit_px": exec_px,
                "ret_pct": ret,
                "reason": reason
            })
            shares=0.0; pos=0; stop=np.nan; entry_px=None; entry_idx=None; took_tp1=False

        # TP2
        if pos != 0 and R_now >= tp2_r:
            px = nxt_open*(1 - slip) if pos==1 else nxt_open*(1 + slip)
            do_exit(px, "TP2"); continue

        # TP1 partial
        if pos != 0 and (not took_tp1) and R_now >= tp1_r:
            px = nxt_open*(1 - slip) if pos==1 else nxt_open*(1 + slip)
            part = tp1_part/100.0
            close_sh = np.floor(shares*part)
            if close_sh >= 1:
                if pos == 1:
                    cash += close_sh*px
                    cash -= abs(close_sh*px)*fee
                else:
                    cash += close_sh*(entry_px - px)
                    cash -= abs(close_sh*px)*fee
                shares -= close_sh
            took_tp1 = True
            if move_be:
                stop = max(stop, entry_px) if pos==1 else min(stop, entry_px)

        # Exit (signal of gap door stop)
        exit_signal = (pos==1 and bool(df_.loc[i,"sell_sig"])) or (pos==-1 and bool(df_.loc[i,"buy_sig"]))
        stop_gap    = (pos==1 and not np.isnan(stop) and nxt_open < stop) or (pos==-1 and not np.isnan(stop) and nxt_open > stop)
        if pos != 0 and (exit_signal or stop_gap):
            px = nxt_open*(1 - slip) if pos==1 else nxt_open*(1 + slip)
            do_exit(px, "signal" if exit_signal and not stop_gap else "stop")
            continue

        # Entries (next open)
        if pos == 0:
            if allow_long and bool(df_.loc[i,"buy_sig"]):
                buy_px = nxt_open*(1 + slip)
                init_stop = buy_px - atr_mult*atr_next
                if risk_pct > 0:
                    risk_amt = cash*(risk_pct/100.0)
                    per_share = max(buy_px - init_stop, 1e-9)
                    shares = np.floor(risk_amt / per_share)
                    cost = shares*buy_px
                    if cost > cash:
                        shares = np.floor(cash/buy_px); cost = shares*buy_px
                else:
                    cost = cash; shares = cost/buy_px
                cash -= cost; cash -= cost*fee
                pos=1; entry_px=buy_px; entry_idx=i+1; stop=init_stop; took_tp1=False

            elif allow_short and bool(df_.loc[i,"short_sig"]):
                sell_px = nxt_open*(1 - slip)
                init_stop = sell_px + atr_mult*atr_next
                if risk_pct > 0:
                    risk_amt = cash*(risk_pct/100.0)
                    per_share = max(init_stop - sell_px, 1e-9)
                    shares = np.floor(risk_amt / per_share)
                else:
                    shares = cash/sell_px
                cash += shares*sell_px
                cash -= abs(shares*sell_px)*fee
                pos=-1; entry_px=sell_px; entry_idx=i+1; stop=init_stop; took_tp1=False

    eq_series = pd.Series(equity_curve, index=pd.to_datetime(df_["date"]), name="equity")
    trades_df = pd.DataFrame(trades)
    return eq_series, trades_df

# --- Combined, Long-only, Short-only curves ---
eq_combined, trades_df = run_backtest(
    d, START_CAP, ATR_MULT, FEE_BPS, SLIP_BPS, RISK_PCT,
    allow_long=True, allow_short=ALLOW_SHORT,
    tp1_r=TP1_R, tp2_r=TP2_R, tp1_part=TP1_PART, move_be=MOVE_BE
)
eq_long, _ = run_backtest(
    d, START_CAP, ATR_MULT, FEE_BPS, SLIP_BPS, RISK_PCT,
    allow_long=True, allow_short=False,
    tp1_r=TP1_R, tp2_r=TP2_R, tp1_part=TP1_PART, move_be=MOVE_BE
)
eq_short, _ = run_backtest(
    d, START_CAP, ATR_MULT, FEE_BPS, SLIP_BPS, RISK_PCT,
    allow_long=False, allow_short=True,
    tp1_r=TP1_R, tp2_r=TP2_R, tp1_part=TP1_PART, move_be=MOVE_BE
)

# Buy & Hold (equity) — ROBUUST
def buyhold_equity(df_: pd.DataFrame, start_cap: float) -> pd.Series:
    dfx = df_.copy().reset_index(drop=True)
    dfx["date"] = pd.to_datetime(dfx["date"])
    if len(dfx) < 2:
        s = pd.Series([start_cap], index=ddfx["date"])
        s.name = "bh_equity"
        return s
    buy_px = float(dfx.loc[1, "open"])  # start op eerstvolgende open
    shares = start_cap / buy_px
    eq = shares * dfx["close"]
    eq.index = dfx["date"]
    eq.name = "bh_equity"
    return eq

bh_equity = buyhold_equity(d, START_CAP)

# ====== ZORG DAT INDEXES DATETIME ZIJN VOOR METRICS ======
eq_combined.index = pd.to_datetime(eq_combined.index)
eq_long.index     = pd.to_datetime(eq_long.index)
eq_short.index    = pd.to_datetime(eq_short.index)
bh_equity.index   = pd.to_datetime(bh_equity.index)

# ----- metrics (ROBUUST) -----
def perf_metrics(eq: pd.Series, trading_days_per_year: int = 252):
    eq = eq.dropna()
    if len(eq) < 2:
        return {"CAGR": np.nan, "Vol": np.nan, "Sharpe": np.nan, "MaxDD": np.nan}

    idx = pd.to_datetime(eq.index)
    span_days = (idx[-1] - idx[0]) / np.timedelta64(1, "D")
    years = max(float(span_days) / 365.25, 1e-9)

    rets = eq.pct_change().fillna(0.0)
    mu = rets.mean()
    sigma = rets.std()
    vol_ann = sigma * np.sqrt(trading_days_per_year)
    sharpe = (mu * trading_days_per_year) / (vol_ann + 1e-12)

    total_ret = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
    cagr = (1.0 + total_ret) ** (1.0 / years) - 1.0 if years > 0 else np.nan

    roll_max = eq.cummax()
    dd = eq / roll_max - 1.0
    maxdd = dd.min()

    return {"CAGR": cagr, "Vol": vol_ann, "Sharpe": sharpe, "MaxDD": maxdd}

metrics = pd.DataFrame({
    "Buy&Hold": perf_metrics(bh_equity),
    "Combined": perf_metrics(eq_combined),
    "Long-only": perf_metrics(eq_long),
    "Short-only": perf_metrics(eq_short)
}).T

# =========================
# KPI’s
# =========================
last = d.iloc[-1]
up_regime_now   = (last["close"] > last["ema200"]) and (last["ema50"] > last["ema200"])
down_regime_now = (last["close"] < last["ema200"]) and (last["ema50"] < last["ema200"])
regime = "Bullish" if up_regime_now else "Bearish" if down_regime_now else "Neutraal"
ytd_full, pytd_full = ytd_return_full(df), pytd_return_full(df)
volatility = d['delta_pct'].std()
strategy_vs_bh = (eq_combined.iloc[-1]/eq_combined.iloc[0] - bh_equity.iloc[-1]/bh_equity.iloc[0]) * 100
num_buys, num_sells = d['buy_sig'].sum(), d['sell_sig'].sum()

k1,k2,k3,k4,k5,k6,k7,k8,k9 = st.columns(9)
k1.metric("Laatste close", f"{last['close']:.2f}")
k2.metric("Δ % (dag)", f"{(d['close'].pct_change().iloc[-1]*100):.2f}%")
k3.metric("VIX (close)", f"{last.get('vix_close', np.nan):.2f}")
k4.metric("Regime", regime)
k5.metric("YTD Return",  f"{ytd_full:.2f}%" if ytd_full is not None else "—")
k6.metric("PYTD Return", f"{pytd_full:.2f}%" if pytd_full is not None else "—")
k7.metric("Volatiliteit (std Δ%)", f"{volatility:.2f}%")
k8.metric("Strategy vs B&H", f"{strategy_vs_bh:.1f}%")
k9.metric("Signals (buy/sell)", f"{int(num_buys)} / {int(num_sells)}")

# =========================
# Alerts (kort)
# =========================
alerts = []
if "vix_close" in d.columns and pd.notna(last.get("vix_close", np.nan)):
    if last["vix_close"] > DEFAULTS["vix_high"]:
        alerts.append(("red","VIX hoog", f"VIX {last['vix_close']:.1f} > {DEFAULTS['vix_high']}"))
    elif last["vix_close"] < DEFAULTS["vix_low"]:
        alerts.append(("blue","VIX laag", f"VIX {last['vix_close']:.1f} < {DEFAULTS['vix_low']}"))
if up_regime_now and (last["adx14"] > DEFAULTS["adx_threshold"]):
    alerts.append(("green","Trend ↑ (sterk)", f"ADX {last['adx14']:.1f}"))
elif down_regime_now and (last["adx14"] > DEFAULTS["adx_threshold"]):
    alerts.append(("red","Trend ↓ (sterk)", f"ADX {last['adx14']:.1f}"))
elif last["adx14"] <= DEFAULTS["adx_threshold"]:
    alerts.append(("gray","Trend zwak", f"ADX {last['adx14']:.1f}"))
if crossed_up(d["macd_hist"]).iloc[-1]:
    alerts.append(("green","Momentum draait ↑","MACD-hist cross ↑ 0"))
elif crossed_down(d["macd_hist"]).iloc[-1]:
    alerts.append(("red","Momentum draait ↓","MACD-hist cross ↓ 0"))

def badge(color, text):
    colors = {"green":"#00A65A","red":"#D55E00","orange":"#E69F00","blue":"#1f77b4","gray":"#6c757d"}
    return f"""<span style="background:{colors[color]};color:white;padding:2px 8px;border-radius:12px;font-size:0.9rem;">{text}</span>"""

with st.expander("⚠️ Alerts & Signal status", expanded=True):
    if not alerts:
        st.success("Geen alerts op dit moment.")
    else:
        for color, title, msg in alerts:
            st.markdown(f"{badge(color, title)} &nbsp; {msg}", unsafe_allow_html=True)

# =========================
# Paneel 1 – HA + Supertrend + Donchian (+ VIX) + GELE WINDOWS
# =========================
fig1 = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]],
                     subplot_titles=["S&P 500 Heikin-Ashi + Supertrend (10,1) + Donchian" + (" + VIX (2e y-as)" if show_vix else "")])

# Gele balken eerst (zodat traces erboven liggen)
if highlight_on and win_list:
    for (s, e) in win_list:
        fig1.add_vrect(x0=s, x1=e, fillcolor="rgba(255,215,0,0.18)", line_width=0, layer="below")

fig1.add_trace(go.Candlestick(x=d["date"], open=d["ha_open"], high=d["ha_high"], low=d["ha_low"], close=d["ha_close"],
                              name="SPX (Heikin-Ashi)"), row=1, col=1, secondary_y=False)
fig1.add_trace(go.Scatter(x=d["date"], y=d["dc_high"], mode="lines",
                          line=dict(dash="dot", width=2), name="DC High"),
              row=1, col=1, secondary_y=False)
fig1.add_trace(go.Scatter(x=d["date"], y=d["dc_low"], mode="lines",
                          line=dict(dash="dot", width=2), name="DC Low"),
              row=1, col=1, secondary_y=False)
st_up = d["st_line"].where(d["st_trend"]==1); st_dn = d["st_line"].where(d["st_trend"]==-1)
fig1.add_trace(go.Scatter(x=d["date"], y=st_up, mode="lines", line=dict(width=3, color="green"), name="Supertrend ↑ (10,1)"),
              row=1, col=1, secondary_y=False)
fig1.add_trace(go.Scatter(x=d["date"], y=st_dn, mode="lines", line=dict(width=3, color="red"), name="Supertrend ↓ (10,1)"),
              row=1, col=1, secondary_y=False)
if show_vix and ("vix_close" in d.columns and d["vix_close"].notna().any()):
    fig1.add_trace(go.Scatter(x=d["date"], y=d["vix_close"], mode="lines", name="VIX (sec. y)", line=dict(width=2)),
                  row=1, col=1, secondary_y=True)

fig1.update_layout(
    height=700,
    margin=dict(l=60, r=60, t=80, b=40),
    legend_orientation="h", legend_yanchor="top", legend_y=1.08, legend_x=0,
    yaxis=dict(title="Index (HA)", tickfont=dict(size=13)),
    yaxis2=dict(title="VIX", tickfont=dict(size=13))
)
fig1.update_xaxes(rangeslider_visible=False, tickfont=dict(size=13))
st.plotly_chart(fig1, use_container_width=True)

# =========================
# Δ-aggregatie
# =========================
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

# =========================
# Paneel 2–6: Δ, Close+EMA, MACD, ADX, RSI
# =========================
fig2 = make_subplots(
    rows=5, cols=1, shared_xaxes=True,
    subplot_titles=[
        f"{'Δ (%)' if delta_mode=='Δ %' else 'Δ (punten)'} — {agg_mode.lower()}{' — MA'+str(ma_window) if smooth_on else ''}",
        "Close + EMA(20/50/200) — regime, buy/sell",
        "MACD(12,26,9) — lijn/signal/hist",
        f"ADX(14) + DI± — drempel={DEFAULTS['adx_threshold']}",
        "RSI(14) — >70 overbought, <30 oversold"
    ],
    row_heights=[0.18, 0.32, 0.2, 0.2, 0.18],
    vertical_spacing=0.06
)

# (1) Δ bars
fig2.add_trace(go.Bar(x=delta_x, y=delta_series.values, name=delta_legend,
                      marker=dict(color=delta_colors), opacity=0.9), row=1, col=1)

# (2) Close + EMA + signals
fig2.add_trace(go.Scatter(x=d["date"], y=d["close"], mode="lines", name="Close", line=dict(width=2)), row=2, col=1)
fig2.add_trace(go.Scatter(x=d["date"], y=d["ema20"], mode="lines", name="EMA20", line=dict(width=2)), row=2, col=1)
fig2.add_trace(go.Scatter(x=d["date"], y=d["ema50"], mode="lines", name="EMA50", line=dict(width=2)), row=2, col=1)
fig2.add_trace(go.Scatter(x=d["date"], y=d["ema200"], mode="lines", name="EMA200", line=dict(width=2)), row=2, col=1)

buys  = d.loc[d["buy_sig"]]
sells = d.loc[d["sell_sig"]]
fig2.add_trace(go.Scatter(
    x=buys["date"], y=buys["close"], mode="markers", name="Buy",
    marker=dict(symbol="triangle-up", size=12, color="#00A65A", line=dict(width=1, color="black"))
), row=2, col=1)
fig2.add_trace(go.Scatter(
    x=sells["date"], y=sells["close"], mode="markers", name="Sell",
    marker=dict(symbol="triangle-down", size=12, color="#D55E00", line=dict(width=1, color="black"))
), row=2, col=1)

# (2) optioneel: grijze balkjes achteraf als referentie
if highlight_on and win_list:
    for (s, e) in win_list:
        fig2.add_vrect(x0=s, x1=e, fillcolor="rgba(255,215,0,0.12)", line_width=0, row=2, col=1)

# (3) MACD
fig2.add_trace(go.Scatter(x=d["date"], y=d["macd_line"],   mode="lines", name="MACD",   line=dict(width=2)), row=3, col=1)
fig2.add_trace(go.Scatter(x=d["date"], y=d["macd_signal"], mode="lines", name="Signal", line=dict(width=2)), row=3, col=1)
fig2.add_trace(go.Bar(x=d["date"], y=d["macd_hist"], name="Hist",
                      marker_color=np.where(d["macd_hist"]>=0, "rgba(16,150,24,0.6)", "rgba(219,64,82,0.6)")),
               row=3, col=1)
fig2.add_hline(y=0, line_dash="dot", row=3, col=1)

# (4) ADX + DI±
fig2.add_trace(go.Scatter(x=d["date"], y=d["adx14"],   mode="lines", name="ADX(14)", line=dict(width=2)), row=4, col=1)
fig2.add_trace(go.Scatter(x=d["date"], y=d["di_plus"], mode="lines", name="+DI", line=dict(width=2)),     row=4, col=1)
fig2.add_trace(go.Scatter(x=d["date"], y=d["di_minus"],mode="lines", name="−DI", line=dict(width=2)),     row=4, col=1)
fig2.add_hline(y=DEFAULTS["adx_threshold"], line_dash="dot", row=4, col=1)

# (5) RSI
fig2.add_trace(go.Scatter(x=d["date"], y=d["rsi14"], mode="lines", name="RSI(14)", line=dict(width=2)), row=5, col=1)
fig2.add_hline(y=DEFAULTS["rsi_ob"], line_dash="dash", line_color="red", row=5, col=1)
fig2.add_hline(y=DEFAULTS["rsi_os"], line_dash="dash", line_color="green", row=5, col=1)

# Leesbaarheid
fig2.update_layout(height=1550, margin=dict(l=60, r=60, t=70, b=50),
                   legend_orientation="h", legend_yanchor="top", legend_y=1.06, legend_x=0)
fig2.update_xaxes(rangeslider_visible=False, tickfont=dict(size=13))
for rr in range(1,6):
    fig2.update_yaxes(row=rr, col=1, tickfont=dict(size=13))
fig2.update_yaxes(title_text="Δ", row=1, col=1)
fig2.update_yaxes(title_text="Close/EMA", row=2, col=1)
fig2.update_yaxes(title_text="MACD", row=3, col=1)
fig2.update_yaxes(title_text="ADX / DI", row=4, col=1)
fig2.update_yaxes(title_text="RSI", row=5, col=1, range=[0,100])
st.plotly_chart(fig2, use_container_width=True)

# =========================
# Equity curves: B&H + Combined + Long-only + Short-only
# =========================
st.subheader("Equity Curves — % sinds startkapitaal")
def equity_to_pct(eq: pd.Series): return (eq/eq.iloc[0]-1)*100.0
fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=eq_combined.index, y=equity_to_pct(bh_equity), name="Buy & Hold", line=dict(width=3)))
fig3.add_trace(go.Scatter(x=eq_combined.index, y=equity_to_pct(eq_combined), name="Strategy (Combined)", line=dict(width=3, dash="dot")))
fig3.add_trace(go.Scatter(x=eq_combined.index, y=equity_to_pct(eq_long), name="Strategy (Long-only)", line=dict(width=2)))
fig3.add_trace(go.Scatter(x=eq_combined.index, y=equity_to_pct(eq_short), name="Strategy (Short-only)", line=dict(width=2, dash="dash")))
fig3.update_layout(
    height=540,
    margin=dict(l=60, r=60, t=50, b=50),
    yaxis=dict(title="% sinds start (equity)", tickfont=dict(size=13)),
    xaxis=dict(tickfont=dict(size=13)),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0)
)
st.plotly_chart(fig3, use_container_width=True)

# =========================
# Metrics tabel
# =========================
st.subheader("Performance-metrics")
show = pd.DataFrame({
    "CAGR": metrics["CAGR"].map(lambda v: "—" if pd.isna(v) else f"{v*100:,.2f}%"),
    "Vol (ann.)": metrics["Vol"].map(lambda v: "—" if pd.isna(v) else f"{v*100:,.2f}%"),
    "Sharpe": metrics["Sharpe"].map(lambda v: "—" if pd.isna(v) else f"{v:,.2f}"),
    "MaxDD": metrics["MaxDD"].map(lambda v: "—" if pd.isna(v) else f"{v*100:,.2f}%")
})
st.dataframe(show, use_container_width=True)

# =========================
# Rolling correlatie met VIX
# =========================
st.markdown("#### Rolling correlatie met VIX")
corr_df = d.copy().set_index("date")
spx_ret = corr_df["delta_pct"]
vix_series = (corr_df["vix_close"].pct_change()*100.0) if corr_vs=="% change" else corr_df["vix_close"]
corr_join = pd.concat([spx_ret.rename("spx"), vix_series.rename("vix")], axis=1).dropna()
rolling_corr = corr_join["spx"].rolling(corr_win).corr(corr_join["vix"])

fig_corr = go.Figure()
fig_corr.add_trace(go.Scatter(x=rolling_corr.index, y=rolling_corr.values, mode="lines", name="Rolling corr", line=dict(width=2)))
fig_corr.add_hline(y=0.0, line_dash="dot")
fig_corr.add_hrect(y0=-1, y1=-0.5, fillcolor="rgba(255,0,0,0.06)", line_width=0)
fig_corr.add_hrect(y0=0.5, y1=1,   fillcolor="rgba(0,128,0,0.06)", line_width=0)
fig_corr.update_layout(height=400, margin=dict(l=60,r=60,t=50,b=40),
                       yaxis=dict(range=[-1,1], title="corr", tickfont=dict(size=13)),
                       xaxis=dict(tickfont=dict(size=13)))
st.plotly_chart(fig_corr, use_container_width=True)

# =========================
# Heatmap
# =========================
heat_on = True if 'heat_on' not in globals() else heat_on  # safeguard
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
    heat.update_layout(height=480, margin=dict(l=60, r=60, t=50, b=50),
                       coloraxis=dict(colorscale="RdBu", cauto=True, colorbar_title=value_title),
                       xaxis=dict(title="Maand", tickfont=dict(size=13)),
                       yaxis=dict(title="Jaar", tickfont=dict(size=13)))
    st.plotly_chart(heat, use_container_width=True)

# =========================
# Histogrammen
# =========================
if 'hist_on' not in globals():
    hist_on = True
if hist_on:
    st.subheader("Histogram dagrendementen")
    col_a, col_b = st.columns(2)
    hist_df = d.dropna(subset=["delta_abs","delta_pct"]).copy()
    with col_a:
        fig_abs = go.Figure([go.Histogram(x=hist_df["delta_abs"], nbinsx=60)])
        fig_abs.update_layout(title="Δ abs (punten)", height=420, bargap=0.02,
                              margin=dict(l=50,r=50,t=60,b=50),
                              xaxis=dict(tickfont=dict(size=13)),
                              yaxis=dict(tickfont=dict(size=13)))
        st.plotly_chart(fig_abs, use_container_width=True)
    with col_b:
        fig_pct = go.Figure([go.Histogram(x=hist_df["delta_pct"], nbinsx=60)])
        fig_pct.update_layout(title="Δ %", height=420, bargap=0.02,
                              margin=dict(l=50,r=50,t=60,b=50),
                              xaxis=dict(tickfont=dict(size=13)),
                              yaxis=dict(tickfont=dict(size=13)))
        st.plotly_chart(fig_pct, use_container_width=True)

# =========================
# Options-proxy (dagelijks) — windows vs outside
# =========================
st.subheader("Options-proxy — PnL (windows vs outside)")

# Per dag volgende close: |Δ| en ATR
d["next_close"] = d["close"].shift(-1)
valid = d.dropna(subset=["next_close","atr14"]).copy()
valid["abs_move"] = (valid["next_close"] - valid["open"]).abs()

# Long Straddle proxy: |Δ| − cost×ATR
valid["straddle_pnl"] = valid["abs_move"] - (straddle_cost_atr * valid["atr14"])

# Short Strangle proxy:
# PnL = premium×ATR − max(0, |Δ| − width×ATR)
overflow = (valid["abs_move"] - strangle_width_atr * valid["atr14"]).clip(lower=0.0)
valid["strangle_pnl"] = (strangle_prem_atr * valid["atr14"]) - overflow

valid["in_window"] = in_any_window(valid["date"])

def _sumstats(df_, col):
    x = df_[col]
    return pd.Series({
        "N dagen": len(x),
        "Hit % (>0)": (x > 0).mean()*100 if len(x) else np.nan,
        "Gem. PnL": x.mean() if len(x) else np.nan,
        "Totaal PnL": x.sum() if len(x) else np.nan
    })

sum_win = pd.concat([
    _sumstats(valid[valid["in_window"]], "straddle_pnl").rename("Straddle (in windows)"),
    _sumstats(valid[~valid["in_window"]], "straddle_pnl").rename("Straddle (outside)"),
    _sumstats(valid[valid["in_window"]], "strangle_pnl").rename("Strangle (in windows)"),
    _sumstats(valid[~valid["in_window"]], "strangle_pnl").rename("Strangle (outside)")
], axis=1).T

# Netjes tonen
def _fmt(x, pct=False):
    if pd.isna(x): return "—"
    return f"{x:,.2f}%" if pct else f"{x:,.2f}"

show_opt = pd.DataFrame({
    "N": sum_win["N dagen"].map(lambda v: f"{int(v)}"),
    "Hit %": sum_win["Hit % (>0)"].map(lambda v: _fmt(v, pct=True)),
    "Gem. PnL (pts)": sum_win["Gem. PnL"].map(_fmt),
    "Totaal PnL (pts)": sum_win["Totaal PnL"].map(_fmt)
})
st.dataframe(show_opt, use_container_width=True)

# =========================
# Trades (geavanceerde backtest)
# =========================
st.subheader("Trades (geavanceerde backtest)")
if len(trades_df):
    show_tr = trades_df.copy()
    show_tr["ret_pct"] = show_tr["ret_pct"].map(lambda x: f"{x:,.2f}%")
    st.dataframe(show_tr[["side","entry_date","entry_px","exit_date","exit_px","ret_pct","reason"]],
                 use_container_width=True)
else:
    st.info("Geen trades in de geselecteerde periode.")
