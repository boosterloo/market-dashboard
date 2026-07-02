# =========================
try:
    if not bq_ping():
        st.error("Geen BigQuery-verbinding.")
        st.stop()
except Exception as e:
    st.error("Geen BigQuery-verbinding.")
    st.caption(f"Details: {e}")
    st.stop()

# =========================
# Data
# =========================
@st.cache_data(ttl=1800, show_spinner=False)
def load_aex():
    return run_query(f"SELECT * FROM `{AEX_VIEW}` ORDER BY date")


with st.spinner("AEX data laden..."):
    df = load_aex()

if df.empty:
    st.warning("Geen data in view.")
    st.stop()

df["date"] = pd.to_datetime(df["date"])
df = df[df["date"].dt.weekday < 5].copy()

for c in ["open", "high", "low", "close", "vix_close", "delta_abs", "delta_pct"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

if "delta_abs" not in df.columns or df["delta_abs"].isna().all():
    df["delta_abs"] = df["close"].diff()

if "delta_pct" not in df.columns or df["delta_pct"].isna().all():
    df["delta_pct"] = df["close"].pct_change() * 100.0

# =========================
# Defaults
# =========================
DEFAULTS = {
    "ema_spans": [20, 50, 200],
    "macd": (12, 26, 9),
    "rsi_period": 14,
    "adx_length": 14,
    "donchian_n": 20,
    "corr_win_default": 20,
    "rsi_dyn_win": 252,
}

# =========================
# Helpers
# =========================
def ema(s: pd.Series, span: int):
    return s.ewm(span=span, adjust=False, min_periods=span).mean()


def atr_rma(high, low, close, length: int):
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / length, adjust=False).mean()


def donchian(d: pd.DataFrame, n=20):
    return (
        d["high"].rolling(n, min_periods=n).max(),
        d["low"].rolling(n, min_periods=n).min(),
    )


def macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = series.ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def adx(df_: pd.DataFrame, length: int = 14):
    high, low, close = df_["high"], df_["low"], df_["close"]

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr1 = (high - low).abs()
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1 / length, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df_.index).ewm(alpha=1 / length, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=df_.index).ewm(alpha=1 / length, adjust=False).mean() / atr

    denom = (plus_di + minus_di).replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / denom
    adx_val = dx.ewm(alpha=1 / length, adjust=False).mean()

    return plus_di, minus_di, adx_val


def rma(x: pd.Series, length: int):
    return x.ewm(alpha=1 / length, adjust=False).mean()


def rsi_wilder(close: pd.Series, length: int = 14):
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = rma(up, length)
    roll_down = rma(down, length)
    rs = roll_up / roll_down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def rolling_percentile(s: pd.Series, q: float, win: int = 252):
    return s.rolling(win, min_periods=max(20, int(win * 0.6))).quantile(q)


def zscore(series: pd.Series, win: int):
    mean = series.rolling(win, min_periods=max(20, int(win * 0.6))).mean()
    std = series.rolling(win, min_periods=max(20, int(win * 0.6))).std()
    return (series - mean) / std.replace(0, np.nan)


def slope_pct(series: pd.Series, lookback: int = 10):
    return ((series / series.shift(lookback)) - 1.0) * 100.0


def safe_last(series: pd.Series):
    x = series.dropna()
    if len(x) == 0:
        return np.nan
    return x.iloc[-1]


def fmt_num(x, nd=2, suffix=""):
    if pd.isna(x):
        return "-"
    return f"{x:.{nd}f}{suffix}"


def ytd_return_full(full_df: pd.DataFrame):
    sub = full_df.dropna(subset=["date", "close"]).copy()
    if sub.empty:
        return None
    max_d = sub["date"].max()
    start = pd.Timestamp(date(max_d.year, 1, 1))
    sub = sub[sub["date"] >= start]
    return (sub["close"].iloc[-1] / sub["close"].iloc[0] - 1) * 100 if len(sub) >= 2 else None


def pytd_return_full(full_df: pd.DataFrame):
    sub = full_df.dropna(subset=["date", "close"]).copy()
    if sub.empty:
        return None
    max_d = sub["date"].max()
    prev_year = max_d.year - 1
    start = pd.Timestamp(date(prev_year, 1, 1))
    try:
        end = max_d.replace(year=prev_year)
    except Exception:
        end = pd.Timestamp(date(prev_year, 12, 31))
    sub = sub[(sub["date"] >= start) & (sub["date"] <= end)]
    return (sub["close"].iloc[-1] / sub["close"].iloc[0] - 1) * 100 if len(sub) >= 2 else None


def heikin_ashi(src: pd.DataFrame):
    ha = src.copy()
    ha["ha_close"] = (ha["open"] + ha["high"] + ha["low"] + ha["close"]) / 4.0

    ha_open = pd.Series(index=ha.index, dtype=float)
    first_valid_idx = ha[["open", "close"]].dropna().index.min()

    if pd.isna(first_valid_idx):
        ha["ha_open"] = np.nan
        ha["ha_high"] = np.nan
        ha["ha_low"] = np.nan
        return ha[["ha_open", "ha_high", "ha_low", "ha_close"]]

    ha_open.loc[first_valid_idx] = (ha.loc[first_valid_idx, "open"] + ha.loc[first_valid_idx, "close"]) / 2.0
    start_pos = ha.index.get_loc(first_valid_idx)

    for i in range(start_pos + 1, len(ha)):
        prev_idx = ha.index[i - 1]
        cur_idx = ha.index[i]
        if pd.notna(ha_open.loc[prev_idx]) and pd.notna(ha.loc[prev_idx, "ha_close"]):
            ha_open.loc[cur_idx] = (ha_open.loc[prev_idx] + ha.loc[prev_idx, "ha_close"]) / 2.0
        else:
            ha_open.loc[cur_idx] = np.nan

    ha["ha_open"] = ha_open
    ha["ha_high"] = pd.concat([ha["high"], ha["ha_open"], ha["ha_close"]], axis=1).max(axis=1)
    ha["ha_low"] = pd.concat([ha["low"], ha["ha_open"], ha["ha_close"]], axis=1).min(axis=1)

    return ha[["ha_open", "ha_high", "ha_low", "ha_close"]]


# =========================
# Indicator calculation
# =========================
@st.cache_data(ttl=1800)
def compute_indicators(full_df: pd.DataFrame):
    dfx = full_df.copy()

    for span in DEFAULTS["ema_spans"]:
        dfx[f"ema{span}"] = ema(dfx["close"], span)

    dfx["atr14"] = atr_rma(dfx["high"], dfx["low"], dfx["close"], 14)
    dfx["macd_line"], dfx["macd_signal"], dfx["macd_hist"] = macd(dfx["close"], *DEFAULTS["macd"])
    dfx["di_plus"], dfx["di_minus"], dfx["adx14"] = adx(dfx, DEFAULTS["adx_length"])

    dfx["rsi14"] = rsi_wilder(dfx["close"], DEFAULTS["rsi_period"])
    dfx["rsi14_s"] = dfx["rsi14"].ewm(span=5, adjust=False).mean()
    dfx["rsi_dyn_hi"] = rolling_percentile(dfx["rsi14"], 0.80, DEFAULTS["rsi_dyn_win"])
    dfx["rsi_dyn_lo"] = rolling_percentile(dfx["rsi14"], 0.20, DEFAULTS["rsi_dyn_win"])

    dfx["dc_high"], dfx["dc_low"] = donchian(dfx, DEFAULTS["donchian_n"])
    dfx["ema20_slope_10"] = slope_pct(dfx["ema20"], 10)
    dfx["ema50_slope_10"] = slope_pct(dfx["ema50"], 10)

    dfx["stretch_ema20_atr"] = (dfx["close"] - dfx["ema20"]) / dfx["atr14"].replace(0, np.nan)
    dfx["stretch_ema50_atr"] = (dfx["close"] - dfx["ema50"]) / dfx["atr14"].replace(0, np.nan)
    dfx["z20"] = zscore(dfx["close"], 20)
    dfx["z50"] = zscore(dfx["close"], 50)
    dfx["atr_pct_close"] = (dfx["atr14"] / dfx["close"]) * 100.0

    dfx["rv_10"] = dfx["close"].pct_change().rolling(10).std() * np.sqrt(252) * 100.0
    dfx["rv_20"] = dfx["close"].pct_change().rolling(20).std() * np.sqrt(252) * 100.0

    if "vix_close" in dfx.columns:
        vix_ma20 = dfx["vix_close"].rolling(20, min_periods=20).mean()
        vix_sd20 = dfx["vix_close"].rolling(20, min_periods=20).std()
        dfx["vix_z"] = (dfx["vix_close"] - vix_ma20) / vix_sd20.replace(0, np.nan)
        dfx["vix_change_5d"] = dfx["vix_close"].pct_change(5) * 100.0
        dfx["vix_rv20_spread"] = dfx["vix_close"] - dfx["rv_20"]
    else:
        dfx["vix_close"] = np.nan
        dfx["vix_z"] = np.nan