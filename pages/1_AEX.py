# pages/2_AEX.py
# AEX - Market State Dashboard
# Focus op regime, trend, momentum, uitputting en forward returns.

import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =========================
# BigQuery helpers
# =========================
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
st.set_page_config(page_title="AEX - Market State Dashboard", layout="wide")
st.title("AEX - Market State Dashboard")

AEX_VIEW = st.secrets.get("tables", {}).get(
    "aex_view", "nth-pier-468314-p7.marketdata.aex_with_vix_v"
)

# =========================
# Health
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
        dfx["vix_change_5d"] = np.nan
        dfx["vix_rv20_spread"] = np.nan

    ha = heikin_ashi(dfx)
    dfx[["ha_open", "ha_high", "ha_low", "ha_close"]] = ha[["ha_open", "ha_high", "ha_low", "ha_close"]]

    dfx["delta_abs"] = dfx["delta_abs"].fillna(0)
    dfx["delta_pct"] = dfx["delta_pct"].fillna(0)

    return dfx


df = compute_indicators(df)

# =========================
# Periode + sidebar
# =========================
min_d = df["date"].min().date()
max_d = df["date"].max().date()
default_start = max((df["date"].max() - timedelta(days=365)).date(), min_d)

c1, c2, c3 = st.columns([0.08, 0.84, 0.08])
with c2:
    start_date, end_date = st.slider(
        "Periode",
        min_value=min_d,
        max_value=max_d,
        value=(default_start, max_d),
        format="YYYY-MM-DD",
    )

with st.sidebar:
    st.markdown("### Instellingen")

    st.markdown("#### Grafieken")
    show_vix = st.toggle("Toon VIX in hoofdgrafiek", value=True)
    show_donchian = st.toggle("Toon Donchian", value=True)
    show_regime_shading = st.toggle("Toon regime shading", value=True)

    st.divider()
    st.markdown("#### Delta")
    delta_mode = st.radio("Weergave", ["Delta punten", "Delta %"], index=0)
    agg_mode = st.selectbox("Aggregatie", ["Dagelijks", "Wekelijks", "Maandelijks"], index=0)
    smooth_on = st.checkbox("Smoothing MA", value=False)
    ma_window = st.slider("MA-window", 2, 60, 5, step=1, disabled=not smooth_on)

    st.divider()
    st.markdown("#### Correlatie")
    corr_vs = st.radio("Rolling correlatie vs VIX", ["% change", "level"], index=0)
    corr_win = st.slider("Correlatie-window", 5, 90, DEFAULTS["corr_win_default"], step=1)

    st.divider()
    st.markdown("#### Forward returns")
    future_horizons = st.multiselect(
        "Horizons",
        options=[1, 3, 5, 10, 20],
        default=[1, 3, 5, 10],
    )

# =========================
# Filter subset
# =========================
d = df[
    (df["date"].dt.date >= start_date) &
    (df["date"].dt.date <= end_date)
].reset_index(drop=True).copy()

if d.empty or len(d) < 30:
    st.warning("Te weinig data in de gekozen periode.")
    st.stop()

# =========================
# State engine
# =========================
def classify_trend(row):
    if pd.isna(row["ema20"]) or pd.isna(row["ema50"]) or pd.isna(row["ema200"]):
        return "Onvoldoende data"

    c = row["close"]
    e20 = row["ema20"]
    e50 = row["ema50"]
    e200 = row["ema200"]
    s20 = row["ema20_slope_10"]
    s50 = row["ema50_slope_10"]

    if c > e20 > e50 > e200 and s20 > 0 and s50 > 0:
        return "Strong Bull"
    if c > e50 > e200:
        return "Bull"
    if c < e20 < e50 < e200 and s20 < 0 and s50 < 0:
        return "Strong Bear"
    if c < e50 < e200:
        return "Bear"
    return "Neutral"


def classify_trend_strength(row):
    adx_value = row["adx14"]
    spread1 = abs((row["ema20"] / row["ema50"] - 1) * 100) if pd.notna(row["ema20"]) and pd.notna(row["ema50"]) and row["ema50"] != 0 else np.nan
    spread2 = abs((row["ema50"] / row["ema200"] - 1) * 100) if pd.notna(row["ema50"]) and pd.notna(row["ema200"]) and row["ema200"] != 0 else np.nan
    score = 0

    if pd.notna(adx_value):
        if adx_value >= 30:
            score += 2
        elif adx_value >= 20:
            score += 1

    if pd.notna(spread1) and spread1 > 1.0:
        score += 1
    if pd.notna(spread2) and spread2 > 2.0:
        score += 1

    if score >= 4:
        return "Sterk"
    if score >= 2:
        return "Gemiddeld"
    return "Zwak"


def classify_momentum(row):
    hist = row["macd_hist"]
    hist_prev = row["macd_hist_prev"]
    rsi = row["rsi14_s"]

    if pd.isna(hist) or pd.isna(hist_prev) or pd.isna(rsi):
        return "Onvoldoende data"

    if hist > 0 and hist > hist_prev and rsi > 50:
        return "Versnellend omhoog"
    if hist > 0 and hist <= hist_prev and rsi > 50:
        return "Positief maar afzwakkend"
    if hist < 0 and hist < hist_prev and rsi < 50:
        return "Versnellend omlaag"
    if hist < 0 and hist >= hist_prev and rsi < 50:
        return "Negatief maar afzwakkend"
    return "Neutraal"


def classify_exhaustion(row):
    rsi = row["rsi14_s"]
    dyn_hi = row["rsi_dyn_hi"]
    dyn_lo = row["rsi_dyn_lo"]
    stretch = abs(row["stretch_ema20_atr"])
    z20 = abs(row["z20"])

    flags = 0

    if pd.notna(rsi) and pd.notna(dyn_hi) and rsi >= dyn_hi:
        flags += 1
    if pd.notna(rsi) and pd.notna(dyn_lo) and rsi <= dyn_lo:
        flags += 1
    if pd.notna(stretch) and stretch >= 1.75:
        flags += 1
    if pd.notna(z20) and z20 >= 2.0:
        flags += 1

    if flags >= 3:
        return "Hoog"
    if flags >= 1:
        return "Oplopend"
    return "Laag"


def classify_vol_regime(row):
    vix = row["vix_close"]
    vix_z = row["vix_z"]
    atr_pct = row["atr_pct_close"]

    if pd.isna(vix) and pd.isna(atr_pct):
        return "Onvoldoende data"

    score = 0
    if pd.notna(vix):
        if vix >= 35:
            score += 3
        elif vix >= 25:
            score += 2
        elif vix >= 18:
            score += 1

    if pd.notna(vix_z):
        if vix_z >= 2:
            score += 2
        elif vix_z >= 1:
            score += 1

    if pd.notna(atr_pct):
        if atr_pct >= 2.5:
            score += 2
        elif atr_pct >= 1.5:
            score += 1

    if score >= 5:
        return "Panic"
    if score >= 3:
        return "Stress"
    if score >= 1:
        return "Alert"
    return "Rustig"


def build_summary(row):
    trend = classify_trend(row)
    strength = classify_trend_strength(row)
    momentum = classify_momentum(row)
    exhaustion = classify_exhaustion(row)
    vol = classify_vol_regime(row)

    parts = []

    if trend in ["Strong Bull", "Bull"]:
        parts.append(f"Trend is {trend.lower()} met {strength.lower()} trendkracht")
    elif trend in ["Strong Bear", "Bear"]:
        parts.append(f"Trend is {trend.lower()} met {strength.lower()} trendkracht")
    else:
        parts.append("Markt zit meer in een neutrale of overgangsfase")

    if momentum != "Onvoldoende data":
        parts.append(f"momentum oogt {momentum.lower()}")

    if exhaustion == "Hoog":
        parts.append("uitputting is hoog")
    elif exhaustion == "Oplopend":
        parts.append("uitputting loopt op")
    else:
        parts.append("uitputting blijft beperkt")

    if vol != "Onvoldoende data":
        parts.append(f"volatiliteitsregime staat op {vol.lower()}")

    return ". ".join(parts) + "."


d["macd_hist_prev"] = d["macd_hist"].shift(1)
d["trend_label"] = d.apply(classify_trend, axis=1)
d["trend_strength"] = d.apply(classify_trend_strength, axis=1)
d["momentum_label"] = d.apply(classify_momentum, axis=1)
d["exhaustion_label"] = d.apply(classify_exhaustion, axis=1)
d["vol_regime"] = d.apply(classify_vol_regime, axis=1)

state_ready = d.dropna(subset=["close"]).copy()
last_state_row = state_ready.iloc[-1] if not state_ready.empty else d.iloc[-1]

# =========================
# Regime shading helper
# =========================
def add_regime_spans(fig, data, row=1, col=1):
    colors = {
        "Strong Bull": "rgba(0,140,70,0.08)",
        "Bull": "rgba(0,180,90,0.05)",
        "Neutral": "rgba(130,130,130,0.05)",
        "Bear": "rgba(220,90,0,0.05)",
        "Strong Bear": "rgba(200,0,0,0.08)",
    }
    lbl = data["trend_label"].fillna("Neutral")
    if lbl.empty:
        return

    start_idx = 0
    current = lbl.iloc[0]

    for i in range(1, len(lbl)):
        if lbl.iloc[i] != current:
            fig.add_vrect(
                x0=data["date"].iloc[start_idx],
                x1=data["date"].iloc[i - 1],
                fillcolor=colors.get(current, "rgba(130,130,130,0.04)"),
                line_width=0,
                row=row,
                col=col,
                layer="below",
            )
            start_idx = i
            current = lbl.iloc[i]

    fig.add_vrect(
        x0=data["date"].iloc[start_idx],
        x1=data["date"].iloc[len(lbl) - 1],
        fillcolor=colors.get(current, "rgba(130,130,130,0.04)"),
        line_width=0,
        row=row,
        col=col,
        layer="below",
    )


# =========================
# Top diagnostics
# =========================
summary_text = build_summary(last_state_row)
ytd_full = ytd_return_full(df)

last_close_val = safe_last(d["close"])
last_delta_day = safe_last(d["close"].pct_change() * 100.0)
last_vix_val = safe_last(d["vix_close"]) if "vix_close" in d.columns else np.nan
last_rsi_val = safe_last(d["rsi14_s"])
last_adx_val = safe_last(d["adx14"])
last_macd_hist_val = safe_last(d["macd_hist"])
last_stretch_val = safe_last(d["stretch_ema20_atr"])
last_z20_val = safe_last(d["z20"])
last_atr_pct_val = safe_last(d["atr_pct_close"])

k1, k2, k3, k4, k5, k6, k7, k8, k9 = st.columns(9)
k1.metric("Laatste close", fmt_num(last_close_val))
k2.metric("Delta % dag", fmt_num(last_delta_day, 2, "%"))
k3.metric("Trend", last_state_row["trend_label"])
k4.metric("Trendkracht", last_state_row["trend_strength"])
k5.metric("Momentum", last_state_row["momentum_label"])
k6.metric("Uitputting", last_state_row["exhaustion_label"])
k7.metric("Vol-regime", last_state_row["vol_regime"])
k8.metric("YTD", fmt_num(ytd_full, 2, "%") if ytd_full is not None else "-")
k9.metric("VIX", fmt_num(last_vix_val))

st.info(summary_text)

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("RSI(14) smoothed", fmt_num(last_rsi_val))
m2.metric("ADX(14)", fmt_num(last_adx_val))
m3.metric("MACD hist", fmt_num(last_macd_hist_val, 3))
m4.metric("Stretch vs EMA20", fmt_num(last_stretch_val, 2, " ATR"))
m5.metric("Z-score 20d", fmt_num(last_z20_val, 2))
m6.metric("ATR % close", fmt_num(last_atr_pct_val, 2, "%"))

# =========================
# Main chart: Heikin Ashi
# =========================
fig1 = make_subplots(
    rows=1,
    cols=1,
    specs=[[{"secondary_y": True}]],
    subplot_titles=["AEX Heikin Ashi + EMA20/50/200" + (" + VIX" if show_vix else "")]
)

if show_regime_shading:
    add_regime_spans(fig1, d, row=1, col=1)

ha_plot = d.dropna(subset=["ha_open", "ha_high", "ha_low", "ha_close"]).copy()

fig1.add_trace(
    go.Candlestick(
        x=ha_plot["date"],
        open=ha_plot["ha_open"],
        high=ha_plot["ha_high"],
        low=ha_plot["ha_low"],
        close=ha_plot["ha_close"],
        name="Heikin Ashi"
    ),
    row=1, col=1, secondary_y=False
)

for span in DEFAULTS["ema_spans"]:
    fig1.add_trace(
        go.Scatter(x=d["date"], y=d[f"ema{span}"], mode="lines", name=f"EMA{span}", line=dict(width=2)),
        row=1, col=1, secondary_y=False
    )

if show_donchian:
    fig1.add_trace(
        go.Scatter(x=d["date"], y=d["dc_high"], mode="lines", name="DC High", line=dict(width=1, dash="dot")),
        row=1, col=1, secondary_y=False
    )
    fig1.add_trace(
        go.Scatter(x=d["date"], y=d["dc_low"], mode="lines", name="DC Low", line=dict(width=1, dash="dot")),
        row=1, col=1, secondary_y=False
    )

if show_vix and "vix_close" in d.columns and d["vix_close"].notna().any():
    fig1.add_trace(
        go.Scatter(x=d["date"], y=d["vix_close"], mode="lines", name="VIX", line=dict(width=2)),
        row=1, col=1, secondary_y=True
    )

fig1.update_layout(
    height=700,
    margin=dict(l=60, r=60, t=80, b=40),
    legend_orientation="h",
    legend_yanchor="top",
    legend_y=1.08,
    legend_x=0,
    xaxis_rangeslider_visible=False,
)
fig1.update_xaxes(tickfont=dict(size=13))
fig1.update_yaxes(title_text="AEX", row=1, col=1, tickfont=dict(size=13), secondary_y=False)
fig1.update_yaxes(title_text="VIX", row=1, col=1, tickfont=dict(size=13), secondary_y=True)
st.plotly_chart(fig1, use_container_width=True)

# =========================
# Momentum dashboard
# =========================
fig2 = make_subplots(
    rows=4,
    cols=1,
    shared_xaxes=True,
    subplot_titles=[
        "RSI(14) Wilder + dynamische zones",
        "MACD(12,26,9)",
        "ADX(14) + DI",
        "Stretch vs EMA20 in ATR",
    ],
    row_heights=[0.24, 0.24, 0.24, 0.24],
    vertical_spacing=0.06,
)

fig2.add_trace(go.Scatter(x=d["date"], y=d["rsi14_s"], mode="lines", name="RSI smoothed", line=dict(width=2)), row=1, col=1)
fig2.add_trace(go.Scatter(x=d["date"], y=d["rsi_dyn_hi"], mode="lines", name="RSI dyn-hi", line=dict(width=1, dash="dot")), row=1, col=1)
fig2.add_trace(go.Scatter(x=d["date"], y=d["rsi_dyn_lo"], mode="lines", name="RSI dyn-lo", line=dict(width=1, dash="dot")), row=1, col=1)
fig2.add_hline(y=70, line_dash="dash", line_color="red", row=1, col=1)
fig2.add_hline(y=50, line_dash="dot", row=1, col=1)
fig2.add_hline(y=30, line_dash="dash", line_color="green", row=1, col=1)
fig2.add_hrect(y0=70, y1=100, fillcolor="rgba(255,0,0,0.05)", line_width=0, row=1, col=1)
fig2.add_hrect(y0=0, y1=30, fillcolor="rgba(0,128,0,0.05)", line_width=0, row=1, col=1)

fig2.add_trace(go.Scatter(x=d["date"], y=d["macd_line"], mode="lines", name="MACD", line=dict(width=2)), row=2, col=1)
fig2.add_trace(go.Scatter(x=d["date"], y=d["macd_signal"], mode="lines", name="Signal", line=dict(width=2)), row=2, col=1)
fig2.add_trace(
    go.Bar(
        x=d["date"],
        y=d["macd_hist"],
        name="Hist",
        marker_color=np.where(d["macd_hist"] >= 0, "rgba(16,150,24,0.6)", "rgba(219,64,82,0.6)"),
    ),
    row=2,
    col=1,
)
fig2.add_hline(y=0, line_dash="dot", row=2, col=1)

fig2.add_trace(go.Scatter(x=d["date"], y=d["adx14"], mode="lines", name="ADX", line=dict(width=2)), row=3, col=1)
fig2.add_trace(go.Scatter(x=d["date"], y=d["di_plus"], mode="lines", name="+DI", line=dict(width=2)), row=3, col=1)
fig2.add_trace(go.Scatter(x=d["date"], y=d["di_minus"], mode="lines", name="-DI", line=dict(width=2)), row=3, col=1)
fig2.add_hline(y=20, line_dash="dot", row=3, col=1)
fig2.add_hline(y=30, line_dash="dot", row=3, col=1)

fig2.add_trace(go.Scatter(x=d["date"], y=d["stretch_ema20_atr"], mode="lines", name="Stretch EMA20", line=dict(width=2)), row=4, col=1)
fig2.add_hline(y=0, line_dash="dot", row=4, col=1)
fig2.add_hline(y=1.5, line_dash="dash", row=4, col=1)
fig2.add_hline(y=-1.5, line_dash="dash", row=4, col=1)
fig2.add_hrect(y0=1.5, y1=4, fillcolor="rgba(255,0,0,0.05)", line_width=0, row=4, col=1)
fig2.add_hrect(y0=-4, y1=-1.5, fillcolor="rgba(0,128,0,0.05)", line_width=0, row=4, col=1)

fig2.update_layout(
    height=1350,
    margin=dict(l=60, r=60, t=70, b=50),
    legend_orientation="h",
    legend_yanchor="top",
    legend_y=1.05,
    legend_x=0,
)
fig2.update_xaxes(rangeslider_visible=False, tickfont=dict(size=13))
for rr in range(1, 5):
    fig2.update_yaxes(row=rr, col=1, tickfont=dict(size=13))
fig2.update_yaxes(title_text="RSI", row=1, col=1, range=[0, 100])
fig2.update_yaxes(title_text="MACD", row=2, col=1)
fig2.update_yaxes(title_text="ADX / DI", row=3, col=1)
fig2.update_yaxes(title_text="ATR stretch", row=4, col=1)
st.plotly_chart(fig2, use_container_width=True)

# =========================
# Delta aggregation
# =========================
def aggregate_delta(_df: pd.DataFrame, mode: str, how: str) -> pd.Series:
    t = _df.copy().set_index("date")

    if how == "Dagelijks":
        series = t["delta_pct"].dropna() if mode == "Delta %" else t["delta_abs"].dropna()
        return series.reindex(t.index, fill_value=0)

    rule = "W-FRI" if how == "Wekelijks" else "ME"

    if mode == "Delta %":
        res = t["delta_pct"].groupby(pd.Grouper(freq=rule)).apply(
            lambda g: (np.prod((g.dropna() / 100.0 + 1.0)) - 1.0) * 100.0 if len(g.dropna()) else np.nan
        )
        return res.fillna(0)
    return t["delta_abs"].groupby(pd.Grouper(freq=rule)).sum(min_count=1).fillna(0)


delta_series = aggregate_delta(d, delta_mode, agg_mode)
if smooth_on:
    delta_series = delta_series.rolling(ma_window, min_periods=1).mean()

delta_x = delta_series.index
delta_legend = "Delta (%)" if delta_mode == "Delta %" else "Delta (punten)"
if smooth_on:
    delta_legend += f" MA{ma_window}"

delta_colors = np.where(delta_series.values >= 0, "rgba(16,150,24,0.7)", "rgba(219,64,82,0.7)")

fig_delta = go.Figure()
fig_delta.add_trace(go.Bar(x=delta_x, y=delta_series.values, name=delta_legend, marker=dict(color=delta_colors), opacity=0.9))
fig_delta.update_layout(
    title=f"{delta_legend} - {agg_mode.lower()}",
    height=420,
    margin=dict(l=60, r=60, t=60, b=50),
    xaxis=dict(tickfont=dict(size=13)),
    yaxis=dict(tickfont=dict(size=13)),
)
st.plotly_chart(fig_delta, use_container_width=True)

# =========================
# Forward returns
# =========================
st.subheader("Forward returns per marktstate")

fr = d.copy()
for h in [1, 3, 5, 10, 20]:
    fr[f"fwd_{h}d_pct"] = (fr["close"].shift(-h) / fr["close"] - 1.0) * 100.0

fr["state_oversold"] = (
    ((fr["rsi14_s"] <= fr["rsi_dyn_lo"]) | (fr["z20"] <= -2.0) | (fr["stretch_ema20_atr"] <= -1.5))
)
fr["state_overbought"] = (
    ((fr["rsi14_s"] >= fr["rsi_dyn_hi"]) | (fr["z20"] >= 2.0) | (fr["stretch_ema20_atr"] >= 1.5))
)
fr["state_bull_trend"] = (
    (fr["close"] > fr["ema20"]) &
    (fr["ema20"] > fr["ema50"]) &
    (fr["ema50"] > fr["ema200"]) &
    (fr["adx14"] >= 20)
)
fr["state_bear_trend"] = (
    (fr["close"] < fr["ema20"]) &
    (fr["ema20"] < fr["ema50"]) &
    (fr["ema50"] < fr["ema200"]) &
    (fr["adx14"] >= 20)
)
fr["state_stress"] = ((fr["vix_z"] >= 1.5) | (fr["vol_regime"].isin(["Stress", "Panic"])))

state_map = {
    "Oversold / mean reversion": "state_oversold",
    "Overbought / stretch": "state_overbought",
    "Bull trend continuation": "state_bull_trend",
    "Bear trend continuation": "state_bear_trend",
    "Stress / risk-off": "state_stress",
}


def forward_stats(mask: pd.Series, label: str, horizons: list[int]):
    sub = fr[mask].copy()
    row = {"State": label, "N": int(mask.fillna(False).sum())}

    for h in horizons:
        s = sub[f"fwd_{h}d_pct"].dropna()
        row[f"{h}d avg"] = s.mean() if len(s) else np.nan
        row[f"{h}d med"] = s.median() if len(s) else np.nan
        row[f"{h}d hit"] = (s > 0).mean() * 100 if len(s) else np.nan

    return row


rows = [forward_stats(fr[col], label, future_horizons) for label, col in state_map.items()]
forward_table = pd.DataFrame(rows)

fmt_forward = forward_table.copy()
for c in fmt_forward.columns:
    if c == "State":
        continue
    if c == "N":
        fmt_forward[c] = fmt_forward[c].astype("Int64")
        continue
    if "hit" in c:
        fmt_forward[c] = fmt_forward[c].map(lambda x: "-" if pd.isna(x) else f"{x:.1f}%")
    else:
        fmt_forward[c] = fmt_forward[c].map(lambda x: "-" if pd.isna(x) else f"{x:.2f}%")

st.dataframe(fmt_forward, use_container_width=True)

# =========================
# Rolling correlatie met VIX
# =========================
st.subheader("Rolling correlatie met VIX")

corr_df = d.copy().set_index("date")
aex_ret = corr_df["delta_pct"]
vix_series = (corr_df["vix_close"].pct_change() * 100.0) if corr_vs == "% change" else corr_df["vix_close"]
corr_join = pd.concat([aex_ret.rename("aex"), vix_series.rename("vix")], axis=1).dropna()
rolling_corr = corr_join["aex"].rolling(corr_win).corr(corr_join["vix"])

fig_corr = go.Figure()
fig_corr.add_trace(go.Scatter(x=rolling_corr.index, y=rolling_corr.values, mode="lines", name="Rolling corr", line=dict(width=2)))
fig_corr.add_hline(y=0.0, line_dash="dot")
fig_corr.add_hrect(y0=-1, y1=-0.5, fillcolor="rgba(255,0,0,0.06)", line_width=0)
fig_corr.add_hrect(y0=0.5, y1=1, fillcolor="rgba(0,128,0,0.06)", line_width=0)
fig_corr.update_layout(
    height=400,
    margin=dict(l=60, r=60, t=50, b=40),
    yaxis=dict(range=[-1, 1], title="corr", tickfont=dict(size=13)),
    xaxis=dict(tickfont=dict(size=13)),
)
st.plotly_chart(fig_corr, use_container_width=True)

# =========================
# Heatmap
# =========================
st.subheader("Maand/jaar-heatmap van Delta")

t = d.copy().set_index("date")
if delta_mode == "Delta %":
    monthly = t["delta_pct"].groupby(pd.Grouper(freq="ME")).apply(
        lambda g: (np.prod((g.dropna() / 100.0 + 1.0)) - 1.0) * 100.0 if len(g.dropna()) else np.nan
    )
    value_title = "Delta% maand"
else:
    monthly = t["delta_abs"].groupby(pd.Grouper(freq="ME")).sum(min_count=1)
    value_title = "Delta punten maand"

hm = pd.DataFrame({
    "year": monthly.index.year,
    "month": monthly.index.month,
    "value": monthly.values,
}).dropna()

month_names = {
    1: "jan", 2: "feb", 3: "mrt", 4: "apr", 5: "mei", 6: "jun",
    7: "jul", 8: "aug", 9: "sep", 10: "okt", 11: "nov", 12: "dec",
}

hm["mname"] = hm["month"].map(month_names)
pivot = hm.pivot_table(index="year", columns="mname", values="value", aggfunc="first")
pivot = pivot.reindex(columns=[month_names[m] for m in range(1, 13)])

heat = go.Figure(
    data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=pivot.index,
        coloraxis="coloraxis",
        hovertemplate="Jaar %{y} - %{x}: %{z:.2f}<extra></extra>",
    )
)
heat.update_layout(
    height=480,
    margin=dict(l=60, r=60, t=50, b=50),
    coloraxis=dict(colorscale="RdBu", cauto=True, colorbar_title=value_title),
    xaxis=dict(title="Maand", tickfont=dict(size=13)),
    yaxis=dict(title="Jaar", tickfont=dict(size=13)),
)
st.plotly_chart(heat, use_container_width=True)

# =========================
# Histograms
# =========================
st.subheader("Histogram dagrendementen")

col_a, col_b = st.columns(2)
hist_df = d.dropna(subset=["delta_abs", "delta_pct"]).copy()

with col_a:
    fig_abs = go.Figure([go.Histogram(x=hist_df["delta_abs"], nbinsx=60)])
    fig_abs.update_layout(
        title="Delta abs (punten)",
        height=420,
        bargap=0.02,
        margin=dict(l=50, r=50, t=60, b=50),
        xaxis=dict(tickfont=dict(size=13)),
        yaxis=dict(tickfont=dict(size=13)),
    )
    st.plotly_chart(fig_abs, use_container_width=True)

with col_b:
    fig_pct = go.Figure([go.Histogram(x=hist_df["delta_pct"], nbinsx=60)])
    fig_pct.update_layout(
        title="Delta %",
        height=420,
        bargap=0.02,
        margin=dict(l=50, r=50, t=60, b=50),
        xaxis=dict(tickfont=dict(size=13)),
        yaxis=dict(tickfont=dict(size=13)),
    )
    st.plotly_chart(fig_pct, use_container_width=True)

# =========================
# Raw diagnostics table
# =========================
st.subheader("Laatste diagnose")

diag_cols = [
    "date", "close", "ema20", "ema50", "ema200",
    "adx14", "rsi14_s", "macd_hist", "stretch_ema20_atr",
    "z20", "atr_pct_close", "vix_close", "vix_z",
    "trend_label", "trend_strength", "momentum_label",
    "exhaustion_label", "vol_regime",
]
show_diag = d[diag_cols].tail(20).copy()
st.dataframe(show_diag, use_container_width=True)
