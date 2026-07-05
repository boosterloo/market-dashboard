# pages/Silver_Analytics.py
# Silver Analytics - price, trend, macro drivers, gold/silver ratio, correlations and beta.

import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from google.api_core.exceptions import NotFound, BadRequest, Forbidden

# ---------- BigQuery helpers ----------
try:
    from utils.bq import run_query, bq_ping
except Exception:
    import google.cloud.bigquery as bq
    from google.oauth2 import service_account

    @st.cache_resource(show_spinner=False)
    def _bq_client_from_secrets():
        creds = st.secrets["gcp_service_account"]
        credentials = service_account.Credentials.from_service_account_info(creds)
        return bq.Client(credentials=credentials, project=creds["project_id"])

    @st.cache_data(ttl=300, show_spinner=False)
    def run_query(sql: str) -> pd.DataFrame:
        client = _bq_client_from_secrets()
        return client.query(sql).to_dataframe()

    def bq_ping() -> bool:
        try:
            _bq_client_from_secrets().query("SELECT 1").result(timeout=10)
            return True
        except Exception:
            return False


# ---------- Page config ----------
st.set_page_config(page_title="Silver Analytics", layout="wide")
st.title("Silver Analytics - Price, Gold Ratio, Dollar, Rates and Industrial Drivers")

# ---------- Sources ----------
TABLES = st.secrets.get("tables", {})

COM_WIDE_VIEW = TABLES.get(
    "commodities_wide_view",
    "nth-pier-468314-p7.marketdata.commodity_prices_wide_v",
)
DRIVERS_WIDE_VIEW = TABLES.get(
    "silver_drivers_wide_view",
    TABLES.get("gold_drivers_wide_view", "nth-pier-468314-p7.marketdata.gold_drivers_wide_v"),
)
AEX_VIEW = TABLES.get("aex_view", "nth-pier-468314-p7.marketdata.aex_with_vix_v")
CRYPTO_WIDE_VIEW = TABLES.get("crypto_daily_wide", "nth-pier-468314-p7.marketdata.crypto_daily_wide_v")
FX_WIDE_VIEW = TABLES.get("fx_wide_view", "nth-pier-468314-p7.marketdata.fx_daily_wide_v")
US_YIELD_VIEW = TABLES.get("us_yield_view", "nth-pier-468314-p7.marketdata.us_yields_daily_wide_v")
MACRO_VIEW = TABLES.get("macro_view", "nth-pier-468314-p7.marketdata.macro_series_wide_monthly_fill_v")

with st.expander("Debug: gebruikte bronnen", expanded=False):
    st.write(
        {
            "commodities_wide_view": COM_WIDE_VIEW,
            "drivers_wide_view": DRIVERS_WIDE_VIEW,
            "aex_view (VIX fallback)": AEX_VIEW,
            "crypto_daily_wide (BTC fallback)": CRYPTO_WIDE_VIEW,
            "fx_wide_view (DXY/EURUSD fallback)": FX_WIDE_VIEW,
            "us_yield_view (US10Y/TIPS fallback)": US_YIELD_VIEW,
            "macro_view (M2 fallback)": MACRO_VIEW,
        }
    )

# ---------- Health ----------
try:
    if not bq_ping():
        st.error("Geen BigQuery-verbinding. Controleer Secrets ([gcp_service_account]).")
        st.stop()
except Exception as e:
    st.error("Geen BigQuery-verbinding.")
    st.caption(f"Details: {e}")
    st.stop()


# ---------- Loaders ----------
def _numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    for c in df.columns:
        if c != "date":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def best_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        if c.lower() in lower:
            return lower[c.lower()]
    return None


@st.cache_data(ttl=300, show_spinner=False)
def load_com() -> pd.DataFrame:
    df = run_query(f"SELECT * FROM `{COM_WIDE_VIEW}` ORDER BY date")
    return _numeric_df(df)


@st.cache_data(ttl=300, show_spinner=False)
def load_drv_main() -> pd.DataFrame:
    try:
        df = run_query(f"SELECT * FROM `{DRIVERS_WIDE_VIEW}` ORDER BY date")
        return _numeric_df(df)
    except (NotFound, Forbidden, BadRequest):
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=600, show_spinner=False)
def load_vix_fallback() -> pd.DataFrame:
    try:
        d = run_query(f"SELECT date, vix_close FROM `{AEX_VIEW}` ORDER BY date")
        return _numeric_df(d)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=600, show_spinner=False)
def load_crypto_fallback() -> pd.DataFrame:
    candidates = [
        f"SELECT date, price_btc AS btc_close FROM `{CRYPTO_WIDE_VIEW}` ORDER BY date",
        f"SELECT date, price_BTC AS btc_close FROM `{CRYPTO_WIDE_VIEW}` ORDER BY date",
        f"SELECT date, btc_close FROM `{CRYPTO_WIDE_VIEW}` ORDER BY date",
    ]
    for sql in candidates:
        try:
            d = run_query(sql)
            if not d.empty and "btc_close" in d.columns:
                return _numeric_df(d)
        except Exception:
            continue
    return pd.DataFrame()


@st.cache_data(ttl=600, show_spinner=False)
def load_fx_fallback() -> pd.DataFrame:
    merged = None
    for sql in [
        f"SELECT date, dxy_close FROM `{FX_WIDE_VIEW}` ORDER BY date",
        f"SELECT date, dxy AS dxy_close FROM `{FX_WIDE_VIEW}` ORDER BY date",
        f"SELECT date, DXY AS dxy_close FROM `{FX_WIDE_VIEW}` ORDER BY date",
        f"SELECT date, eurusd_close FROM `{FX_WIDE_VIEW}` ORDER BY date",
        f"SELECT date, eurusd AS eurusd_close FROM `{FX_WIDE_VIEW}` ORDER BY date",
        f"SELECT date, EURUSD AS eurusd_close FROM `{FX_WIDE_VIEW}` ORDER BY date",
    ]:
        try:
            d = run_query(sql)
            if d.empty:
                continue
            d = _numeric_df(d)
            merged = d if merged is None else pd.merge(merged, d, on="date", how="outer")
        except Exception:
            continue
    if merged is not None:
        return merged
    try:
        d_all = run_query(f"SELECT * FROM `{FX_WIDE_VIEW}` ORDER BY date")
        if d_all.empty:
            return pd.DataFrame()
        dxy_col = best_col(d_all, ["dxy_close", "dxy", "DXY", "dollar_index", "usd_index"])
        eurusd_col = best_col(d_all, ["eurusd_close", "eurusd", "EURUSD", "eur_usd"])
        keep = ["date"]
        rename = {}
        if dxy_col:
            keep.append(dxy_col)
            rename[dxy_col] = "dxy_close"
        if eurusd_col and eurusd_col not in keep:
            keep.append(eurusd_col)
            rename[eurusd_col] = "eurusd_close"
        if len(keep) <= 1:
            return pd.DataFrame()
        return _numeric_df(d_all[keep].rename(columns=rename))
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=600, show_spinner=False)
def load_yield_fallback() -> pd.DataFrame:
    merged = None
    for sql in [
        f"SELECT date, y_10y AS us10y, tips10y_real FROM `{US_YIELD_VIEW}` ORDER BY date",
        f"SELECT date, y_10y_synth AS us10y, tips10y_real FROM `{US_YIELD_VIEW}` ORDER BY date",
        f"SELECT date, us10y, tips10y_real FROM `{US_YIELD_VIEW}` ORDER BY date",
        f"SELECT date, us_10y AS us10y, tips10y_real FROM `{US_YIELD_VIEW}` ORDER BY date",
        f"SELECT date, y_10y AS us10y FROM `{US_YIELD_VIEW}` ORDER BY date",
        f"SELECT date, y_10y_synth AS us10y FROM `{US_YIELD_VIEW}` ORDER BY date",
        f"SELECT date, us_10y AS us10y FROM `{US_YIELD_VIEW}` ORDER BY date",
    ]:
        try:
            d = run_query(sql)
            if d.empty:
                continue
            d = _numeric_df(d)
            merged = d if merged is None else pd.merge(merged, d, on="date", how="outer")
        except Exception:
            continue
    if merged is not None:
        merged = merged.loc[:, ~merged.columns.duplicated()]
        return merged
    try:
        d_all = run_query(f"SELECT * FROM `{US_YIELD_VIEW}` ORDER BY date")
        if d_all.empty:
            return pd.DataFrame()
        us10y_col = best_col(d_all, ["us10y", "y_10y", "y_10y_synth", "us_10y", "rate_10y", "tenor_10y", "yield_10y"])
        real10y_col = best_col(d_all, ["tips10y_real", "real10y", "real_10y", "tips_10y", "tips10y", "y_10y_real"])
        keep = ["date"]
        rename = {}
        if us10y_col:
            keep.append(us10y_col)
            rename[us10y_col] = "us10y"
        if real10y_col and real10y_col not in keep:
            keep.append(real10y_col)
            rename[real10y_col] = "tips10y_real"
        if len(keep) <= 1:
            return pd.DataFrame()
        return _numeric_df(d_all[keep].rename(columns=rename))
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=1800, show_spinner=False)
def load_m2_fallback() -> pd.DataFrame:
    candidates = [
        "m2",
        "m2_ma3",
        "m2_real",
        "m2_real_ma3",
        "m2_yoy",
        "m2_real_yoy",
        "m2_vel",
        "m2_vel_ma3",
        "m2_vel_yoy",
    ]
    try:
        cols_sql = ", ".join(candidates)
        d = run_query(f"SELECT date, {cols_sql} FROM `{MACRO_VIEW}` ORDER BY date")
        return _numeric_df(d)
    except Exception:
        try:
            d_all = run_query(f"SELECT * FROM `{MACRO_VIEW}` ORDER BY date")
            if d_all.empty:
                return pd.DataFrame()
            lower_map = {c.lower(): c for c in d_all.columns}
            keep = ["date"]
            for c in candidates:
                if c in d_all.columns:
                    keep.append(c)
                elif c.lower() in lower_map:
                    keep.append(lower_map[c.lower()])
            if len(keep) <= 1:
                return pd.DataFrame()
            d = d_all[keep].copy()
            rename = {lower_map[c.lower()]: c for c in candidates if c.lower() in lower_map and lower_map[c.lower()] != c}
            d = d.rename(columns=rename)
            return _numeric_df(d)
        except Exception:
            return pd.DataFrame()


def merge_new_cols(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    if right is None or right.empty:
        return left
    keep = ["date"] + [c for c in right.columns if c != "date" and c not in left.columns]
    if len(keep) <= 1:
        return left
    return pd.merge(left, right[keep], on="date", how="outer")


# ---------- Load & merge ----------
df_com = load_com()
if df_com.empty:
    st.warning("Geen data in commodities wide view.")
    st.stop()

base_prefixes = ("silver_", "gold_", "copper_")
base_cols = ["date"] + [c for c in df_com.columns if c.startswith(base_prefixes)]
df = df_com[base_cols].copy()

if "silver_close" not in df.columns:
    st.error("Geen `silver_close` kolom gevonden in commodities wide view.")
    st.stop()

drv_main = load_drv_main()
drivers_view_status = "fallback"
if not drv_main.empty:
    drivers_view_status = "loaded"
    df = merge_new_cols(df, drv_main)

for fallback in [load_vix_fallback(), load_crypto_fallback(), load_fx_fallback(), load_yield_fallback(), load_m2_fallback()]:
    df = merge_new_cols(df, fallback)

df = df.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)


# ---------- Helpers ----------
def ema(series: pd.Series, span: int) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    return s.ewm(span=span, adjust=False, min_periods=span).mean()


def rma(x: pd.Series, length: int) -> pd.Series:
    return x.ewm(alpha=1 / length, adjust=False).mean()


def rsi_wilder(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    rs = rma(up, length) / rma(down, length).replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast_line = series.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow_line = series.ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd_line = ema_fast_line - ema_slow_line
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def heikin_ashi(src: pd.DataFrame) -> pd.DataFrame:
    ha = src.copy()
    ha["ha_close"] = (ha["open"] + ha["high"] + ha["low"] + ha["close"]) / 4.0

    ha_open = pd.Series(index=ha.index, dtype=float)
    first_valid_idx = ha[["open", "close"]].dropna().index.min()
    if pd.isna(first_valid_idx):
        return pd.DataFrame(
            {
                "ha_open": np.nan,
                "ha_high": np.nan,
                "ha_low": np.nan,
                "ha_close": np.nan,
            },
            index=ha.index,
        )

    ha_open.loc[first_valid_idx] = (ha.loc[first_valid_idx, "open"] + ha.loc[first_valid_idx, "close"]) / 2.0
    start_pos = ha.index.get_loc(first_valid_idx)
    for i in range(start_pos + 1, len(ha)):
        prev_idx = ha.index[i - 1]
        cur_idx = ha.index[i]
        if pd.notna(ha_open.loc[prev_idx]) and pd.notna(ha.loc[prev_idx, "ha_close"]):
            ha_open.loc[cur_idx] = (ha_open.loc[prev_idx] + ha.loc[prev_idx, "ha_close"]) / 2.0

    ha["ha_open"] = ha_open
    ha["ha_high"] = pd.concat([ha["high"], ha["ha_open"], ha["ha_close"]], axis=1).max(axis=1)
    ha["ha_low"] = pd.concat([ha["low"], ha["ha_open"], ha["ha_close"]], axis=1).min(axis=1)
    return ha[["ha_open", "ha_high", "ha_low", "ha_close"]]


def zscore(s: pd.Series, win: int = 20) -> pd.Series:
    mu = s.rolling(win, min_periods=win).mean()
    sd = s.rolling(win, min_periods=win).std()
    return (s - mu) / sd.replace(0, np.nan)


def slope_pct(series: pd.Series, lookback: int = 10) -> pd.Series:
    return ((series / series.shift(lookback)) - 1.0) * 100.0


def normalize_100(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    valid = s.dropna()
    if valid.empty or valid.iloc[0] == 0:
        return s
    return s / valid.iloc[0] * 100.0


def padded_range(series_list: list[pd.Series], pad_ratio: float = 0.08):
    values = pd.concat([pd.to_numeric(s, errors="coerce") for s in series_list], axis=0).dropna()
    if values.empty:
        return None
    y_min = float(values.min())
    y_max = float(values.max())
    if not np.isfinite(y_min) or not np.isfinite(y_max):
        return None
    if y_min == y_max:
        pad = abs(y_max) * pad_ratio if y_max != 0 else 1.0
    else:
        pad = (y_max - y_min) * pad_ratio
    return [y_min - pad, y_max + pad]


def pct_as_str(x):
    return "-" if pd.isna(x) else f"{x:.2f}%"


def last_val(data: pd.DataFrame, col: str):
    if col not in data.columns or data[col].notna().sum() == 0:
        return np.nan
    return float(data[col].dropna().iloc[-1])


def latest_pct(data: pd.DataFrame, close_col: str, delta_col: str | None = None):
    if delta_col and delta_col in data.columns and data[delta_col].notna().any():
        return float(data[delta_col].dropna().iloc[-1] * 100.0)
    s = data[close_col].dropna()
    return float(s.pct_change().dropna().iloc[-1] * 100.0) if len(s) >= 2 else np.nan


def rolling_corr(a: pd.Series, b: pd.Series, win: int, returns: bool = True) -> pd.Series:
    x = a.pct_change() if returns else a
    y = b.pct_change() if returns else b
    join = pd.concat([x.rename("a"), y.rename("b")], axis=1).dropna()
    if len(join) < win:
        return pd.Series(dtype=float)
    return join["a"].rolling(win).corr(join["b"]).dropna()


def regress_xy(xv: np.ndarray, yv: np.ndarray):
    slope, intercept = np.polyfit(xv, yv, 1)
    y_hat = slope * xv + intercept
    ss_res = np.sum((yv - y_hat) ** 2)
    ss_tot = np.sum((yv - np.mean(yv)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan
    corr = np.corrcoef(xv, yv)[0, 1] if len(xv) > 1 else np.nan
    return slope, intercept, r2, corr


def trend_delta(data: pd.DataFrame, col: str, lookback: int):
    if col not in data.columns or data[col].notna().sum() < lookback + 1:
        return np.nan
    s = data[col].dropna()
    return float(s.iloc[-1] - s.iloc[-1 - lookback])


# ---------- Sidebar ----------
DRIVER_MAP = {
    "Gold (USD/oz)": "gold_close",
    "Gold/Silver ratio": "gold_silver_ratio",
    "Copper (industrial proxy)": "copper_close",
    "DXY (Dollar Index)": "dxy_close",
    "EURUSD": "eurusd_close",
    "US 10Y (yield %)": "us10y",
    "US 10Y Real (TIPS %)": "tips10y_real",
    "VIX": "vix_close",
    "BTC (BTCUSD)": "btc_close",
    "M2 YoY (%)": "m2_yoy",
    "Real M2 YoY (%)": "m2_real_yoy",
    "M2 level": "m2",
    "M2 velocity": "m2_vel",
}

with st.sidebar:
    st.header("Instellingen")
    view_mode = st.radio("Overlayschaling", ["Genormaliseerd (=100)", "Eigen schaal (2e y-as)"], index=0)
    show_delta = st.checkbox("Delta%-bars tonen (Silver)", value=True)

    st.divider()
    st.markdown("#### Trend")
    ema_fast = st.number_input("EMA fast", 5, 100, 20, step=1)
    ema_mid = st.number_input("EMA mid", 10, 150, 50, step=1)
    ema_slow = st.number_input("EMA slow", 50, 400, 200, step=5)
    show_emas = st.checkbox("EMA's tonen in hoofdgrafiek", value=False)
    selected_emas = st.multiselect(
        "Welke EMA's",
        options=["fast", "mid", "slow"],
        default=["mid", "slow"],
        disabled=not show_emas,
    )
    rsi_period = st.slider("RSI periode", 5, 40, 14, 1)

    st.divider()
    st.markdown("#### Drivers & alerts")
    trend_look = st.slider("Driver trend lookback (dagen)", 5, 90, 20, 1)
    dxy_drop_thr = st.number_input("DXY bullish als delta lager dan", -10.0, 10.0, 0.0, step=0.1)
    yield_drop_thr = st.number_input("Yield bullish als delta lager dan", -5.0, 5.0, 0.0, step=0.1)
    ratio_hi = st.slider("Gold/Silver ratio hoog", 50.0, 120.0, 85.0, 1.0)
    ratio_lo = st.slider("Gold/Silver ratio laag", 30.0, 90.0, 65.0, 1.0)
    m2_impulse_thr = st.number_input("M2 liquidity impulse bullish boven", -5.0, 5.0, 0.0, step=0.1)

    st.divider()
    st.markdown("#### Correlatie / beta")
    corr_win = st.slider("Rolling corr window", 10, 180, 60, 5)
    corr_transform = st.radio("Correlatiebasis", ["Returns / changes", "Levels"], index=0)
    lead_lag_max = st.slider("Lead/lag max dagen", 5, 60, 20, 5)
    scatter_mode = st.radio("Scatter transformatie", ["%-returns vs %-returns", "Levels vs Levels"], index=0)

# ---------- Period ----------
if "date" not in df.columns or df["date"].isna().all():
    st.error("Geen geldige date-kolom na merge.")
    st.stop()

min_d = df["date"].min().date()
max_d = df["date"].max().date()
default_start = max(max_d - timedelta(days=365), min_d)
start_d, end_d = st.slider(
    "Periode",
    min_value=min_d,
    max_value=max_d,
    value=(default_start, max_d),
    step=timedelta(days=1),
    format="YYYY-MM-DD",
)

mask = (df["date"].dt.date >= start_d) & (df["date"].dt.date <= end_d)
d = df.loc[mask].copy()
if d.empty:
    st.info("Geen data in de gekozen periode.")
    st.stop()

# ---------- Derived series ----------
if "silver_delta_pct" not in d.columns or d["silver_delta_pct"].isna().all():
    d["silver_delta_pct"] = d["silver_close"].pct_change()

if "gold_close" in d.columns:
    d["gold_silver_ratio"] = d["gold_close"] / d["silver_close"].replace(0, np.nan)
    d["silver_gold_rel"] = normalize_100(d["silver_close"]) - normalize_100(d["gold_close"])

ema_spans = sorted(set([ema_fast, ema_mid, ema_slow]))
for span in ema_spans:
    d[f"silver_ema{span}"] = ema(d["silver_close"], span)

d["silver_rsi"] = rsi_wilder(d["silver_close"], rsi_period)
d["silver_z20"] = zscore(d["silver_close"], 20)
d["silver_macd_line"], d["silver_macd_signal"], d["silver_macd_hist"] = macd(d["silver_close"])
d["silver_macd_hist_prev"] = d["silver_macd_hist"].shift(1)
d["silver_ema_fast_slope_10"] = slope_pct(d[f"silver_ema{ema_fast}"], 10)
d["silver_ema_mid_slope_10"] = slope_pct(d[f"silver_ema{ema_mid}"], 10)
d["silver_stretch_fast_pct"] = (d["silver_close"] / d[f"silver_ema{ema_fast}"] - 1.0) * 100.0


def classify_silver_trend(row):
    close = row["silver_close"]
    e_fast = row[f"silver_ema{ema_fast}"]
    e_mid = row[f"silver_ema{ema_mid}"]
    e_slow = row[f"silver_ema{ema_slow}"]
    s_fast = row["silver_ema_fast_slope_10"]
    s_mid = row["silver_ema_mid_slope_10"]

    if pd.isna(close) or pd.isna(e_fast) or pd.isna(e_mid) or pd.isna(e_slow):
        return "Onvoldoende data"
    if close > e_fast > e_mid > e_slow and s_fast > 0 and s_mid > 0:
        return "Strong Bull"
    if close > e_mid > e_slow:
        return "Bull"
    if close < e_fast < e_mid < e_slow and s_fast < 0 and s_mid < 0:
        return "Strong Bear"
    if close < e_mid < e_slow:
        return "Bear"
    return "Neutral"


def classify_silver_strength(row):
    spread_fast_mid = abs((row[f"silver_ema{ema_fast}"] / row[f"silver_ema{ema_mid}"] - 1) * 100) if pd.notna(row[f"silver_ema{ema_mid}"]) and row[f"silver_ema{ema_mid}"] != 0 else np.nan
    spread_mid_slow = abs((row[f"silver_ema{ema_mid}"] / row[f"silver_ema{ema_slow}"] - 1) * 100) if pd.notna(row[f"silver_ema{ema_slow}"]) and row[f"silver_ema{ema_slow}"] != 0 else np.nan
    slope_fast = abs(row["silver_ema_fast_slope_10"]) if pd.notna(row["silver_ema_fast_slope_10"]) else np.nan
    score = 0
    if pd.notna(spread_fast_mid) and spread_fast_mid > 2.0:
        score += 1
    if pd.notna(spread_mid_slow) and spread_mid_slow > 4.0:
        score += 1
    if pd.notna(slope_fast) and slope_fast > 2.0:
        score += 1
    if pd.notna(row["silver_stretch_fast_pct"]) and abs(row["silver_stretch_fast_pct"]) > 5:
        score += 1
    if score >= 3:
        return "Sterk"
    if score >= 2:
        return "Gemiddeld"
    return "Zwak"


def classify_silver_momentum(row):
    hist = row["silver_macd_hist"]
    hist_prev = row["silver_macd_hist_prev"]
    rsi = row["silver_rsi"]
    if pd.isna(hist) or pd.isna(hist_prev) or pd.isna(rsi):
        return "Onvoldoende data"
    if hist > 0 and hist > hist_prev and rsi > 55:
        return "Versnellend omhoog"
    if hist > 0 and hist <= hist_prev and rsi > 50:
        return "Positief maar afzwakkend"
    if hist < 0 and hist < hist_prev and rsi < 45:
        return "Versnellend omlaag"
    if hist < 0 and hist >= hist_prev and rsi < 50:
        return "Negatief maar afzwakkend"
    return "Neutraal"


def classify_silver_exhaustion(row):
    flags = 0
    if pd.notna(row["silver_rsi"]) and (row["silver_rsi"] >= 72 or row["silver_rsi"] <= 28):
        flags += 1
    if pd.notna(row["silver_z20"]) and abs(row["silver_z20"]) >= 2.0:
        flags += 1
    if pd.notna(row["silver_stretch_fast_pct"]) and abs(row["silver_stretch_fast_pct"]) >= 6.0:
        flags += 1
    if flags >= 2:
        return "Hoog"
    if flags == 1:
        return "Oplopend"
    return "Laag"


def classify_macro_bias(row):
    score = 0
    reasons = []
    dxy_delta = trend_delta(d, "dxy_close", trend_look)
    real_yield_delta = trend_delta(d, "tips10y_real", trend_look)
    m2_delta = trend_delta(d, "m2_real_yoy", trend_look)
    copper_mom = d["copper_close"].pct_change(trend_look).dropna().iloc[-1] if "copper_close" in d.columns and d["copper_close"].notna().sum() > trend_look else np.nan

    if pd.notna(dxy_delta):
        score += 1 if dxy_delta < 0 else -1
        reasons.append(f"DXY {dxy_delta:+.2f}")
    if pd.notna(real_yield_delta):
        score += 1 if real_yield_delta < 0 else -1
        reasons.append(f"real yield {real_yield_delta:+.2f}")
    if pd.notna(m2_delta):
        score += 1 if m2_delta > 0 else -1
        reasons.append(f"real M2 YoY {m2_delta:+.2f}pp")
    if pd.notna(copper_mom):
        score += 1 if copper_mom > 0 else -1
        reasons.append(f"copper {copper_mom * 100:+.2f}%")

    if score >= 2:
        return "Macro tailwind", " | ".join(reasons)
    if score <= -2:
        return "Macro headwind", " | ".join(reasons)
    return "Macro mixed", " | ".join(reasons) if reasons else "Onvoldoende macrodata"


def build_silver_summary(row):
    trend = row["silver_trend"]
    strength = row["silver_trend_strength"]
    momentum = row["silver_momentum"]
    exhaustion = row["silver_exhaustion"]
    macro = row["macro_bias"]

    parts = []
    if trend in ["Strong Bull", "Bull"]:
        parts.append(f"Zilver zit technisch in een {trend.lower()} fase met {strength.lower()} trendkracht")
    elif trend in ["Strong Bear", "Bear"]:
        parts.append(f"Zilver zit technisch in een {trend.lower()} fase met {strength.lower()} trendkracht")
    else:
        parts.append("Zilver zit technisch in een neutrale of overgangsfase")
    if momentum != "Onvoldoende data":
        parts.append(f"momentum is {momentum.lower()}")
    parts.append(f"uitputting is {exhaustion.lower()}")
    parts.append(f"macrobeeld: {macro.lower()}")
    return ". ".join(parts) + "."


d["silver_trend"] = d.apply(classify_silver_trend, axis=1)
d["silver_trend_strength"] = d.apply(classify_silver_strength, axis=1)
d["silver_momentum"] = d.apply(classify_silver_momentum, axis=1)
d["silver_exhaustion"] = d.apply(classify_silver_exhaustion, axis=1)
macro_bias, macro_detail = classify_macro_bias(d.iloc[-1])
d["macro_bias"] = macro_bias
d["macro_detail"] = macro_detail

driver_choices = [name for name, col in DRIVER_MAP.items() if col in d.columns and d[col].notna().any()]
default_overlay = [
    x for x in ["Gold (USD/oz)", "Gold/Silver ratio", "Copper (industrial proxy)", "DXY (Dollar Index)", "US 10Y (yield %)", "US 10Y Real (TIPS %)", "Real M2 YoY (%)", "VIX"]
    if x in driver_choices
]
default_scatter = [
    x for x in ["Gold (USD/oz)", "Copper (industrial proxy)", "DXY (Dollar Index)", "US 10Y (yield %)", "US 10Y Real (TIPS %)", "Real M2 YoY (%)"]
    if x in driver_choices
]

# ---------- Debug ----------
with st.expander("Debug: driver-kolommen en non-null counts", expanded=False):
    dbg_cols = [
        "silver_close", "gold_close", "gold_silver_ratio", "copper_close",
        "dxy_close", "eurusd_close", "us10y", "tips10y_real", "vix_close", "btc_close",
        "m2", "m2_real", "m2_yoy", "m2_real_yoy", "m2_vel", "m2_vel_yoy",
    ]
    st.write(
        {
            "drivers_view_status": drivers_view_status,
            "drivers_view": DRIVERS_WIDE_VIEW,
            "present": {c: c in df.columns or c in d.columns for c in dbg_cols},
            "notna_count_in_range": {c: int(d[c].notna().sum()) if c in d.columns else 0 for c in dbg_cols},
        }
    )

# ---------- KPI ----------
st.subheader("Silver Market State")
silver_last = last_val(d, "silver_close")
silver_dpct = latest_pct(d, "silver_close", "silver_delta_pct")
gold_last = last_val(d, "gold_close")
ratio_last = last_val(d, "gold_silver_ratio")
copper_dpct = latest_pct(d, "copper_close", "copper_delta_pct") if "copper_close" in d.columns else np.nan
state_ready = d.dropna(subset=["silver_close"]).copy()
last_state = state_ready.iloc[-1] if not state_ready.empty else d.iloc[-1]
summary_text = build_silver_summary(last_state)

k1, k2, k3, k4, k5, k6, k7, k8 = st.columns(8)
k1.metric("Silver (USD/oz)", f"{silver_last:,.2f}" if pd.notna(silver_last) else "-", pct_as_str(silver_dpct))
k2.metric("Trend", last_state["silver_trend"])
k3.metric("Trendkracht", last_state["silver_trend_strength"])
k4.metric("Momentum", last_state["silver_momentum"])
k5.metric("Uitputting", last_state["silver_exhaustion"])
k6.metric("Macro bias", last_state["macro_bias"])
k7.metric("Gold/Silver", f"{ratio_last:,.1f}" if pd.notna(ratio_last) else "-")
k8.metric("Real M2 YoY", pct_as_str(last_val(d, "m2_real_yoy")))

st.info(summary_text)
if last_state.get("macro_detail", ""):
    st.caption(f"Macro detail ({trend_look}d): {last_state['macro_detail']}")

d1, d2, d3, d4, d5, d6 = st.columns(6)
d1.metric("RSI", f"{last_val(d, 'silver_rsi'):.1f}" if pd.notna(last_val(d, "silver_rsi")) else "-")
d2.metric("MACD hist", f"{last_val(d, 'silver_macd_hist'):.3f}" if pd.notna(last_val(d, "silver_macd_hist")) else "-")
d3.metric("Z-score 20d", f"{last_val(d, 'silver_z20'):.2f}" if pd.notna(last_val(d, "silver_z20")) else "-")
d4.metric("Stretch vs EMA fast", f"{last_val(d, 'silver_stretch_fast_pct'):.2f}%" if pd.notna(last_val(d, "silver_stretch_fast_pct")) else "-")
d5.metric("Copper delta", pct_as_str(copper_dpct))
d6.metric("DXY", f"{last_val(d, 'dxy_close'):.2f}" if pd.notna(last_val(d, "dxy_close")) else "-")

with st.expander("Hoe wordt deze diagnose gelezen?", expanded=False):
    st.markdown(
        """
        - **Trend** kijkt naar silver close versus EMA fast/mid/slow en de richting van de EMA's.
        - **Momentum** combineert MACD-histogram en RSI.
        - **Uitputting** kijkt naar RSI-extremen, z-score en stretch versus de snelle EMA.
        - **Macro bias** combineert DXY, real yield, real M2 YoY en koper over de gekozen driver-lookback.
        """
    )

macro_watch = [
    ("DXY", "dxy_close", "level", "Dollar sterker is meestal druk op metalen."),
    ("US 10Y", "us10y", "pp", "Nominale rente; hogere rente verhoogt opportunity cost."),
    ("US 10Y real", "tips10y_real", "pp", "Reele rente is vaak belangrijker voor precious metals."),
    ("M2 YoY", "m2_yoy", "pp", "Liquiditeitsgroei kan risk assets en metals ondersteunen."),
    ("Real M2 YoY", "m2_real_yoy", "pp", "Inflatie-gecorrigeerde liquiditeitsimpuls."),
]
available_macro_watch = [(label, col, unit, note) for label, col, unit, note in macro_watch if col in d.columns and d[col].notna().any()]
if available_macro_watch:
    st.subheader("Dollar, rates & liquidity")
    macro_cols = st.columns(min(5, len(available_macro_watch)))
    for i, (label, col, unit, _) in enumerate(available_macro_watch[:5]):
        val = last_val(d, col)
        delta = trend_delta(d, col, trend_look)
        if unit == "pp":
            value_text = pct_as_str(val)
            delta_text = f"{delta:+.2f}pp / {trend_look}d" if pd.notna(delta) else None
        else:
            value_text = f"{val:,.2f}" if pd.notna(val) else "-"
            delta_text = f"{delta:+.2f} / {trend_look}d" if pd.notna(delta) else None
        macro_cols[i].metric(label, value_text, delta_text)

    fig_macro = go.Figure()
    fig_macro.add_trace(
        go.Scatter(
            x=d["date"],
            y=normalize_100(d["silver_close"]),
            name="Silver (=100)",
            line=dict(width=2.5, color="#111111"),
        )
    )
    macro_palette = {
        "dxy_close": "#ef4444",
        "us10y": "#2563eb",
        "tips10y_real": "#7c3aed",
        "m2_yoy": "#16a34a",
        "m2_real_yoy": "#059669",
    }
    for label, col, _, _ in available_macro_watch:
        fig_macro.add_trace(
            go.Scatter(
                x=d["date"],
                y=normalize_100(d[col]),
                name=f"{label} (=100)",
                line=dict(width=2, color=macro_palette.get(col, "#6b7280")),
            )
        )
    fig_macro.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        yaxis_title="Index (=100)",
    )
    st.plotly_chart(fig_macro, use_container_width=True)

    with st.expander("Waarom deze macrofactoren?", expanded=False):
        for label, _, _, note in available_macro_watch:
            st.markdown(f"- **{label}**: {note}")

# ---------- Signals ----------
st.subheader("Signal dashboard")
alerts = []

if d["silver_close"].notna().sum() >= 2:
    close_now = last_val(d, "silver_close")
    ma_mid_now = last_val(d, f"silver_ema{ema_mid}")
    ma_slow_now = last_val(d, f"silver_ema{ema_slow}")
    if pd.notna(close_now) and pd.notna(ma_mid_now) and pd.notna(ma_slow_now):
        if close_now > ma_slow_now and ma_mid_now > ma_slow_now:
            alerts.append(("green", "Bullish silver regime", f"Close > EMA{ema_slow} en EMA{ema_mid} > EMA{ema_slow}"))
        elif close_now < ma_slow_now and ma_mid_now < ma_slow_now:
            alerts.append(("red", "Bearish silver regime", f"Close < EMA{ema_slow} en EMA{ema_mid} < EMA{ema_slow}"))

if pd.notna(ratio_last):
    if ratio_last >= ratio_hi:
        alerts.append(("orange", "Gold/Silver ratio hoog", f"Ratio {ratio_last:.1f}: zilver relatief goedkoop vs goud"))
    elif ratio_last <= ratio_lo:
        alerts.append(("blue", "Gold/Silver ratio laag", f"Ratio {ratio_last:.1f}: zilver relatief duur vs goud"))

for col, label, threshold in [
    ("dxy_close", "DXY daalt", dxy_drop_thr),
    ("us10y", "US 10Y daalt", yield_drop_thr),
    ("tips10y_real", "Real yield daalt", yield_drop_thr),
]:
    delta = trend_delta(d, col, trend_look)
    if pd.notna(delta) and delta < threshold:
        alerts.append(("green", label, f"Delta {trend_look}d = {delta:.2f}"))

for col, label in [("m2_yoy", "M2 YoY verbetert"), ("m2_real_yoy", "Real M2 YoY verbetert")]:
    delta = trend_delta(d, col, trend_look)
    if pd.notna(delta) and delta > m2_impulse_thr:
        alerts.append(("green", label, f"Delta {trend_look}d = {delta:.2f}pp"))

if "copper_close" in d.columns and d["copper_close"].notna().sum() > trend_look:
    c_delta = d["copper_close"].pct_change(trend_look).dropna()
    if not c_delta.empty and c_delta.iloc[-1] > 0:
        alerts.append(("green", "Copper momentum positief", f"Copper {trend_look}d {c_delta.iloc[-1] * 100:.2f}%"))

if "vix_close" in d.columns and d["vix_close"].notna().sum() >= 21:
    vz = zscore(d["vix_close"], 20).dropna()
    if not vz.empty and vz.iloc[-1] >= 2:
        alerts.append(("orange", "VIX spike", f"VIX z-score {vz.iloc[-1]:.2f}"))


def badge(color, text):
    colors = {"green": "#00A65A", "red": "#D55E00", "orange": "#E69F00", "blue": "#1f77b4", "gray": "#6c757d"}
    return f"""<span style="background:{colors[color]};color:white;padding:2px 8px;border-radius:12px;font-size:0.9rem;">{text}</span>"""


if not alerts:
    st.success("Geen alerts op dit moment.")
else:
    for color, title, msg in alerts:
        st.markdown(f"{badge(color, title)} &nbsp; {msg}", unsafe_allow_html=True)

st.divider()

# ---------- Price + overlays ----------
st.subheader("Silver - Price and macro overlays")
sel = st.multiselect(
    "Kies drivers voor overlay",
    options=driver_choices,
    default=default_overlay,
    help="Genormaliseerd toont relatieve performance. Eigen schaal zet drivers op de tweede y-as.",
)

fig = make_subplots(specs=[[{"secondary_y": view_mode.startswith("Eigen")}]] )
primary_range_series = []
secondary_range_series = []

if view_mode.startswith("Genormaliseerd"):
    silver_plot = normalize_100(d["silver_close"])
    fig.add_trace(go.Scatter(x=d["date"], y=silver_plot, name="Silver (=100)", line=dict(width=2.5, color="#111111")))
    primary_range_series.append(silver_plot)
else:
    fig.add_trace(go.Scatter(x=d["date"], y=d["silver_close"], name="Silver (USD/oz)", line=dict(width=2.5, color="#111111")))
    primary_range_series.append(d["silver_close"])

ema_config = {
    "fast": (ema_fast, "#E69F00"),
    "mid": (ema_mid, "#009E73"),
    "slow": (ema_slow, "#0072B2"),
}
if show_emas:
    for key in selected_emas:
        span, color = ema_config[key]
        ema_series = d[f"silver_ema{span}"]
        y = normalize_100(ema_series) if view_mode.startswith("Genormaliseerd") else ema_series
        fig.add_trace(go.Scatter(x=d["date"], y=y, name=f"EMA{span}", line=dict(width=1.8, color=color)))
        primary_range_series.append(y)

palette = ["#6B7280", "#8B5CF6", "#EF4444", "#10B981", "#A855F7", "#F59E0B", "#3B82F6"]
for i, name in enumerate(sel):
    col = DRIVER_MAP[name]
    series = d[col]
    if series.notna().sum() == 0:
        continue
    if view_mode.startswith("Genormaliseerd"):
        y = normalize_100(series)
        fig.add_trace(go.Scatter(x=d["date"], y=y, name=name, line=dict(width=2, color=palette[i % len(palette)])))
        primary_range_series.append(y)
    else:
        fig.add_trace(
            go.Scatter(x=d["date"], y=series, name=name, line=dict(width=2, color=palette[i % len(palette)])),
            secondary_y=True,
        )
        secondary_range_series.append(series)

fig.update_yaxes(
    title_text="Index (=100)" if view_mode.startswith("Genormaliseerd") else "Silver USD/oz",
    range=padded_range(primary_range_series),
    secondary_y=False,
)
if view_mode.startswith("Eigen"):
    fig.update_yaxes(title_text="Drivers", range=padded_range(secondary_range_series), secondary_y=True)
fig.update_layout(height=560, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
st.plotly_chart(fig, use_container_width=True)

if show_delta:
    bars = d[["date", "silver_delta_pct"]].dropna().copy()
    bars["pct"] = bars["silver_delta_pct"] * 100.0
    colors = ["#10B981" if v >= 0 else "#EF4444" for v in bars["pct"]]
    fig_d = go.Figure()
    fig_d.add_trace(go.Bar(x=bars["date"], y=bars["pct"], name="Silver daily delta %", marker_color=colors, opacity=0.9))
    fig_d.add_hline(y=0, line_dash="dot", opacity=0.6)
    fig_d.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    fig_d.update_yaxes(title_text="Delta % dag")
    st.plotly_chart(fig_d, use_container_width=True)

# ---------- TA chart ----------
st.subheader("Silver TA - Heikin Ashi, trend and momentum")
real_ohlc_cols = ["silver_open", "silver_high", "silver_low", "silver_close"]
has_real_ohlc = all(c in d.columns and d[c].notna().any() for c in real_ohlc_cols)

if has_real_ohlc:
    ta_ohlc = d[["date", "silver_open", "silver_high", "silver_low", "silver_close"]].rename(
        columns={
            "silver_open": "open",
            "silver_high": "high",
            "silver_low": "low",
            "silver_close": "close",
        }
    ).copy()
    ohlc_note = "Heikin Ashi op basis van echte OHLC-kolommen."
else:
    ta_ohlc = d[["date", "silver_close"]].copy()
    ta_ohlc["open"] = ta_ohlc["silver_close"].shift(1).fillna(ta_ohlc["silver_close"])
    ta_ohlc["close"] = ta_ohlc["silver_close"]
    ta_ohlc["high"] = ta_ohlc[["open", "close"]].max(axis=1)
    ta_ohlc["low"] = ta_ohlc[["open", "close"]].min(axis=1)
    ohlc_note = "Geen echte silver OHLC-kolommen gevonden; candles zijn synthetisch uit close-to-close beweging."

ha = heikin_ashi(ta_ohlc[["open", "high", "low", "close"]])
ta_plot = pd.concat([ta_ohlc[["date", "close"]], ha], axis=1).dropna(subset=["ha_open", "ha_high", "ha_low", "ha_close"])

fig_ta = make_subplots(
    rows=3,
    cols=1,
    shared_xaxes=True,
    subplot_titles=["Heikin Ashi + trend EMA's", f"RSI({rsi_period})", "MACD(12,26,9)"],
    row_heights=[0.58, 0.20, 0.22],
    vertical_spacing=0.06,
)

fig_ta.add_trace(
    go.Candlestick(
        x=ta_plot["date"],
        open=ta_plot["ha_open"],
        high=ta_plot["ha_high"],
        low=ta_plot["ha_low"],
        close=ta_plot["ha_close"],
        name="Heikin Ashi",
    ),
    row=1,
    col=1,
)

for span, color in [(ema_fast, "#E69F00"), (ema_mid, "#009E73"), (ema_slow, "#0072B2")]:
    ema_col = f"silver_ema{span}"
    if ema_col in d.columns and d[ema_col].notna().any():
        fig_ta.add_trace(
            go.Scatter(x=d["date"], y=d[ema_col], name=f"EMA{span}", line=dict(width=2, color=color)),
            row=1,
            col=1,
        )

fig_ta.add_trace(go.Scatter(x=d["date"], y=d["silver_rsi"], name="RSI", line=dict(width=2)), row=2, col=1)
fig_ta.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
fig_ta.add_hline(y=50, line_dash="dot", row=2, col=1)
fig_ta.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

fig_ta.add_trace(go.Scatter(x=d["date"], y=d["silver_macd_line"], name="MACD", line=dict(width=2)), row=3, col=1)
fig_ta.add_trace(go.Scatter(x=d["date"], y=d["silver_macd_signal"], name="Signal", line=dict(width=2)), row=3, col=1)
fig_ta.add_trace(
    go.Bar(
        x=d["date"],
        y=d["silver_macd_hist"],
        name="Hist",
        marker_color=np.where(d["silver_macd_hist"] >= 0, "rgba(16,150,24,0.65)", "rgba(239,68,68,0.65)"),
    ),
    row=3,
    col=1,
)
fig_ta.add_hline(y=0, line_dash="dot", row=3, col=1)

price_range = padded_range([ta_plot["ha_low"], ta_plot["ha_high"], d[f"silver_ema{ema_mid}"], d[f"silver_ema{ema_slow}"]])
fig_ta.update_layout(
    height=900,
    margin=dict(l=10, r=10, t=70, b=30),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    xaxis_rangeslider_visible=False,
)
fig_ta.update_yaxes(title_text="Silver", range=price_range, row=1, col=1)
fig_ta.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)
fig_ta.update_yaxes(title_text="MACD", row=3, col=1)
fig_ta.update_xaxes(rangeslider_visible=False)
st.plotly_chart(fig_ta, use_container_width=True)
st.caption(ohlc_note)

# ---------- Ratio and momentum ----------
if "gold_silver_ratio" in d.columns:
    st.subheader("Gold/Silver ratio and relative strength")
    fig_r = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=["Gold/Silver ratio", "Silver minus Gold relative performance (=100 spread)"])
    fig_r.add_trace(go.Scatter(x=d["date"], y=d["gold_silver_ratio"], name="Gold/Silver ratio", line=dict(width=2)), row=1, col=1)
    fig_r.add_hline(y=ratio_hi, line_dash="dash", line_color="#E69F00", row=1, col=1)
    fig_r.add_hline(y=ratio_lo, line_dash="dash", line_color="#1f77b4", row=1, col=1)
    if "silver_gold_rel" in d.columns:
        fig_r.add_trace(go.Scatter(x=d["date"], y=d["silver_gold_rel"], name="Silver - Gold rel", line=dict(width=2, color="#6B7280")), row=2, col=1)
        fig_r.add_hline(y=0, line_dash="dot", row=2, col=1)
    fig_r.update_layout(height=620, margin=dict(l=10, r=10, t=60, b=10), legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    fig_r.update_yaxes(title_text="Ratio", row=1, col=1)
    fig_r.update_yaxes(title_text="pp", row=2, col=1)
    st.plotly_chart(fig_r, use_container_width=True)

st.subheader("Momentum diagnostics")
fig_m = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=[f"RSI({rsi_period})", "Z-score 20d"])
fig_m.add_trace(go.Scatter(x=d["date"], y=d["silver_rsi"], name="RSI", line=dict(width=2)), row=1, col=1)
fig_m.add_hline(y=70, line_dash="dash", line_color="red", row=1, col=1)
fig_m.add_hline(y=50, line_dash="dot", row=1, col=1)
fig_m.add_hline(y=30, line_dash="dash", line_color="green", row=1, col=1)
fig_m.add_trace(go.Scatter(x=d["date"], y=d["silver_z20"], name="Z20", line=dict(width=2)), row=2, col=1)
fig_m.add_hline(y=2, line_dash="dash", line_color="red", row=2, col=1)
fig_m.add_hline(y=0, line_dash="dot", row=2, col=1)
fig_m.add_hline(y=-2, line_dash="dash", line_color="green", row=2, col=1)
fig_m.update_layout(height=560, margin=dict(l=10, r=10, t=60, b=10), legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
fig_m.update_yaxes(title_text="RSI", row=1, col=1, range=[0, 100])
fig_m.update_yaxes(title_text="Z-score", row=2, col=1)
st.plotly_chart(fig_m, use_container_width=True)

st.divider()

# ---------- Liquidity ----------
liquidity_cols = [c for c in ["m2_yoy", "m2_real_yoy", "m2_vel_yoy"] if c in d.columns and d[c].notna().any()]
level_cols = [c for c in ["m2", "m2_real", "m2_vel"] if c in d.columns and d[c].notna().any()]
if liquidity_cols or level_cols:
    st.subheader("Liquidity - M2 and velocity")
    fig_l = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        subplot_titles=["M2 YoY / Real M2 YoY / Velocity YoY", "Silver vs liquidity levels (=100)"],
        vertical_spacing=0.10,
    )
    liq_palette = {"m2_yoy": "#2563eb", "m2_real_yoy": "#16a34a", "m2_vel_yoy": "#dc2626"}
    for c in liquidity_cols:
        fig_l.add_trace(
            go.Scatter(x=d["date"], y=d[c], name=c, line=dict(width=2, color=liq_palette.get(c, "#6B7280"))),
            row=1,
            col=1,
        )
    fig_l.add_hline(y=0, line_dash="dot", row=1, col=1)
    fig_l.add_trace(
        go.Scatter(x=d["date"], y=normalize_100(d["silver_close"]), name="Silver (=100)", line=dict(width=2.5, color="#111111")),
        row=2,
        col=1,
    )
    for c in level_cols:
        fig_l.add_trace(go.Scatter(x=d["date"], y=normalize_100(d[c]), name=f"{c} (=100)", line=dict(width=2)), row=2, col=1)
    fig_l.update_layout(height=620, margin=dict(l=10, r=10, t=60, b=10), legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    fig_l.update_yaxes(title_text="YoY %", row=1, col=1)
    fig_l.update_yaxes(title_text="Index", row=2, col=1)
    st.plotly_chart(fig_l, use_container_width=True)

    notes = []
    for c in ["m2_yoy", "m2_real_yoy"]:
        val = last_val(d, c)
        delta = trend_delta(d, c, trend_look)
        if pd.notna(val) and pd.notna(delta):
            direction = "verbetert" if delta > 0 else "verslechtert" if delta < 0 else "vlak"
            notes.append(f"{c}: {val:.2f}% en {direction} over {trend_look} dagen ({delta:+.2f}pp)")
    if notes:
        st.info("Liquidity read-through: " + " | ".join(notes))

st.divider()

# ---------- Rolling correlations ----------
st.subheader("Correlation map - Silver and drivers")
sel_corr = st.multiselect("Kies drivers voor correlatie", options=driver_choices, default=default_overlay)
if sel_corr:
    corr_cols = {"Silver": "silver_close"}
    corr_cols.update({name: DRIVER_MAP[name] for name in sel_corr if DRIVER_MAP[name] in d.columns})

    corr_frame = pd.DataFrame({"date": d["date"]})
    for name, col in corr_cols.items():
        raw = pd.to_numeric(d[col], errors="coerce")
        if corr_transform.startswith("Returns"):
            if name in ["US 10Y (yield %)", "US 10Y Real (TIPS %)", "VIX", "Gold/Silver ratio", "M2 YoY (%)", "Real M2 YoY (%)"]:
                corr_frame[name] = raw.diff()
            else:
                corr_frame[name] = raw.pct_change() * 100.0
        else:
            corr_frame[name] = raw

    corr_frame = corr_frame.set_index("date")
    corr_data = corr_frame.dropna(how="all")
    corr_matrix = corr_data.corr(min_periods=max(10, int(corr_win * 0.5)))

    if corr_matrix.empty or "Silver" not in corr_matrix.columns:
        st.info("Geen voldoende overlappende data voor een correlatie-heatmap.")
    else:
        fig_h = go.Figure(
            data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.index,
                zmin=-1,
                zmax=1,
                colorscale="RdBu",
                reversescale=True,
                colorbar=dict(title="corr"),
                text=np.round(corr_matrix.values, 2),
                texttemplate="%{text}",
                hovertemplate="%{y} vs %{x}: %{z:.2f}<extra></extra>",
            )
        )
        fig_h.update_layout(height=520, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_h, use_container_width=True)

        rel_rows = []
        for name in corr_matrix.columns:
            if name == "Silver":
                continue
            val = corr_matrix.loc["Silver", name] if "Silver" in corr_matrix.index else np.nan
            if pd.notna(val):
                rel_rows.append({"Driver": name, "Corr vs Silver": val, "Abs corr": abs(val)})
        if rel_rows:
            rel_df = pd.DataFrame(rel_rows).sort_values("Abs corr", ascending=False)
            top_txt = []
            for _, row in rel_df.head(3).iterrows():
                direction = "positief" if row["Corr vs Silver"] > 0 else "negatief"
                top_txt.append(f"{row['Driver']}: {direction} ({row['Corr vs Silver']:.2f})")
            st.info("Sterkste samenhang in deze periode: " + " | ".join(top_txt))
            show_rel = rel_df.drop(columns=["Abs corr"]).copy()
            show_rel["Corr vs Silver"] = show_rel["Corr vs Silver"].map(lambda x: f"{x:.2f}")
            st.dataframe(show_rel, use_container_width=True, hide_index=True)

    st.subheader("Lead/lag heatmap - welke driver loopt voor?")
    lag_rows = []
    lag_values = list(range(-lead_lag_max, lead_lag_max + 1, 5))
    silver_series = corr_frame["Silver"]
    for name in sel_corr:
        if name not in corr_frame.columns:
            continue
        driver_series = corr_frame[name]
        row_vals = []
        for lag in lag_values:
            # Positieve lag: driver beweegt eerder dan silver; negatieve lag: silver beweegt eerder.
            shifted_driver = driver_series.shift(lag)
            joined = pd.concat([silver_series.rename("silver"), shifted_driver.rename("driver")], axis=1).dropna()
            row_vals.append(joined["silver"].corr(joined["driver"]) if len(joined) >= 20 else np.nan)
        lag_rows.append(row_vals)

    if lag_rows:
        lag_matrix = np.array(lag_rows, dtype=float)
        fig_lag = go.Figure(
            data=go.Heatmap(
                z=lag_matrix,
                x=lag_values,
                y=sel_corr,
                zmin=-1,
                zmax=1,
                colorscale="RdBu",
                reversescale=True,
                colorbar=dict(title="corr"),
                hovertemplate="Driver %{y}<br>Lag %{x}d: %{z:.2f}<extra></extra>",
            )
        )
        fig_lag.update_layout(
            height=max(360, 70 + 34 * len(sel_corr)),
            margin=dict(l=10, r=10, t=30, b=40),
            xaxis_title="Lag in dagen: + betekent driver leidt silver",
        )
        st.plotly_chart(fig_lag, use_container_width=True)

        best_rows = []
        for driver_name, values in zip(sel_corr, lag_matrix):
            if np.all(np.isnan(values)):
                continue
            idx = int(np.nanargmax(np.abs(values)))
            best_rows.append(
                {
                    "Driver": driver_name,
                    "Beste lag": lag_values[idx],
                    "Corr": values[idx],
                    "Interpretatie": "driver leidt silver" if lag_values[idx] > 0 else "silver leidt driver" if lag_values[idx] < 0 else "zelfde dag",
                }
            )
        if best_rows:
            best_df = pd.DataFrame(best_rows).sort_values("Corr", key=lambda s: s.abs(), ascending=False)
            best_df["Corr"] = best_df["Corr"].map(lambda x: f"{x:.2f}")
            st.dataframe(best_df, use_container_width=True, hide_index=True)

    st.subheader("Rolling correlaties - Silver vs drivers")
    figc = go.Figure()
    any_line = False
    for name in sel_corr:
        if name not in corr_frame.columns:
            continue
        joined = pd.concat([corr_frame["Silver"].rename("silver"), corr_frame[name].rename("driver")], axis=1).dropna()
        if len(joined) < corr_win:
            continue
        rc = joined["silver"].rolling(corr_win).corr(joined["driver"]).dropna()
        figc.add_trace(go.Scatter(x=rc.index, y=rc.values, mode="lines", name=name, line=dict(width=2)))
        any_line = True
    figc.add_hline(y=0.0, line_dash="dot")
    figc.add_hrect(y0=-1, y1=-0.5, fillcolor="rgba(255,0,0,0.06)", line_width=0)
    figc.add_hrect(y0=0.5, y1=1, fillcolor="rgba(0,128,0,0.06)", line_width=0)
    figc.update_layout(height=430, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    figc.update_yaxes(title_text="Rolling corr", range=[-1, 1])
    if any_line:
        st.plotly_chart(figc, use_container_width=True)
    else:
        st.info("Geen voldoende overlappende data voor correlaties in de gekozen periode.")
else:
    st.info("Selecteer minimaal een driver.")

# ---------- Scatter / beta ----------
st.subheader("Beta / scatter - Silver vs drivers")
scatter_drivers = st.multiselect("Drivers voor scatter/beta", options=driver_choices, default=default_scatter)
for name in scatter_drivers:
    col = DRIVER_MAP[name]
    if col not in d.columns or d[col].notna().sum() < 10 or d["silver_close"].notna().sum() < 10:
        st.info(f"Onvoldoende data voor scatter: {name}")
        continue

    if scatter_mode.startswith("%"):
        y = d["silver_close"].pct_change() * 100.0
        x = d[col].pct_change() * 100.0
        x_label = f"{name} delta %"
        y_label = "Silver delta %"
    else:
        y = d["silver_close"]
        x = d[col]
        x_label = name
        y_label = "Silver (USD/oz)"

    scatter_df = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    if len(scatter_df) < 10:
        st.info(f"Te weinig punten voor regressie: {name}")
        continue

    X = scatter_df["x"].values
    Y = scatter_df["y"].values
    slope, intercept, r2, corr = regress_xy(X, Y)
    with st.expander(f"Scatter: Silver vs {name} - beta={slope:.3f}, R2={r2:.3f}, corr={corr:.3f}", expanded=False):
        fig_s = go.Figure()
        fig_s.add_trace(go.Scatter(x=X, y=Y, mode="markers", name="Waarnemingen", opacity=0.6))
        x_line = np.linspace(X.min(), X.max(), 100)
        fig_s.add_trace(go.Scatter(x=x_line, y=slope * x_line + intercept, mode="lines", name=f"Fit: y = {slope:.3f}x + {intercept:.3f}"))
        fig_s.update_layout(height=460, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
        fig_s.update_xaxes(title=x_label)
        fig_s.update_yaxes(title=y_label)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Beta", f"{slope:.3f}")
        c2.metric("Intercept", f"{intercept:.3f}")
        c3.metric("R2", f"{r2:.3f}" if pd.notna(r2) else "-")
        c4.metric("Corr", f"{corr:.3f}" if pd.notna(corr) else "-")
        st.plotly_chart(fig_s, use_container_width=True)

st.divider()

# ---------- Table ----------
st.subheader("Laatste rijen")
show_cols = [
    "date",
    "silver_close",
    f"silver_ema{ema_fast}",
    f"silver_ema{ema_mid}",
    f"silver_ema{ema_slow}",
    "silver_delta_pct",
    "silver_rsi",
    "silver_z20",
    "gold_close",
    "gold_silver_ratio",
    "copper_close",
    "dxy_close",
    "eurusd_close",
    "us10y",
    "tips10y_real",
    "m2",
    "m2_real",
    "m2_yoy",
    "m2_real_yoy",
    "m2_vel",
    "m2_vel_yoy",
    "vix_close",
    "btc_close",
]
show_cols = [c for c in show_cols if c in d.columns]
tail = d[show_cols].tail(200).copy()
if "silver_delta_pct" in tail.columns:
    tail["silver_delta_pct"] = (tail["silver_delta_pct"] * 100).round(2)
st.dataframe(tail, use_container_width=True)
