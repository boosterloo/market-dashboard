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

with st.expander("Debug: gebruikte bronnen", expanded=False):
    st.write(
        {
            "commodities_wide_view": COM_WIDE_VIEW,
            "drivers_wide_view": DRIVERS_WIDE_VIEW,
            "aex_view (VIX fallback)": AEX_VIEW,
            "crypto_daily_wide (BTC fallback)": CRYPTO_WIDE_VIEW,
            "fx_wide_view (DXY/EURUSD fallback)": FX_WIDE_VIEW,
            "us_yield_view (US10Y/TIPS fallback)": US_YIELD_VIEW,
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
        f"SELECT date, DXY AS dxy_close FROM `{FX_WIDE_VIEW}` ORDER BY date",
        f"SELECT date, eurusd_close FROM `{FX_WIDE_VIEW}` ORDER BY date",
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
    return merged if merged is not None else pd.DataFrame()


@st.cache_data(ttl=600, show_spinner=False)
def load_yield_fallback() -> pd.DataFrame:
    merged = None
    for sql in [
        f"SELECT date, y_10y AS us10y, tips10y_real FROM `{US_YIELD_VIEW}` ORDER BY date",
        f"SELECT date, y_10y_synth AS us10y, tips10y_real FROM `{US_YIELD_VIEW}` ORDER BY date",
        f"SELECT date, us10y, tips10y_real FROM `{US_YIELD_VIEW}` ORDER BY date",
        f"SELECT date, y_10y AS us10y FROM `{US_YIELD_VIEW}` ORDER BY date",
        f"SELECT date, y_10y_synth AS us10y FROM `{US_YIELD_VIEW}` ORDER BY date",
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
    return merged if merged is not None else pd.DataFrame()


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
if drv_main.empty:
    st.warning(f"Drivers-view niet gevonden of geen rechten: `{DRIVERS_WIDE_VIEW}`. Fallback-bronnen ingeschakeld.")
else:
    df = merge_new_cols(df, drv_main)

for fallback in [load_vix_fallback(), load_crypto_fallback(), load_fx_fallback(), load_yield_fallback()]:
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


def zscore(s: pd.Series, win: int = 20) -> pd.Series:
    mu = s.rolling(win, min_periods=win).mean()
    sd = s.rolling(win, min_periods=win).std()
    return (s - mu) / sd.replace(0, np.nan)


def normalize_100(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    valid = s.dropna()
    if valid.empty or valid.iloc[0] == 0:
        return s
    return s / valid.iloc[0] * 100.0


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
    rsi_period = st.slider("RSI periode", 5, 40, 14, 1)

    st.divider()
    st.markdown("#### Drivers & alerts")
    trend_look = st.slider("Driver trend lookback (dagen)", 5, 90, 20, 1)
    dxy_drop_thr = st.number_input("DXY bullish als delta lager dan", -10.0, 10.0, 0.0, step=0.1)
    yield_drop_thr = st.number_input("Yield bullish als delta lager dan", -5.0, 5.0, 0.0, step=0.1)
    ratio_hi = st.slider("Gold/Silver ratio hoog", 50.0, 120.0, 85.0, 1.0)
    ratio_lo = st.slider("Gold/Silver ratio laag", 30.0, 90.0, 65.0, 1.0)

    st.divider()
    st.markdown("#### Correlatie / beta")
    corr_win = st.slider("Rolling corr window", 10, 180, 60, 5)
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

for span in [ema_fast, ema_mid, ema_slow]:
    d[f"silver_ema{span}"] = ema(d["silver_close"], span)

d["silver_rsi"] = rsi_wilder(d["silver_close"], rsi_period)
d["silver_z20"] = zscore(d["silver_close"], 20)

driver_choices = [name for name, col in DRIVER_MAP.items() if col in d.columns and d[col].notna().any()]
default_overlay = [
    x for x in ["Gold (USD/oz)", "Gold/Silver ratio", "Copper (industrial proxy)", "DXY (Dollar Index)", "US 10Y Real (TIPS %)", "VIX"]
    if x in driver_choices
]
default_scatter = [
    x for x in ["Gold (USD/oz)", "Copper (industrial proxy)", "DXY (Dollar Index)", "US 10Y (yield %)", "US 10Y Real (TIPS %)"]
    if x in driver_choices
]

# ---------- Debug ----------
with st.expander("Debug: driver-kolommen en non-null counts", expanded=False):
    dbg_cols = [
        "silver_close", "gold_close", "gold_silver_ratio", "copper_close",
        "dxy_close", "eurusd_close", "us10y", "tips10y_real", "vix_close", "btc_close",
    ]
    st.write(
        {
            "present": {c: c in df.columns or c in d.columns for c in dbg_cols},
            "notna_count_in_range": {c: int(d[c].notna().sum()) if c in d.columns else 0 for c in dbg_cols},
        }
    )

# ---------- KPI ----------
st.subheader("KPI's")
silver_last = last_val(d, "silver_close")
silver_dpct = latest_pct(d, "silver_close", "silver_delta_pct")
gold_last = last_val(d, "gold_close")
ratio_last = last_val(d, "gold_silver_ratio")
copper_dpct = latest_pct(d, "copper_close", "copper_delta_pct") if "copper_close" in d.columns else np.nan

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Silver (USD/oz)", f"{silver_last:,.2f}" if pd.notna(silver_last) else "-", pct_as_str(silver_dpct))
k2.metric("Gold (USD/oz)", f"{gold_last:,.2f}" if pd.notna(gold_last) else "-")
k3.metric("Gold/Silver ratio", f"{ratio_last:,.1f}" if pd.notna(ratio_last) else "-")
k4.metric("Copper delta", pct_as_str(copper_dpct))
k5.metric("DXY", f"{last_val(d, 'dxy_close'):.2f}" if pd.notna(last_val(d, "dxy_close")) else "-")
k6.metric("Real 10Y", pct_as_str(last_val(d, "tips10y_real")))

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
st.subheader("Silver - Price, EMA and macro overlays")
sel = st.multiselect(
    "Kies drivers voor overlay",
    options=driver_choices,
    default=default_overlay,
    help="Genormaliseerd toont relatieve performance. Eigen schaal zet drivers op de tweede y-as.",
)

fig = make_subplots(specs=[[{"secondary_y": view_mode.startswith("Eigen")}]] )
fig.add_trace(go.Scatter(x=d["date"], y=d["silver_close"], name="Silver (USD/oz)", line=dict(width=2.5, color="#111111")))
fig.add_trace(go.Scatter(x=d["date"], y=d[f"silver_ema{ema_fast}"], name=f"EMA{ema_fast}", line=dict(width=2, color="#E69F00")))
fig.add_trace(go.Scatter(x=d["date"], y=d[f"silver_ema{ema_mid}"], name=f"EMA{ema_mid}", line=dict(width=2, color="#009E73")))
fig.add_trace(go.Scatter(x=d["date"], y=d[f"silver_ema{ema_slow}"], name=f"EMA{ema_slow}", line=dict(width=2, color="#0072B2")))

palette = ["#6B7280", "#8B5CF6", "#EF4444", "#10B981", "#A855F7", "#F59E0B", "#3B82F6"]
base_added = False
for i, name in enumerate(sel):
    col = DRIVER_MAP[name]
    series = d[col]
    if series.notna().sum() == 0:
        continue
    if view_mode.startswith("Genormaliseerd"):
        if not base_added:
            fig.add_trace(
                go.Scatter(
                    x=d["date"],
                    y=normalize_100(d["silver_close"]),
                    name="Silver (=100)",
                    line=dict(width=1.8, color="#111111", dash="dot"),
                )
            )
            base_added = True
        y = normalize_100(series)
        fig.add_trace(go.Scatter(x=d["date"], y=y, name=name, line=dict(width=2, color=palette[i % len(palette)])))
    else:
        fig.add_trace(
            go.Scatter(x=d["date"], y=series, name=name, line=dict(width=2, color=palette[i % len(palette)])),
            secondary_y=True,
        )

fig.update_yaxes(title_text="Index (=100)" if view_mode.startswith("Genormaliseerd") else "Silver USD/oz", secondary_y=False)
if view_mode.startswith("Eigen"):
    fig.update_yaxes(title_text="Drivers", secondary_y=True)
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

# ---------- Rolling correlations ----------
st.subheader("Rolling correlaties - Silver vs drivers")
sel_corr = st.multiselect("Kies drivers voor correlatie", options=driver_choices, default=default_overlay)
if sel_corr:
    figc = go.Figure()
    any_line = False
    for name in sel_corr:
        col = DRIVER_MAP[name]
        returns = name not in ["US 10Y (yield %)", "US 10Y Real (TIPS %)", "VIX", "Gold/Silver ratio"]
        rc = rolling_corr(d["silver_close"], d[col], corr_win, returns=returns)
        if rc.empty:
            continue
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
    "vix_close",
    "btc_close",
]
show_cols = [c for c in show_cols if c in d.columns]
tail = d[show_cols].tail(200).copy()
if "silver_delta_pct" in tail.columns:
    tail["silver_delta_pct"] = (tail["silver_delta_pct"] * 100).round(2)
st.dataframe(tail, use_container_width=True)
