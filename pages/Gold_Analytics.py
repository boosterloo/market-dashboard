# pages/Gold_Analytics.py
# 🏅 Gold Analytics — Price • TA • Drivers • Correlations • Multi-Scatter/Beta • Tunable Alerts

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
st.set_page_config(page_title="🏅 Gold Analytics", layout="wide")
st.title("🏅 Gold Analytics — Price • TA • Drivers • Correlations • Multi-Scatter/Beta • Alerts")

# ---------- Sources ----------
TABLES = st.secrets.get("tables", {})

# Basis commodity view (gold/silver)
COM_WIDE_VIEW      = TABLES.get("commodities_wide_view",   "nth-pier-468314-p7.marketdata.commodity_prices_wide_v")
# ‘Full’ drivers view (indien je die hebt)
DRIVERS_WIDE_VIEW  = TABLES.get("gold_drivers_wide_view",  "nth-pier-468314-p7.marketdata.gold_drivers_wide_v")

# Fallbacks — hier heb ik NU defaults ingevuld
AEX_VIEW           = TABLES.get("aex_view",               "nth-pier-468314-p7.marketdata.aex_with_vix_v")          # voor VIX
CRYPTO_WIDE_VIEW   = TABLES.get("crypto_daily_wide",      "nth-pier-468314-p7.marketdata.crypto_daily_wide_v")     # voor BTC
FX_WIDE_VIEW       = TABLES.get("fx_wide_view",           "nth-pier-468314-p7.marketdata.fx_daily_wide_v")         # voor DXY/EURUSD
US_YIELD_VIEW      = TABLES.get("us_yield_view",          "nth-pier-468314-p7.marketdata.us_yields_daily_wide_v")  # voor US10Y/TIPS

with st.expander("🔎 Debug: gebruikte bronnen", expanded=False):
    st.write({
        "commodities_wide_view": COM_WIDE_VIEW,
        "gold_drivers_wide_view": DRIVERS_WIDE_VIEW,
        "aex_view (VIX fallback)": AEX_VIEW,
        "crypto_daily_wide (BTC fallback)": CRYPTO_WIDE_VIEW,
        "fx_wide_view (EURUSD/DXY fallback)": FX_WIDE_VIEW,
        "us_yield_view (US10Y/TIPS fallback)": US_YIELD_VIEW,
    })

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
@st.cache_data(ttl=300, show_spinner=False)
def load_com() -> pd.DataFrame:
    df = run_query(f"SELECT * FROM `{COM_WIDE_VIEW}` ORDER BY date")
    if df.empty: return df
    df["date"] = pd.to_datetime(df["date"])
    for c in df.columns:
        if c != "date": df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

@st.cache_data(ttl=300, show_spinner=False)
def load_drv_main() -> pd.DataFrame:
    try:
        df = run_query(f"SELECT * FROM `{DRIVERS_WIDE_VIEW}` ORDER BY date")
        if df.empty: return df
        df["date"] = pd.to_datetime(df["date"])
        for c in df.columns:
            if c != "date": df[c] = pd.to_numeric(df[c], errors="coerce")
        return df
    except (NotFound, Forbidden, BadRequest):
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=600, show_spinner=False)
def load_vix_fallback() -> pd.DataFrame:
    try:
        d = run_query(f"SELECT date, vix_close FROM `{AEX_VIEW}` ORDER BY date")
        if d.empty: return pd.DataFrame()
        d["date"] = pd.to_datetime(d["date"])
        d["vix_close"] = pd.to_numeric(d["vix_close"], errors="coerce")
        return d
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
            if d.empty or "btc_close" not in d.columns: 
                continue
            d["date"] = pd.to_datetime(d["date"])
            d["btc_close"] = pd.to_numeric(d["btc_close"], errors="coerce")
            return d
        except Exception:
            continue
    return pd.DataFrame()

@st.cache_data(ttl=600, show_spinner=False)
def load_fx_fallback() -> pd.DataFrame:
    merged = None
    for sql in [
        f"SELECT date, eurusd_close FROM `{FX_WIDE_VIEW}` ORDER BY date",
        f"SELECT date, dxy_close FROM `{FX_WIDE_VIEW}` ORDER BY date",
        f"SELECT date, EURUSD AS eurusd_close FROM `{FX_WIDE_VIEW}` ORDER BY date",
        f"SELECT date, DXY AS dxy_close FROM `{FX_WIDE_VIEW}` ORDER BY date",
    ]:
        try:
            d = run_query(sql)
            if d.empty: 
                continue
            d["date"] = pd.to_datetime(d["date"])
            for c in d.columns:
                if c != "date": d[c] = pd.to_numeric(d[c], errors="coerce")
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
            d["date"] = pd.to_datetime(d["date"])
            for c in d.columns:
                if c != "date": d[c] = pd.to_numeric(d[c], errors="coerce")
            merged = d if merged is None else pd.merge(merged, d, on="date", how="outer")
        except Exception:
            continue
    if merged is not None:
        cols, seen = [], set()
        for c in merged.columns:
            if c == "date" or c not in seen:
                cols.append(c); seen.add(c)
        merged = merged[cols]
    return merged if merged is not None else pd.DataFrame()

# ---------- Load & merge ----------
df_com = load_com()
if df_com.empty:
    st.warning("Geen data in commodities wide view."); st.stop()

base_cols = ["date"] + [c for c in df_com.columns if c.startswith("gold_") or c == "silver_close"]
df = df_com[base_cols].copy()

drv_main = load_drv_main()
if drv_main.empty:
    st.warning(f"Drivers-view niet gevonden of geen rechten: `{DRIVERS_WIDE_VIEW}`. Fallback-bronnen ingeschakeld.")
else:
    df = pd.merge(df, drv_main, on="date", how="outer")

for fb in [load_vix_fallback(), load_crypto_fallback(), load_fx_fallback(), load_yield_fallback()]:
    if fb is not None and not fb.empty:
        df = pd.merge(df, fb, on="date", how="outer")

df = df.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)

# ---------- Helpers ----------
def ema(series: pd.Series, span: int) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    return s.ewm(span=span, adjust=False, min_periods=span).mean()

def zscore(s: pd.Series, win: int = 20) -> pd.Series:
    mu = s.rolling(win, min_periods=win).mean()
    sd = s.rolling(win, min_periods=win).std()
    return (s - mu) / sd

def normalize_100(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    base = s.dropna().iloc[0] if s.notna().any() else np.nan
    return s/base*100.0 if pd.notna(base) and base not in (0, np.nan) else s

def pct_as_str(x): return "—" if pd.isna(x) else f"{x:.2f}%"

def regress_xy(xv: np.ndarray, yv: np.ndarray):
    slope, intercept = np.polyfit(xv, yv, 1)
    y_hat = slope*xv + intercept
    ss_res = np.sum((yv - y_hat)**2)
    ss_tot = np.sum((yv - np.mean(yv))**2)
    r2 = 1 - ss_res/ss_tot if ss_tot != 0 else np.nan
    corr = np.corrcoef(xv, yv)[0,1] if len(xv) > 1 else np.nan
    return slope, intercept, r2, corr

# ---------- Sidebar ----------
with st.sidebar:
    st.header("⚙️ Instellingen")

    DRIVER_MAP = {
        "BTC (BTCUSD)": "btc_close",
        "DXY (Dollar Index)": "dxy_close",
        "EURUSD": "eurusd_close",
        "US 10Y (yield %)": "us10y",
        "US 10Y Real (TIPS %)": "tips10y_real",
        "VIX": "vix_close",
        "Silver (USD/oz)": "silver_close",
    }
    # 👉 Toon optie zodra kolom bestaat (niet filteren op notna())
    driver_choices = [n for n,c in DRIVER_MAP.items() if c in df.columns]

    view_mode = st.radio("Overlayschaling", ["Genormaliseerd (=100)", "Eigen schaal (2e y-as)"], index=0)
    show_delta = st.checkbox("Δ%-bars tonen (Gold)", value=True)

    st.divider(); st.markdown("#### 🔔 Signals & thresholds")
    ema_fast = st.number_input("EMA fast", 5, 100, 20, step=1)
    ema_mid  = st.number_input("EMA mid", 10, 150, 50, step=1)
    ema_slow = st.number_input("EMA slow", 50, 400, 200, step=5)

    vix_z_hi = st.slider("VIX z-score (20d) — spike drempel", 0.5, 4.0, 2.0, 0.1)
    vix_z_lo = st.slider("VIX z-score (20d) — 'laag' drempel", -3.0, -0.5, -1.5, 0.1)

    trend_look = st.slider("Lookback (dagen) voor trend drivers", 5, 60, 20, 1)
    trend_thr  = st.number_input("Trend drempel (Δ over lookback, units driver)", -5.0, 5.0, 0.0, step=0.1)

    corr_win = st.slider("Rolling corr window (dagen)", 10, 180, 60, 5)
    corr_neg_dxy = st.slider("Negatieve corr-drempel (Gold–DXY)", -1.0, 0.0, -0.6, 0.05)
    corr_neg_yld = st.slider("Negatieve corr-drempel (Gold–US10Y)", -1.0, 0.0, -0.5, 0.05)

    st.divider(); st.markdown("#### VIX & plot")
    vix_avail = ("vix_close" in df.columns)
    show_vix = st.toggle("VIX overlay in prijs-paneel", value=vix_avail, disabled=not vix_avail)
    vix_as_z = st.checkbox("Plot VIX als z-score (20d)", value=False, disabled=not vix_avail)

    st.divider(); st.markdown("#### Scatter/Beta")
    scatter_mode = st.radio("Transformatie", ["%-returns vs %-returns", "Levels vs Levels"], index=0)
    scatter_drivers = st.multiselect(
        "Drivers voor scatter/beta",
        options=driver_choices,
        default=[x for x in ["US 10Y (yield %)","DXY (Dollar Index)","BTC (BTCUSD)","VIX","Silver (USD/oz)"] if x in driver_choices],
        disabled=(len(driver_choices) == 0)
    )

# ---------- Periode ----------
if "date" not in df.columns or df["date"].isna().all():
    st.error("Geen geldige 'date' kolom na merge — check views."); st.stop()

min_d, max_d = df["date"].min().date(), df["date"].max().date()
default_start = max(max_d - timedelta(days=365), min_d)
start_d, end_d = st.slider("📅 Periode", min_value=min_d, max_value=max_d,
                           value=(default_start, max_d), step=timedelta(days=1), format="YYYY-MM-DD")

mask = (df["date"].dt.date >= start_d) & (df["date"].dt.date <= end_d)
d = df.loc[mask].copy()
if d.empty:
    st.info("Geen data in de gekozen periode."); st.stop()

# ---------- Debug: presence & counts ----------
with st.expander("🧪 Debug: driver-kolommen en non-null counts (na merge)", expanded=False):
    dbg_cols = ["silver_close","btc_close","dxy_close","eurusd_close","us10y","tips10y_real","vix_close"]
    present = {c: (c in df.columns) for c in dbg_cols}
    counts_all = {c: (int(df[c].notna().sum()) if c in df.columns else 0) for c in dbg_cols}
    counts_range = {c: (int(d[c].notna().sum()) if c in d.columns else 0) for c in dbg_cols}
    st.write({"present(all)": present, "notna_count(all)": counts_all, "notna_count(in_range)": counts_range})

# ---------- EMAs ----------
for span in [ema_fast, ema_mid, ema_slow]:
    col = f"gold_ma{span}"
    if col not in d.columns: d[col] = ema(d["gold_close"], span)

# ---------- KPI’s ----------
st.subheader("KPI’s")
last = d.dropna(subset=["gold_close"]).tail(1)
gold_last = float(last["gold_close"].iloc[0]) if not last.empty else np.nan
if "gold_delta_pct" in d.columns and d["gold_delta_pct"].notna().any():
    gold_dpct = d["gold_delta_pct"].dropna().iloc[-1]*100.0
else:
    gold_dpct = (d["gold_close"].pct_change().dropna().iloc[-1]*100.0) if d["gold_close"].notna().sum()>=2 else np.nan

def last_val(col):
    return (float(d[col].dropna().iloc[-1]) if col in d.columns and d[col].notna().any() else np.nan)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Gold (USD/oz)", f"{gold_last:,.2f}" if pd.notna(gold_last) else "—", pct_as_str(gold_dpct))
k2.metric("US 10Y", pct_as_str(last_val("us10y")))
k3.metric("Real 10Y", pct_as_str(last_val("tips10y_real")))
k4.metric("VIX", f"{last_val('vix_close'):.2f}" if pd.notna(last_val('vix_close')) else "—")

# ---------- Alerts ----------
st.subheader("📣 Signal Alerts (config via sidebar)")
alerts = []
maF, maM, maS = d[f"gold_ma{ema_fast}"], d[f"gold_ma{ema_mid}"], d[f"gold_ma{ema_slow}"]
if len(d) >= 2:
    close_now = d["gold_close"].iloc[-1]
    up_regime   = (close_now > maS.iloc[-1]) and (maM.iloc[-1] > maS.iloc[-1])
    down_regime = (close_now < maS.iloc[-1]) and (maM.iloc[-1] < maS.iloc[-1])
    if up_regime:   alerts.append(("green","Bullish regime", f"Close > EMA{ema_slow} en EMA{ema_mid} > EMA{ema_slow}"))
    elif down_regime: alerts.append(("red","Bearish regime", f"Close < EMA{ema_slow} en EMA{ema_mid} < EMA{ema_slow}"))

def crossed(a: pd.Series, b: pd.Series, up=True):
    if len(a) < 2 or len(b) < 2: return False
    if up:  return (a.shift(1).iloc[-1] <= b.shift(1).iloc[-1]) and (a.iloc[-1] > b.iloc[-1])
    return  (a.shift(1).iloc[-1] >= b.shift(1).iloc[-1]) and (a.iloc[-1] < b.iloc[-1])

if crossed(maM, maS, up=True):  alerts.append(("green","Golden cross", f"EMA{ema_mid} ↑ EMA{ema_slow}"))
if crossed(maM, maS, up=False): alerts.append(("red","Death cross",  f"EMA{ema_mid} ↓ EMA{ema_slow}"))

def trend_down(col: str, look: int, thr: float) -> bool:
    if col not in d.columns or d[col].notna().sum() < look+1: return False
    s = d[col].dropna()
    return (s.iloc[-1] - s.iloc[-1-look]) < thr

if trend_down("tips10y_real", trend_look, trend_thr): alerts.append(("green","Real yield ↓", f"TIPS10Y Δ{trend_look}d < {trend_thr:.2f}"))
if trend_down("us10y",       trend_look, trend_thr): alerts.append(("green","US 10Y ↓",     f"US10Y Δ{trend_look}d < {trend_thr:.2f}"))
if trend_down("dxy_close",   trend_look, trend_thr): alerts.append(("green","DXY ↓",        f"DXY Δ{trend_look}d < {trend_thr:.2f}"))

if "vix_close" in d.columns and d["vix_close"].notna().sum() >= 21:
    vz = zscore(d["vix_close"], 20).iloc[-1]
    if pd.notna(vz) and vz >= vix_z_hi: alerts.append(("orange","VIX spike", f"z≥{vix_z_hi:.1f}"))
    if pd.notna(vz) and vz <= vix_z_lo: alerts.append(("blue","VIX laag", f"z≤{vix_z_lo:.1f}"))

def rolling_corr(a: pd.Series, b: pd.Series, win: int) -> float:
    x = pd.concat([a.pct_change(), b.pct_change()], axis=1).dropna()
    if len(x) < win: return np.nan
    return x.iloc[-win:, 0].corr(x.iloc[-win:, 1])

if all(c in d.columns for c in ["gold_close","dxy_close"]):
    cD = rolling_corr(d["gold_close"], d["dxy_close"], corr_win)
    if pd.notna(cD) and cD <= corr_neg_dxy: alerts.append(("green","Sterk negatief Gold–DXY", f"corr≈{cD:.2f} (≤ {corr_neg_dxy:.2f})"))
if all(c in d.columns for c in ["gold_close","us10y"]):
    cY = rolling_corr(d["gold_close"], d["us10y"], corr_win)
    if pd.notna(cY) and cY <= corr_neg_yld: alerts.append(("green","Negatief Gold–US10Y", f"corr≈{cY:.2f} (≤ {corr_neg_yld:.2f})"))

def badge(color, text):
    colors = {"green":"#00A65A","red":"#D55E00","orange":"#E69F00","blue":"#1f77b4","gray":"#6c757d"}
    return f"""<span style="background:{colors[color]};color:white;padding:2px 8px;border-radius:12px;font-size:0.9rem;">{text}</span>"""

with st.container():
    if not alerts:
        st.success("Geen alerts op dit moment.")
    else:
        for color, title, msg in alerts:
            st.markdown(f"{badge(color, title)} &nbsp; {msg}", unsafe_allow_html=True)

st.divider()

# ---------- Paneel 1: Gold + EMA + Drivers overlay ----------
st.subheader("Gold — Price & EMA + Drivers overlay")
sel = st.multiselect(
    "Kies drivers voor overlay/vergelijking",
    options=driver_choices,
    default=[x for x in ["US 10Y (yield %)","US 10Y Real (TIPS %)","BTC (BTCUSD)","DXY (Dollar Index)","Silver (USD/oz)","VIX"] if x in driver_choices],
    help="Bij ‘Genormaliseerd (=100)’ delen alle lijnen dezelfde schaal; anders krijgen drivers de 2e y-as.",
    disabled=(len(driver_choices) == 0)
)

fig = make_subplots(specs=[[{"secondary_y": (view_mode.startswith("Eigen"))}]])
# Gold
fig.add_trace(go.Scatter(x=d["date"], y=d["gold_close"], name="Gold (USD/oz)", line=dict(width=2, color="#111111")))
fig.add_trace(go.Scatter(x=d["date"], y=d[f"gold_ma{ema_fast}"],  name=f"EMA{ema_fast}", line=dict(width=2, color="#E69F00")))
fig.add_trace(go.Scatter(x=d["date"], y=d[f"gold_ma{ema_mid}"],   name=f"EMA{ema_mid}",  line=dict(width=2, color="#009E73")))
fig.add_trace(go.Scatter(x=d["date"], y=d[f"gold_ma{ema_slow}"],  name=f"EMA{ema_slow}", line=dict(width=2, color="#0072B2")))

palette = ["#6B7280","#9CA3AF","#8B5CF6","#EF4444","#10B981","#A855F7","#F59E0B","#3B82F6"]
pi = 0

def add_overlay(name, series, color, secondary=False):
    if view_mode.startswith("Genormaliseerd"):
        base_gold = normalize_100(d["gold_close"])
        if "Gold (=100)" not in [t.name for t in fig.data]:
            fig.add_trace(go.Scatter(x=d["date"], y=base_gold, name="Gold (=100)",
                                     line=dict(width=1.8, color="#111111", dash="dot")))
        fig.add_trace(go.Scatter(x=d["date"], y=normalize_100(series), name=name, line=dict(width=2, color=color)))
    else:
        fig.add_trace(go.Scatter(x=d["date"], y=series, name=name, line=dict(width=2, color=color)), secondary_y=secondary)

# VIX overlay (aparte toggle)
if ("vix_close" in d.columns) and show_vix:
    vseries = d["vix_close"]
    if vseries.notna().any():
        if vix_as_z: vseries = zscore(vseries, 20)
        add_overlay("VIX (z)" if vix_as_z else "VIX", vseries, "#A855F7", secondary=True)

for name in sel:
    col = DRIVER_MAP[name]
    if col not in d.columns: 
        continue
    ser = d[col]
    if ser.notna().sum() == 0:
        continue
    add_overlay(name, ser, palette[pi % len(palette)], secondary=True)
    pi += 1

yl = "USD/oz" if not view_mode.startswith("Genormaliseerd") else "Index (=100)"
fig.update_yaxes(title_text=yl, secondary_y=False)
if view_mode.startswith("Eigen"):
    fig.update_yaxes(title_text="Drivers (2e as)", secondary_y=True)
fig.update_layout(height=520, margin=dict(l=10, r=10, t=30, b=10),
                  legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
st.plotly_chart(fig, use_container_width=True)

# ---------- Δ%-bars ----------
if show_delta:
    if "gold_delta_pct" in d.columns and d["gold_delta_pct"].notna().any():
        bars = d[["date","gold_delta_pct"]].dropna().copy()
        bars["pct"] = bars["gold_delta_pct"] * 100.0
    else:
        tmp = d[["date","gold_close"]].dropna().copy()
        tmp["pct"] = tmp["gold_close"].pct_change() * 100.0
        bars = tmp.dropna()
    colors = ["#10B981" if v >= 0 else "#EF4444" for v in bars["pct"]]
    fig_d = go.Figure()
    fig_d.add_trace(go.Bar(x=bars["date"], y=bars["pct"], name="Δ% per dag", marker_color=colors, opacity=0.9))
    fig_d.add_hline(y=0, line_dash="dot", opacity=0.6)
    fig_d.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    fig_d.update_yaxes(title_text="Δ% dag")
    st.plotly_chart(fig_d, use_container_width=True)

st.divider()

# ---------- Rolling correlaties ----------
st.subheader("Rolling correlaties (Gold ↔ drivers)")
if len(driver_choices) == 0:
    st.info("Geen drivers beschikbaar in dataset.")
else:
    sel_corr = st.multiselect(
        "Kies drivers voor correlatie",
        options=driver_choices,
        default=[x for x in ["US 10Y (yield %)","US 10Y Real (TIPS %)","DXY (Dollar Index)","BTC (BTCUSD)","VIX","Silver (USD/oz)"] if x in driver_choices]
    )
    if sel_corr:
        gold_ret = d["gold_close"].pct_change()
        figc = go.Figure(); any_line = False
        for name in sel_corr:
            col = DRIVER_MAP.get(name)
            if not col or col not in d.columns: continue
            series = d[col]
            as_return = name in ["BTC (BTCUSD)", "Silver (USD/oz)"]
            x = series.pct_change() if as_return else series
            join = pd.concat([gold_ret.rename("g"), x.rename("x")], axis=1).dropna()
            if join.empty or len(join) < max(10, corr_win): continue
            rc = join["g"].rolling(corr_win).corr(join["x"]).dropna()
            if rc.empty: continue
            figc.add_trace(go.Scatter(x=rc.index, y=rc.values, mode="lines", name=name, line=dict(width=2)))
            any_line = True
        figc.add_hline(y=0.0, line_dash="dot")
        figc.add_hrect(y0=-1, y1=-0.5, fillcolor="rgba(255,0,0,0.06)", line_width=0)
        figc.add_hrect(y0=0.5, y1=1,   fillcolor="rgba(0,128,0,0.06)", line_width=0)
        figc.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10),
                           legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
        figc.update_yaxes(title_text="Rolling corr", range=[-1, 1])
        if any_line: st.plotly_chart(figc, use_container_width=True)
        else:        st.info("Geen voldoende overlappende data om correlaties te tekenen in de gekozen periode.")
    else:
        st.info("Selecteer minimaal één driver voor correlaties.")

# ---------- Multi-Scatter / Beta ----------
st.subheader("β / Scatter — Gold vs meerdere drivers")
if len(scatter_drivers) == 0:
    st.info("Geen drivers geselecteerd voor scatter/beta.")
else:
    for name in scatter_drivers:
        drv_col = DRIVER_MAP.get(name)
        if drv_col not in d.columns or d[drv_col].notna().sum() < 5 or d["gold_close"].notna().sum() < 5:
            st.info(f"Onvoldoende data voor scatter: {name}"); continue
        if scatter_mode.startswith("%"):
            y = d["gold_close"].pct_change()*100.0
            x = d[drv_col].pct_change()*100.0
            x_label = f"{name} Δ%"; y_label = "Gold Δ%"
        else:
            y = d["gold_close"]; x = d[drv_col]
            x_label = name; y_label = "Gold (USD/oz)"
        scatter_df = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
        if len(scatter_df) < 10:
            st.info(f"Te weinig punten voor regressie: {name}"); continue
        X = scatter_df["x"].values; Y = scatter_df["y"].values
        slope, intercept, r2, corr = regress_xy(X, Y)
        with st.expander(f"Scatter: Gold vs {name} — β={slope:.3f}, R²={r2:.3f}, corr={corr:.3f}", expanded=False):
            fig_s = go.Figure()
            fig_s.add_trace(go.Scatter(x=X, y=Y, mode="markers", name="Waarnemingen", opacity=0.6))
            x_line = np.linspace(X.min(), X.max(), 100); y_line = slope*x_line + intercept
            fig_s.add_trace(go.Scatter(x=x_line, y=y_line, mode="lines",
                                       name=f"Fit: y = {slope:.3f}x + {intercept:.3f}"))
            fig_s.update_layout(height=460, margin=dict(l=10,r=10,t=30,b=10),
                                legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
            fig_s.update_xaxes(title=x_label); fig_s.update_yaxes(title=y_label)
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("β (slope)", f"{slope:.3f}")
            c2.metric("Intercept", f"{intercept:.3f}")
            c3.metric("R²", f"{r2:.3f}" if pd.notna(r2) else "—")
            c4.metric("Corr", f"{corr:.3f}" if pd.notna(corr) else "—")
            st.plotly_chart(fig_s, use_container_width=True)

st.divider()

# ---------- Tabel ----------
st.subheader("Laatste rijen (gefilterd bereik)")
show_cols = ["date", "gold_close", f"gold_ma{ema_fast}", f"gold_ma{ema_mid}", f"gold_ma{ema_slow}",
             "gold_delta_pct", "silver_close", "btc_close", "dxy_close", "eurusd_close", "us10y", "tips10y_real", "vix_close"]
show_cols = [c for c in show_cols if c in d.columns]
tail = d[show_cols].tail(200).copy()
if "gold_delta_pct" in tail.columns:
    tail["gold_delta_pct"] = (tail["gold_delta_pct"]*100).round(2)
st.dataframe(tail, use_container_width=True)
