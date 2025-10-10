# pages/Gold_Analytics.py
# 🏅 Gold Analytics — Price • TA • Drivers • Correlations • Beta/Scatter • Alerts

import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
st.title("🏅 Gold Analytics — Price • TA • Drivers • Correlations • Beta/Scatter • Alerts")

# ---------- Sources ----------
TABLES = st.secrets.get("tables", {})
COM_WIDE_VIEW     = TABLES.get("commodities_wide_view", "nth-pier-468314-p7.marketdata.commodity_prices_wide_v")
DRIVERS_WIDE_VIEW = TABLES.get("gold_drivers_wide_view", "nth-pier-468314-p7.marketdata.gold_drivers_wide_v")

# ---------- Health ----------
try:
    if not bq_ping():
        st.error("Geen BigQuery-verbinding. Controleer Secrets ([gcp_service_account]).")
        st.stop()
except Exception as e:
    st.error("Geen BigQuery-verbinding.")
    st.caption(f"Details: {e}")
    st.stop()

# ---------- Load data ----------
@st.cache_data(ttl=300, show_spinner=False)
def load_com() -> pd.DataFrame:
    df = run_query(f"SELECT * FROM `{COM_WIDE_VIEW}` ORDER BY date")
    if df.empty: return df
    df["date"] = pd.to_datetime(df["date"])
    for c in df.columns:
        if c != "date": df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

@st.cache_data(ttl=300, show_spinner=False)
def load_drv() -> pd.DataFrame:
    df = run_query(f"SELECT * FROM `{DRIVERS_WIDE_VIEW}` ORDER BY date")
    if df.empty: return df
    df["date"] = pd.to_datetime(df["date"])
    for c in df.columns:
        if c != "date": df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

df_com = load_com()
df_drv = load_drv()
if df_com.empty:
    st.warning("Geen data in commodities wide view.")
    st.stop()

have = set(df_com.columns) | set(df_drv.columns)
if "gold_close" not in have:
    st.error("Vereiste kolom ontbreekt: gold_close")
    st.stop()

# Merge (outer) & housekeeping
df = pd.merge(
    df_com[["date"] + [c for c in df_com.columns if c.startswith("gold_") or c in ["silver_close"]]],
    df_drv, on="date", how="outer"
).sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)

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

def pct_as_str(x):
    return "—" if pd.isna(x) else f"{x:.2f}%"

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
        # "S&P 500": "spx_close",  # optioneel
    }
    driver_choices = [n for n,c in DRIVER_MAP.items() if c in df.columns and df[c].notna().any()]

    view_mode = st.radio("Overlayschaling", ["Genormaliseerd (=100)", "Eigen schaal (2e y-as)"], index=0)
    show_delta = st.checkbox("Δ%-bars tonen (Gold)", value=True)

    st.divider()
    st.markdown("#### VIX & correlatie")
    vix_avail = ("vix_close" in df.columns) and df["vix_close"].notna().any()
    show_vix = st.toggle("VIX overlay in prijs-paneel", value=vix_avail, disabled=not vix_avail)
    vix_as_z = st.checkbox("Plot VIX als z-score (20d)", value=False, disabled=not vix_avail)
    corr_win = st.slider("Rolling corr window (dagen)", 10, 180, 60, 5)

    st.divider()
    st.markdown("#### Beta/Scatter")
    scatter_driver = st.selectbox("Driver voor scatter", driver_choices, index=driver_choices.index("US 10Y (yield %)") if "US 10Y (yield %)" in driver_choices else 0)
    scatter_mode = st.radio("Transformatie", ["%-returns vs %-returns", "Levels vs Levels"], index=0, horizontal=False)
    # hint: voor yields/vol is 'Levels' vaak zinvoller; voor assets '%-returns'

# ---------- Periode (bovenaan) ----------
min_d, max_d = df["date"].min().date(), df["date"].max().date()
default_start = max(max_d - timedelta(days=365), min_d)
start_d, end_d = st.slider("📅 Periode", min_value=min_d, max_value=max_d,
                           value=(default_start, max_d), step=timedelta(days=1), format="YYYY-MM-DD")
mask = (df["date"].dt.date >= start_d) & (df["date"].dt.date <= end_d)
d = df.loc[mask].copy()
if d.empty:
    st.info("Geen data in de gekozen periode."); st.stop()

# ---------- KPI’s ----------
st.subheader("KPI’s")
if not {"gold_ma20","gold_ma50","gold_ma200"} <= set(d.columns):
    d["gold_ma20"] = ema(d["gold_close"], 20)
    d["gold_ma50"] = ema(d["gold_close"], 50)
    d["gold_ma200"] = ema(d["gold_close"], 200)

last = d.dropna(subset=["gold_close"]).tail(1)
gold_last = float(last["gold_close"].iloc[0]) if not last.empty else np.nan
gold_dpct = (d["gold_delta_pct"].dropna().iloc[-1]*100.0) if ("gold_delta_pct" in d.columns and d["gold_delta_pct"].notna().any()) else ((d["gold_close"].pct_change().dropna().iloc[-1])*100.0 if d["gold_close"].notna().sum()>=2 else np.nan)

def last_val(col): 
    return (float(d[col].dropna().iloc[-1]) if col in d.columns and d[col].notna().any() else np.nan)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Gold (USD/oz)", f"{gold_last:,.2f}" if pd.notna(gold_last) else "—", pct_as_str(gold_dpct))
k2.metric("US 10Y", pct_as_str(last_val("us10y")))
k3.metric("Real 10Y", pct_as_str(last_val("tips10y_real")))
k4.metric("VIX", f"{last_val('vix_close'):.2f}" if pd.notna(last_val("vix_close")) else "—")

# ---------- Alerts ----------
st.subheader("📣 Signal Alerts")
alerts = []

# Trend regime + crosses
up_regime   = (d["gold_close"].iloc[-1] > d["gold_ma200"].iloc[-1]) and (d["gold_ma50"].iloc[-1] > d["gold_ma200"].iloc[-1])
down_regime = (d["gold_close"].iloc[-1] < d["gold_ma200"].iloc[-1]) and (d["gold_ma50"].iloc[-1] < d["gold_ma200"].iloc[-1])
if up_regime: alerts.append(("green","Bullish regime","Close > EMA200 en EMA50 > EMA200"))
elif down_regime: alerts.append(("red","Bearish regime","Close < EMA200 en EMA50 < EMA200"))

if len(d) >= 2:
    cross_up = (d["gold_ma50"].shift(1).iloc[-1] <= d["gold_ma200"].shift(1).iloc[-1]) and (d["gold_ma50"].iloc[-1] > d["gold_ma200"].iloc[-1])
    cross_dn = (d["gold_ma50"].shift(1).iloc[-1] >= d["gold_ma200"].shift(1).iloc[-1]) and (d["gold_ma50"].iloc[-1] < d["gold_ma200"].iloc[-1])
    if cross_up: alerts.append(("green","Golden cross","EMA50 kruist ↑ EMA200"))
    if cross_dn: alerts.append(("red","Death cross","EMA50 kruist ↓ EMA200"))

# Macro drivers: daling yields/DXY ondersteunt goud
def trend_down(col, look=10, thr=0.0):
    if col not in d.columns or d[col].notna().sum() < 2: return False
    s = d[col].dropna()
    return (s.iloc[-1] - s.iloc[max(len(s)-look,0)]) < thr

if trend_down("tips10y_real", look=20, thr=0): alerts.append(("green","Real yield ↓","TIPS 10Y daalt over ~20d"))
if trend_down("us10y", look=20, thr=0): alerts.append(("green","US 10Y ↓","Nominale 10Y daalt over ~20d"))
if trend_down("dxy_close", look=20, thr=0): alerts.append(("green","DXY ↓","Dollar index daalt over ~20d"))

# VIX regime
if "vix_close" in d.columns and d["vix_close"].notna().sum() >= 21:
    vz = zscore(d["vix_close"], 20).iloc[-1]
    if pd.notna(vz) and vz >= 2.0: alerts.append(("orange","VIX-spike","VIX z≥2 (20d)"))
    elif pd.notna(vz) and vz <= -1.5: alerts.append(("blue","VIX laag","VIX z≤−1.5 (20d)"))

# Corr regime (Gold vs DXY/US10Y)
def rolling_corr(a, b, win=60):
    x = pd.concat([a, b], axis=1).dropna()
    if len(x) < win: return np.nan
    return x[a.name].pct_change().rolling(win).corr(x[b.name].pct_change())

if all(c in d.columns for c in ["gold_close","dxy_close"]) and d["gold_close"].notna().sum() > 60 and d["dxy_close"].notna().sum() > 60:
    c_dxy = rolling_corr(d["gold_close"].rename("g"), d["dxy_close"].rename("dxy"), corr_win).dropna()
    if len(c_dxy):
        val = c_dxy.iloc[-1]
        if val <= -0.6: alerts.append(("green","Sterk neg. corr (Gold–DXY)", f"corr≈{val:.2f}"))
if all(c in d.columns for c in ["gold_close","us10y"]) and d["us10y"].notna().sum() > 60:
    c_y = rolling_corr(d["gold_close"].rename("g"), d["us10y"].rename("y"), corr_win).dropna()
    if len(c_y):
        val = c_y.iloc[-1]
        if val <= -0.5: alerts.append(("green","Neg. corr (Gold–US10Y)", f"corr≈{val:.2f}"))

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
st.subheader("Gold — Price & MA + Drivers overlay")

sel = st.multiselect(
    "Kies drivers voor overlay/vergelijking",
    options=driver_choices,
    default=[x for x in ["US 10Y (yield %)","US 10Y Real (TIPS %)","BTC (BTCUSD)","DXY (Dollar Index)"] if x in driver_choices],
    help="Bij ‘Genormaliseerd (=100)’ delen alle lijnen dezelfde schaal; anders krijgen drivers de 2e y-as."
)

fig = make_subplots(specs=[[{"secondary_y": (view_mode.startswith("Eigen"))}]])
# Gold base
fig.add_trace(go.Scatter(x=d["date"], y=d["gold_close"], name="Gold (USD/oz)", line=dict(width=2, color="#111111")))
fig.add_trace(go.Scatter(x=d["date"], y=d["gold_ma20"],  name="EMA20", line=dict(width=2, color="#E69F00")))
fig.add_trace(go.Scatter(x=d["date"], y=d["gold_ma50"],  name="EMA50", line=dict(width=2, color="#009E73")))
fig.add_trace(go.Scatter(x=d["date"], y=d["gold_ma200"], name="EMA200", line=dict(width=2, color="#0072B2")))

palette = ["#6B7280","#9CA3AF","#8B5CF6","#EF4444","#10B981","#A855F7","#F59E0B","#3B82F6"]
pi = 0

def add_overlay(name, series, color, secondary=False):
    if view_mode.startswith("Genormaliseerd"):
        base_gold = normalize_100(d["gold_close"])
        # teken Gold (=100) eenmalig
        if "Gold (=100)" not in [t.name for t in fig.data]:
            fig.add_trace(go.Scatter(x=d["date"], y=base_gold, name="Gold (=100)",
                                     line=dict(width=1.8, color="#111111", dash="dot")))
        fig.add_trace(go.Scatter(x=d["date"], y=normalize_100(series), name=name, line=dict(width=2, color=color)))
    else:
        fig.add_trace(go.Scatter(x=d["date"], y=series, name=name, line=dict(width=2, color=color)), secondary_y=secondary)

for name in sel:
    col = DRIVER_MAP[name]
    if col not in d.columns or d[col].notna().sum() == 0:
        continue
    ser = d[col]
    if name == "VIX" and show_vix:
        if vix_as_z: ser = zscore(ser, 20)
        add_overlay("VIX (z)" if vix_as_z else "VIX", ser, palette[pi % len(palette)], secondary=True)
    else:
        add_overlay(name, ser, palette[pi % len(palette)], secondary=True)
    pi += 1

yl = "USD/oz" if not view_mode.startswith("Genormaliseerd") else "Index (=100)"
fig.update_yaxes(title_text=yl, secondary_y=False)
if view_mode.startswith("Eigen"):
    fig.update_yaxes(title_text="Drivers (2e as)", secondary_y=True)
fig.update_layout(height=520, margin=dict(l=10, r=10, t=30, b=10),
                  legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
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
if not driver_choices:
    st.info("Geen drivers beschikbaar in dataset.")
else:
    sel_corr = st.multiselect("Kies drivers voor correlatie", options=driver_choices,
                              default=[x for x in ["US 10Y (yield %)","US 10Y Real (TIPS %)","DXY (Dollar Index)","BTC (BTCUSD)","VIX","Silver (USD/oz)"] if x in driver_choices])
    if sel_corr:
        gold_ret = d["gold_close"].pct_change()
        figc = go.Figure()
        for name in sel_corr:
            col = DRIVER_MAP[name]
            if col not in d.columns: 
                continue
            series = d[col]
            # assets -> returns; yields/vol -> levels
            as_return = name in ["BTC (BTCUSD)", "Silver (USD/oz)"]
            x = series.pct_change() if as_return else series
            corr = pd.concat([gold_ret.rename("g"), x.rename("x")], axis=1).dropna().rolling(corr_win).corr().reset_index()
            corr = corr[corr["level_1"]=="x"][["date","g"]].rename(columns={"g":name})
            if not corr.empty:
                figc.add_trace(go.Scatter(x=corr["date"], y=corr[name], mode="lines", name=name))
        figc.add_hline(y=0.0, line_dash="dot")
        figc.add_hrect(y0=-1, y1=-0.5, fillcolor="rgba(255,0,0,0.06)", line_width=0)
        figc.add_hrect(y0=0.5, y1=1,   fillcolor="rgba(0,128,0,0.06)", line_width=0)
        figc.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10),
                           legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
        figc.update_yaxes(title_text="Rolling corr", range=[-1,1])
        st.plotly_chart(figc, use_container_width=True)
    else:
        st.info("Selecteer minimaal één driver voor correlaties.")

st.divider()

# ---------- Beta/Scatter ----------
st.subheader("β / Scatter — Gold vs geselecteerde driver")
drv_col = DRIVER_MAP.get(scatter_driver)
if drv_col not in d.columns or d[drv_col].notna().sum() < 5 or d["gold_close"].notna().sum() < 5:
    st.info("Onvoldoende data voor scatter.")
else:
    if scatter_mode.startswith("%"):
        y = d["gold_close"].pct_change()*100.0
        x = d[drv_col].pct_change()*100.0
        x_label = f"{scatter_driver} Δ%"; y_label = "Gold Δ%"
    else:
        y = d["gold_close"]
        x = d[drv_col]
        x_label = scatter_driver; y_label = "Gold (USD/oz)"

    scatter_df = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    if len(scatter_df) < 10:
        st.info("Te weinig overlappende punten voor regressie.")
    else:
        X = scatter_df["x"].values
        Y = scatter_df["y"].values
        # Regressie (OLS via polyfit) + stats
        slope, intercept = np.polyfit(X, Y, 1)
        y_hat = slope*X + intercept
        # R^2
        ss_res = np.sum((Y - y_hat)**2)
        ss_tot = np.sum((Y - np.mean(Y))**2)
        r2 = 1 - ss_res/ss_tot if ss_tot != 0 else np.nan
        # corr
        corr = np.corrcoef(X, Y)[0,1] if len(scatter_df) > 1 else np.nan

        fig_s = go.Figure()
        fig_s.add_trace(go.Scatter(x=X, y=Y, mode="markers", name="Waarnemingen", opacity=0.6))
        # regressielijn
        x_line = np.linspace(X.min(), X.max(), 100)
        y_line = slope*x_line + intercept
        fig_s.add_trace(go.Scatter(x=x_line, y=y_line, mode="lines", name=f"Fit: y = {slope:.3f}x + {intercept:.3f}"))

        fig_s.update_layout(height=520, margin=dict(l=10,r=10,t=30,b=10),
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
        fig_s.update_xaxes(title=x_label); fig_s.update_yaxes(title=y_label)
        st.plotly_chart(fig_s, use_container_width=True)

        # Metrics-kaartjes
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("β (slope)", f"{slope:.3f}")
        m2.metric("Intercept", f"{intercept:.3f}")
        m3.metric("R²", f"{r2:.3f}" if pd.notna(r2) else "—")
        m4.metric("Corr", f"{corr:.3f}" if pd.notna(corr) else "—")
        st.caption("Let op: β is het gevoeligheidscoëfficiënt in de gekozen transformatie. Bij %-returns interpreteer je β als %-punt reactie van Gold op 1 %-punt in de driver.")

st.divider()

# ---------- Tabel (laatste rijen) ----------
st.subheader("Laatste rijen (gefilterd bereik)")
show_cols = ["date", "gold_close", "gold_ma20", "gold_ma50", "gold_ma200", "gold_delta_pct",
             "silver_close", "btc_close", "dxy_close", "eurusd_close", "us10y", "tips10y_real", "vix_close"]
show_cols = [c for c in show_cols if c in d.columns]
tail = d[show_cols].tail(200).copy()
if "gold_delta_pct" in tail.columns:
    tail["gold_delta_pct"] = (tail["gold_delta_pct"]*100).round(2)
st.dataframe(tail, use_container_width=True)
