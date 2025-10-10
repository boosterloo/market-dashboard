# pages/Gold_Analytics.py
# 🥇 Gold Analytics — Macro • TA • Correlaties
# Verwerkt wensen:
#  - Grote periodebalk bovenaan (ipv sidebar)
#  - Macro-drivers: multi-select (BTC, 10Y, DXY, VIX) + invert + normaliseren

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

    @st.cache_data(ttl=600, show_spinner=False)
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
st.set_page_config(page_title="🥇 Gold Analytics", layout="wide")
st.title("🥇 Gold Analytics — Macro • TA • Correlaties")

# ---------- Secrets / View names ----------
PROJECT_ID = st.secrets["gcp_service_account"]["project_id"]
TABLES     = st.secrets.get("tables", {})

COM_WIDE_VIEW   = TABLES.get("commodities_wide_view", f"{PROJECT_ID}.marketdata.commodity_prices_wide_v")
CRYPTO_WIDE     = TABLES.get("crypto_daily_wide",     f"{PROJECT_ID}.marketdata.crypto_daily_wide")
US_YIELD_VIEW   = TABLES.get("us_yield_view",         f"{PROJECT_ID}.marketdata.us_yield_curve_enriched_v")
MACRO_WIDE_M    = TABLES.get("macro_view",            f"{PROJECT_ID}.marketdata.macro_series_wide_monthly_fill_v")
FX_WIDE_VIEW    = TABLES.get("fx_wide_view",          f"{PROJECT_ID}.marketdata.fx_rates_dashboard_v")  # optioneel

with st.expander("🔎 Debug bronnen"):
    st.write({
        "COM_WIDE_VIEW": COM_WIDE_VIEW,
        "CRYPTO_WIDE": CRYPTO_WIDE,
        "US_YIELD_VIEW": US_YIELD_VIEW,
        "MACRO_WIDE_M": MACRO_WIDE_M,
        "FX_WIDE_VIEW (optioneel)": FX_WIDE_VIEW
    })

# ---------- Health ----------
try:
    if not bq_ping():
        st.error("Geen BigQuery-verbinding. Controleer Secrets ([gcp_service_account])."); st.stop()
except Exception as e:
    st.error("Geen BigQuery-verbinding."); st.caption(f"Details: {e}"); st.stop()

# ---------- Helpers ----------
def to_date(df: pd.DataFrame, col: str = "date") -> pd.DataFrame:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col]).dt.date
    return df

def _f(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype(float)

def ema(series: pd.Series, n: int) -> pd.Series:
    s = _f(series)
    return s.ewm(span=n, adjust=False, min_periods=n).mean()

def sma(series: pd.Series, n: int) -> pd.Series:
    s = _f(series)
    return s.rolling(n, min_periods=n).mean()

def rsi(series: pd.Series, n: int = 14) -> pd.Series:
    s = _f(series)
    delta = s.diff()
    up = delta.clip(lower=0); down = -delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/n, adjust=False).mean()
    ma_down = down.ewm(alpha=1/n, adjust=False).mean()
    rs = ma_up / (ma_down.replace(0, np.nan))
    return 100 - (100 / (1 + rs))

def hv(series: pd.Series, window: int = 20, annualize: int = 252) -> float | None:
    ret = _f(series).pct_change()
    if ret.notna().sum() < window:
        return None
    return float(ret.rolling(window).std().iloc[-1] * np.sqrt(annualize) * 100.0)

def rolling_corr(s1: pd.Series, s2: pd.Series, window: int) -> pd.Series:
    return s1.rolling(window).corr(s2)

def best_col(cols: list[str] | set[str], preferred: list[str]) -> str | None:
    cl = {c.lower(): c for c in cols}
    for p in preferred:
        if p in cl: return cl[p]
    return None

def padded_range(s: pd.Series, pad_ratio: float = 0.05):
    x = pd.to_numeric(s, errors="coerce").dropna()
    if x.empty: return None
    lo, hi = float(x.min()), float(x.max())
    if not np.isfinite(lo) or not np.isfinite(hi): return None
    if lo == hi:
        pad = abs(lo) * pad_ratio if lo != 0 else 1.0
        return [lo - pad, hi + pad]
    span = hi - lo; pad = span * pad_ratio
    return [lo - pad, hi + pad]

# ---------- Data laden ----------
@st.cache_data(ttl=600, show_spinner=False)
def load_com() -> pd.DataFrame:
    return to_date(run_query(f"SELECT * FROM `{COM_WIDE_VIEW}` ORDER BY date"))

@st.cache_data(ttl=600, show_spinner=False)
def load_crypto() -> pd.DataFrame:
    return to_date(run_query(f"SELECT * FROM `{CRYPTO_WIDE}` ORDER BY date"))

@st.cache_data(ttl=600, show_spinner=False)
def load_us_yields() -> pd.DataFrame:
    return to_date(run_query(f"SELECT * FROM `{US_YIELD_VIEW}` ORDER BY date"))

@st.cache_data(ttl=1200, show_spinner=False)
def load_macro() -> pd.DataFrame:
    return run_query(f"SELECT * FROM `{MACRO_WIDE_M}` ORDER BY date")

@st.cache_data(ttl=600, show_spinner=False)
def load_fx_optional() -> pd.DataFrame:
    try:
        return to_date(run_query(f"SELECT * FROM `{FX_WIDE_VIEW}` ORDER BY date"))
    except Exception:
        return pd.DataFrame()

com     = load_com()
crypto  = load_crypto()
us_y    = load_us_yields()
macro_m = load_macro()
fx_wide = load_fx_optional()

if com.empty or ("gold_close" not in com.columns):
    st.error("Geen goudkolom gevonden in commodities-view (verwacht: gold_close)."); st.stop()

# ---------- Periodebalk BOVENAAN ----------
all_min, all_max = com["date"].min(), com["date"].max()
default_start = max(all_max - timedelta(days=365), all_min)

preset = st.radio("Periode preset", ["6M","1Y","3Y","5Y","YTD","Max","Custom"], horizontal=True, index=1)
def clamp(ts): 
    return max(all_min, ts)

if preset == "6M":
    start, end = clamp(all_max - timedelta(days=182)), all_max
elif preset == "1Y":
    start, end = clamp(all_max - timedelta(days=365)), all_max
elif preset == "3Y":
    start, end = clamp(all_max - timedelta(days=3*365)), all_max
elif preset == "5Y":
    start, end = clamp(all_max - timedelta(days=5*365)), all_max
elif preset == "YTD":
    start, end = pd.to_datetime(f"{pd.to_datetime(all_max).year}-01-01").date(), all_max
elif preset == "Max":
    start, end = all_min, all_max
else:
    start, end = st.slider(
        "📅 Periode",
        min_value=all_min, max_value=all_max,
        value=(default_start, all_max),
        step=timedelta(days=1),
        format="YYYY-MM-DD",
    )

# ---------- Sidebar voor overige instellingen ----------
with st.sidebar:
    st.header("⚙️ Instellingen")
    ma_mode = st.radio("Gemiddelden", ["EMA", "SMA"], index=0, horizontal=True)
    corr_win = st.radio("Correlatievenster", ["30","60","90"], index=1, horizontal=True)
    daily_interp = st.checkbox("Macro maanddata interpoleren naar dag", value=True)

# ---------- Kolommen detecteren ----------
y10_real_col = best_col(us_y.columns, ["y_10y_real"])
y10_nom_col  = best_col(us_y.columns, ["y_10y_synth","y_10y"])
y10_col      = y10_real_col or y10_nom_col

macro_cols = [c.lower() for c in macro_m.columns]
vix_col  = best_col(macro_cols, ["vix","vix_close","cboe_vix"])
dxy_col  = best_col(macro_cols, ["dxy","dxy_close","dollar_index","usd_index"])

if dxy_col is None and not fx_wide.empty:
    dxy_col = best_col([c.lower() for c in fx_wide.columns], ["dxy","dxy_close","usd_index","dollar_index"])

btc_price_col = best_col([c.lower() for c in crypto.columns], ["price_btc","btc_price"])

# ---------- Filter & frames ----------
df_gold = com.loc[(com["date"] >= start) & (com["date"] <= end)][["date","gold_close"]].dropna().copy()
df_gold["gold_close"] = _f(df_gold["gold_close"])

df_btc = pd.DataFrame()
if btc_price_col and btc_price_col in [c.lower() for c in crypto.columns]:
    bc = {c.lower(): c for c in crypto.columns}[btc_price_col]
    df_btc = crypto[["date", bc]].rename(columns={bc: "BTC"}).dropna()
    df_btc = df_btc.loc[(df_btc["date"] >= start) & (df_btc["date"] <= end)].copy()
    df_btc["BTC"] = _f(df_btc["BTC"])

df_y10 = pd.DataFrame()
if y10_col:
    yc = {c.lower(): c for c in us_y.columns}[y10_col]
    nm = "US 10Y (real)" if y10_col == y10_real_col else "US 10Y"
    df_y10 = us_y[["date", yc]].rename(columns={yc: nm}).dropna()
    df_y10 = df_y10.loc[(df_y10["date"] >= start) & (df_y10["date"] <= end)].copy()
    df_y10[nm] = _f(df_y10[nm])

# Macro monthly → daily?
df_macro = pd.DataFrame()
if not macro_m.empty:
    mm = macro_m.copy()
    if "date" in mm.columns and not np.issubdtype(mm["date"].dtype, np.datetime64):
        mm["date"] = pd.to_datetime(mm["date"])
    if daily_interp:
        mm = mm.set_index("date").asfreq("D").interpolate(method="time").reset_index()
    mm["date"] = mm["date"].dt.date
    keep = ["date"]
    ren  = {}
    if vix_col:
        keep.append({c.lower():c for c in mm.columns}[vix_col])
        ren[{c.lower():c for c in mm.columns}[vix_col]] = "VIX"
    if dxy_col:
        orig = {c.lower():c for c in mm.columns}.get(dxy_col, dxy_col)
        if orig in mm.columns:
            keep.append(orig); ren[orig] = "DXY"
    df_macro = mm[keep].rename(columns=ren).dropna(how="all")

if (("DXY" not in df_macro.columns) and (not fx_wide.empty)):
    fx = fx_wide.loc[(fx_wide["date"] >= start) & (fx_wide["date"] <= end)].copy()
    cand = best_col([c.lower() for c in fx.columns], ["dxy_close","dxy","usd_index","dollar_index"])
    if cand:
        colname = {c.lower():c for c in fx.columns}[cand]
        add = fx[["date", colname]].rename(columns={colname:"DXY"})
        df_macro = (df_macro if not df_macro.empty else fx[["date"]]).merge(add, on="date", how="outer")

# ---------- KPI’s ----------
st.subheader("KPI’s")
kcols = st.columns(5)

last_gold = df_gold["gold_close"].dropna().iloc[-1] if not df_gold.empty else np.nan
prev_gold = df_gold["gold_close"].dropna().iloc[-2] if len(df_gold.dropna())>=2 else np.nan
delta_gold = (last_gold/prev_gold - 1.0)*100.0 if np.isfinite(last_gold) and np.isfinite(prev_gold) and prev_gold!=0 else None
kcols[0].metric("Gold (USD/oz)", f"{last_gold:,.2f}" if np.isfinite(last_gold) else "—",
                delta=(f"{delta_gold:+.2f}%" if delta_gold is not None else "—"))

if not df_y10.empty:
    ycol = [c for c in df_y10.columns if c != "date"][0]
    kcols[1].metric(ycol, f"{float(df_y10[ycol].iloc[-1]):.2f}%")
else:
    kcols[1].metric("US 10Y", "—")

if not df_macro.empty and "DXY" in df_macro.columns:
    kcols[2].metric("DXY", f"{float(df_macro['DXY'].dropna().iloc[-1]):.2f}")
else:
    kcols[2].metric("DXY", "—")

if not df_btc.empty:
    last_btc = float(df_btc["BTC"].iloc[-1])
    prev_btc = float(df_btc["BTC"].iloc[-2]) if len(df_btc)>=2 else np.nan
    d_btc = (last_btc/prev_btc - 1.0)*100.0 if np.isfinite(prev_btc) and prev_btc!=0 else None
    kcols[3].metric("BTC (USD)", f"{last_btc:,.0f}", delta=(f"{d_btc:+.2f}%" if d_btc is not None else "—"))
else:
    kcols[3].metric("BTC (USD)", "—")

hv20 = hv(df_gold["gold_close"], window=20, annualize=252)
kcols[4].metric("Gold HV20 (ann.)", f"{hv20:.1f}%" if hv20 is not None else "—")

st.divider()

# ---------- TA: Price + MAs + BB + RSI ----------
st.subheader("Technische analyse")
ma20 = ema(df_gold["gold_close"], 20) if ma_mode=="EMA" else sma(df_gold["gold_close"], 20)
ma50 = ema(df_gold["gold_close"], 50) if ma_mode=="EMA" else sma(df_gold["gold_close"], 50)
ma200= ema(df_gold["gold_close"],200) if ma_mode=="EMA" else sma(df_gold["gold_close"],200)

s20 = _f(df_gold["gold_close"]).rolling(20, min_periods=20)
bb_mid = s20.mean(); bb_sd = s20.std()
bb_up  = bb_mid + 2*bb_sd; bb_lo = bb_mid - 2*bb_sd

fig_ta = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.7, 0.3])
fig_ta.add_trace(go.Scatter(x=df_gold["date"], y=df_gold["gold_close"], name="Gold",
                            line=dict(width=2, color="#111111")), row=1, col=1)
fig_ta.add_trace(go.Scatter(x=df_gold["date"], y=ma20, name=("EMA20" if ma_mode=="EMA" else "MA20"),
                            line=dict(width=2, color="#E69F00")), row=1, col=1)
fig_ta.add_trace(go.Scatter(x=df_gold["date"], y=ma50, name=("EMA50" if ma_mode=="EMA" else "MA50"),
                            line=dict(width=2, color="#009E73")), row=1, col=1)
fig_ta.add_trace(go.Scatter(x=df_gold["date"], y=ma200, name=("EMA200" if ma_mode=="EMA" else "MA200"),
                            line=dict(width=2, color="#0072B2")), row=1, col=1)
fig_ta.add_trace(go.Scatter(x=df_gold["date"], y=bb_up, name="BB +2σ",
                            line=dict(width=1, dash="dot", color="#999999"), showlegend=False), row=1, col=1)
fig_ta.add_trace(go.Scatter(x=df_gold["date"], y=bb_mid, name="BB mid",
                            line=dict(width=1, dash="dot", color="#BBBBBB"), showlegend=False), row=1, col=1)
fig_ta.add_trace(go.Scatter(x=df_gold["date"], y=bb_lo, name="BB −2σ",
                            line=dict(width=1, dash="dot", color="#999999"), fill="tonexty", opacity=0.15, showlegend=False), row=1, col=1)

rsi14 = rsi(df_gold["gold_close"], 14)
fig_ta.add_trace(go.Scatter(x=df_gold["date"], y=rsi14, name="RSI(14)",
                            line=dict(width=2, color="#CC79A7")), row=2, col=1)
fig_ta.add_hline(y=70, line_dash="dot", line_color="#D55E00", row=2, col=1)
fig_ta.add_hline(y=30, line_dash="dot", line_color="#009E73", row=2, col=1)

fig_ta.update_layout(margin=dict(l=10,r=10,t=10,b=10),
                     legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
st.plotly_chart(fig_ta, use_container_width=True)

st.divider()

# ---------- Macro-drivers (NIEUW: multi-select + invert + normaliseren) ----------
st.subheader("Macro-drivers (selecteer meerdere)")

# Stel beschikbare drivers samen
available = {}
if not df_y10.empty:
    ynm = [c for c in df_y10.columns if c != "date"][0]
    available["US 10Y"] = df_y10.rename(columns={ynm: "US 10Y"})[["date","US 10Y"]]
if not df_macro.empty and "DXY" in df_macro.columns:
    available["DXY"] = df_macro[["date","DXY"]]
if not df_macro.empty and "VIX" in df_macro.columns:
    available["VIX"] = df_macro[["date","VIX"]]
if not df_btc.empty:
    available["BTC"] = df_btc  # reeds "BTC"

if not available:
    st.info("Geen drivers gevonden (DXY/10Y/VIX/BTC).")
else:
    order_pref = ["US 10Y","DXY","VIX","BTC"]
    ordered_opts = [o for o in order_pref if o in available] + [o for o in available.keys() if o not in order_pref]

    left, right = st.columns([2,1])
    with left:
        sel = st.multiselect("Kies drivers om te tonen (rechts)", options=ordered_opts, default=[o for o in ordered_opts if o in ["US 10Y","DXY","BTC"]])
    with right:
        normalize = st.checkbox("Normaliseer drivers (Z-score, per selectie)", value=True)
        invert_these = st.multiselect("Inverteer", options=ordered_opts, default=[o for o in ["US 10Y","DXY"] if o in ordered_opts], help="Handig omdat dalende rente/sterke USD vaak negatief met goud correleren.")

    # Merge gold + gekozen drivers
    base = df_gold.rename(columns={"gold_close":"Gold"})[["date","Gold"]].copy()
    for name in sel:
        d = available[name].copy()
        base = base.merge(d, on="date", how="inner")

    if base.shape[1] <= 2:
        st.info("Geen overlap tussen Gold en de gekozen drivers in de geselecteerde periode.")
    else:
        # Voor drivers: optioneel invert + normaliseren naar Z-score
        drivers_cols = [c for c in base.columns if c not in ["date","Gold"]]
        plot_df = base.copy()

        # invert
        for c in drivers_cols:
            if c in invert_these:
                plot_df[c] = -_f(plot_df[c])

        # Z-score normaliseren (alleen drivers, Gold blijft op linkeras)
        if normalize:
            for c in drivers_cols:
                x = _f(plot_df[c])
                mu, sd = x.mean(), x.std()
                plot_df[c] = (x - mu) / (sd if sd not in [0, None] else 1.0)

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        # Gold links
        fig.add_trace(go.Scatter(x=plot_df["date"], y=_f(plot_df["Gold"]), name="Gold (USD/oz)",
                                 line=dict(width=2, color="#111111")), secondary_y=False)
        # Drivers rechts
        palette = ["#0072B2","#E69F00","#CC79A7","#009E73","#D55E00","#56B4E9"]
        for i, c in enumerate(drivers_cols):
            fig.add_trace(go.Scatter(x=plot_df["date"], y=_f(plot_df[c]), name=c,
                                     line=dict(width=2, dash="dash", color=palette[i % len(palette)])),
                          secondary_y=True)

        fig.update_yaxes(title_text="Gold (USD/oz)", secondary_y=False)
        if normalize:
            fig.update_yaxes(title_text="Drivers (Z-score)", secondary_y=True)
        else:
            fig.update_yaxes(title_text="Drivers (units)", secondary_y=True)

        fig.update_layout(margin=dict(l=10,r=10,t=10,b=10),
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
        st.plotly_chart(fig, use_container_width=True)

st.divider()

# ---------- Rolling correlaties ----------
st.subheader("Rolling correlaties (Gold met drivers)")
win = int(st.session_state.get("corr_win_val", 0) or 0)
try:
    win = int((corr_win if isinstance(corr_win, str) else str(corr_win)))
except Exception:
    win = 60

# Maak samengevoegd frame
base_corr = df_gold.rename(columns={"gold_close":"gold"})[["date","gold"]].copy()

# koppel de drivers die we hebben
if not df_y10.empty:
    base_corr = base_corr.merge(df_y10.rename(columns={c:c for c in df_y10.columns}), on="date", how="left")
if not df_macro.empty and "DXY" in df_macro.columns:
    base_corr = base_corr.merge(df_macro[["date","DXY"]], on="date", how="left")
if not df_btc.empty:
    base_corr = base_corr.merge(df_btc.rename(columns={"BTC":"BTC"}), on="date", how="left")

# Corrs
corr_plots = []
lbl_map = {}
# kolomnamen
y10_lbl = [c for c in base_corr.columns if c.startswith("US 10Y")]
if y10_lbl:
    col = y10_lbl[0]
    base_corr["corr_gold_y10"] = rolling_corr(_f(base_corr["gold"]).pct_change(), _f(base_corr[col]).pct_change(), win)
    corr_plots.append(("corr_gold_y10", f"Gold ~ {col}"))
if "DXY" in base_corr.columns:
    base_corr["corr_gold_dxy"] = rolling_corr(_f(base_corr["gold"]).pct_change(), _f(base_corr["DXY"]).pct_change(), win)
    corr_plots.append(("corr_gold_dxy", "Gold ~ DXY"))
if "BTC" in base_corr.columns:
    base_corr["corr_gold_btc"] = rolling_corr(_f(base_corr["gold"]).pct_change(), _f(base_corr["BTC"]).pct_change(), win)
    corr_plots.append(("corr_gold_btc", "Gold ~ BTC"))

if not corr_plots:
    st.info("Geen voldoende data voor correlaties.")
else:
    figc = go.Figure()
    for i,(col,name) in enumerate(corr_plots):
        figc.add_trace(go.Scatter(x=base_corr["date"], y=base_corr[col], name=name, line=dict(width=2)))
    figc.add_hline(y=0, line_dash="dot", opacity=0.5)
    figc.update_layout(margin=dict(l=10,r=10,t=10,b=10),
                       legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                       yaxis=dict(title=f"ρ (rolling {win}d)", range=[-1,1]))
    st.plotly_chart(figc, use_container_width=True)

st.divider()

# ---------- Quick OLS ----------
st.subheader("Quick OLS — Gold gevoeligheid")
ols_df = base_corr.rename(columns={"gold":"Gold"}).copy()
drivers = []
if y10_lbl: drivers.append(y10_lbl[0])
if "DXY" in ols_df.columns: drivers.append("DXY")
if "BTC" in ols_df.columns: drivers.append("BTC")

if len(drivers) == 0:
    st.info("Onvoldoende drivers beschikbaar voor OLS.")
else:
    ols = ols_df[["date","Gold"] + drivers].dropna().copy()
    ols["ret_gold"] = _f(ols["Gold"]).pct_change()
    for d in drivers:
        ols[f"ret_{d}"] = _f(ols[d]).pct_change()
    ols = ols.dropna()
    if len(ols) < 50:
        st.info("Te weinig datapunten voor een stabiele OLS in de gekozen periode.")
    else:
        X = np.column_stack([ols[f"ret_{d}"].values for d in drivers])
        y = ols["ret_gold"].values
        X = np.nan_to_num(X); y = np.nan_to_num(y)
        Xi = np.column_stack([np.ones(len(X)), X])
        beta, *_ = np.linalg.lstsq(Xi, y, rcond=None)
        y_hat = Xi @ beta
        ss_res = np.sum((y - y_hat)**2)
        ss_tot = np.sum((y - y.mean())**2)
        r2 = 1 - ss_res/ss_tot if ss_tot > 0 else np.nan

        cols = st.columns(2)
        with cols[0]:
            st.markdown("**Sensitiviteiten (β)**")
            st.write({ "intercept": float(beta[0]), **{f"β_{d}": float(beta[i+1]) for i,d in enumerate(drivers)} })
        with cols[1]:
            st.metric("R² (in-sample)", f"{r2:.2f}")
        st.caption("NB: eenvoudige lineaire schatting met dagelijkse returns; regimesensitief, niet causaal.")

st.divider()

# ---------- Tabel ----------
st.subheader("Laatste rijen (gefilterd bereik)")
df_tbl = df_gold.rename(columns={"gold_close":"Gold"}).copy()
if not df_y10.empty:
    df_tbl = df_tbl.merge(df_y10, how="left", on="date")
if not df_macro.empty and "DXY" in df_macro.columns:
    df_tbl = df_tbl.merge(df_macro[["date","DXY"]], how="left", on="date")
if not df_btc.empty:
    df_tbl = df_tbl.merge(df_btc, how="left", on="date")
st.dataframe(df_tbl.tail(200), use_container_width=True)
