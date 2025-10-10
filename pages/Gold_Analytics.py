# pages/Gold_Analytics.py
# Goud (GC) — Macro-determinanten, TA en correlaties
# Sluit aan op bestaande views:
# - Commodities wide:  marketdata.commodity_prices_wide_v  (gold_close, gold_delta_pct, etc.)
# - Crypto wide:       marketdata.crypto_daily_wide        (price_btc, delta_pct_btc, ...)
# - US yields:         marketdata.us_yield_curve_enriched_v (y_10y_real | y_10y, y_2y_*)
# - Macro monthly:     marketdata.macro_series_wide_monthly_fill_v (CPI/PCE/VIX/DXY indien aanwezig)
# - FX (optioneel):    marketdata.fx_rates_dashboard_v (voor DXY, indien aanwezig in jouw view)
#
# Functies:
#   • KPI-balk: Gold, 10Y (real/nom), DXY (indien aanwezig), BTC, HV20
#   • TA: EMA/SMA 20/50/200, Bollinger Bands, RSI(14)
#   • Macro-drivers: Gold vs 10Y (real of nominaal/inverted), Gold vs DXY (inverted)
#   • Rolling correlaties: Gold~(DXY, 10Y, BTC) met venster 30/60/90
#   • Quick OLS: Gold ~ 10Y + DXY (+ VIX indien beschikbaar), met betas & R²
#   • Tabel: laatste rijen in selectie
#
# Notities:
# - DXY is optioneel: we zoeken in macro/FX kolommen 'dxy'/'dollar_index'. Anders verbergen we blokken die DXY vereisen.
# - 10Y real: we zoeken kolom 'y_10y_real'; fallback: 'y_10y'.
# - VIX optioneel via macro (zoek 'vix').
# - Alle onderdelen degraderen gracieus als een driver ontbreekt.

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

# ---------- Secrets / View names (met fallbacks) ----------
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
    # maandelijkse wide – interpolatie naar dag doen we downstream indien aangevinkt
    return run_query(f"SELECT * FROM `{MACRO_WIDE_M}` ORDER BY date")

@st.cache_data(ttl=600, show_spinner=False)
def load_fx_optional() -> pd.DataFrame:
    try:
        return to_date(run_query(f"SELECT * FROM `{FX_WIDE_VIEW}` ORDER BY date"))
    except Exception:
        return pd.DataFrame()

com = load_com()
crypto = load_crypto()
us_y = load_us_yields()
macro_m = load_macro()
fx_wide = load_fx_optional()

if com.empty or ("gold_close" not in com.columns):
    st.error("Geen goudkolom gevonden in commodities-view (verwacht: gold_close)."); st.stop()

# ---------- UI: Sidebar ----------
with st.sidebar:
    st.header("⚙️ Instellingen")
    all_min, all_max = com["date"].min(), com["date"].max()
    default_start = max(all_max - timedelta(days=365), all_min)
    date_range = st.slider(
        "📅 Periode",
        min_value=all_min, max_value=all_max,
        value=(default_start, all_max),
        step=timedelta(days=1),
        format="YYYY-MM-DD",
    )

    ma_mode = st.radio("Gemiddelden", ["EMA", "SMA"], index=0, horizontal=True)
    corr_win = st.radio("Correlatievenster", ["30", "60", "90"], index=1, horizontal=True)
    show_dxy_invert = st.checkbox("DXY omgekeerd tonen (logisch voor goud)", value=True)
    show_10y_invert = st.checkbox("10Y omgekeerd tonen (lagere 10Y = bullish)", value=True)
    daily_interp = st.checkbox("Macro maanddata interpoleren naar dag", value=True)

start, end = date_range

# ---------- Kolommen detecteren ----------
# US 10Y: real vs nominal
y10_real_col = best_col(us_y.columns, ["y_10y_real"])
y10_nom_col  = best_col(us_y.columns, ["y_10y_synth", "y_10y"])
y10_col      = y10_real_col or y10_nom_col

# Macro candidates
macro_cols = [c.lower() for c in macro_m.columns]
vix_col  = best_col(macro_cols, ["vix","vix_close","cboe_vix"])
dxy_col  = best_col(macro_cols, ["dxy","dxy_close","dollar_index","usd_index"])

# Zo niet in macro, probeer fx-view (soms staat DXY daar in een wide-v)
if dxy_col is None and not fx_wide.empty:
    dxy_col = best_col([c.lower() for c in fx_wide.columns], ["dxy","dxy_close","usd_index","dollar_index"])

# BTC
btc_price_col = best_col([c.lower() for c in crypto.columns], ["price_btc","btc_price"])

# ---------- Filter ----------
df_gold = com.loc[(com["date"] >= start) & (com["date"] <= end)][["date","gold_close"]].dropna().copy()
df_gold["gold_close"] = _f(df_gold["gold_close"])

df_btc = pd.DataFrame()
if btc_price_col and btc_price_col in [c.lower() for c in crypto.columns]:
    bc = {c.lower(): c for c in crypto.columns}[btc_price_col]
    df_btc = crypto[["date", bc]].rename(columns={bc: "btc"}).dropna()
    df_btc = df_btc.loc[(df_btc["date"] >= start) & (df_btc["date"] <= end)].copy()
    df_btc["btc"] = _f(df_btc["btc"])

df_y10 = pd.DataFrame()
if y10_col:
    yc = {c.lower(): c for c in us_y.columns}[y10_col]
    df_y10 = us_y[["date", yc]].rename(columns={yc: "y10"}).dropna()
    df_y10 = df_y10.loc[(df_y10["date"] >= start) & (df_y10["date"] <= end)].copy()
    df_y10["y10"] = _f(df_y10["y10"])

# Macro monthly → daily?
df_macro = pd.DataFrame()
if not macro_m.empty:
    mm = macro_m.copy()
    if "date" in mm.columns and not np.issubdtype(mm["date"].dtype, np.datetime64):
        mm["date"] = pd.to_datetime(mm["date"])
    if daily_interp:
        # Her-sample naar dag en lineair interpoleren
        mm = mm.set_index("date").asfreq("D").interpolate(method="time").reset_index()
    mm["date"] = mm["date"].dt.date
    # Neem alleen kolommen die we eventueel gebruiken
    keep = ["date"]
    if vix_col: keep.append({c.lower():c for c in mm.columns}[vix_col])
    if dxy_col: keep.append({c.lower():c for c in mm.columns}.get(dxy_col, dxy_col))
    df_macro = mm[keep].dropna(how="all")
    # Hernoem naar gestandaardiseerde namen
    ren = {}
    if vix_col: ren[{c.lower():c for c in mm.columns}[vix_col]] = "vix"
    if dxy_col:
        orig = {c.lower():c for c in mm.columns}.get(dxy_col, dxy_col)
        if orig in mm.columns: ren[orig] = "dxy"
    df_macro = df_macro.rename(columns=ren)

# DXY vanuit fx-view als macro geen DXY had
if (dxy_col is None) and (not fx_wide.empty):
    fx = fx_wide.copy()
    fx = fx.loc[(fx["date"] >= start) & (fx["date"] <= end)]
    # zoek kolom 'dxy_close' of 'dxy'
    cand = best_col([c.lower() for c in fx.columns], ["dxy_close","dxy","usd_index","dollar_index"])
    if cand:
        df_macro = (df_macro if not df_macro.empty else fx[["date"]]).merge(
            fx[["date", {c.lower():c for c in fx.columns}[cand]]].rename(columns={{c.lower():c for c in fx.columns}[cand]: "dxy"}),
            on="date", how="outer"
        )
        dxy_col = "dxy"

# ---------- KPI’s ----------
st.subheader("KPI’s")
kcols = st.columns(5)

# Gold
last_gold = df_gold["gold_close"].dropna().iloc[-1] if not df_gold.empty else np.nan
prev_gold = df_gold["gold_close"].dropna().iloc[-2] if len(df_gold.dropna())>=2 else np.nan
delta_gold = (last_gold/prev_gold - 1.0)*100.0 if np.isfinite(last_gold) and np.isfinite(prev_gold) and prev_gold!=0 else None
kcols[0].metric("Gold (USD/oz)", f"{last_gold:,.2f}" if np.isfinite(last_gold) else "—",
                delta=(f"{delta_gold:+.2f}%" if delta_gold is not None else "—"))

# 10Y
if not df_y10.empty:
    last_y10 = float(df_y10["y10"].iloc[-1])
    kcols[1].metric(("US 10Y (real)" if y10_col==y10_real_col else "US 10Y (nom)"), f"{last_y10:.2f}%")
else:
    kcols[1].metric("US 10Y", "—")

# DXY
last_dxy = None
if not df_macro.empty and "dxy" in df_macro.columns:
    last_dxy = df_macro["dxy"].dropna().iloc[-1]
kcols[2].metric("DXY", f"{last_dxy:.2f}" if last_dxy is not None and np.isfinite(last_dxy) else "—")

# BTC
if not df_btc.empty:
    last_btc = float(df_btc["btc"].iloc[-1])
    prev_btc = float(df_btc["btc"].iloc[-2]) if len(df_btc)>=2 else np.nan
    d_btc = (last_btc/prev_btc - 1.0)*100.0 if np.isfinite(prev_btc) and prev_btc!=0 else None
    kcols[3].metric("BTC (USD)", f"{last_btc:,.0f}", delta=(f"{d_btc:+.2f}%" if d_btc is not None else "—"))
else:
    kcols[3].metric("BTC (USD)", "—")

# HV20
hv20 = hv(df_gold["gold_close"], window=20, annualize=252)
kcols[4].metric("Gold HV20 (ann.)", f"{hv20:.1f}%" if hv20 is not None else "—")

st.divider()

# ---------- TA: Price + MAs + BB + RSI ----------
st.subheader("Technische analyse")
ma20 = ema(df_gold["gold_close"], 20) if ma_mode=="EMA" else sma(df_gold["gold_close"], 20)
ma50 = ema(df_gold["gold_close"], 50) if ma_mode=="EMA" else sma(df_gold["gold_close"], 50)
ma200= ema(df_gold["gold_close"],200) if ma_mode=="EMA" else sma(df_gold["gold_close"],200)

# Bollinger Bands (20, 2σ)
s20 = _f(df_gold["gold_close"]).rolling(20, min_periods=20)
bb_mid = s20.mean()
bb_sd  = s20.std()
bb_up  = bb_mid + 2*bb_sd
bb_lo  = bb_mid - 2*bb_sd

fig_ta = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                       row_heights=[0.7, 0.3])
# Price pane
fig_ta.add_trace(go.Scatter(x=df_gold["date"], y=df_gold["gold_close"], name="Gold",
                            line=dict(width=2, color="#111111")), row=1, col=1)
fig_ta.add_trace(go.Scatter(x=df_gold["date"], y=ma20, name=("EMA20" if ma_mode=="EMA" else "MA20"),
                            line=dict(width=2, color="#E69F00")), row=1, col=1)
fig_ta.add_trace(go.Scatter(x=df_gold["date"], y=ma50, name=("EMA50" if ma_mode=="EMA" else "MA50"),
                            line=dict(width=2, color="#009E73")), row=1, col=1)
fig_ta.add_trace(go.Scatter(x=df_gold["date"], y=ma200, name=("EMA200" if ma_mode=="EMA" else "MA200"),
                            line=dict(width=2, color="#0072B2")), row=1, col=1)
# BBands
fig_ta.add_trace(go.Scatter(x=df_gold["date"], y=bb_up, name="BBand +2σ",
                            line=dict(width=1, dash="dot", color="#999999"), showlegend=False), row=1, col=1)
fig_ta.add_trace(go.Scatter(x=df_gold["date"], y=bb_mid, name="BBand mid",
                            line=dict(width=1, dash="dot", color="#BBBBBB"), showlegend=False), row=1, col=1)
fig_ta.add_trace(go.Scatter(x=df_gold["date"], y=bb_lo, name="BBand -2σ",
                            line=dict(width=1, dash="dot", color="#999999"), fill="tonexty", opacity=0.15, showlegend=False), row=1, col=1)

# RSI pane
rsi14 = rsi(df_gold["gold_close"], 14)
fig_ta.add_trace(go.Scatter(x=df_gold["date"], y=rsi14, name="RSI(14)",
                            line=dict(width=2, color="#CC79A7")), row=2, col=1)
fig_ta.add_hline(y=70, line_dash="dot", line_color="#D55E00", row=2, col=1)
fig_ta.add_hline(y=30, line_dash="dot", line_color="#009E73", row=2, col=1)

fig_ta.update_layout(margin=dict(l=10,r=10,t=10,b=10),
                     legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
st.plotly_chart(fig_ta, use_container_width=True)

st.divider()

# ---------- Macro drivers: Gold vs 10Y & DXY ----------
st.subheader("Macro-drivers")
tabs = st.tabs(["Gold vs 10Y", "Gold vs DXY (optioneel)"])

# Gold vs 10Y
with tabs[0]:
    if df_y10.empty:
        st.info("Geen 10Y-kolom gevonden in US-yield-view.")
    else:
        loc = df_gold.merge(df_y10, on="date", how="inner")
        if loc.empty:
            st.info("Geen overlap Gold–10Y in de gekozen periode.")
        else:
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Scatter(x=loc["date"], y=loc["gold_close"], name="Gold (USD/oz)",
                                     line=dict(width=2, color="#111111")), secondary_y=False)
            y10_series = -loc["y10"] if show_10y_invert else loc["y10"]
            fig.add_trace(go.Scatter(x=loc["date"], y=y10_series, name=("−US 10Y" if show_10y_invert else "US 10Y"),
                                     line=dict(width=2, color="#0072B2", dash="dash")), secondary_y=True)
            fig.update_yaxes(title_text="Gold (USD/oz)", secondary_y=False)
            fig.update_yaxes(title_text="US 10Y (%)" if not show_10y_invert else "−US 10Y (%)", secondary_y=True)
            fig.update_layout(margin=dict(l=10,r=10,t=10,b=10),
                              legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
            st.plotly_chart(fig, use_container_width=True)

# Gold vs DXY
with tabs[1]:
    if df_macro.empty or "dxy" not in df_macro.columns:
        st.info("DXY niet gevonden in macro/fx-view — blok overgeslagen.")
    else:
        loc = df_gold.merge(df_macro[["date","dxy"]], on="date", how="inner")
        if loc.empty:
            st.info("Geen overlap Gold–DXY in de gekozen periode.")
        else:
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Scatter(x=loc["date"], y=loc["gold_close"], name="Gold (USD/oz)",
                                     line=dict(width=2, color="#111111")), secondary_y=False)
            dxy_series = -loc["dxy"] if show_dxy_invert else loc["dxy"]
            fig.add_trace(go.Scatter(x=loc["date"], y=dxy_series, name=("−DXY" if show_dxy_invert else "DXY"),
                                     line=dict(width=2, color="#E69F00", dash="dash")), secondary_y=True)
            fig.update_yaxes(title_text="Gold (USD/oz)", secondary_y=False)
            fig.update_yaxes(title_text="DXY" if not show_dxy_invert else "−DXY", secondary_y=True)
            fig.update_layout(margin=dict(l=10,r=10,t=10,b=10),
                              legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
            st.plotly_chart(fig, use_container_width=True)

st.divider()

# ---------- Rolling correlaties ----------
st.subheader("Rolling correlaties (Gold met drivers)")
win = int(corr_win)

# Maak één samengevoegd frame
base = df_gold.rename(columns={"gold_close":"gold"}).copy()
def merge_in(df_in: pd.DataFrame, name: str) -> None:
    global base
    if not df_in.empty and (name in df_in.columns or df_in.shape[1]==2):
        if name not in df_in.columns:  # verwacht vorm (date, <name>)
            n = [c for c in df_in.columns if c != "date"][0]
            dfm = df_in[["date", n]].rename(columns={n:name})
        else:
            dfm = df_in[["date", name]]
        base = base.merge(dfm, on="date", how="left")

merge_in(df_y10.rename(columns={"y10":"y10"}), "y10")
if not df_macro.empty and "dxy" in df_macro.columns:
    merge_in(df_macro[["date","dxy"]], "dxy")
if not df_btc.empty:
    merge_in(df_btc.rename(columns={"btc":"btc"}), "btc")

# Corr berekenen
corr_plots = []
labels = []
if "y10" in base.columns and base["y10"].notna().sum() > win:
    base["corr_gold_y10"] = rolling_corr(_f(base["gold"]).pct_change(), _f(base["y10"]).pct_change(), win)
    corr_plots.append(("corr_gold_y10", "Gold ~ US 10Y"))
if "dxy" in base.columns and base["dxy"].notna().sum() > win:
    base["corr_gold_dxy"] = rolling_corr(_f(base["gold"]).pct_change(), _f(base["dxy"]).pct_change(), win)
    corr_plots.append(("corr_gold_dxy", "Gold ~ DXY"))
if "btc" in base.columns and base["btc"].notna().sum() > win:
    base["corr_gold_btc"] = rolling_corr(_f(base["gold"]).pct_change(), _f(base["btc"]).pct_change(), win)
    corr_plots.append(("corr_gold_btc", "Gold ~ BTC"))

if not corr_plots:
    st.info("Geen voldoende data voor correlaties (drivers ontbreken of te korte periode).")
else:
    figc = go.Figure()
    for i,(col,name) in enumerate(corr_plots):
        figc.add_trace(go.Scatter(x=base["date"], y=base[col], name=name, line=dict(width=2)))
    figc.add_hline(y=0, line_dash="dot", opacity=0.5)
    figc.update_layout(margin=dict(l=10,r=10,t=10,b=10),
                       legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                       yaxis=dict(title=f"ρ (rolling {win}d)", range=[-1,1]))
    st.plotly_chart(figc, use_container_width=True)

st.divider()

# ---------- Quick OLS sensitiviteiten ----------
st.subheader("Quick OLS — Gold gevoeligheid")
# Doel: Δgold ~ a + b1*Δy10 + b2*Δdxy + b3*Δvix (indien beschikbaar)
ols_df = base[["date","gold"]].copy()
for nm in ["y10","dxy"]:
    if nm in base.columns:
        ols_df[nm] = base[nm]
if not df_macro.empty and vix_col and "vix" in df_macro.columns:
    # Als we eerder vix in macro hebben gestopt
    pass
elif not macro_m.empty and vix_col:
    # Als we in df_macro geen vix hadden (bijv. geen interpolatie), probeer alsnog uit macro_m
    mm = macro_m.copy()
    c = {c.lower():c for c in mm.columns}.get(vix_col)
    if c:
        mm["date"] = pd.to_datetime(mm["date"]).dt.date
        if daily_interp:
            mm = mm.set_index("date").asfreq("D").interpolate(method="time").reset_index()
            mm["date"] = mm["date"].dt.date
        mm = mm.rename(columns={c:"vix"})
        ols_df = ols_df.merge(mm[["date","vix"]], on="date", how="left")

# Vereist minimaal 2 drivers
drivers = [c for c in ["y10","dxy","vix"] if c in ols_df.columns]
if len(drivers) == 0:
    st.info("Onvoldoende drivers beschikbaar voor OLS (geen 10Y/DXY/VIX).")
else:
    # Dagelijkse returns/diffs
    ols_frame = ols_df.dropna().copy()
    ols_frame["ret_gold"] = _f(ols_frame["gold"]).pct_change()
    for d in drivers:
        ols_frame[f"ret_{d}"] = _f(ols_frame[d]).pct_change()
    ols_frame = ols_frame.dropna()

    if len(ols_frame) < 50:
        st.info("Te weinig datapunten voor een stabiele OLS in de gekozen periode.")
    else:
        # X (drivers), y (ret_gold)
        X = np.column_stack([ols_frame[f"ret_{d}"].values for d in drivers])
        y = ols_frame["ret_gold"].values
        X = np.nan_to_num(X); y = np.nan_to_num(y)
        # Voeg intercept
        Xi = np.column_stack([np.ones(len(X)), X])
        beta, *_ = np.linalg.lstsq(Xi, y, rcond=None)
        y_hat = Xi @ beta
        # R^2
        ss_res = np.sum((y - y_hat)**2)
        ss_tot = np.sum((y - y.mean())**2)
        r2 = 1 - ss_res/ss_tot if ss_tot > 0 else np.nan

        cols = st.columns(2)
        with cols[0]:
            st.markdown("**Sensitiviteiten (β)**")
            st.write({ "intercept": float(beta[0]), **{f"β_{d}": float(beta[i+1]) for i,d in enumerate(drivers)} })
        with cols[1]:
            st.metric("R² (in-sample)", f"{r2:.2f}")

        st.caption("Let op: dit is een eenvoudige, lineaire benadering met dagelijkse returns; interpretatie is regimesensitief en niet causaal.")

st.divider()

# ---------- Tabel ----------
st.subheader("Laatste rijen (gefilterd bereik)")
df_tbl = df_gold.merge(df_y10, how="left", on="date")
if not df_macro.empty and "dxy" in df_macro.columns:
    df_tbl = df_tbl.merge(df_macro[["date","dxy"]], how="left", on="date")
if not df_btc.empty:
    df_tbl = df_tbl.merge(df_btc, how="left", on="date")
st.dataframe(df_tbl.tail(200), use_container_width=True)
