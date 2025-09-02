# pages/4_Greeks_3D.py
import streamlit as st
import numpy as np
import pandas as pd
import math
from datetime import date, datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from google.cloud import bigquery
from google.oauth2 import service_account

# —— r/q helpers (zoals op opties-pagina) ——
from utils.rates import get_r_curve_for_snapshot, get_q_curve_const

# --- Sidebar navigatie ---
from utils.nav import sidebar_nav

NAV_ENTRIES = [
    {"label": "Home",          "page": "Home",                         "icon": "🏠"},
    {"label": "SPX Options",   "page": "pages/3_SPX_Options.py",       "icon": "🧩"},
    {"label": "3D Greeks",     "page": "pages/4_Greeks_3D.py",         "icon": "🧮"},
    # Voeg hier jouw andere pagina’s toe (pas paden/namen aan):
    # {"label": "Yield Curve",   "page": "pages/2_Yield_Curve.py",       "icon": "📈"},
    # {"label": "Macro",         "page": "pages/5_Macro_Dashboard.py",   "icon": "🌍"},
]

# Zet current_slug naar (een deel van) de zichtbare label-naam van deze pagina
sidebar_nav(NAV_ENTRIES, section_title="📚 Trading Dashboard", current_slug="SPX Options")  # op de SPX-pagina
# sidebar_nav(NAV_ENTRIES, section_title="📚 Trading Dashboard", current_slug="3D Greeks")  # op de Greeks-pagina


# ─────────────────────────── Helpers ───────────────────────────
def norm_cdf(x):  # Φ(x)
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def norm_pdf(x):  # φ(x)
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

def d1_d2(S, K, T, r, q, sigma):
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return np.nan, np.nan
    srt = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / srt
    d2 = d1 - srt
    return d1, d2

# —— Greeks (Espen Haug formules, continuous r/q) ——
def greeks(S, K, T, r, q, sigma, is_call=True):
    d1, d2 = d1_d2(S, K, T, r, q, sigma)
    if np.isnan(d1):  # guard
        return {k: np.nan for k in ["delta","gamma","vega","theta","rho","vanna","vomma","speed","zomma","charm","color"]}
    phi = norm_pdf(d1)
    Nd1 = norm_cdf(d1)
    Nd2 = norm_cdf(d2)
    disc_q = math.exp(-q * T)
    disc_r = math.exp(-r * T)

    # Base
    if is_call:
        delta = disc_q * Nd1
        rho   =  T * disc_r * norm_cdf(d2)
    else:
        delta = disc_q * (Nd1 - 1.0)
        rho   = -T * disc_r * norm_cdf(-d2)

    gamma = disc_q * phi / (S * sigma * math.sqrt(T))
    vega  = S * disc_q * phi * math.sqrt(T)  # per vol absolute (not %)
    # Commonly reported per 1% vol: vega/100; we leave absolute

    # Theta (per year)
    term1 = - (S * disc_q * phi * sigma) / (2.0 * math.sqrt(T))
    if is_call:
        theta = term1 - r * (K * disc_r) * norm_cdf(d2) + q * (S * disc_q) * norm_cdf(d1)
    else:
        theta = term1 + r * (K * disc_r) * norm_cdf(-d2) - q * (S * disc_q) * norm_cdf(-d1)

    # Higher orders
    vanna = disc_q * phi * math.sqrt(T) * (1.0 - d1 / (sigma * math.sqrt(T)))
    # alternative closed form: vanna = vega * (1 - d1/(sigma*sqrtT)) / S
    vomma = vega * d1 * d2 / sigma  # aka volga
    speed = -gamma / S * (1 + d1 / (sigma * math.sqrt(T)))
    # zomma: dGamma/dVol
    zomma = gamma * (d1 * d2 - 1) / sigma
    # charm (dDelta/dt under BS; per year):
    charm = disc_q * (phi * (2*(r - q)*T - d2*sigma*math.sqrt(T)) / (2*T*sigma*math.sqrt(T))) \
            - (q * disc_q * (Nd1 if is_call else (Nd1 - 1)))
    # color (dGamma/dt)
    color = -disc_q * (phi / (2*S*T*sigma*math.sqrt(T))) * (1 + (2*(r - q)*T - d1*sigma*math.sqrt(T)) * d1)

    return dict(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho,
                vanna=vanna, vomma=vomma, speed=speed, zomma=zomma, charm=charm, color=color)

def bs_price(S,K,T,r,q,sigma,is_call=True):
    d1, d2 = d1_d2(S,K,T,r,q,sigma)
    if np.isnan(d1): return np.nan
    disc_q = math.exp(-q*T); disc_r = math.exp(-r*T)
    if is_call:
        return S*disc_q*norm_cdf(d1) - K*disc_r*norm_cdf(d2)
    return K*disc_r*norm_cdf(-d2) - S*disc_q*norm_cdf(-d1)

# Robust mid price
def best_px(row):
    for col in ("mid_price","last_price","bid","ask"):
        v = row.get(col, np.nan)
        if v is not None and not pd.isna(v) and float(v) > 0:
            return float(v)
    return np.nan

# ─────────────────────────── BigQuery ─────────────────────────────
def get_bq_client():
    sa_info = st.secrets["gcp_service_account"]
    creds = service_account.Credentials.from_service_account_info(sa_info)
    project_id = st.secrets.get("PROJECT_ID", sa_info.get("project_id"))
    return bigquery.Client(project=project_id, credentials=creds)
_bq_client = get_bq_client()

def _bq_param(name, value):
    if isinstance(value, (list, tuple)):
        if len(value) == 0: return bigquery.ArrayQueryParameter(name, "STRING", [])
        e = value[0]
        if isinstance(e, int):   return bigquery.ArrayQueryParameter(name, "INT64", list(value))
        if isinstance(e, float): return bigquery.ArrayQueryParameter(name, "FLOAT64", list(value))
        if isinstance(e, (date, pd.Timestamp, datetime)):
            return bigquery.ArrayQueryParameter(name, "DATE", [str(pd.to_datetime(v).date()) for v in value])
        return bigquery.ArrayQueryParameter(name, "STRING", [str(v) for v in value])
    if isinstance(value, bool):                 return bigquery.ScalarQueryParameter(name, "BOOL", value)
    if isinstance(value, (int, np.integer)):    return bigquery.ScalarQueryParameter(name, "INT64", int(value))
    if isinstance(value, (float, np.floating)): return bigquery.ScalarQueryParameter(name, "FLOAT64", float(value))
    if isinstance(value, datetime):             return bigquery.ScalarQueryParameter(name, "TIMESTAMP", value)
    if isinstance(value, (date, pd.Timestamp)): return bigquery.ScalarQueryParameter(name, "DATE", str(pd.to_datetime(value).date()))
    return bigquery.ScalarQueryParameter(name, "STRING", str(value))

def run_query(sql: str, params: dict | None = None) -> pd.DataFrame:
    job_config = None
    if params:
        job_config = bigquery.QueryJobConfig(query_parameters=[_bq_param(k, v) for k, v in params.items()])
    return _bq_client.query(sql, job_config=job_config).to_dataframe()

# ─────────────────────────── UI setup ─────────────────────────────
st.set_page_config(page_title="3D Greeks & Surfaces", layout="wide")
st.title("🧮 3D Greeks met r/q-toggles & IV-interpolatie")

VIEW = "marketdata.spx_options_enriched_v"  # pas aan indien nodig
YIELD_VIEW = "nth-pier-468314-p7.marketdata.yield_curve_analysis_wide"  # check jouw dataset

PLOTLY_CONFIG = {"scrollZoom": True, "doubleClick": "reset", "displaylogo": False, "modeBarButtonsToRemove": ["lasso2d","select2d"]}

# ─────────────────────────── Filters ──────────────────────────────
@st.cache_data(ttl=600, show_spinner=False)
def load_date_bounds():
    df = run_query(f"SELECT MIN(CAST(snapshot_date AS DATE)) min_date, MAX(CAST(snapshot_date AS DATE)) max_date FROM `{VIEW}`")
    return df["min_date"].iloc[0], df["max_date"].iloc[0]
min_date, max_date = load_date_bounds()
default_start = max(min_date, max_date - timedelta(days=30))

colA, colB, colC, colD = st.columns([1.2, 1, 1, 1.1])
with colA:
    start_date, end_date = st.date_input("Periode (snapshot_date)",
        value=(default_start, max_date), min_value=min_date, max_value=max_date, format="YYYY-MM-DD")
with colB:
    sel_type = st.radio("Optietype", ["call","put"], index=0, horizontal=True)
with colC:
    dte_pick = st.slider("Doel DTE (dagen)", 1, 120, 30, step=1)
with colD:
    snap_pick_mode = st.radio("Peildatum", ["Laatste in range","Kies exact"], index=0, horizontal=True)

@st.cache_data(ttl=600, show_spinner=True)
def load_chain(start_date, end_date, typ):
    sql = f"""
      SELECT
        TIMESTAMP_TRUNC(snapshot_date, MINUTE) AS snap_m,
        snapshot_date, type, expiration, days_to_exp,
        strike, underlying_price, implied_volatility,
        open_interest, volume, last_price, mid_price, bid, ask
      FROM `{VIEW}`
      WHERE DATE(snapshot_date) BETWEEN @start AND @end
        AND LOWER(type) = @t
    """
    df = run_query(sql, {"start": start_date, "end": end_date, "t": typ})
    if df.empty: return df
    df["snap_m"] = pd.to_datetime(df["snap_m"])
    df["expiration"] = pd.to_datetime(df["expiration"]).dt.date
    for c in ["days_to_exp","strike","underlying_price","implied_volatility","open_interest","volume","last_price","mid_price","bid","ask"]:
        if c in df: df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

df = load_chain(start_date, end_date, sel_type)
if df.empty:
    st.warning("Geen data in deze selectie.")
    st.stop()

snapshots = sorted(df["snap_m"].unique())
if snap_pick_mode == "Kies exact":
    sel_snapshot = st.selectbox("Kies snapshot (min-resolution)", options=snapshots, index=len(snapshots)-1, format_func=lambda x: pd.to_datetime(x).strftime("%Y-%m-%d %H:%M"))
else:
    sel_snapshot = snapshots[-1]

df_s = df[df["snap_m"] == sel_snapshot].copy()
if df_s.empty:
    st.warning("Geen records op dit snapshot.")
    st.stop()

underlying_now = float(df_s["underlying_price"].median())
exps = sorted(df_s["expiration"].unique().tolist())

colE1, colE2 = st.columns([1,1])
with colE1:
    exp_choice = st.selectbox("Expiratie", options=exps, index=min(range(len(exps)), key=lambda i: abs(int(df_s[df_s["expiration"]==exps[i]]["days_to_exp"].median()) - dte_pick)) if exps else 0)
with colE2:
    liq_min_oi = st.slider("Min OI (filter)", 0, 50, 1, step=1)

df_e = df_s[(df_s["expiration"]==exp_choice) & ((df_s["open_interest"]>=liq_min_oi) | (df_s["volume"]>0))].copy()
if df_e.empty:
    st.warning("Geen (liquide) ketting voor deze expiratie.")
    st.stop()

dte = int(pd.to_numeric(df_e["days_to_exp"], errors="coerce").median())
T = max(dte,1)/365.0

# ─────────────────────────── r/q bron ————————————————————————
tw1, tw2, tw3 = st.columns([1.2, 1, 1])
with tw1:
    use_yield_r = st.toggle("Gebruik r uit yield-curve", value=True)
with tw2:
    r_manual_simple = st.number_input("r (p.j., simple) — handmatig", min_value=-0.02, max_value=0.10, value=0.02, step=0.001, format="%.3f", disabled=use_yield_r)
with tw3:
    q_mode = st.radio("q-bron", ["Constante q","Implied via C−P"], index=0, horizontal=True)

# r(T)
if use_yield_r:
    r_T = get_r_curve_for_snapshot(snapshot_date=pd.to_datetime(sel_snapshot), T_years=np.array([T]), view=YIELD_VIEW,
                                   compounding_in="simple", extrapolate=True)[0]
    r_cont = math.log1p(r_T)  # simple → continuous
else:
    r_cont = math.log1p(float(r_manual_simple))

# q(T)
q_const_simple = st.number_input("q (p.j., simple) — constante", min_value=0.0, max_value=0.10, value=0.016, step=0.001, format="%.3f",
                                 disabled=(q_mode!="Constante q"))

def implied_q_from_parity(df_slice, S, T, r_cont):
    if df_slice.empty or S<=0 or T<=0: return np.nan
    tmp = df_slice.copy()
    tmp["px"] = tmp.apply(lambda r: best_px(r), axis=1)
    piv = tmp.dropna(subset=["px"]).pivot_table(index="strike", columns="type", values="px", aggfunc="median")
    if "call" not in piv.columns or "put" not in piv.columns or piv.empty:
        return np.nan
    piv["atm_abs"] = (piv.index - S).abs()
    piv = piv.dropna(subset=["call","put"]).sort_values("atm_abs")
    if piv.empty: return np.nan
    K = float(piv.index[0]); C = float(piv.iloc[0]["call"]); P = float(piv.iloc[0]["put"])
    try:
        val = (C - P + K * math.exp(-r_cont * T)) / S
        if val <= 0: return np.nan
        q_cont = -math.log(val) / T
        if not (-0.10 <= q_cont <= 0.10): return np.nan
        return q_cont
    except Exception:
        return np.nan

if q_mode == "Implied via C−P":
    q_cont = implied_q_from_parity(df_e, underlying_now, T, r_cont)
    if np.isnan(q_cont):
        st.warning("Kon q niet betrouwbaar afleiden uit put–call-pariteit. Val terug op constante q.")
        q_cont = get_q_curve_const(np.array([T]), q_const=q_const_simple, to_continuous=True)[0]
else:
    q_cont = get_q_curve_const(np.array([T]), q_const=q_const_simple, to_continuous=True)[0]

# ─────────────────────────── IV-bron & interpolatie ——————————————
st.markdown("### IV-bron & interpolatie")
ivcol1, ivcol2, ivcol3 = st.columns([1.2, 1, 1])
with ivcol1:
    iv_source = st.radio("IV-bron", ["Flat ATM-IV","Smile (strike-interp)","Term+Smile (bivariaat)"], index=1, horizontal=False)
with ivcol2:
    strike_band = st.slider("Strike-band (± punten rond spot)", 50, 600, 300, step=50)
with ivcol3:
    grid_S_pct = st.slider("S-raster (±% rond spot)", 5, 30, 15, step=1)

# bouw ketting rond strike-band
df_band = df_e[(df_e["strike"].between(underlying_now - strike_band, underlying_now + strike_band))].copy()
if df_band.empty:
    st.warning("Geen data binnen de gekozen strike-band.")
    st.stop()

# ATM-IV
atm_rows = df_band[np.abs(df_band["strike"] - underlying_now) == np.abs(df_band["strike"] - underlying_now).min()]
iv_atm = float(pd.to_numeric(atm_rows["implied_volatility"], errors="coerce").median())

# Smile data
smile = (df_band.groupby("strike", as_index=False)["implied_volatility"].median().dropna().sort_values("strike"))
strikes = smile["strike"].values.astype(float)
iv_strike = smile["implied_volatility"].values.astype(float)

# eenvoudige 1D lineaire interpolatie over strikes
def iv_from_smile(K):
    if len(strikes) < 2 or np.isnan(iv_atm): return iv_atm
    K = float(K)
    if K <= strikes[0]: return iv_strike[0]
    if K >= strikes[-1]: return iv_strike[-1]
    i = np.searchsorted(strikes, K)
    x0,x1 = strikes[i-1], strikes[i]
    y0,y1 = iv_strike[i-1], iv_strike[i]
    t = (K - x0)/(x1 - x0) if x1!=x0 else 0.0
    return float(y0 + t*(y1 - y0))

# term+smile: simpele tilt op basis van lokale term-structure (median IV vs DTE rond T)
if iv_source == "Term+Smile (bivariaat)":
    term = (df_s.groupby("days_to_exp", as_index=False)["implied_volatility"].median().dropna().sort_values("days_to_exp"))
    if len(term) >= 2:
        # lineaire interp in DTE
        dte_arr = term["days_to_exp"].values.astype(float)
        ivt_arr = term["implied_volatility"].values.astype(float)
        def iv_term(dte):
            if dte <= dte_arr[0]: return ivt_arr[0]
            if dte >= dte_arr[-1]: return ivt_arr[-1]
            j = np.searchsorted(dte_arr, dte)
            x0,x1 = dte_arr[j-1], dte_arr[j]; y0,y1 = ivt_arr[j-1], ivt_arr[j]
            u = (dte - x0)/(x1 - x0) if x1!=x0 else 0.0
            return float(y0 + u*(y1 - y0))
        iv_T = iv_term(dte)
    else:
        iv_T = iv_atm
else:
    iv_T = iv_atm

def iv_func(K):
    if iv_source == "Flat ATM-IV":
        return iv_atm
    elif iv_source == "Smile (strike-interp)":
        return iv_from_smile(K)
    else:  # Term+Smile
        base_k = iv_from_smile(K)
        # tilt richting term iv_T t.o.v. atm:
        # base_k_adj = base_k + (iv_T - iv_atm)  (behoud relatieve smile)
        return float(base_k + (iv_T - iv_atm))

# ─────────────────────────── Mesh & Greeks ————————————————
S_min = underlying_now * (1 - grid_S_pct/100)
S_max = underlying_now * (1 + grid_S_pct/100)
S_grid = np.linspace(S_min, S_max, 41)  # 41 x 51 grid default
K_grid = np.linspace(max(5, underlying_now - strike_band), underlying_now + strike_band, 51)
SS, KK = np.meshgrid(S_grid, K_grid, indexing="xy")

# Compute selected Greek on grid
st.markdown("### 3D Surface instellingen")
gcol1, gcol2, gcol3 = st.columns([1.2, 1, 1])
with gcol1:
    greek_name = st.selectbox("Welke Greek?", ["delta","gamma","vega","theta","rho","vanna","vomma","speed","zomma","charm","color"], index=2)
with gcol2:
    show_wire = st.checkbox("Wireframe overlay", value=False)
with gcol3:
    is_call = (sel_type == "call")

Z = np.empty_like(SS, dtype=float)
IVm = np.empty_like(SS, dtype=float)

for i in range(SS.shape[0]):
    for j in range(SS.shape[1]):
        S = float(SS[i,j]); K = float(KK[i,j])
        sigma = float(max(1e-6, iv_func(K)))  # guard
        IVm[i,j] = sigma
        g = greeks(S, K, T, r_cont, q_cont, sigma, is_call=is_call)
        Z[i,j] = g.get(greek_name, np.nan)

# ─────────────────────────── Plot 3D ————————————————
fig = go.Figure(data=[go.Surface(x=SS, y=KK, z=Z, showscale=True, contours={"z": {"show": False}})])
if show_wire:
    fig.add_trace(go.Surface(x=SS, y=KK, z=Z, colorscale="Greys", opacity=0.25, showscale=False,
                             contours={"z": {"show": True, "usecolormap": False, "highlightcolor": "black", "project_z": True}}))
fig.update_layout(title=f"3D Surface — {greek_name.upper()} | exp {exp_choice} | DTE≈{dte} | r={r_cont:.3%} (cont), q={q_cont:.3%}",
                  scene=dict(
                      xaxis_title="S (onderliggende)",
                      yaxis_title="K (strike)",
                      zaxis_title=greek_name),
                  height=700)
st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

# ─────────────────────────── 2D slices ————————————————
st.markdown("### 2D Slices")
scol1, scol2 = st.columns(2)

with scol1:
    K_slice = float(pd.to_numeric(df_band["strike"], errors="coerce").median())
    K_sel = st.slider("Slice @ K", int(K_grid.min()), int(K_grid.max()), int(round(K_slice)), step=5)
    # S→Greek at fixed K
    S_line = np.linspace(S_min, S_max, 200)
    Z_line = []
    for S in S_line:
        sig = float(max(1e-6, iv_func(K_sel)))
        g = greeks(S, K_sel, T, r_cont, q_cont, sig, is_call=is_call)
        Z_line.append(g.get(greek_name, np.nan))
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=S_line, y=Z_line, mode="lines", name=f"{greek_name} @ K={K_sel:.0f}"))
    fig1.update_layout(title=f"{greek_name.upper()} vs S @ K={K_sel:.0f}", xaxis_title="S", yaxis_title=greek_name, height=380)
    st.plotly_chart(fig1, use_container_width=True, config=PLOTLY_CONFIG)

with scol2:
    S_sel = float(underlying_now)
    S_sel = st.slider("Slice @ S", int(S_min), int(S_max), int(round(S_sel)), step=10)
    K_line = np.linspace(K_grid.min(), K_grid.max(), 220)
    Zk_line = []
    for K in K_line:
        sig = float(max(1e-6, iv_func(K)))
        g = greeks(S_sel, K, T, r_cont, q_cont, sig, is_call=is_call)
        Zk_line.append(g.get(greek_name, np.nan))
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=K_line, y=Zk_line, mode="lines", name=f"{greek_name} @ S={S_sel:.0f}"))
    fig2.update_layout(title=f"{greek_name.upper()} vs K @ S={S_sel:.0f}", xaxis_title="K", yaxis_title=greek_name, height=380)
    st.plotly_chart(fig2, use_container_width=True, config=PLOTLY_CONFIG)

# ─────────────────────────── IV Surface (optioneel) ————————————
with st.expander("🔬 Toon IV-matrix (ter controle van de smile)", expanded=False):
    fig_iv = go.Figure(data=[go.Surface(x=SS, y=KK, z=IVm, showscale=True)])
    fig_iv.update_layout(title="IV over (S,K) — indirect (K-afhankelijk; S-onafhankelijk weergegeven)", 
                         scene=dict(xaxis_title="S", yaxis_title="K", zaxis_title="IV"), height=520)
    st.plotly_chart(fig_iv, use_container_width=True, config=PLOTLY_CONFIG)
    st.caption("NB: in dit model is IV alleen functie van K (en T); S-as is getoond om dezelfde grid te delen met de Greeks-plot.")

# ─────────────────────────── Sanity Cards ————————————————
c1, c2, c3, c4 = st.columns(4)
with c1: st.metric("Spot (S)", f"{underlying_now:,.2f}")
with c2: st.metric("DTE", f"{dte}")
with c3: st.metric("ATM-IV", f"{iv_atm:.2%}")
with c4: st.metric("q (cont.)", f"{q_cont:.2%}")

st.caption("Tip: kies **Term+Smile** om te zien hoe Δ/Γ/Vega-velden verschuiven wanneer de term structure steiler/vlakker is. "
           "Gebruik de r/q-toggels om carry-effecten (discount/dividend) te testen.")
