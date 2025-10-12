# pages/3_SPX_Options.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import math
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly import colors as pc

from google.cloud import bigquery
from google.oauth2 import service_account

# —— NIEUW: r/q helpers uit yield-view ——
from utils.rates import get_r_curve_for_snapshot, get_q_curve_const

# ─────────────────────────── Convenience ───────────────────────────
def norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

def norm_pdf(z: float) -> float:
    return (1.0/math.sqrt(2*math.pi)) * math.exp(-0.5*z*z)

def pick_closest_date(options: list[date], target: date):
    if not options: return None
    return min(options, key=lambda d: abs(pd.Timestamp(d) - pd.Timestamp(target)))

def pick_first_on_or_after(options: list[date], target: date):
    after = [d for d in options if d >= target]
    return after[0] if after else (options[-1] if options else None)

def pick_closest_value(options: list[float], target: float, fallback: float | None = None):
    if not options: return fallback
    return float(min(options, key=lambda x: abs(float(x) - float(target))))

def smooth_series(y: pd.Series, window: int = 3) -> pd.Series:
    if len(y) < 3: return y
    return y.rolling(window, center=True, min_periods=1).median()

def annualize_std(s: pd.Series, periods_per_year: int = 252) -> float:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) < 2: return np.nan
    return float(s.std(ddof=0) * math.sqrt(periods_per_year))

# —— Black–Scholes met continue r en q ——
def bs_delta(S: float, K: float, iv: float, T_years: float, r_cont: float, q_cont: float, is_call: bool) -> float:
    # S, K > 0; iv >= 0; T_years > 0
    if any(x is None or np.isnan(x) or x <= 0 for x in [S, K]) or np.isnan(iv) or T_years <= 0:
        return np.nan
    sqrtT = math.sqrt(T_years)
    denom = iv * sqrtT if iv > 0 else float("inf")
    d1 = (math.log(S / K) + (r_cont - q_cont + 0.5 * iv * iv) * T_years) / denom
    Nd1 = norm_cdf(d1)
    disc_q = math.exp(-q_cont * T_years)
    return (disc_q * Nd1) if is_call else (disc_q * (Nd1 - 1.0))

def bs_gamma(S: float, K: float, iv: float, T_years: float, r_cont: float, q_cont: float) -> float:
    if any(x is None or np.isnan(x) or x <= 0 for x in [S, K]) or np.isnan(iv) or iv <= 0 or T_years <= 0:
        return np.nan
    sqrtT = math.sqrt(T_years)
    d1 = (math.log(S / K) + (r_cont - q_cont + 0.5 * iv * iv) * T_years) / (iv * sqrtT)
    disc_q = math.exp(-q_cont * T_years)
    return disc_q * norm_pdf(d1) / (S * iv * sqrtT)

def strangle_payoff_at_expiry(S, Kp, Kc, credit_pts, multiplier=100) -> float:
    return (credit_pts - max(Kp - S, 0.0) - max(S - Kc, 0.0)) * multiplier

def span_like_margin(S, Kp, Kc, credit_pts, down=0.15, up=0.10, multiplier=100) -> float:
    S_down, S_up = S*(1-down), S*(1+up)
    loss_down = (max(Kp - S_down, 0.0) - credit_pts) * multiplier
    loss_up   = (max(S_up - Kc, 0.0) - credit_pts) * multiplier
    return float(max(0.0, loss_down, loss_up))

def regt_strangle_margin(S, Kp, Kc, put_px_pts, call_px_pts, multiplier=100) -> float:
    otm_call, otm_put = max(Kc - S, 0.0), max(S - Kp, 0.0)
    base_call = max(0.20 * S - otm_call, 0.10 * S)
    base_put  = max(0.20 * S - otm_put,  0.10 * S)
    req_call  = (call_px_pts + base_call) * multiplier
    req_put   = (put_px_pts  + base_put ) * multiplier
    worst_leg = max(req_call, req_put)
    other_leg = put_px_pts if worst_leg == req_call else call_px_pts
    return float(worst_leg + other_leg * multiplier)

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

# ─────────────────────────── Page setup ───────────────────────────
st.set_page_config(page_title="SPX Options Dashboard", layout="wide")
st.title("🧩 SPX Options Dashboard")
VIEW = "marketdata.spx_options_enriched_v"

# Plotly UI
PLOTLY_CONFIG = {
    "scrollZoom": True,
    "doubleClick": "reset",
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"]
}

with st.expander("📌 Workflow (kort): van data → strangle-keuze", expanded=False):
    st.markdown(
        """
1) Kies **periode, type, DTE & moneyness**.  
2) Bekijk **Serie-selectie** om gevoel te krijgen voor prijs/PPD en liquiditeit.  
3) Gebruik **PPD vs Afstand** & **PPD vs DTE** om een sweet spot te vinden.  
4) Check **Vol & Risk** (IV-Rank/VRP/Expected Move).  
5) In **Strangle Helper** kies je σ/Δ-doelen of gebruik **Auto-pick**.  
6) Valideer met **Margin & Payoff** en test een **Roll**.
        """
    )

# ─────────────────────────── Filters ──────────────────────────────
@st.cache_data(ttl=600, show_spinner=False)
def load_date_bounds():
    df = run_query(f"SELECT MIN(CAST(snapshot_date AS DATE)) min_date, MAX(CAST(snapshot_date AS DATE)) max_date FROM `{VIEW}`")
    return df["min_date"].iloc[0], df["max_date"].iloc[0]

min_date, max_date = load_date_bounds()
default_start = max(min_date, max_date - timedelta(days=365))

colA, colB, colC, colD, colE = st.columns([1.3, 0.8, 1, 1, 1.2])
with colA:
    start_date, end_date = st.date_input("Periode (snapshot_date)",
        value=(default_start, max_date), min_value=min_date, max_value=max_date, format="YYYY-MM-DD")
with colB:
    sel_type = st.radio("Type", ["call", "put"], index=1, horizontal=True)
with colC:
    dte_range = st.slider("DTE", 0, 365, (0, 60), step=1)
with colD:
    mny_range = st.slider("Moneyness (K/S − 1)", -0.20, 0.20, (-0.10, 0.10), step=0.01)
with colE:
    show_underlying = st.toggle("Overlay S&P500", value=True)

# Liquidity guardrails
colL1, colL2, colL3 = st.columns([1, 1, 1])
with colL1:
    min_oi = st.slider("Min Open Interest (filter)", 0, 50, 1, step=1)
with colL2:
    min_vol = st.slider("Min Volume (filter)", 0, 50, 1, step=1)
with colL3:
    min_per_bin = st.slider("Min punten per bin (aggr)", 1, 10, 3, step=1, help="Voor PPD-aggregaties per afstand/DTE.")

# expirations
@st.cache_data(ttl=600, show_spinner=False)
def load_expirations(start_date: date, end_date: date, sel_type: str):
    df = run_query(f"""
        SELECT DISTINCT expiration
        FROM `{VIEW}`
        WHERE DATE(snapshot_date) BETWEEN @start AND @end
          AND LOWER(type) = @t
        ORDER BY expiration
    """, {"start": start_date, "end": end_date, "t": sel_type})
    return sorted(pd.to_datetime(df["expiration"]).dt.date.unique())

exps = load_expirations(start_date, end_date, sel_type)

# data
@st.cache_data(ttl=600, show_spinner=True)
def load_filtered(start_date, end_date, sel_type, dte_min, dte_max, mny_min, mny_max):
    sql = f"""
    WITH base AS (
      SELECT
        snapshot_date, contract_symbol, type, expiration, days_to_exp,
        strike, underlying_price,
        SAFE_DIVIDE(CAST(strike AS FLOAT64), NULLIF(underlying_price, 0)) - 1.0 AS moneyness,
        (CAST(strike AS FLOAT64) - CAST(underlying_price AS FLOAT64)) AS dist_points,
        in_the_money, last_price, bid, ask, mid_price,
        implied_volatility, open_interest, volume, vix, ppd
      FROM `{VIEW}`
      WHERE DATE(snapshot_date) BETWEEN @start AND @end
        AND LOWER(type) = @t
        AND days_to_exp BETWEEN @dte_min AND @dte_max
        AND SAFE_DIVIDE(CAST(strike AS FLOAT64), NULLIF(underlying_price, 0)) - 1.0
            BETWEEN @mny_min AND @mny_max
    )
    SELECT * FROM base
    """
    params = {"start": start_date, "end": end_date, "t": sel_type,
              "dte_min": int(dte_min), "dte_max": int(dte_max),
              "mny_min": float(mny_min), "mny_max": float(mny_max)}
    df = run_query(sql, params=params)
    if not df.empty:
        df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
        df["expiration"]    = pd.to_datetime(df["expiration"]).dt.date
        num_cols = ["days_to_exp","implied_volatility","open_interest","volume","ppd",
                    "strike","underlying_price","last_price","mid_price","bid","ask","dist_points"]
        for c in num_cols:
            if c in df: df[c] = pd.to_numeric(df[c], errors="coerce")
        df["moneyness_pct"] = 100 * df["moneyness"]
        df["abs_dist_pct"]  = (np.abs(df["dist_points"]) / df["underlying_price"]) * 100.0
        df["snap_min"] = df["snapshot_date"].dt.floor("min")
    return df

df = load_filtered(start_date, end_date, sel_type, dte_range[0], dte_range[1], mny_range[0], mny_range[1])
if df.empty:
    st.warning("Geen data voor de huidige filters.")
    st.stop()

# Liquidity mask
liq_mask = ((df["open_interest"].fillna(0) >= min_oi) | (df["volume"].fillna(0) >= min_vol))

# KPIs (basis)
c1, c2, c3, c4 = st.columns(4)
with c1: st.metric("Records", f"{len(df):,}")
with c2: st.metric("Gem. IV", f"{df['implied_volatility'].mean():.2%}")
with c3: st.metric("Som Volume", f"{int(df['volume'].sum()):,}")
with c4: st.metric("Som OI", f"{int(df['open_interest'].sum()):,}")
st.markdown("---")

# ─────────────────────────── Nieuw: Sentiment & Positioning ───────────────────────────
st.header("🧭 Sentiment & Positioning")

# — Settings voor r/q (lightweight en robuust)
sent_co1, sent_co2, sent_co3 = st.columns([1.1, 1, 1])
with sent_co1:
    q_const_simple = st.number_input("Dividendrendement q (p.j., simple)", min_value=0.0, max_value=0.10, value=0.016, step=0.001, format="%.3f")
with sent_co2:
    dte_for_skew = st.slider("DTE-range voor 25Δ Skew", 10, 60, (20, 40), step=5)
with sent_co3:
    atm_band_skew = st.slider("ATM-band voor term-slope (±%)", 0.5, 5.0, 1.0, step=0.5)

def to_T_years(dte):
    return float(max(pd.to_numeric(dte, errors="coerce"), 1)) / 365.0

def _delta_vec(S, K, iv, T, is_call):
    # r=0, q≈const (to continuous)
    q_T = float(get_q_curve_const(np.array([T], dtype=float), q_const=q_const_simple, to_continuous=True)[0]) if not np.isnan(T) else 0.0
    return bs_delta(S, K, iv, T, r_cont=0.0, q_cont=q_T, is_call=is_call)

def _gamma_vec(S, K, iv, T):
    q_T = float(get_q_curve_const(np.array([T], dtype=float), q_const=q_const_simple, to_continuous=True)[0]) if not np.isnan(T) else 0.0
    return bs_gamma(S, K, iv, T, r_cont=0.0, q_cont=q_T)

# — Tijdreeks PCR (vol/oi)
pcr_df = (df.assign(day=df["snapshot_date"].dt.date)
            .groupby(["day","type"], as_index=False)
            .agg(vol=("volume","sum"), oi=("open_interest","sum")))
if not pcr_df.empty:
    pv = (pcr_df.pivot_table(index="day", columns="type", values=["vol","oi"], aggfunc="sum")
                 .sort_index().sort_index(axis=1)).fillna(0.0)
    pv.columns = [f"{a}_{b}" for a,b in pv.columns.to_flat_index()]
    for col in ["vol_put","vol_call","oi_put","oi_call"]:
        if col not in pv.columns: pv[col] = 0.0
    pv["PCR_vol"] = pv["vol_put"] / pv["vol_call"].replace(0, np.nan)
    pv["PCR_oi"]  = pv["oi_put"]  / pv["oi_call"].replace(0, np.nan)
else:
    pv = pd.DataFrame(columns=["PCR_vol","PCR_oi"])

# — Delta-gewogen PCR (tijdreeks; beperkt tot redelijke strikes & DTE)
dg_mask = (df["days_to_exp"].between(7, 45)) & (df["moneyness"].abs() <= 0.20) & liq_mask
dg = df[dg_mask].copy()
if not dg.empty:
    # Vectorized BS-delta
    T = (pd.to_numeric(dg["days_to_exp"], errors="coerce").fillna(0)/365.0).astype(float)
    S = pd.to_numeric(dg["underlying_price"], errors="coerce").astype(float)
    K = pd.to_numeric(dg["strike"], errors="coerce").astype(float)
    IV = pd.to_numeric(dg["implied_volatility"], errors="coerce").astype(float)

    is_call = (dg["type"].str.lower()=="call").astype(bool)
    deltas = np.vectorize(bs_delta)(
        S.values, K.values, IV.values, T.values,
        np.zeros_like(T.values),  # r
        np.vectorize(lambda t: float(get_q_curve_const(np.array([t],dtype=float), q_const=q_const_simple, to_continuous=True)[0]))(T.values),
        is_call.values
    )
    dg["delta_abs"] = np.abs(deltas)
    # Volume-gewogen is gevoeliger voor “wat er vandaag speelt”
    dg["dw_vol"] = dg["delta_abs"] * pd.to_numeric(dg["volume"], errors="coerce").fillna(0)
    dg["dw_oi"]  = dg["delta_abs"] * pd.to_numeric(dg["open_interest"], errors="coerce").fillna(0)

    dgt = (dg.assign(day=dg["snapshot_date"].dt.date)
             .groupby(["day","type"], as_index=False)
             .agg(dw_vol=("dw_vol","sum"), dw_oi=("dw_oi","sum")))
    dgp = (dgt.pivot_table(index="day", columns="type", values=["dw_vol","dw_oi"], aggfunc="sum")
               .sort_index().sort_index(axis=1)).fillna(0.0)
    if ("dw_vol_put" in dgp.columns) and ("dw_vol_call" in dgp.columns):
        dgp["PCR_delta_vol"] = dgp["dw_vol_put"] / dgp["dw_vol_call"].replace(0, np.nan)
    else:
        dgp["PCR_delta_vol"] = np.nan
    if ("dw_oi_put" in dgp.columns) and ("dw_oi_call" in dgp.columns):
        dgp["PCR_delta_oi"] = dgp["dw_oi_put"] / dgp["dw_oi_call"].replace(0, np.nan)
    else:
        dgp["PCR_delta_oi"] = np.nan
else:
    dgp = pd.DataFrame(columns=["PCR_delta_vol","PCR_delta_oi"])

# — 25Δ Skew (laatste snapshot; mediane skew over ~30D expiries)
snapshots_all = sorted(df["snap_min"].unique())
default_snapshot = snapshots_all[-1] if snapshots_all else None
df_last = df[(df["snap_min"] == default_snapshot) & liq_mask].copy() if default_snapshot is not None else pd.DataFrame()

def compute_25d_skew(df_last, dte_lo:int, dte_hi:int) -> float:
    if df_last.empty: return np.nan
    z = df_last[df_last["days_to_exp"].between(dte_lo, dte_hi)].copy()
    if z.empty: return np.nan
    # Delta per rij
    z["T"] = pd.to_numeric(z["days_to_exp"], errors="coerce").fillna(0)/365.0
    z["delta"] = z.apply(lambda r: bs_delta(
        float(r["underlying_price"]), float(r["strike"]),
        float(r["implied_volatility"]), float(r["T"]),
        0.0, float(get_q_curve_const(np.array([r["T"]],dtype=float), q_const=q_const_simple, to_continuous=True)[0]),
        is_call=(str(r["type"]).lower()=="call")), axis=1)
    if z["delta"].isna().all(): return np.nan
    # dichtst bij -0.25 (put) en +0.25 (call) per expiratie
    rows = []
    for e, g in z.groupby("expiration"):
        gp = g[g["type"].str.lower()=="put"].copy()
        gc = g[g["type"].str.lower()=="call"].copy()
        if not gp.empty:
            gp["dist"] = (gp["delta"] + 0.25).abs()
            rowp = gp.loc[gp["dist"].idxmin()]
        else:
            rowp = None
        if not gc.empty:
            gc["dist"] = (gc["delta"] - 0.25).abs()
            rowc = gc.loc[gc["dist"].idxmin()]
        else:
            rowc = None
        if rowp is not None and rowc is not None:
            rows.append(float(rowp["implied_volatility"]) - float(rowc["implied_volatility"]))
    return float(np.nanmedian(rows)) if rows else np.nan

skew_25d = compute_25d_skew(df_last, dte_for_skew[0], dte_for_skew[1])

# — IV term-slope (ATM-band) en 1w expected move
def iv_term_slope(df_scope, band_pct: float) -> float:
    if df_scope.empty: return np.nan
    a = df_scope[(df_scope["moneyness"].abs() <= band_pct/100.0)]
    short = a[a["days_to_exp"].between(7, 15)]["implied_volatility"].median()
    mid   = a[a["days_to_exp"].between(30, 60)]["implied_volatility"].median()
    if np.isnan(short) or np.isnan(mid): return np.nan
    return float(short - mid)

iv_slope = iv_term_slope(df_last, atm_band_skew)

underlying_now = float(df_last["underlying_price"].median()) if not df_last.empty else float(pd.to_numeric(df["underlying_price"], errors="coerce").dropna().iloc[-1])
iv_1w = float(df_last[df_last["days_to_exp"].between(5,10)]["implied_volatility"].median()) if not df_last.empty else np.nan
em_1w_pts = (underlying_now * iv_1w * math.sqrt(7/365.0)) if (not np.isnan(underlying_now) and not np.isnan(iv_1w)) else np.nan

# — KPI-kaarten voor sentiment
def last_and_delta(series: pd.Series, lookback:int=20):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty: return np.nan, None
    last = float(s.iloc[-1])
    ref  = float(s.tail(lookback).mean()) if len(s) >= lookback else float(s.mean())
    return last, last - ref

k1, k2, k3, k4, k5 = st.columns(5)
pcr_v_last, pcr_v_delta = last_and_delta(pv["PCR_vol"] if "PCR_vol" in pv else pd.Series(dtype=float))
pcr_oi_last, pcr_oi_delta = last_and_delta(pv["PCR_oi"] if "PCR_oi" in pv else pd.Series(dtype=float))
with k1: st.metric("PCR (Volume)", f"{pcr_v_last:.2f}" if not np.isnan(pcr_v_last) else "—",
                   delta=(f"{pcr_v_delta:+.2f} vs 20d" if pcr_v_delta is not None else None))
with k2: st.metric("PCR (OI)", f"{pcr_oi_last:.2f}" if not np.isnan(pcr_oi_last) else "—",
                   delta=(f"{pcr_oi_delta:+.2f} vs 20d" if pcr_oi_delta is not None else None))
with k3: st.metric("25Δ Skew (put−call)", f"{skew_25d:.2%}" if not np.isnan(skew_25d) else "—")
with k4: st.metric("IV Term Slope (7–15d − 30–60d)", f"{iv_slope:.2%}" if not np.isnan(iv_slope) else "—")
with k5:
    em_txt = f"±{em_1w_pts:,.0f} pts ({em_1w_pts/underlying_now:.2%})" if (not np.isnan(em_1w_pts) and not np.isnan(underlying_now)) else "—"
    st.metric("Expected Move ~1w", em_txt)

# — Visuals: PCR & delta-gewogen PCR tijdreeks
if not pv.empty:
    fig_pcr_ts = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                               subplot_titles=("Put/Call Ratio — Volume", "Put/Call Ratio — Open Interest"))
    fig_pcr_ts.add_trace(go.Scatter(x=pv.index, y=pv["PCR_vol"], mode="lines", name="PCR (Vol)"), row=1, col=1)
    fig_pcr_ts.add_hline(y=1.0, line=dict(dash="dot"), row=1, col=1)
    fig_pcr_ts.add_trace(go.Scatter(x=pv.index, y=pv["PCR_oi"],  mode="lines", name="PCR (OI)"), row=2, col=1)
    fig_pcr_ts.add_hline(y=1.0, line=dict(dash="dot"), row=2, col=1)
    fig_pcr_ts.update_layout(height=520, title_text="Put/Call Ratio — Ontwikkeling", dragmode="zoom")
    st.plotly_chart(fig_pcr_ts, use_container_width=True, config=PLOTLY_CONFIG)

if not dgp.empty:
    fig_dpcr = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                             subplot_titles=("Delta-gewogen PCR — Volume", "Delta-gewogen PCR — Open Interest"))
    fig_dpcr.add_trace(go.Scatter(x=dgp.index, y=dgp.get("PCR_delta_vol"), mode="lines", name="Δ-PCR (Vol)"), row=1, col=1)
    fig_dpcr.add_hline(y=1.0, line=dict(dash="dot"), row=1, col=1)
    fig_dpcr.add_trace(go.Scatter(x=dgp.index, y=dgp.get("PCR_delta_oi"),  mode="lines", name="Δ-PCR (OI)"),  row=2, col=1)
    fig_dpcr.add_hline(y=1.0, line=dict(dash="dot"), row=2, col=1)
    fig_dpcr.update_layout(height=520, title_text="Delta-gewogen Put/Call Ratio — Ontwikkeling", dragmode="zoom")
    st.plotly_chart(fig_dpcr, use_container_width=True, config=PLOTLY_CONFIG)
st.caption("**Interpretatie:** PCR ↑ (zeker Δ-gewogen) wijst op defensievere positioning (meer put-bescherming). Een oplopende 25Δ-skew bevestigt verhoogde downside-premies.")

# — Gamma-exposure (proxy) per strike @ laatste snapshot
st.subheader("Gamma Exposure (proxy) — laatste snapshot")
if df_last.empty:
    st.info("Geen data op het laatste snapshot.")
else:
    gdf = df_last.copy()
    gdf["T"]  = pd.to_numeric(gdf["days_to_exp"], errors="coerce").fillna(0)/365.0
    gdf["Γ"]  = gdf.apply(lambda r: bs_gamma(
        float(r["underlying_price"]), float(r["strike"]), float(r["implied_volatility"]),
        float(r["T"]), 0.0, float(get_q_curve_const(np.array([r["T"]],dtype=float), q_const=q_const_simple, to_continuous=True)[0])
    ), axis=1)
    # Exposure ≈ Γ * OI * multiplier (100) * S^2  (S^2 om unit-neutraliteit te benaderen)
    mult = 100.0
    gdf["gex_raw"] = gdf["Γ"] * pd.to_numeric(gdf["open_interest"], errors="coerce").fillna(0) * mult * (pd.to_numeric(gdf["underlying_price"], errors="coerce")**2)
    gex = (gdf.groupby("strike", as_index=False)["gex_raw"].sum().sort_values("strike"))
    if gex.empty:
        st.info("Geen (liquide) data voor een GEX-profiel.")
    else:
        fig_gex = go.Figure(go.Bar(x=gex["strike"], y=gex["gex_raw"], name="Gamma exposure (proxy)"))
        fig_gex.add_vline(x=underlying_now, line=dict(dash="dot"), annotation_text="Spot")
        fig_gex.update_layout(height=420, title=f"GEX (proxy) — {pd.to_datetime(default_snapshot).strftime('%Y-%m-%d %H:%M')}",
                              xaxis_title="Strike", yaxis_title="Exposure (units, relatieve schaal)", dragmode="zoom")
        st.plotly_chart(fig_gex, use_container_width=True, config=PLOTLY_CONFIG)
st.markdown("---")

# ─────────────────────────── A) Serie-selectie ────────────────────
st.subheader("Serie-selectie — volg één optiereeks door de tijd")
# (de rest blijft ongewijzigd)
strikes_all = sorted([float(x) for x in df["strike"].dropna().unique().tolist()])

today = pd.Timestamp(date.today())
default_snapshot = (max([s for s in snapshots_all if pd.to_datetime(s).date()==today.date()])
                    if any(pd.to_datetime(s).date()==today.date() for s in snapshots_all)
                    else (snapshots_all[-1] if snapshots_all else None))

if default_snapshot is not None:
    sub_u = df[df["snap_min"] == default_snapshot]["underlying_price"].dropna()
    underlying_now = float(sub_u.mean()) if not sub_u.empty else float(df["underlying_price"].dropna().iloc[-1])
else:
    underlying_now = float(df["underlying_price"].dropna().iloc[-1])

def choose_best_strike(df_all: pd.DataFrame, typ: str, underlying: float) -> float:
    if np.isnan(underlying) or df_all.empty: return 6000.0
    target = underlying - 300.0 if typ == "put" else underlying + 200.0
    w = 200.0
    cand = df_all[(df_all["strike"] >= target - w) & (df_all["strike"] <= target + w)].copy()
    if cand.empty:
        return pick_closest_value(strikes_all, target, fallback=6000.0)
    grp = (cand.groupby("strike", as_index=False)
               .agg(volume=("volume","sum"), oi=("open_interest","sum")))
    for c in ["volume","oi"]:
        v = grp[c].astype(float)
        grp[c+"_n"] = (v - v.min()) / (v.max() - v.min()) if v.max() > v.min() else 0.0
    grp["dist_n"] = np.abs(grp["strike"] - target)
    grp["dist_n"] = (grp["dist_n"] - grp["dist_n"].min()) / (grp["dist_n"].max() - grp["dist_n"].min()) if grp["dist_n"].max() > grp["dist_n"].min() else 0.0
    grp["score"] = 1.0*grp["volume_n"] + 0.6*grp["oi_n"] - 0.2*grp["dist_n"]
    return float(grp.sort_values("score", ascending=False)["strike"].iloc[0])

default_series_strike = choose_best_strike(df, sel_type, underlying_now)
exps_all = exps
target_exp = date.today() + timedelta(days=14)
default_series_exp = pick_first_on_or_after(exps_all, target_exp) or (pick_closest_date(exps_all, target_exp) if exps_all else None)

cS1, cS2, cS3, cS4 = st.columns([1, 1, 1, 1.6])
with cS1:
    series_strike = st.selectbox("Serie Strike", options=strikes_all or [6000.0],
                                 index=(strikes_all.index(default_series_strike) if default_series_strike in strikes_all else 0))
with cS2:
    series_exp = st.selectbox("Serie Expiratie", options=exps_all or [date.today()],
                              index=(exps_all.index(default_series_exp) if (exps_all and default_series_exp in exps_all) else 0))
with cS3:
    series_price_col = st.radio("Prijsbron", ["last_price","mid_price"], index=0, horizontal=True)
with cS4:
    st.caption(f"🔧 Defaults: {'PUT −300' if sel_type=='put' else 'CALL +200'} rond onderliggende ~{underlying_now:.0f} • Exp ~{target_exp}")

serie = df[(df["strike"]==series_strike) & (df["expiration"]==series_exp) & liq_mask].copy().sort_values("snapshot_date")
if serie.empty:
    st.info("Geen (genoeg) liquiditeit voor deze combinatie binnen de huidige filters.")
else:
    a1, a2 = st.columns(2)
    with a1:
        fig_price = make_subplots(specs=[[{"secondary_y": True}]])
        fig_price.add_trace(go.Scatter(x=serie["snapshot_date"],
                                       y=serie[series_price_col],
                                       name="Price", mode="lines+markers", connectgaps=True), secondary_y=False)
        if show_underlying:
            fig_price.add_trace(go.Scatter(x=serie["snapshot_date"], y=serie["underlying_price"],
                                           name="SP500", mode="lines", line=dict(dash="dot"), connectgaps=True), secondary_y=True)
        fig_price.update_layout(title=f"{sel_type.UPPER()} {series_strike} — exp {series_exp} | Price vs SP500",
                                height=420, hovermode="x unified", dragmode="zoom")
        fig_price.update_xaxes(title_text="Meetmoment")
        fig_price.update_yaxes(title_text="Price", secondary_y=False, rangemode="tozero")
        fig_price.update_yaxes(title_text="SP500", secondary_y=True)
        st.plotly_chart(fig_price, use_container_width=True, config=PLOTLY_CONFIG)
        st.caption("**Interpretatie:** stabiel verloop met smalle bid-ask en voldoende volume/oi wijst op een ‘werkbare’ serie.")
    with a2:
        def ppd_series(df_like: pd.DataFrame, unit_bp: bool=False) -> pd.Series:
            s = pd.to_numeric(df_like["ppd"], errors="coerce")
            if unit_bp:
                u = pd.to_numeric(df_like["underlying_price"], errors="coerce")
                s = 10000.0 * s / u
            return s.replace(0.0, np.nan)
        fig_ppd = make_subplots(specs=[[{"secondary_y": True}]])
        fig_ppd.add_trace(go.Scatter(x=serie["snapshot_date"],
                                     y=ppd_series(serie, unit_bp=False),
                                     name="PPD", mode="lines+markers", connectgaps=True), secondary_y=False)
        if show_underlying:
            fig_ppd.add_trace(go.Scatter(x=serie["snapshot_date"], y=serie["underlying_price"],
                                         name="SP500", mode="lines", line=dict(dash="dot"), connectgaps=True), secondary_y=True)
        fig_ppd.update_layout(title=f"{sel_type.UPPER()} {series_strike} — exp {series_exp} | PPD vs SP500",
                              height=420, hovermode="x unified", dragmode="zoom")
        fig_ppd.update_xaxes(title_text="Meetmoment")
        fig_ppd.update_yaxes(title_text="PPD (points/day)", secondary_y=False, rangemode="tozero")
        fig_ppd.update_yaxes(title_text="SP500", secondary_y=True)
        st.plotly_chart(fig_ppd, use_container_width=True, config=PLOTLY_CONFIG)
        st.caption("**Gebruik:** stijgende PPD bij gelijkblijvende DTE → oplopende IV of vraag naar bescherming; daling → omgekeerd.")

# ─────────────────────────── B) PPD vs Afstand ────────────────────
st.subheader("PPD & Afstand tot Uitoefenprijs (ATM→OTM/ITM)")
default_idx = snapshots_all.index(default_snapshot) if default_snapshot in snapshots_all else len(snapshots_all)-1
sel_snapshot = st.selectbox("Peildatum (snapshot)", options=snapshots_all, index=max(default_idx,0),
                            format_func=lambda x: pd.to_datetime(x).strftime("%Y-%m-%d %H:%M"))

df_last = df[(df["snap_min"] == sel_snapshot) & liq_mask].copy()
if df_last.empty:
    st.info("Geen data op dit snapshot (na liquiditeit-filter).")
else:
    df_last["abs_dist_pct"] = ((df_last["dist_points"].abs() / df_last["underlying_price"]) * 100.0)
    df_last = df_last.assign(ppd_u=pd.to_numeric(df_last["ppd"], errors="coerce"))
    bins = np.arange(0, 18.5, 0.5)
    df_last["dist_bin"] = pd.cut(df_last["abs_dist_pct"], bins=bins, include_lowest=True)
    g = (df_last.groupby("dist_bin").agg(ppd=("ppd_u","median"), n=("ppd_u","count")).reset_index())
    g = g[g["n"] >= min_per_bin].copy()
    g["bin_mid"] = g["dist_bin"].apply(lambda iv: iv.mid if pd.notna(iv) else np.nan)
    g = g.dropna(subset=["bin_mid"]).sort_values("bin_mid")
    g["ppd_s"] = smooth_series(g["ppd"], window=3)
    best_idx = g["ppd_s"].idxmax() if not g.empty else None

    fig_ppd_dist = go.Figure()
    fig_ppd_dist.add_vrect(x0=-0.5, x1=0.5, fillcolor="lightgrey", opacity=0.25, line_width=0,
                           annotation_text="ATM-zone", annotation_position="top left")
    if not np.isnan(underlying_now) and not np.isnan(series_strike):
        cur_dist = abs(float(series_strike) - underlying_now) / underlying_now * 100.0
        fig_ppd_dist.add_vline(x=cur_dist, line=dict(dash="dot"),
                               annotation_text=f"gekozen strike ≈ {cur_dist:.2f}%", annotation_position="top right")
    fig_ppd_dist.add_trace(go.Scatter(x=g["bin_mid"], y=g["ppd"],   mode="markers", name="PPD (median/bin)", opacity=0.85))
    fig_ppd_dist.add_trace(go.Scatter(x=g["bin_mid"], y=g["ppd_s"], mode="lines",   name="Smoothed"))
    if best_idx is not None and pd.notna(g.loc[best_idx,"ppd_s"]):
        x_b, y_b = float(g.loc[best_idx,"bin_mid"]), float(g.loc[best_idx,"ppd_s"])
        fig_ppd_dist.add_annotation(x=x_b, y=y_b, text=f"sweet spot ≈ {y_b:.2f} @ {x_b:.2f}%", showarrow=True, arrowhead=2)
    fig_ppd_dist.update_layout(title=f"PPD vs Afstand — {pd.to_datetime(sel_snapshot).strftime('%Y-%m-%d %H:%M')}",
                               xaxis_title="Afstand |K−S|/S (%)", yaxis_title="PPD (points/day)",
                               height=420, dragmode="zoom")
    st.plotly_chart(fig_ppd_dist, use_container_width=True, config=PLOTLY_CONFIG)

    if best_idx is not None and pd.notna(g.loc[best_idx,"ppd_s"]):
        st.info(f"**Advies (afstand):** over dit snapshot ligt de **sweet spot** rond **{g.loc[best_idx,'bin_mid']:.1f}%** van ATM "
                f"met **PPD ≈ {g.loc[best_idx,'ppd_s']:.2f}**. Combineer met DTE-inzicht hieronder.")

# ─────────────────────────── C) Price/PPD vs Exp Date ─────────────
st.subheader("Ontwikkeling Prijs per Expiratiedatum (laatste snapshot)")
df_last_strike = df_last[df_last["strike"] == series_strike].copy()
if df_last_strike.empty:
    st.info("Geen data voor deze strike op dit snapshot (na liquiditeit-filter).")
else:
    exp_curve = (df_last_strike.assign(ppd_u=pd.to_numeric(df_last_strike["ppd"], errors="coerce"))
                 .groupby("expiration", as_index=False)
                 .agg(price=(series_price_col, "median"), ppd=("ppd_u", "median"))
                 .sort_values("expiration"))
    fig_exp = make_subplots(specs=[[{"secondary_y": True}]])
    fig_exp.add_trace(go.Scatter(x=exp_curve["expiration"], y=exp_curve["price"],
                                 name="Price", mode="lines+markers"), secondary_y=False)
    fig_exp.add_trace(go.Scatter(x=exp_curve["expiration"], y=exp_curve["ppd"],
                                 name="PPD", mode="lines+markers"), secondary_y=True)
    fig_exp.update_layout(title=f"{sel_type.UPPER()} — Strike {series_strike} — peildatum {pd.to_datetime(sel_snapshot).strftime('%Y-%m-%d %H:%M')}",
                          height=420, hovermode="x unified", dragmode="zoom")
    fig_exp.update_xaxes(title_text="Expiratiedatum")
    fig_exp.update_yaxes(title_text="Price", secondary_y=False, rangemode="tozero")
    fig_exp.update_yaxes(title_text="PPD (points/day)", secondary_y=True)
    st.plotly_chart(fig_exp, use_container_width=True, config=PLOTLY_CONFIG)
    st.caption("**Interpretatie:** hogere PPD bij langere DTE = meer time value/dag; spikes per datum zijn vaak events (CPI/Fed) of smile-effecten.")

# ─────────────────────────── D) PPD vs DTE ────────────────────────
st.subheader("PPD vs DTE — opbouw van premium per dag")
m1, m2, m3, m4, m5 = st.columns([1.2, 1, 1, 1, 1])
with m1:
    ppd_mode = st.radio("Bereik", ["ATM-band (moneyness)", "Rond gekozen strike"], index=0)
with m2:
    atm_band = st.slider("ATM-band (± moneyness %)", 0.01, 0.10, 0.02, step=0.01)
with m3:
    strike_window = st.slider("Strike-venster rond gekozen strike (punten)", 10, 200, 50, step=10)
with m4:
    use_last_snap = st.checkbox("Alleen laatste snapshot", value=True)
with m5:
    robust_scale = st.checkbox("Robust scale (95e pct)", value=True)

base_df = df_last if use_last_snap else df
base_df = base_df[liq_mask]
if ppd_mode.startswith("ATM"):
    df_ppd = base_df[np.abs(base_df["moneyness"]) <= atm_band].copy()
else:
    df_ppd = base_df[(base_df["strike"] >= series_strike - strike_window) & (base_df["strike"] <= series_strike + strike_window)].copy()

if df_ppd.empty:
    st.info("Geen data voor PPD vs DTE met deze instellingen.")
else:
    df_ppd = df_ppd.assign(ppd_u=pd.to_numeric(df_ppd["ppd"], errors="coerce"))
    ppd_curve = (df_ppd.groupby("days_to_exp", as_index=False)
                        .agg(ppd=("ppd_u","median"), n=("ppd_u","count"))
                        .query("n >= @min_per_bin")
                        .sort_values("days_to_exp"))
    ppd_curve["ppd_s"] = smooth_series(ppd_curve["ppd"], window=3)
    y_range = None
    if robust_scale and ppd_curve["ppd"].notna().any():
        hi = float(np.nanpercentile(ppd_curve["ppd"], 95))
        lo = float(np.nanpercentile(ppd_curve["ppd"], 5))
        pad = (hi - lo) * 0.10
        y_range = [max(lo - pad, 0.0), hi + pad]
    fig_ppd_dte = go.Figure()
    fig_ppd_dte.add_trace(go.Scatter(x=ppd_curve["days_to_exp"], y=ppd_curve["ppd"], mode="markers", name="PPD (median)", opacity=0.85))
    fig_ppd_dte.add_trace(go.Scatter(x=ppd_curve["days_to_exp"], y=ppd_curve["ppd_s"], mode="lines", name="Smoothed"))
    fig_ppd_dte.update_layout(title="PPD vs Days To Expiration", xaxis_title="Days to Expiration",
                              yaxis_title="PPD (points/day)", height=420, dragmode="zoom")
    if y_range: fig_ppd_dte.update_yaxes(range=y_range)
    st.plotly_chart(fig_ppd_dte, use_container_width=True, config=PLOTLY_CONFIG)

    sweet_row = ppd_curve.loc[ppd_curve["ppd_s"].idxmax()] if not ppd_curve.empty else None
    if sweet_row is not None and pd.notna(sweet_row["ppd_s"]):
        st.info(f"**Advies (DTE):** de PPD-sweet spot ligt rond **{int(sweet_row['days_to_exp'])} dagen** "
                f"met **PPD ≈ {sweet_row['ppd_s']:.2f}**. Combineer dit met de afstand-sweet spot hierboven.")

st.markdown("---")

# ─────────────────────────── E) Matrix — meetmoment × strike ──────
st.subheader("Matrix — meetmoment × strike")
cM1, cM2, cM3 = st.columns([1, 1, 1])
with cM1:
    matrix_exp = st.selectbox("Expiratie (matrix)", options=sorted(df["expiration"].unique().tolist()), index=0, key="mx_exp")
with cM2:
    matrix_metric = st.radio("Waarde", ["last_price", "mid_price", "ppd"], horizontal=True, index=0, key="mx_metric")
with cM3:
    max_rows = st.slider("Max. meetmomenten (recentste)", 50, 500, 200, step=50, key="mx_rows")

mx = df[(df["expiration"]==matrix_exp) & liq_mask].copy().sort_values("snapshot_date").tail(max_rows)
if mx.empty:
    st.info("Geen matrix-data voor de gekozen expiratie.")
else:
    if matrix_metric == "ppd":
        mx = mx.assign(ppd_u=pd.to_numeric(mx["ppd"], errors="coerce")); value_col = "ppd_u"
    else:
        value_col = matrix_metric
    mx["snap_s"] = mx["snapshot_date"].dt.strftime("%Y-%m-%d %H:%M")
    pivot = mx.pivot_table(index="snap_s", columns="strike", values=value_col, aggfunc="median").sort_index(ascending=False).round(3)
    arr = pivot.values.astype(float)
    tab_hm, tab_tbl = st.tabs(["Heatmap", "Tabel"])
    with tab_hm:
        fig_mx = go.Figure(data=go.Heatmap(z=arr, x=pivot.columns.astype(float), y=pivot.index.tolist(),
                                           colorbar_title=value_col))
        fig_mx.update_layout(title=f"Heatmap — {sel_type.UPPER()} exp {matrix_exp} — {value_col}",
                             xaxis_title="Strike", yaxis_title="Meetmoment", height=520, dragmode="zoom")
        st.plotly_chart(fig_mx, use_container_width=True, config=PLOTLY_CONFIG)
    with tab_tbl:
        st.dataframe(pivot, use_container_width=True)

st.markdown("---")

# ─────────────────────────── F) Term structure & Smile ────────────
term = df[liq_mask].groupby("days_to_exp", as_index=False)["implied_volatility"].median().sort_values("days_to_exp")
fig_term = go.Figure(go.Scatter(x=term["days_to_exp"], y=term["implied_volatility"], mode="lines+markers", name=f"IV {sel_type.UPPER()}"))
fig_term.update_layout(title="Term Structure — mediane IV", xaxis_title="DTE", yaxis_title="Implied Volatility", height=380, dragmode="zoom")
st.plotly_chart(fig_term, use_container_width=True, config=PLOTLY_CONFIG)
st.caption("**Interpretatie:** stijgende term structure → hogere vol voor langere looptijd; dalend → omgekeerd.")

st.subheader("IV Smile (laatste snapshot)")
exp_for_smile = st.selectbox("Expiratie voor IV Smile", options=exps or [None], index=0)
sm = df_last[(df_last["expiration"] == exp_for_smile) & liq_mask].copy()
if sm.empty:
    st.info("Geen (liquide) data voor IV Smile.")
else:
    sm = sm.groupby("strike", as_index=False)["implied_volatility"].median().sort_values("strike")
    if len(sm) >= 5:
        lo, hi = sm["implied_volatility"].quantile([0.02, 0.98])
        sm["implied_volatility"] = sm["implied_volatility"].clip(lower=lo, upper=hi)
    fig_sm = go.Figure(go.Scatter(x=sm["strike"], y=sm["implied_volatility"], mode="lines+markers", name="IV"))
    fig_sm.update_layout(title=f"IV Smile — {sel_type.UPPER()} exp {exp_for_smile}",
                         xaxis_title="Strike", yaxis_title="Implied Volatility", height=420, dragmode="zoom")
    st.plotly_chart(fig_sm, use_container_width=True, config=PLOTLY_CONFIG)
    st.caption("**Gebruik:** kies strangles in een **glad** stuk van de smile (weinig spikes) → betere liquiditeit & eerlijkere mid-prices.")

# ─────────────────────────── G) Put/Call-ratio per exp ────────────
st.subheader("Put/Call-ratio per expiratie")
p = (df[liq_mask].groupby(["expiration","type"], as_index=False)
       .agg(vol=("volume","sum"), oi=("open_interest","sum")))
if p.empty:
    st.info("Geen data voor PCR.")
else:
    pv2 = (p.pivot_table(index="expiration", columns="type", values=["vol","oi"], aggfunc="sum")
             .sort_index().sort_index(axis=1)).fillna(0.0)
    pv2.columns = [f"{a}_{b}" for a,b in pv2.columns.to_flat_index()]
    for col in ["vol_put","vol_call","oi_put","oi_call"]:
        if col not in pv2.columns: pv2[col] = 0.0
    pv2["PCR_vol"] = pv2["vol_put"] / pv2["vol_call"].replace(0, np.nan)
    pv2["PCR_oi"]  = pv2["oi_put"]  / pv2["oi_call"].replace(0, np.nan)
    pv2 = pv2.replace([np.inf,-np.inf], np.nan).dropna(subset=["PCR_vol","PCR_oi"], how="all")
    if pv2.empty:
        st.info("Niet genoeg data (alleen puts of alleen calls in de selectie). Probeer andere filters/expiraties.")
    else:
        fig_pcr = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                                subplot_titles=("PCR op Volume", "PCR op Open Interest"))
        fig_pcr.add_trace(go.Bar(x=pv2.index, y=pv2["PCR_vol"], name="PCR (Vol)"), row=1, col=1)
        fig_pcr.add_trace(go.Bar(x=pv2.index, y[pv2["PCR_oi"]],  name="PCR (OI)"),  row=2, col=1)  # Avoids accidental shadowing
        fig_pcr.update_layout(height=520, title_text="Put/Call-ratio per Expiratie", dragmode="zoom")
        st.plotly_chart(fig_pcr, use_container_width=True, config=PLOTLY_CONFIG)
        st.caption("**Interpretatie:** PCR↑ → defensiever; PCR↓ → meer call-speculatie. Gebruik dit om strangle-afstand/DTE conservatiever of agressiever te kiezen.")

# ─────────────────────────── H) Vol & Risk ────────────────────────
st.markdown("### 📊 Vol & Risk (ATM-IV, HV, VRP, IV-Rank, Expected-Move)")
u_daily = (df.assign(dte=df["snapshot_date"].dt.date).sort_values(["dte","snapshot_date"])
             .groupby("dte", as_index=False).agg(close=("underlying_price","last")))
u_daily["ret"] = u_daily["close"].pct_change()
hv20 = annualize_std(u_daily["ret"].tail(21).dropna())
near_atm = df_last[(df_last["days_to_exp"].between(20, 40)) & (df_last["moneyness"].abs() <= 0.01)]
iv_atm = float(near_atm["implied_volatility"].median()) if not near_atm.empty else float(df_last["implied_volatility"].median())
iv_hist = (df.assign(day=df["snapshot_date"].dt.date)
             .query("days_to_exp>=20 and days_to_exp<=40 and abs(moneyness)<=0.01")
             .groupby("day", as_index=False)["implied_volatility"].median()
             .rename(columns={"implied_volatility":"iv"}))
iv_1y = iv_hist.tail(252)["iv"] if not iv_hist.empty else pd.Series(dtype=float)
iv_rank = float((iv_1y <= iv_1y.iloc[-1]).mean()) if not iv_1y.empty else np.nan
dte_selected = int(pd.to_numeric(df_last[df_last["expiration"]==pick_first_on_or_after(exps, date.today()+timedelta(days=30))]["days_to_exp"], errors="coerce").median()) if not df_last.empty else 30
em_sigma = (underlying_now * iv_atm * math.sqrt(max(dte_selected,1)/365.0)) if (not np.isnan(underlying_now) and not np.isnan(iv_atm)) else np.nan
cv1, cv2, cv3, cv4, cv5 = st.columns(5)
with cv1: st.metric("ATM-IV (~30D)", f"{iv_atm:.2%}" if not np.isnan(iv_atm) else "—")
with cv2: st.metric("HV20", f"{hv20:.2%}" if not np.isnan(hv20) else "—")
with cv3: st.metric("VRP (IV−HV)", f"{(iv_atm-hv20):.2%}" if (not np.isnan(iv_atm) and not np.isnan(hv20)) else "—")
with cv4: st.metric("IV-Rank (1y)", f"{iv_rank*100:.0f}%" if not np.isnan(iv_rank) else "—")
with cv5:
    em_txt = f"±{em_sigma:,.0f} pts ({em_sigma/underlying_now:.2%})" if (not np.isnan(em_sigma) and not np.isnan(underlying_now)) else "—"
    st.metric("Expected Move (σ)", em_txt)
st.caption("**VRP**>0: IV boven gerealiseerde → gunstiger voor **short vol**. **IV-Rank** hoog → premie dikker (events checken).")

# ─────────────────────────── I) Strangle Helper ───────────────────
st.markdown("### 🧠 Strangle Helper (σ- of Δ-doel / quick pick)")
# (… ongewijzigd; jouw bestaande Strangle Helper, Roll, Payoff, etc. code hieronder …)
# (Vanwege de lengte is de rest van de pagina identiek aan je huidige versie; alleen de nieuwe sectie is ingevoegd.)
# Je kunt desgewenst de volledige helper/roll-secties uit je huidige bestand laten staan (ze zijn compatible).
