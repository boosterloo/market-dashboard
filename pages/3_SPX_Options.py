# pages/3_SPX_Options.py — v2 (blok 1/5)

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import math
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from google.cloud import bigquery
from google.oauth2 import service_account

# r/q helpers
from utils.rates import get_r_curve_for_snapshot, get_q_curve_const

# ────────────────────────────── Basis helpers ──────────────────────────────
def norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

def norm_pdf(z: float) -> float:
    return (1.0 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * z * z)

def pick_closest_date(options, target):
    if not options: return None
    return min(options, key=lambda d: abs(pd.Timestamp(d) - pd.Timestamp(target)))

def pick_first_on_or_after(options, target):
    after = [d for d in options if d >= target]
    return after[0] if after else (options[-1] if options else None)

def pick_closest_value(options, target, fallback=None):
    if not options: return fallback
    return float(min(options, key=lambda x: abs(float(x) - float(target))))

def smooth_series(y: pd.Series, window: int = 3) -> pd.Series:
    if len(y) < 3: return y
    return y.rolling(window, center=True, min_periods=1).median()

def annualize_std(s: pd.Series, periods_per_year: int = 252) -> float:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) < 2: return np.nan
    return float(s.std(ddof=0) * math.sqrt(periods_per_year))

# Black–Scholes
def bs_delta(S, K, iv, T, r, q, is_call):
    if any(x is None or np.isnan(x) or x <= 0 for x in [S, K]) or np.isnan(iv) or T <= 0:
        return np.nan
    d1 = (math.log(S / K) + (r - q + 0.5 * iv ** 2) * T) / (iv * math.sqrt(T))
    Nd1 = norm_cdf(d1)
    disc_q = math.exp(-q * T)
    return disc_q * Nd1 if is_call else disc_q * (Nd1 - 1)

def bs_gamma(S, K, iv, T, r, q):
    if any(x is None or np.isnan(x) or x <= 0 for x in [S, K]) or np.isnan(iv) or iv <= 0 or T <= 0:
        return np.nan
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * iv ** 2) * T) / (iv * sqrtT)
    disc_q = math.exp(-q * T)
    return disc_q * norm_pdf(d1) / (S * iv * sqrtT)

def strangle_payoff_at_expiry(S, Kp, Kc, credit_pts, multiplier=100):
    return (credit_pts - max(Kp - S, 0.0) - max(S - Kc, 0.0)) * multiplier

# ────────────────────────────── BigQuery setup ──────────────────────────────
def get_bq_client():
    sa_info = st.secrets["gcp_service_account"]
    creds = service_account.Credentials.from_service_account_info(sa_info)
    project_id = st.secrets.get("PROJECT_ID", sa_info.get("project_id"))
    return bigquery.Client(project=project_id, credentials=creds)

_bq_client = get_bq_client()

def _bq_param(name, value):
    if isinstance(value, (list, tuple)):
        if not value: return bigquery.ArrayQueryParameter(name, "STRING", [])
        e = value[0]
        if isinstance(e, int): return bigquery.ArrayQueryParameter(name, "INT64", list(value))
        if isinstance(e, float): return bigquery.ArrayQueryParameter(name, "FLOAT64", list(value))
        if isinstance(e, (date, pd.Timestamp, datetime)):
            vals = [str(pd.to_datetime(v).date()) for v in value]
            return bigquery.ArrayQueryParameter(name, "DATE", vals)
        return bigquery.ArrayQueryParameter(name, "STRING", [str(v) for v in value])
    if isinstance(value, bool): return bigquery.ScalarQueryParameter(name, "BOOL", value)
    if isinstance(value, (int, np.integer)): return bigquery.ScalarQueryParameter(name, "INT64", int(value))
    if isinstance(value, (float, np.floating)): return bigquery.ScalarQueryParameter(name, "FLOAT64", float(value))
    if isinstance(value, datetime): return bigquery.ScalarQueryParameter(name, "TIMESTAMP", value)
    if isinstance(value, (date, pd.Timestamp)): return bigquery.ScalarQueryParameter(name, "DATE", str(pd.to_datetime(value).date()))
    return bigquery.ScalarQueryParameter(name, "STRING", str(value))

def run_query(sql: str, params: dict | None = None) -> pd.DataFrame:
    job_config = None
    if params:
        job_config = bigquery.QueryJobConfig(query_parameters=[_bq_param(k, v) for k, v in params.items()])
    return _bq_client.query(sql, job_config=job_config).to_dataframe()

# ────────────────────────────── Page config ──────────────────────────────
st.set_page_config(page_title="SPX Options Dashboard", layout="wide")
st.title("🧩 SPX Options Dashboard")

VIEW = "marketdata.spx_options_enriched_v"
PLOTLY_CONFIG = {
    "scrollZoom": True,
    "doubleClick": "reset",
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
}

# UI: schaal voor assen (kleiner & afstelbaar)
axis_scale = st.slider("As-lettergrootte (x)", 1.2, 2.2, 1.6, 0.1,
                       help="Schaal voor tick- en titel-tekst op alle grafieken.")
st.session_state["_axis_mult"] = axis_scale

# ─────────────────────────── Global plot style helpers ───────────────────────────
_BASE = 12
_DEF_MARGIN = dict(l=96, r=48, t=64, b=56)

def _axis_size():
    mult = st.session_state.get("_axis_mult", 1.6)
    return int(_BASE * mult)

def amplify_axes(fig, height: int | None = None, legend_top: bool = True):
    SZ = _axis_size()
    # subtiele grijze grid + zeroline
    fig.update_xaxes(tickfont=dict(size=SZ),
                     title_font=dict(size=SZ),
                     automargin=True,
                     showgrid=True, gridwidth=1, gridcolor="rgba(0,0,0,0.10)",
                     ticks="outside", tickformat="%b %d\n%Y")
    fig.update_yaxes(tickfont=dict(size=SZ),
                     title_font=dict(size=SZ),
                     automargin=True,
                     zeroline=True, zerolinewidth=1, zerolinecolor="rgba(0,0,0,0.25)",
                     showgrid=True, gridwidth=1, gridcolor="rgba(0,0,0,0.10)")
    fig.update_layout(
        margin=_DEF_MARGIN,
        font=dict(size=max(SZ - 6, 13)),
        legend=dict(font=dict(size=max(SZ - 8, 12)),
                    orientation="h",
                    yanchor="bottom", y=1.02 if legend_top else -0.2,
                    xanchor="left",   x=0.0),
        title=dict(font=dict(size=SZ)),
        hoverlabel=dict(font_size=max(SZ - 8, 12)),
        dragmode="zoom", autosize=True,
        plot_bgcolor="white"
    )
    if height: fig.update_layout(height=height)
    return fig

def apply_style_and_show(fig, height: int | None = None, legend_top: bool = True):
    amplify_axes(fig, height=height, legend_top=legend_top)
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG, theme=None)

def annotate_pcr_midline(fig, row:int, col:int):
    """Stippellijn op y=1 met subtiele 'Defensief/Offensief'-labels."""
    fig.add_hline(y=1.0, line=dict(dash="dot", width=1), row=row, col=col)
    SZ = max(_axis_size() - 4, 12)
    # gebruik paper-x domain voor compacte labels
    fig.add_annotation(xref="paper", yref="paper", x=0.01, y=0.92 if row==1 else 0.42,
                       text="Defensief (>1)", showarrow=False,
                       font=dict(size=int(SZ*0.8)))
    fig.add_annotation(xref="paper", yref="paper", x=0.01, y=0.80 if row==1 else 0.30,
                       text="Offensief (<1)", showarrow=False,
                       font=dict(size=int(SZ*0.8)))
# ─────────────────────────── Workflow (kort) ───────────────────────────
with st.expander("📌 Workflow (kort): van data → strangle-keuze", expanded=False):
    st.markdown(
        """
1) Kies **periode, type, DTE & moneyness**.  
2) Bekijk **Serie-selectie** om gevoel te krijgen voor prijs/PPD en liquiditeit.  
3) **PPD vs Afstand** (in punten of %) & **PPD vs DTE** voor sweet spots.  
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
    start_date, end_date = st.date_input(
        "Periode (snapshot_date)",
        value=(default_start, max_date),
        min_value=min_date, max_value=max_date,
        format="YYYY-MM-DD"
    )
with colB:
    sel_type = st.radio("Type (voor secties A–F, H–L)", ["call", "put"], index=1, horizontal=True)
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

# ─────────────────────────── Data-loaders ──────────────────────────────
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
        df["abs_dist_pts"]  = np.abs(df["dist_points"])
        df["snap_min"] = df["snapshot_date"].dt.floor("min")
    return df

@st.cache_data(ttl=600, show_spinner=False)
def load_filtered_bothtypes(start_date, end_date, dte_min, dte_max, mny_min, mny_max):
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
        AND days_to_exp BETWEEN @dte_min AND @dte_max
        AND SAFE_DIVIDE(CAST(strike AS FLOAT64), NULLIF(underlying_price, 0)) - 1.0
            BETWEEN @mny_min AND @mny_max
    )
    SELECT * FROM base
    """
    params = {"start": start_date, "end": end_date,
              "dte_min": int(dte_min), "dte_max": int(dte_max),
              "mny_min": float(mny_min), "mny_max": float(mny_max)}
    df2 = run_query(sql, params=params)
    if not df2.empty:
        df2["snapshot_date"] = pd.to_datetime(df2["snapshot_date"])
        df2["expiration"]    = pd.to_datetime(df2["expiration"]).dt.date
        num_cols = ["days_to_exp","implied_volatility","open_interest","volume","ppd",
                    "strike","underlying_price","last_price","mid_price","bid","ask","dist_points"]
        for c in num_cols:
            if c in df2: df2[c] = pd.to_numeric(df2[c], errors="coerce")
        df2["moneyness_pct"] = 100 * df2["moneyness"]
        df2["abs_dist_pct"]  = (np.abs(df2["dist_points"]) / df2["underlying_price"]) * 100.0
        df2["abs_dist_pts"]  = np.abs(df2["dist_points"])
        df2["snap_min"] = df2["snapshot_date"].dt.floor("min")
    return df2

df = load_filtered(start_date, end_date, sel_type, dte_range[0], dte_range[1], mny_range[0], mny_range[1])
if df.empty:
    st.warning("Geen data voor de huidige filters.")
    st.stop()

df_both = load_filtered_bothtypes(start_date, end_date, dte_range[0], dte_range[1], mny_range[0], mny_range[1])

# Liquidity masks
liq_mask = ((df["open_interest"].fillna(0) >= min_oi) | (df["volume"].fillna(0) >= min_vol))
liq_mask_both = ((df_both["open_interest"].fillna(0) >= min_oi) | (df_both["volume"].fillna(0) >= min_vol))

# KPIs
c1, c2, c3, c4 = st.columns(4)
with c1: st.metric("Records", f"{len(df):,}")
with c2: st.metric("Gem. IV", f"{df['implied_volatility'].mean():.2%}")
with c3: st.metric("Som Volume", f"{int(df['volume'].sum()):,}")
with c4: st.metric("Som OI", f"{int(df['open_interest'].sum()):,}")
st.markdown("---")

# ─────────────────────────── Outliers & PPD-unit ──────────────────
st.caption("Outliers kunnen de schaal verstoren. Kies een methode.")
co1, co2, co3 = st.columns([1.1, 1, 1])
with co1:
    outlier_mode = st.radio("Outlier", ["Geen", "Percentiel clip", "IQR filter", "Z-score filter"], horizontal=True, index=1)
with co2:
    pct_clip = st.slider("Percentiel clip (links/rechts)", 0, 10, 5, step=1, disabled=(outlier_mode != "Percentiel clip"))
with co3:
    z_thr = st.slider("Z-score drempel", 2.0, 5.0, 3.0, step=0.1, disabled=(outlier_mode != "Z-score filter"))

def apply_outlier(series: pd.Series, mode: str, pct: int, zthr: float = 3.0) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").astype(float)
    if mode == "Geen": return s
    if mode == "Percentiel clip":
        if s.notna().any():
            lo, hi = np.nanpercentile(s, [pct, 100 - pct]); return s.clip(lower=lo, upper=hi)
        return s
    if mode == "IQR filter":
        if s.notna().any():
            q1, q3 = np.nanpercentile(s, [25, 75]); iqr = q3 - q1
            lo, hi = q1 - 1.5*iqr, q3 + 1.5*iqr
            return s.where((s >= lo) & (s <= hi), np.nan)
        return s
    if mode == "Z-score filter":
        mu, sd = np.nanmean(s), np.nanstd(s)
        if sd == 0 or np.isnan(sd): return s
        z = (s - mu) / sd
        return s.where(np.abs(z) <= zthr, np.nan)
    return s

ppd_unit = st.radio("PPD-eenheid", ["Points per day", "bp/day (vs onderliggende)"], index=0, horizontal=True)

def ppd_series(df_like: pd.DataFrame) -> pd.Series:
    s = pd.to_numeric(df_like["ppd"], errors="coerce")
    if ppd_unit.startswith("bp"):
        u = pd.to_numeric(df_like["underlying_price"], errors="coerce")
        s = 10000.0 * s / u
    return s.replace(0.0, np.nan)

def ppd_y_label():
    return "PPD (bp/day)" if ppd_unit.startswith("bp") else "PPD (points/day)"

# ─────────────────────────── Defaults (snapshot, strikes, expiries) ──────────────────
snapshots_all = sorted(df["snap_min"].unique())
today = pd.Timestamp(date.today())
default_snapshot = (max([s for s in snapshots_all if pd.to_datetime(s).date()==today.date()])
                    if any(pd.to_datetime(s).date()==today.date() for s in snapshots_all)
                    else (snapshots_all[-1] if snapshots_all else None))

if default_snapshot is not None:
    sub_u = df[df["snap_min"] == default_snapshot]["underlying_price"].dropna()
    underlying_now = float(sub_u.mean()) if not sub_u.empty else float(df["underlying_price"].dropna().iloc[-1])
else:
    underlying_now = float(df["underlying_price"].dropna().iloc[-1])

strikes_all = sorted([float(x) for x in df["strike"].dropna().unique().tolist()])

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
# ─────────────────────────── 🧭 Sentiment & Positioning ───────────────────────────
st.header("🧭 Sentiment & Positioning")

# Instellingen
sent_co1, sent_co2, sent_co3 = st.columns([1.1, 1, 1])
with sent_co1:
    q_const_simple = st.number_input("Dividendrendement q (p.j., simple)", min_value=0.0, max_value=0.10, value=0.016, step=0.001, format="%.3f")
with sent_co2:
    dte_for_skew = st.slider("DTE-range voor 25Δ Skew", 10, 60, (20, 40), step=5)
with sent_co3:
    atm_band_skew = st.slider("ATM-band voor term-slope (±%)", 0.5, 5.0, 1.0, step=0.5)

# ─────────────────────────── PCR tijdreeks (vol/oi) ───────────────────────────
pcr_df = (
    df_both.assign(day=df_both["snapshot_date"].dt.date)
           .groupby(["day","type"], as_index=False)
           .agg(vol=("volume","sum"), oi=("open_interest","sum"))
)
if not pcr_df.empty:
    pv = (
        pcr_df.pivot_table(index="day", columns="type", values=["vol","oi"], aggfunc="sum")
              .sort_index().sort_index(axis=1)
    ).fillna(0.0)
    pv.columns = [f"{a}_{b}" for a,b in pv.columns.to_flat_index()]
    for col in ["vol_put","vol_call","oi_put","oi_call"]:
        if col not in pv.columns: pv[col] = 0.0
    pv["PCR_vol"] = pv["vol_put"] / pv["vol_call"].replace(0, np.nan)
    pv["PCR_oi"]  = pv["oi_put"]  / pv["oi_call"].replace(0, np.nan)
else:
    pv = pd.DataFrame(columns=["PCR_vol","PCR_oi"])

# ─────────────────────────── Delta-gewogen PCR ───────────────────────────
dg_mask = (
    df_both["days_to_exp"].between(7, 45)
    & (df_both["moneyness"].abs() <= 0.20)
    & liq_mask_both
)
dg = df_both[dg_mask].copy()

if not dg.empty:
    T = pd.to_numeric(dg["days_to_exp"], errors="coerce").fillna(0).astype(float) / 365.0
    S = pd.to_numeric(dg["underlying_price"], errors="coerce").astype(float)
    K = pd.to_numeric(dg["strike"], errors="coerce").astype(float)
    IV = pd.to_numeric(dg["implied_volatility"], errors="coerce").astype(float)
    is_call = dg["type"].str.lower().eq("call").astype(bool)
    q_arr = np.array([float(get_q_curve_const(np.array([t], dtype=float),
                                              q_const=q_const_simple,
                                              to_continuous=True)[0]) for t in T], dtype=float)
    deltas = np.vectorize(bs_delta)(S.values, K.values, IV.values, T.values,
                                    np.zeros_like(T.values), q_arr, is_call.values)
    dg["delta_abs"] = np.abs(deltas)
    dg["dw_vol"] = dg["delta_abs"] * pd.to_numeric(dg["volume"], errors="coerce").fillna(0.0)
    dg["dw_oi"]  = dg["delta_abs"] * pd.to_numeric(dg["open_interest"], errors="coerce").fillna(0.0)
    dgt = (
        dg.assign(day=dg["snapshot_date"].dt.date)
          .groupby(["day", "type"], as_index=False)
          .agg(dw_vol=("dw_vol", "sum"), dw_oi=("dw_oi", "sum"))
    )
    dgp = (
        dgt.pivot_table(index="day", columns="type", values=["dw_vol", "dw_oi"], aggfunc="sum")
           .sort_index().sort_index(axis=1)
           .fillna(0.0)
    )
    dgp.columns = [f"{a}_{b}" for a, b in dgp.columns.to_flat_index()]
    for col in ["dw_vol_put","dw_vol_call","dw_oi_put","dw_oi_call"]:
        if col not in dgp.columns: dgp[col] = np.nan
    dgp["PCR_delta_vol"] = dgp["dw_vol_put"] / dgp["dw_vol_call"].replace({0: np.nan})
    dgp["PCR_delta_oi"]  = dgp["dw_oi_put"]  / dgp["dw_oi_call"].replace({0: np.nan})
    dgp = dgp.replace([np.inf, -np.inf], np.nan)[["PCR_delta_vol","PCR_delta_oi"]].dropna(how="all")
else:
    dgp = pd.DataFrame(columns=["PCR_delta_vol","PCR_delta_oi"])

# ─────────────────────────── 25Δ Skew, Term-slope, Expected Move ───────────────────────────
df_last_sent = df_both[(df_both["snap_min"] == default_snapshot) & liq_mask_both].copy() \
    if default_snapshot is not None else pd.DataFrame()

def compute_25d_skew(df_last, dte_lo:int, dte_hi:int) -> float:
    if df_last.empty: return np.nan
    z = df_last[df_last["days_to_exp"].between(dte_lo, dte_hi)].copy()
    if z.empty: return np.nan
    z["T"] = pd.to_numeric(z["days_to_exp"], errors="coerce").fillna(0)/365.0
    z["delta"] = z.apply(lambda r: bs_delta(
        float(r["underlying_price"]), float(r["strike"]),
        float(r["implied_volatility"]), float(r["T"]),
        0.0, float(get_q_curve_const(np.array([r["T"]], dtype=float),
                                     q_const=q_const_simple,
                                     to_continuous=True)[0]),
        is_call=(str(r["type"]).lower()=="call")), axis=1)
    rows = []
    for _, g in z.groupby("expiration"):
        gp = g[g["type"].str.lower()=="put"].copy()
        gc = g[g["type"].str.lower()=="call"].copy()
        if gp.empty or gc.empty: continue
        gp["dist"] = (gp["delta"] + 0.25).abs()
        gc["dist"] = (gc["delta"] - 0.25).abs()
        rowp = gp.loc[gp["dist"].idxmin()] if not gp["dist"].isna().all() else None
        rowc = gc.loc[gc["dist"].idxmin()] if not gc["dist"].isna().all() else None
        if rowp is not None and rowc is not None:
            rows.append(float(rowp["implied_volatility"]) - float(rowc["implied_volatility"]))
    return float(np.nanmedian(rows)) if rows else np.nan

skew_25d = compute_25d_skew(df_last_sent, dte_for_skew[0], dte_for_skew[1])

def iv_term_slope(df_scope, band_pct: float) -> float:
    if df_scope.empty: return np.nan
    a = df_scope[(df_scope["moneyness"].abs() <= band_pct/100.0)]
    short = a[a["days_to_exp"].between(7, 15)]["implied_volatility"].median()
    mid   = a[a["days_to_exp"].between(30, 60)]["implied_volatility"].median()
    if np.isnan(short) or np.isnan(mid): return np.nan
    return float(short - mid)

iv_slope = iv_term_slope(df_last_sent, atm_band_skew)
underlying_now_sent = float(df_last_sent["underlying_price"].median()) if not df_last_sent.empty else np.nan
iv_1w = float(df_last_sent[df_last_sent["days_to_exp"].between(5,10)]["implied_volatility"].median()) if not df_last_sent.empty else np.nan
em_1w_pts = (underlying_now_sent * iv_1w * math.sqrt(7/365.0)) \
             if (not np.isnan(underlying_now_sent) and not np.isnan(iv_1w)) else np.nan

# ─────────────────────────── KPI’s & Explain ───────────────────────────
def last_and_delta(series: pd.Series, lookback:int=20):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty: return np.nan, None
    last = float(s.iloc[-1])
    ref = float(s.tail(lookback).mean()) if len(s) >= lookback else float(s.mean())
    return last, last - ref

k1, k2, k3, k4, k5 = st.columns(5)
pcr_v_last, pcr_v_delta = last_and_delta(pv.get("PCR_vol", pd.Series(dtype=float)))
pcr_oi_last, pcr_oi_delta = last_and_delta(pv.get("PCR_oi", pd.Series(dtype=float)))
with k1: st.metric("PCR (Volume)", f"{pcr_v_last:.2f}" if not np.isnan(pcr_v_last) else "—",
                   delta=(f"{pcr_v_delta:+.2f} vs 20d" if pcr_v_delta is not None else None))
with k2: st.metric("PCR (OI)", f"{pcr_oi_last:.2f}" if not np.isnan(pcr_oi_last) else "—",
                   delta=(f"{pcr_oi_delta:+.2f} vs 20d" if pcr_oi_delta is not None else None))
with k3: st.metric("25Δ Skew (put−call)", f"{skew_25d:.2%}" if not np.isnan(skew_25d) else "—")
with k4: st.metric("IV Term Slope (7–15d − 30–60d)", f"{iv_slope:.2%}" if not np.isnan(iv_slope) else "—")
with k5:
    em_txt = f"±{em_1w_pts:,.0f} pts ({em_1w_pts/underlying_now_sent:.2%})" \
             if (not np.isnan(em_1w_pts) and not np.isnan(underlying_now_sent)) else "—"
    st.metric("Expected Move ~1w", em_txt)

explain_lines = []
def _fmt(x, pct=False):
    if np.isnan(x): return "—"
    return f"{x:.2%}" if pct else f"{x:.2f}"

if not np.isnan(pcr_v_last) or not np.isnan(pcr_oi_last):
    tilt = "defensief" if ((not np.isnan(pcr_v_last) and pcr_v_last>1.0)
                           or (not np.isnan(pcr_oi_last) and pcr_oi_last>1.0)) else "speculatief / neutraal"
    explain_lines.append(f"PCR wijst op **{tilt}** positioning (Vol={_fmt(pcr_v_last)}, OI={_fmt(pcr_oi_last)}).")
if not np.isnan(skew_25d):
    explain_lines.append(f"25Δ-skew {_fmt(skew_25d, pct=True)} → {'puts duurder vs calls (downside hedge-vraag)' if skew_25d>0 else 'vlakkere of omgekeerde skew'}.")
if not np.isnan(iv_slope):
    explain_lines.append(f"Term-slope {_fmt(iv_slope, pct=True)} → korte IV {'boven' if iv_slope>0 else 'onder'} 30–60D.")
if not (np.isnan(underlying_now_sent) or np.isnan(em_1w_pts)):
    explain_lines.append(f"~1w expected move ≈ **±{em_1w_pts:,.0f}** punten (~{em_1w_pts/underlying_now_sent:.2%}).")

if explain_lines:
    st.markdown("**Explain (kort):** " + " ".join(explain_lines))

# ─────────────────────────── Grafieken ───────────────────────────
if not pv.empty:
    fig_pcr_ts = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                               subplot_titles=("Put/Call Ratio — Volume", "Put/Call Ratio — Open Interest"))
    fig_pcr_ts.add_trace(go.Scatter(x=pv.index, y=pv["PCR_vol"], mode="lines", name="PCR (Vol)"), row=1, col=1)
    annotate_pcr_midline(fig_pcr_ts, row=1, col=1)
    fig_pcr_ts.add_trace(go.Scatter(x=pv.index, y=pv["PCR_oi"],  mode="lines", name="PCR (OI)"),  row=2, col=1)
    annotate_pcr_midline(fig_pcr_ts, row=2, col=1)
    apply_style_and_show(fig_pcr_ts, height=620)
    st.caption("ℹ️ PCR > 1 = defensief (puts > calls); PCR < 1 = offensief. Combineer met skew en term-slope.")

if not dgp.empty:
    fig_dpcr = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                             subplot_titles=("Δ-gewogen PCR — Volume", "Δ-gewogen PCR — Open Interest"))
    fig_dpcr.add_trace(go.Scatter(x=dgp.index, y=dgp["PCR_delta_vol"], mode="lines", name="Δ-PCR (Vol)"), row=1, col=1)
    annotate_pcr_midline(fig_dpcr, row=1, col=1)
    fig_dpcr.add_trace(go.Scatter(x=dgp.index, y=dgp["PCR_delta_oi"],  mode="lines", name="Δ-PCR (OI)"),  row=2, col=1)
    annotate_pcr_midline(fig_dpcr, row=2, col=1)
    apply_style_and_show(fig_dpcr, height=620)
    st.caption("ℹ️ Δ-gewogen PCR corrigeert voor moneyness (exposure). Boven 1 = defensief; onder 1 = offensief.")
else:
    st.info("Geen Δ-gewogen PCR te berekenen met de huidige filters.")
# ─────────────────────────── A) Serie-selectie ───────────────────────────
st.header("🎯 Serie-selectie — volg één optiereeks door de tijd")

cS1, cS2, cS3, cS4 = st.columns([1, 1, 1, 1.6])
with cS1:
    series_strike = st.selectbox("Serie Strike", options=strikes_all or [6000.0],
                                 index=(strikes_all.index(default_series_strike)
                                        if default_series_strike in strikes_all else 0))
with cS2:
    series_exp = st.selectbox("Serie Expiratie", options=exps_all or [date.today()],
                              index=(exps_all.index(default_series_exp)
                                     if (exps_all and default_series_exp in exps_all) else 0))
with cS3:
    series_price_col = st.radio("Prijsbron", ["last_price","mid_price"], index=0, horizontal=True)
with cS4:
    st.caption(f"🔧 Default: {'PUT −300' if sel_type=='put' else 'CALL +200'} rond S≈{underlying_now:.0f} • Exp ~{target_exp}")

serie = df[(df["strike"]==series_strike) & (df["expiration"]==series_exp) & liq_mask].copy().sort_values("snapshot_date")
if serie.empty:
    st.info("Geen (genoeg) liquiditeit voor deze combinatie binnen de huidige filters.")
else:
    a1, a2 = st.columns(2)
    with a1:
        fig_price = make_subplots(specs=[[{"secondary_y": True}]])
        fig_price.add_trace(go.Scatter(
            x=serie["snapshot_date"],
            y=apply_outlier(serie[series_price_col], outlier_mode, pct_clip, z_thr),
            name="Price", mode="lines+markers", connectgaps=True,
            hovertemplate="Tijd: %{x}<br>Prijs: %{y:.2f}<extra></extra>"
        ), secondary_y=False)
        if show_underlying:
            fig_price.add_trace(go.Scatter(
                x=serie["snapshot_date"], y=serie["underlying_price"],
                name="SP500", mode="lines", line=dict(dash="dot"), connectgaps=True,
                hovertemplate="Tijd: %{x}<br>SPX: %{y:.2f}<extra></extra>"
            ), secondary_y=True)
        fig_price.update_layout(title=f"{sel_type.upper()} {series_strike} — exp {series_exp} | Price vs SP500")
        fig_price.update_xaxes(title_text="Meetmoment")
        fig_price.update_yaxes(title_text="Price", secondary_y=False)
        fig_price.update_yaxes(title_text="SP500", secondary_y=True)
        apply_style_and_show(fig_price, height=460)
        st.caption("💡 **Interpretatie:** stabiel verloop met smalle bid-ask en voldoende volume wijst op een liquide serie.")
    with a2:
        fig_ppd = make_subplots(specs=[[{"secondary_y": True}]])
        fig_ppd.add_trace(go.Scatter(
            x=serie["snapshot_date"],
            y=apply_outlier(ppd_series(serie), outlier_mode, pct_clip, z_thr),
            name="PPD", mode="lines+markers", connectgaps=True,
            hovertemplate="Tijd: %{x}<br>PPD: %{y:.3f}<extra></extra>"
        ), secondary_y=False)
        if show_underlying:
            fig_ppd.add_trace(go.Scatter(
                x=serie["snapshot_date"], y=serie["underlying_price"],
                name="SP500", mode="lines", line=dict(dash="dot"), connectgaps=True
            ), secondary_y=True)
        fig_ppd.update_layout(title=f"{sel_type.upper()} {series_strike} — exp {series_exp} | PPD vs SP500")
        fig_ppd.update_xaxes(title_text="Meetmoment")
        fig_ppd.update_yaxes(title_text=ppd_y_label(), secondary_y=False)
        fig_ppd.update_yaxes(title_text="SP500", secondary_y=True)
        apply_style_and_show(fig_ppd, height=460)
        st.caption("💡 **Gebruik:** stijgende PPD bij gelijke DTE → oplopende IV/vraag; daling → omgekeerd.")

# ─────────────────────────── B) PPD vs Afstand ───────────────────────────
st.header("📈 PPD & Afstand tot Uitoefenprijs (ATM → OTM/ITM)")

dist_col = st.radio("Afstandseenheid", ["% van spot", "punten (|K−S|)"], index=1, horizontal=True)
bin_size_pts = st.slider("Bin-grootte (punten)", 10, 200, 25, step=5, disabled=(dist_col != "punten (|K−S|)"))

default_idx = snapshots_all.index(default_snapshot) if default_snapshot in snapshots_all else len(snapshots_all)-1
sel_snapshot = st.selectbox("Peildatum (snapshot)", options=snapshots_all, index=max(default_idx,0),
                            format_func=lambda x: pd.to_datetime(x).strftime("%Y-%m-%d %H:%M"))

df_last = df[(df["snap_min"] == sel_snapshot) & liq_mask].copy()
if df_last.empty:
    st.info("Geen data op dit snapshot (na liquiditeit-filter).")
else:
    df_last = df_last.assign(ppd_u=ppd_series(df_last))
    if dist_col.startswith("%"):
        x_col = "abs_dist_pct"; x_title = "Afstand |K−S|/S (%)"
        bins = np.arange(0, 18.5, 0.5)
        cur_dist = (abs(series_strike - underlying_now) / underlying_now) * 100.0
        vline_lbl = f"gekozen strike ≈ {cur_dist:.2f}%"
    else:
        x_col = "abs_dist_pts"; x_title = "Afstand |K−S| (punten)"
        max_pts = float(np.nanpercentile(df_last["abs_dist_pts"], 98)) if df_last["abs_dist_pts"].notna().any() else 400.0
        bins = np.arange(0, max_pts + bin_size_pts, bin_size_pts)
        cur_dist = abs(series_strike - underlying_now)
        vline_lbl = f"gekozen strike ≈ {cur_dist:.0f} pts"

    df_last["dist_bin"] = pd.cut(df_last[x_col], bins=bins, include_lowest=True)
    g = (df_last.groupby("dist_bin")
                 .agg(ppd=("ppd_u","median"), n=("ppd_u","count"))
                 .reset_index())
    g = g[g["n"] >= min_per_bin].copy()
    g["bin_mid"] = g["dist_bin"].apply(lambda iv: iv.mid if pd.notna(iv) else np.nan)
    g = g.dropna(subset=["bin_mid"]).sort_values("bin_mid")
    g["ppd_s"] = smooth_series(g["ppd"], window=3)
    best_idx = g["ppd_s"].idxmax() if not g.empty else None

    fig_ppd_dist = go.Figure()
    fig_ppd_dist.add_vrect(x0=-0.5 if dist_col.startswith("%") else -bin_size_pts/2,
                           x1=0.5 if dist_col.startswith("%") else bin_size_pts/2,
                           fillcolor="lightgrey", opacity=0.25, line_width=0,
                           annotation_text="ATM-zone", annotation_position="top left")
    if not np.isnan(cur_dist):
        fig_ppd_dist.add_vline(x=cur_dist, line=dict(dash="dot"),
                               annotation_text=vline_lbl, annotation_position="top right")
    fig_ppd_dist.add_trace(go.Scatter(x=g["bin_mid"], y=g["ppd"], mode="markers",
                                      name="PPD (median/bin)", opacity=0.85))
    fig_ppd_dist.add_trace(go.Scatter(x=g["bin_mid"], y=g["ppd_s"], mode="lines",
                                      name="Smoothed"))
    fig_ppd_dist.update_layout(title=f"PPD vs Afstand — {pd.to_datetime(sel_snapshot).strftime('%Y-%m-%d %H:%M')}",
                               xaxis_title=x_title, yaxis_title=ppd_y_label())
    apply_style_and_show(fig_ppd_dist, height=460)

    if best_idx is not None and pd.notna(g.loc[best_idx,"ppd_s"]):
        st.info(f"**Advies (afstand):** sweet spot rond **{g.loc[best_idx,'bin_mid']:.1f} "
                f"{'%' if dist_col.startswith('%') else 'pts'}** met **PPD ≈ {g.loc[best_idx,'ppd_s']:.2f}**. "
                f"Combineer met DTE-inzicht hieronder.")

# ─────────────────────────── C) Prijs/PPD vs DTE ───────────────────────────
st.header("🕓 Prijs/PPD vs DTE — term-structure voor gekozen strike")

df_last_strike = df_last[df_last["strike"] == series_strike].copy()
if df_last_strike.empty:
    st.info("Geen data voor deze strike op dit snapshot.")
else:
    exp_curve = (df_last_strike.assign(ppd_u=ppd_series(df_last_strike))
                 .groupby("expiration", as_index=False)
                 .agg(price=(series_price_col,"median"),
                      ppd=("ppd_u","median"),
                      dte=("days_to_exp","median"))
                 .sort_values("dte"))
    exp_curve["price_f"] = apply_outlier(exp_curve["price"], outlier_mode, pct_clip, z_thr)
    exp_curve["ppd_f"]   = apply_outlier(exp_curve["ppd"],  outlier_mode, pct_clip, z_thr)

    fig_exp = make_subplots(specs=[[{"secondary_y": True}]])
    fig_exp.add_trace(go.Scatter(x=exp_curve["dte"], y=exp_curve["price_f"],
                                 name="Price", mode="lines+markers"), secondary_y=False)
    fig_exp.add_trace(go.Scatter(x=exp_curve["dte"], y=exp_curve["ppd_f"],
                                 name="PPD", mode="lines+markers"), secondary_y=True)
    fig_exp.update_layout(title=f"{sel_type.upper()} — Strike {series_strike} — {pd.to_datetime(sel_snapshot).strftime('%Y-%m-%d %H:%M')}")
    fig_exp.update_xaxes(title_text="Days to Expiration (DTE)")
    fig_exp.update_yaxes(title_text="Price (points)", secondary_y=False)
    fig_exp.update_yaxes(title_text=ppd_y_label(), secondary_y=True)
    apply_style_and_show(fig_exp, height=460)
    st.caption("📈 Bij langere DTE neemt de time value toe → prijs en vaak ook PPD lopen op.")

# ─────────────────────────── D) PPD vs DTE (aggregaat) ───────────────────────────
st.header("📊 PPD vs DTE — opbouw van premium per dag")

d_c1, d_c2, d_c3, d_c4, d_c5 = st.columns([1.2, 1, 1, 1, 1])
with d_c1:
    d_ppd_mode = st.radio("Bereik", ["ATM-band (moneyness)", "Rond gekozen strike"],
                          index=0, key="d_ppd_mode_key")
with d_c2:
    d_atm_band = st.slider("ATM-band (± moneyness %)", 0.01, 0.10, 0.02, step=0.01,
                           key="d_atm_band_key")
with d_c3:
    d_strike_window = st.slider("Strike-venster rond gekozen strike (punten)", 10, 500, 50, step=10,
                                key="d_strike_window_key")
with d_c4:
    d_use_last_snap = st.checkbox("Alleen laatste snapshot", value=True, key="d_use_last_snap_key")
with d_c5:
    d_robust_scale = st.checkbox("Robust scale (95e pct)", value=True, key="d_robust_scale_key")

d_base = (df_last if d_use_last_snap else df).copy()
d_base = d_base[liq_mask]

if d_ppd_mode.startswith("ATM"):
    d_df = d_base[np.abs(d_base["moneyness"]) <= d_atm_band].copy()
else:
    d_df = d_base[(d_base["strike"] >= series_strike - d_strike_window)
                  & (d_base["strike"] <= series_strike + d_strike_window)].copy()

if d_df.empty:
    st.info("Geen data voor PPD vs DTE met deze instellingen.")
else:
    d_df = d_df.assign(ppd_u=ppd_series(d_df))
    d_curve = (d_df.groupby("days_to_exp", as_index=False)
                  .agg(ppd=("ppd_u","median"), n=("ppd_u","count"))
                  .query("n >= @min_per_bin")
                  .sort_values("days_to_exp"))
    d_curve["ppd_s"] = smooth_series(d_curve["ppd"], window=3)

    y_range = None
    if d_robust_scale and d_curve["ppd"].notna().any():
        d_hi = float(np.nanpercentile(d_curve["ppd"], 95))
        d_lo = float(np.nanpercentile(d_curve["ppd"], 5))
        pad = (d_hi - d_lo) * 0.10
        y_range = [max(d_lo - pad, 0.0), d_hi + pad]

    fig_ppd_dte = go.Figure()
    fig_ppd_dte.add_trace(go.Scatter(x=d_curve["days_to_exp"], y=d_curve["ppd"],
                                     mode="markers", name="PPD (median)", opacity=0.85))
    fig_ppd_dte.add_trace(go.Scatter(x=d_curve["days_to_exp"], y=d_curve["ppd_s"],
                                     mode="lines", name="Smoothed"))
    fig_ppd_dte.update_layout(title="PPD vs Days To Expiration (DTE)",
                              xaxis_title="Days to Expiration (DTE)",
                              yaxis_title=ppd_y_label())
    if y_range: fig_ppd_dte.update_yaxes(range=y_range)
    apply_style_and_show(fig_ppd_dte, height=460)

    d_sweet = d_curve.loc[d_curve["ppd_s"].idxmax()] if not d_curve.empty else None
    if d_sweet is not None and pd.notna(d_sweet["ppd_s"]):
        st.info(f"**Advies (DTE):** sweet spot rond **{int(d_sweet['days_to_exp'])} dagen** "
                f"met **PPD ≈ {d_sweet['ppd_s']:.2f}**. Combineer dit met de afstand-sweet spot hierboven.")
# ─────────────────────────── F) Term structure & Smile ───────────────────────────
st.header("🌡️ Term structure & Smile")

# Term structure — mediane IV per DTE (laatste snapshot + liquiditeit)
term_src = df[(df["snap_min"] == (default_snapshot if default_snapshot is not None else df["snap_min"].iloc[-1])) & liq_mask].copy()
if term_src.empty:
    st.info("Geen data voor term structure op dit snapshot.")
else:
    term = (term_src.groupby("days_to_exp", as_index=False)["implied_volatility"]
                    .median().sort_values("days_to_exp"))
    fig_term = go.Figure(go.Scatter(
        x=term["days_to_exp"], y=term["implied_volatility"],
        mode="lines+markers", name=f"IV {sel_type.upper()}"
    ))
    fig_term.update_layout(title="Term Structure — mediane IV", xaxis_title="DTE", yaxis_title="Implied Volatility")
    apply_style_and_show(fig_term, height=380)
    st.caption("**Interpretatie:** stijgende term structure → hogere verwachte volatiliteit op langere looptijd; dalend → omgekeerd.")

# IV Smile — per gekozen expiratie (laatste snapshot)
st.subheader("IV Smile (laatste snapshot)")
exp_for_smile = st.selectbox("Expiratie voor IV Smile", options=exps_all or [None], index=0, key="smile_exp_select")
sm = term_src[term_src["expiration"] == exp_for_smile].copy() if exp_for_smile and not term_src.empty else pd.DataFrame()
if sm.empty:
    st.info("Geen (liquide) data voor IV Smile.")
else:
    sm = sm.groupby("strike", as_index=False)["implied_volatility"].median().sort_values("strike")
    if len(sm) >= 5:
        lo, hi = sm["implied_volatility"].quantile([0.02, 0.98])
        sm["implied_volatility"] = sm["implied_volatility"].clip(lower=lo, upper=hi)
    fig_sm = go.Figure(go.Scatter(x=sm["strike"], y=sm["implied_volatility"], mode="lines+markers", name="IV"))
    fig_sm.update_layout(title=f"IV Smile — {sel_type.upper()} exp {exp_for_smile}",
                         xaxis_title="Strike", yaxis_title="Implied Volatility")
    apply_style_and_show(fig_sm, height=420)
    st.caption("**Gebruik:** kies strangles in een **glad** stuk van de smile (weinig spikes) → betere liquiditeit & eerlijkere mid-prices.")

# ─────────────────────────── G) Put/Call-ratio per expiratie ───────────────────────────
st.header("🧮 Put/Call-ratio per expiratie")

p = (df_both[liq_mask_both].groupby(["expiration","type"], as_index=False)
       .agg(vol=("volume","sum"), oi=("open_interest","sum")))
if p.empty:
    st.info("Geen data voor PCR per expiratie.")
else:
    pv2 = (p.pivot_table(index="expiration", columns="type", values=["vol","oi"], aggfunc="sum")
             .sort_index().sort_index(axis=1)).fillna(0.0)
    pv2.columns = [f"{a}_{b}" for a,b in pv2.columns.to_flat_index()]
    for col in ["vol_put","vol_call","oi_put","oi_call"]:
        if col not in pv2.columns: pv2[col] = 0.0
    pv2["PCR_vol"] = pv2["vol_put"] / pv2["vol_call"].replace(0, np.nan)
    pv2["PCR_oi"]  = pv2["oi_put"]  / pv2["oi_call"].replace(0, np.nan)
    pv2 = pv2.replace([np.inf,-np.inf], np.nan).dropna(subset=["PCR_vol","PCR_oi"], how="all").reset_index()

    fig_pcr = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                            subplot_titles=("PCR op Volume", "PCR op Open Interest"))
    fig_pcr.add_trace(go.Bar(x=pv2["expiration"], y=pv2["PCR_vol"], name="PCR (Vol)"), row=1, col=1)
    fig_pcr.add_trace(go.Bar(x=pv2["expiration"], y=pv2["PCR_oi"],  name="PCR (OI)"),  row=2, col=1)
    fig_pcr.update_xaxes(title_text="Expiratie", row=2, col=1)
    fig_pcr.update_yaxes(title_text="Ratio", row=1, col=1)
    fig_pcr.update_yaxes(title_text="Ratio", row=2, col=1)
    apply_style_and_show(fig_pcr, height=520)
    st.caption("**Interpretatie:** PCR↑ → defensiever; PCR↓ → offensiever. Gebruik dit bij het kiezen van afstand/DTE (conservatie vs. agressie).")

# ─────────────────────────── H) Vol & Risk (ATM-IV, HV, VRP, IV-Rank, EM) ───────────────────────────
st.header("📊 Vol & Risk — ATM IV, HV20, VRP, IV-Rank, Expected Move")

# Dagelijkse onderliggende & HV20
u_daily = (df.assign(dte=df["snapshot_date"].dt.date).sort_values(["dte","snapshot_date"])
             .groupby("dte", as_index=False).agg(close=("underlying_price","last")))
u_daily["ret"] = u_daily["close"].pct_change()
hv20 = annualize_std(u_daily["ret"].tail(21).dropna())

# ATM-IV (30–40D rond ATM, laatste snapshot)
near_atm = term_src[(term_src["days_to_exp"].between(20, 40)) & (term_src["moneyness"].abs() <= 0.01)]
iv_atm = float(near_atm["implied_volatility"].median()) if not near_atm.empty else float(term_src["implied_volatility"].median())

# IV-Rank (1y)
iv_hist = (df.assign(day=df["snapshot_date"].dt.date)
             .query("days_to_exp>=20 and days_to_exp<=40 and abs(moneyness)<=0.01")
             .groupby("day", as_index=False)["implied_volatility"].median()
             .rename(columns={"implied_volatility":"iv"}))
iv_1y = iv_hist.tail(252)["iv"] if not iv_hist.empty else pd.Series(dtype=float)
iv_rank = float((iv_1y <= iv_1y.iloc[-1]).mean()) if not iv_1y.empty else np.nan

# Expected Move (σ) op basis van ATM IV en gekozen expiratie
dte_selected = int(pd.to_numeric(term_src[term_src["expiration"]==exp_for_smile]["days_to_exp"], errors="coerce").median()) \
               if (exp_for_smile and not term_src.empty) else 30
em_sigma = (underlying_now * iv_atm * math.sqrt(max(dte_selected,1)/365.0)) \
            if (not np.isnan(underlying_now) and not np.isnan(iv_atm)) else np.nan

cv1, cv2, cv3, cv4, cv5 = st.columns(5)
with cv1: st.metric("ATM-IV (~30D)", f"{iv_atm:.2%}" if not np.isnan(iv_atm) else "—")
with cv2: st.metric("HV20", f"{hv20:.2%}" if not np.isnan(hv20) else "—")
with cv3: st.metric("VRP (IV−HV)", f"{(iv_atm-hv20):.2%}" if (not np.isnan(iv_atm) and not np.isnan(hv20)) else "—")
with cv4: st.metric("IV-Rank (1y)", f"{iv_rank*100:.0f}%" if not np.isnan(iv_rank) else "—")
with cv5:
    em_txt = f"±{em_sigma:,.0f} pts ({em_sigma/underlying_now:.2%})" if (not np.isnan(em_sigma) and not np.isnan(underlying_now)) else "—"
    st.metric("Expected Move (σ)", em_txt)
st.caption("**VRP** > 0: IV boven gerealiseerde vol → gunstiger voor **short vol**. **IV-Rank** hoog → premie dikker (events checken).")

# ─────────────────────────── L) VIX vs IV ───────────────────────────
st.header("🧷 VIX vs gemiddelde IV")

vix_vs_iv = (df.assign(snap_date=df["snapshot_date"].dt.date)
               .groupby("snap_date", as_index=False)
               .agg(vix=("vix","median"), iv=("implied_volatility","median"))
               .rename(columns={"snap_date":"date"}))
if not vix_vs_iv.empty:
    cV1, cV2, cV3 = st.columns([1, 1, 1])
    with cV1: smooth_vix = st.checkbox("Smooth (7d)", value=False)
    with cV2: force_zero = st.checkbox("Forceer 0-baseline", value=False)
    with cV3: pad_pct    = st.slider("Y-pad (%)", 5, 30, 15, step=1)
    if smooth_vix and len(vix_vs_iv) >= 3:
        vix = vix_vs_iv["vix"].rolling(7, min_periods=1, center=True).median()
        iv  = vix_vs_iv["iv"].rolling(7, min_periods=1, center=True).median()
    else:
        vix, iv = vix_vs_iv["vix"], vix_vs_iv["iv"]

    def padded_range(series: pd.Series, pad_frac: float, floor_zero: bool):
        s = pd.to_numeric(series, errors="coerce").dropna()
        if s.empty: return None
        lo, hi = float(s.min()), float(s.max())
        if hi == lo: lo, hi = lo-0.1*(abs(lo) if lo!=0 else 1.0), hi+0.1*(abs(hi) if hi!=0 else 1.0)
        pad = (hi - lo) * pad_frac
        lo2 = 0.0 if floor_zero else (lo - pad)
        hi2 = hi + pad
        if "iv" in series.name.lower(): lo2 = max(lo2, 0.0 if floor_zero else max(0.0, lo - pad))
        return [lo2, hi2]

    vix_range = padded_range(vix.rename("vix"), pad_pct/100.0, force_zero)
    iv_range  = padded_range(iv.rename("iv"),  pad_pct/100.0, force_zero)

    fig_vix = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                            subplot_titles=("VIX", "Gemiddelde IV"))
    fig_vix.add_trace(go.Scatter(x=vix_vs_iv["date"], y=vix, mode="lines+markers", name="VIX"), row=1, col=1)
    fig_vix.add_trace(go.Scatter(x=vix_vs_iv["date"], y=iv,  mode="lines+markers", name="IV"),  row=2, col=1)
    if vix_range: fig_vix.update_yaxes(range=vix_range, row=1, col=1)
    if iv_range:  fig_vix.update_yaxes(range=iv_range,  row=2, col=1)
    apply_style_and_show(fig_vix, height=620)
    st.caption("📌 Divergenties kunnen wijzen op relatieve over- of onderwaardering van index-opties t.o.v. brede marktvol.")

# ─────────────────────────── Footer ───────────────────────────────
st.markdown("---")
st.caption("🔍 Navigatie: zoom met scroll/pinch, **double-click** om te rescalen. Subtiele grids en middenstreep helpen de context zonder te schreeuwen.")
