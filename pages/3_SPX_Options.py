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

# —— Vervangen: Black–Scholes Δ met continue r en q ——
def bs_delta(S: float, K: float, iv: float, T_years: float, r_cont: float, q_cont: float, is_call: bool) -> float:
    # S, K > 0; iv >= 0; T_years > 0; r_cont/q_cont zijn continue rentes per jaar
    if any(x is None or np.isnan(x) or x <= 0 for x in [S, K]) or np.isnan(iv) or T_years <= 0:
        return np.nan
    sqrtT = math.sqrt(T_years)
    denom = iv * sqrtT if iv > 0 else float("inf")
    d1 = (math.log(S / K) + (r_cont - q_cont + 0.5 * iv * iv) * T_years) / denom
    Nd1 = norm_cdf(d1)
    disc_q = math.exp(-q_cont * T_years)
    return (disc_q * Nd1) if is_call else (disc_q * (Nd1 - 1.0))

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
3) Gebruik **PPD vs Afstand** & **PPD vs DTE** om een sweet spot in %-afstand en looptijd te vinden.  
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

# Liquidity mask (gebruik: overal waar we aggregeren/tekenen)
liq_mask = ((df["open_interest"].fillna(0) >= min_oi) | (df["volume"].fillna(0) >= min_vol))

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

def ppd_y_label(): return "PPD (bp/day)" if ppd_unit.startswith("bp") else "PPD (points/day)"

# ─────────────────────────── Defaults ─────────────────────────────
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

# ─────────────────────────── A) Serie-selectie ────────────────────
st.subheader("Serie-selectie — volg één optiereeks door de tijd")
cS1, cS2, cS3, cS4 = st.columns([1, 1, 1, 1.6])
with cS1:
    series_strike = st.selectbox("Serie Strike", options=strikes_all or [6000.0],
                                 index=(strikes_all.index(default_series_strike) if default_series_strike in strikes_all else 0))
with cS2:
    series_exp = st.selectbox("Serie Expiratie", options=exps_all or [date.today()],
                              index=(exps_all.index(default_series_exp) if (default_series_exp in exps_all) else 0))
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
                                       y=apply_outlier(serie[series_price_col], outlier_mode, pct_clip, z_thr),
                                       name="Price", mode="lines+markers", connectgaps=True), secondary_y=False)
        if show_underlying:
            fig_price.add_trace(go.Scatter(x=serie["snapshot_date"], y=serie["underlying_price"],
                                           name="SP500", mode="lines", line=dict(dash="dot"), connectgaps=True), secondary_y=True)
        fig_price.update_layout(title=f"{sel_type.upper()} {series_strike} — exp {series_exp} | Price vs SP500",
                                height=420, hovermode="x unified", dragmode="zoom")
        fig_price.update_xaxes(title_text="Meetmoment")
        fig_price.update_yaxes(title_text="Price", secondary_y=False, rangemode="tozero")
        fig_price.update_yaxes(title_text="SP500", secondary_y=True)
        st.plotly_chart(fig_price, use_container_width=True, config=PLOTLY_CONFIG)
        st.caption("**Interpretatie:** stabiel verloop met smalle bid-ask en voldoende volume/oi wijst op een ‘werkbare’ serie.")
    with a2:
        fig_ppd = make_subplots(specs=[[{"secondary_y": True}]])
        fig_ppd.add_trace(go.Scatter(x=serie["snapshot_date"],
                                     y=apply_outlier(ppd_series(serie), outlier_mode, pct_clip, z_thr),
                                     name="PPD", mode="lines+markers", connectgaps=True), secondary_y=False)
        if show_underlying:
            fig_ppd.add_trace(go.Scatter(x=serie["snapshot_date"], y=serie["underlying_price"],
                                         name="SP500", mode="lines", line=dict(dash="dot"), connectgaps=True), secondary_y=True)
        fig_ppd.update_layout(title=f"{sel_type.upper()} {series_strike} — exp {series_exp} | PPD vs SP500",
                              height=420, hovermode="x unified", dragmode="zoom")
        fig_ppd.update_xaxes(title_text="Meetmoment")
        fig_ppd.update_yaxes(title_text=ppd_y_label(), secondary_y=False, rangemode="tozero")
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
    df_last = df_last.assign(ppd_u=ppd_series(df_last))
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
                               xaxis_title="Afstand |K−S|/S (%)", yaxis_title=ppd_y_label(),
                               height=420, dragmode="zoom")
    st.plotly_chart(fig_ppd_dist, use_container_width=True, config=PLOTLY_CONFIG)

    # mini-advies
    if best_idx is not None and pd.notna(g.loc[best_idx,"ppd_s"]):
        st.info(f"**Advies (afstand):** over dit snapshot ligt de **sweet spot** rond **{g.loc[best_idx,'bin_mid']:.1f}%** van ATM "
                f"met **PPD ≈ {g.loc[best_idx,'ppd_s']:.2f}**. Combineer met DTE-inzicht hieronder.")

# ─────────────────────────── C) Price/PPD vs Exp Date ─────────────
st.subheader("Ontwikkeling Prijs per Expiratiedatum (laatste snapshot)")
df_last_strike = df_last[df_last["strike"] == series_strike].copy()
if df_last_strike.empty:
    st.info("Geen data voor deze strike op dit snapshot (na liquiditeit-filter).")
else:
    exp_curve = (df_last_strike.assign(ppd_u=ppd_series(df_last_strike))
                 .groupby("expiration", as_index=False)
                 .agg(price=(series_price_col, "median"), ppd=("ppd_u", "median"))
                 .sort_values("expiration"))
    exp_curve["price_f"] = apply_outlier(exp_curve["price"], outlier_mode, pct_clip, z_thr)
    exp_curve["ppd_f"]   = apply_outlier(exp_curve["ppd"],  outlier_mode, pct_clip, z_thr)
    fig_exp = make_subplots(specs=[[{"secondary_y": True}]])
    fig_exp.add_trace(go.Scatter(x=exp_curve["expiration"], y=exp_curve["price_f"],
                                 name="Price", mode="lines+markers"), secondary_y=False)
    fig_exp.add_trace(go.Scatter(x=exp_curve["expiration"], y=exp_curve["ppd_f"],
                                 name="PPD", mode="lines+markers"), secondary_y=True)
    fig_exp.update_layout(title=f"{sel_type.upper()} — Strike {series_strike} — peildatum {pd.to_datetime(sel_snapshot).strftime('%Y-%m-%d %H:%M')}",
                          height=420, hovermode="x unified", dragmode="zoom")
    fig_exp.update_xaxes(title_text="Expiratiedatum")
    fig_exp.update_yaxes(title_text="Price", secondary_y=False, rangemode="tozero")
    fig_exp.update_yaxes(title_text=ppd_y_label(), secondary_y=True)
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
base_df = base_df[liq_mask]  # respecteer liquiditeit
if ppd_mode.startswith("ATM"):
    df_ppd = base_df[np.abs(base_df["moneyness"]) <= atm_band].copy()
else:
    df_ppd = base_df[(base_df["strike"] >= series_strike - strike_window) & (base_df["strike"] <= series_strike + strike_window)].copy()

if df_ppd.empty:
    st.info("Geen data voor PPD vs DTE met deze instellingen.")
else:
    df_ppd = df_ppd.assign(ppd_u=ppd_series(df_ppd))
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
                              yaxis_title=ppd_y_label(), height=420, dragmode="zoom")
    if y_range: fig_ppd_dte.update_yaxes(range=y_range)
    st.plotly_chart(fig_ppd_dte, use_container_width=True, config=PLOTLY_CONFIG)

    # mini-advies (DTE sweet spot)
    sweet_row = ppd_curve.loc[ppd_curve["ppd_s"].idxmax()] if not ppd_curve.empty else None
    if sweet_row is not None and pd.notna(sweet_row["ppd_s"]):
        st.info(f"**Advies (DTE):** de PPD-sweet spot ligt rond **{int(sweet_row['days_to_exp'])} dagen** "
                f"met **PPD ≈ {sweet_row['ppd_s']:.2f}**. Combineer dit met de afstand-sweet spot hierboven.")

st.markdown("---")

# ─────────────────────────── E) Matrix ────────────────────────────
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
        mx = mx.assign(ppd_u=ppd_series(mx)); value_col = "ppd_u"
    else:
        value_col = matrix_metric
    mx["snap_s"] = mx["snapshot_date"].dt.strftime("%Y-%m-%d %H:%M")
    pivot = mx.pivot_table(index="snap_s", columns="strike", values=value_col, aggfunc="median").sort_index(ascending=False).round(3)
    arr = pivot.values.astype(float)
    tab_hm, tab_tbl = st.tabs(["Heatmap", "Tabel"])
    with tab_hm:
        fig_mx = go.Figure(data=go.Heatmap(z=arr, x=pivot.columns.astype(float), y=pivot.index.tolist(),
                                           colorbar_title=value_col))
        fig_mx.update_layout(title=f"Heatmap — {sel_type.upper()} exp {matrix_exp} — {value_col}",
                             xaxis_title="Strike", yaxis_title="Meetmoment", height=520, dragmode="zoom")
        st.plotly_chart(fig_mx, use_container_width=True, config=PLOTLY_CONFIG)
    with tab_tbl:
        st.dataframe(pivot, use_container_width=True)

st.markdown("---")

# ─────────────────────────── F) Term structure & Smile ────────────
term = df[liq_mask].groupby("days_to_exp", as_index=False)["implied_volatility"].median().sort_values("days_to_exp")
fig_term = go.Figure(go.Scatter(x=term["days_to_exp"], y=term["implied_volatility"], mode="lines+markers", name=f"IV {sel_type.upper()}"))
fig_term.update_layout(title="Term Structure — mediane IV", xaxis_title="DTE", yaxis_title="Implied Volatility", height=380, dragmode="zoom")
st.plotly_chart(fig_term, use_container_width=True, config=PLOTLY_CONFIG)
st.caption("**Interpretatie:** stijgende term structure → hogere vol voor langere looptijd; dalend → omgekeerd.")

st.subheader("IV Smile (laatste snapshot)")
exp_for_smile = st.selectbox("Expiratie voor IV Smile", options=exps_all or [None], index=0)
sm = df_last[(df_last["expiration"] == exp_for_smile) & liq_mask].copy()
if sm.empty:
    st.info("Geen (liquide) data voor IV Smile.")
else:
    sm = sm.groupby("strike", as_index=False)["implied_volatility"].median().sort_values("strike")
    # zachte clipping tegen spikes
    if len(sm) >= 5:
        lo, hi = sm["implied_volatility"].quantile([0.02, 0.98])
        sm["implied_volatility"] = sm["implied_volatility"].clip(lower=lo, upper=hi)
    fig_sm = go.Figure(go.Scatter(x=sm["strike"], y=sm["implied_volatility"], mode="lines+markers", name="IV"))
    fig_sm.update_layout(title=f"IV Smile — {sel_type.upper()} exp {exp_for_smile}",
                         xaxis_title="Strike", yaxis_title="Implied Volatility", height=420, dragmode="zoom")
    st.plotly_chart(fig_sm, use_container_width=True, config=PLOTLY_CONFIG)
    st.caption("**Gebruik:** kies strangles in een **glad** stuk van de smile (weinig spikes) → betere liquiditeit & eerlijkere mid-prices.")

# ─────────────────────────── G) Put/Call-ratio ────────────────────
st.subheader("Put/Call-ratio per expiratie")
p = (df[liq_mask].groupby(["expiration","type"], as_index=False)
       .agg(vol=("volume","sum"), oi=("open_interest","sum")))
if p.empty:
    st.info("Geen data voor PCR.")
else:
    pv = (p.pivot_table(index="expiration", columns="type", values=["vol","oi"], aggfunc="sum")
            .sort_index().sort_index(axis=1)).fillna(0.0)
    pv.columns = [f"{a}_{b}" for a,b in pv.columns.to_flat_index()]
    for col in ["vol_put","vol_call","oi_put","oi_call"]:
        if col not in pv.columns: pv[col] = 0.0
    pv["PCR_vol"] = pv["vol_put"] / pv["vol_call"].replace(0, np.nan)
    pv["PCR_oi"]  = pv["oi_put"]  / pv["oi_call"].replace(0, np.nan)
    pv = pv.replace([np.inf,-np.inf], np.nan).dropna(subset=["PCR_vol","PCR_oi"], how="all")
    if pv.empty:
        st.info("Niet genoeg data (alleen puts of alleen calls in de selectie). Probeer andere filters/expiraties.")
    else:
        fig_pcr = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                                subplot_titles=("PCR op Volume", "PCR op Open Interest"))
        fig_pcr.add_trace(go.Bar(x=pv.index, y=pv["PCR_vol"], name="PCR (Vol)"), row=1, col=1)
        fig_pcr.add_trace(go.Bar(x=pv.index, y=pv["PCR_oi"],  name="PCR (OI)"),  row=2, col=1)
        fig_pcr.update_layout(height=520, title_text="Put/Call-ratio per Expiratie", dragmode="zoom")
        st.plotly_chart(fig_pcr, use_container_width=True, config=PLOTLY_CONFIG)
        st.caption("**Interpretatie:** PCR↑ → defensiever; PCR↓ → meer call-speculatie. Gebruik dit om je strangle-afstand/DTE conservatiever of agressiever te kiezen.")

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
dte_selected = int(pd.to_numeric(df_last[df_last["expiration"]==default_series_exp]["days_to_exp"], errors="coerce").median()) if not df_last.empty else 30
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
cm1, cm2, cm3, cm4 = st.columns([1.2, 1, 1, 1])
with cm1:  str_sel_mode = st.radio("Selectiemodus", ["σ-doel", "Δ-doel"], index=0)
with cm2:  sigma_target = st.slider("σ-doel per zijde", 0.5, 2.5, 1.0, step=0.1)
with cm3:  delta_target = st.slider("Δ-doel (absoluut)", 0.05, 0.30, 0.15, step=0.01)
with cm4:  price_source = st.radio("Prijsbron", ["mid_price","last_price"], index=0, horizontal=True)

ce1, ce2, ce3 = st.columns([1.2, 1, 1])
with ce1:
    default_exp_idx = exps_all.index(default_series_exp) if exps_all and (default_series_exp in exps_all) else 0
    exp_for_str = st.selectbox("Expiratie voor strangle", options=exps_all or [], index=default_exp_idx if exps_all else 0)
with ce2:
    use_smile_iv = st.checkbox("Gebruik strike-IV (smile) voor Δ", value=False)
with ce3:
    show_table = st.checkbox("Toon details tabel", value=False)

@st.cache_data(ttl=300, show_spinner=False)
def load_strangle_slice(expiration, snap_min):
    if expiration is None or snap_min is None:
        return pd.DataFrame()
    sql = f"""
      SELECT TIMESTAMP_TRUNC(snapshot_date, MINUTE) AS snap_m, snapshot_date, type, expiration,
             days_to_exp, strike, underlying_price, implied_volatility, open_interest,
             volume, last_price, mid_price
      FROM `{VIEW}`
      WHERE expiration = @exp AND DATE(snapshot_date) = DATE(@snap)
    """
    all_rows = run_query(sql, {"exp": expiration, "snap": pd.to_datetime(snap_min)})
    if all_rows.empty: return all_rows
    all_rows["snap_m"] = pd.to_datetime(all_rows["snap_m"])
    target = pd.to_datetime(snap_min)
    best_minute = all_rows.loc[(all_rows["snap_m"]-target).abs().idxmin(), "snap_m"]
    return all_rows[all_rows["snap_m"] == best_minute].copy()

df_str = load_strangle_slice(exp_for_str if 'exp_for_str' in locals() else None, sel_snapshot if 'sel_snapshot' in locals() else None)
if not df_str.empty:
    df_str["type"] = df_str["type"].str.lower()
    df_str["mny"] = df_str["strike"]/df_str["underlying_price"] - 1.0
    df_str = df_str[((df_str["open_interest"]>=min_oi) | (df_str["volume"]>=min_vol))]

iv_atm_exp = float(df_str.loc[(df_str["days_to_exp"].between(20,60)) & (df_str["mny"].abs()<=0.01),"implied_volatility"].median()) if not df_str.empty else np.nan
dte_exp = int(pd.to_numeric(df_str["days_to_exp"], errors="coerce").median()) if not df_str.empty else np.nan
T = max(dte_exp,1)/365.0 if not np.isnan(dte_exp) else np.nan
sigma_pts = underlying_now * iv_atm_exp * math.sqrt(T) if (not np.isnan(underlying_now) and not np.isnan(iv_atm_exp) and not np.isnan(T)) else np.nan
strike_iv_map = {}
if not df_str.empty:
    strike_iv_map = (df_str.groupby(["type","strike"], as_index=False)["implied_volatility"].median()
                          .set_index(["type","strike"])["implied_volatility"].to_dict())

def get_iv_for(side: str, K: float) -> float:
    v = strike_iv_map.get((side, K), np.nan) if use_smile_iv else np.nan
    return float(v) if not np.isnan(v) else float(iv_atm_exp)

def nearest_strike(side: str, target_price: float) -> float:
    s_list = sorted(df_str[df_str["type"]==side]["strike"].unique().tolist()) if not df_str.empty else []
    return pick_closest_value(s_list, target_price, fallback=(s_list[len(s_list)//2] if s_list else 6000.0))

# —— NIEUW: r/q instellingen voor Δ-berekening ——
q_const_simple = st.number_input("Dividendrendement q (p.j., simple)", min_value=0.0, max_value=0.10, value=0.016, step=0.001, format="%.3f")
_T = np.array([T if not np.isnan(T) else 30/365.0], dtype=float)  # veilige fallback
r_T = get_r_curve_for_snapshot(
    snapshot_date=pd.to_datetime(sel_snapshot) if 'sel_snapshot' in locals() else pd.Timestamp(date.today()),
    T_years=_T,
    view="nth-pier-468314-p7.marketdata.yield_curve_analysis_wide",
    compounding_in="simple",
    extrapolate=True
)[0]
q_T = get_q_curve_const(_T, q_const=q_const_simple, to_continuous=True)[0]

def pick_by_sigma():
    if np.isnan(sigma_pts): return np.nan, np.nan
    return nearest_strike("put",  underlying_now - sigma_target*sigma_pts), \
           nearest_strike("call", underlying_now + sigma_target*sigma_pts)

def pick_by_delta():
    if any(np.isnan(x) for x in [underlying_now, T]) or df_str.empty: return np.nan, np.nan
    puts  = df_str[df_str["type"]=="put"]["strike"].unique().tolist()
    calls = df_str[df_str["type"]=="call"]["strike"].unique().tolist()
    best_p, best_c, err_p, err_c = np.nan, np.nan, 1e9, 1e9
    for K in puts:
        d = bs_delta(underlying_now, K, get_iv_for("put", K), T, r_T, q_T, is_call=False)
        e = abs(abs(d) - delta_target)
        if not np.isnan(d) and e < err_p: best_p, err_p = K, e
    for K in calls:
        d = bs_delta(underlying_now, K, get_iv_for("call", K), T, r_T, q_T, is_call=True)
        e = abs(d - delta_target)
        if not np.isnan(d) and e < err_c: best_c, err_c = K, e
    return float(best_p), float(best_c)

ac1, ac2 = st.columns([1, 1])
with ac1:
    if st.button("🔮 Auto-pick (σ-doel)"): str_sel_mode = "σ-doel"
with ac2:
    if st.button("🎯 Auto-pick (Δ-doel)"): str_sel_mode = "Δ-doel"

target_put, target_call = (pick_by_sigma() if str_sel_mode.startswith("σ") else pick_by_delta())

def _val(row, col): return float(pd.to_numeric(row[col], errors="coerce").median()) if (not row.empty and col in row) else np.nan
put_row  = df_str[(df_str["type"]=="put")  & (df_str["strike"]==target_put)].copy()  if not (np.isnan(target_put) or df_str.empty)  else pd.DataFrame()
call_row = df_str[(df_str["type"]=="call") & (df_str["strike"]==target_call)].copy() if not (np.isnan(target_call) or df_str.empty) else pd.DataFrame()
put_px, call_px = _val(put_row, price_source), _val(call_row, price_source)
total_credit = (put_px + call_px) if (not np.isnan(put_px) and not np.isnan(call_px)) else np.nan

def sigma_distance(K: float) -> float: return abs(K - underlying_now) / sigma_pts if not np.isnan(sigma_pts) else np.nan
sd_put, sd_call = sigma_distance(target_put), sigma_distance(target_call)
def p_itm_at_exp(sd: float) -> float: return (1.0 - norm_cdf(sd)) if not np.isnan(sd) else np.nan
p_touch_put  = min(1.0, 2.0 * p_itm_at_exp(sd_put))  if not np.isnan(sd_put)  else np.nan
p_touch_call = min(1.0, 2.0 * p_itm_at_exp(sd_call)) if not np.isnan(sd_call) else np.nan
p_both_touch_approx = min(1.0, (p_touch_put if not np.isnan(p_touch_put) else 0.0) + (p_touch_call if not np.isnan(p_touch_call) else 0.0))
ppd_total_pts = float(total_credit / max(dte_exp,1)) if not np.isnan(total_credit) and not np.isnan(dte_exp) else np.nan

km1, km2, km3, km4, km5, km6 = st.columns(6)
with km1: st.metric("Expiratie", str(exp_for_str) if 'exp_for_str' in locals() else "—")
with km2: st.metric("DTE", f"{dte_exp:.0f}" if not np.isnan(dte_exp) else "—")
with km3: st.metric("Strikes", (f"P {target_put:.0f} / C {target_call:.0f}") if not (np.isnan(target_put) or np.isnan(target_call)) else "—")
with km4: st.metric("Credit", f"{total_credit:,.2f}" if not np.isnan(total_credit) else "—")
with km5: st.metric("PPD (tot.)", f"{ppd_total_pts:,.2f}" if not np.isnan(ppd_total_pts) else "—")
with km6: st.metric("~P(touch) max", f"{p_both_touch_approx*100:.0f}%" if not np.isnan(p_both_touch_approx) else "—")
if show_table and not df_str.empty:
    st.dataframe(df_str.sort_values(["type","strike"])[["type","strike","implied_volatility","open_interest","volume","last_price","mid_price"]],
                 use_container_width=True)

# mini-conclusies
verdict = []
if not np.isnan(iv_rank):
    verdict.append("IV-Rank **hoog** → premie-verkopen aantrekkelijker; events checken." if iv_rank >= 0.7
                   else "IV-Rank **laag** → premie dunner; overweeg smaller/kalendar.")
if not any(np.isnan(x) for x in [iv_atm, hv20]) and (iv_atm - hv20) > 0.0:
    verdict.append("**VRP** positief → kans op mean-reversion van vol (goed voor short premium).")
if not np.isnan(p_both_touch_approx):
    verdict.append("~**P(touch)** beide zijden ≲40% → comfortabele band." if p_both_touch_approx <= 0.40
                   else "~**P(touch)** hoog → kies grotere σ-afstand / langere DTE.")
if verdict: st.markdown("- " + "\n- ".join(verdict))

# ─────────────────────────── J) Sizing & Payoff ───────────────────
ready_for_sizing = (not any(np.isnan(x) for x in [underlying_now])) and \
                   (not np.isnan(target_put)) and (not np.isnan(target_call))
st.markdown("### 💳 Margin & Sizing")
if not ready_for_sizing:
    st.info("Kies eerst **strikes** in de *Strangle Helper* (σ of Δ).")
else:
    sm1, sm2, sm3, sm4 = st.columns([1.1, 1, 1, 1])
    with sm1: margin_model = st.radio("Margin model", ["SPAN-like stress", "Reg-T approx"], index=0)
    with sm2: down_shock = st.slider("Down shock (%)", 5, 30, 15, step=1)
    with sm3: up_shock   = st.slider("Up shock (%)", 5, 30, 10, step=1)
    with sm4: multiplier = st.number_input("Contract multiplier", min_value=10, max_value=250, value=100, step=10)

    sb1, sb2, sb3 = st.columns([1, 1, 1])
    with sb1: risk_budget = st.number_input("Max risico budget €/$", min_value=1000.0, value=10000.0, step=1000.0, format="%.0f")
    with sb2: show_payoff = st.checkbox("Toon payoff (1x)", value=True)
    with sb3: pass

    if margin_model.startswith("SPAN"):
        est_margin = span_like_margin(underlying_now, float(target_put), float(target_call),
                                      float(total_credit) if not np.isnan(total_credit) else 0.0,
                                      down=down_shock/100.0, up=up_shock/100.0, multiplier=multiplier)
    else:
        call_px_pts = float(call_px) if not np.isnan(call_px) else 0.0
        put_px_pts  = float(put_px)  if not np.isnan(put_px)  else 0.0
        est_margin = regt_strangle_margin(underlying_now, float(target_put), float(target_call),
                                          put_px_pts, call_px_pts, multiplier=multiplier)

    n_contracts = int(np.floor(risk_budget / est_margin)) if (est_margin and est_margin > 0) else 0
    tot_credit_cash = (float(total_credit) if not np.isnan(total_credit) else 0.0) * multiplier
    credit_per_margin = (tot_credit_cash / est_margin) if est_margin > 0 else np.nan
    ppd_total = (float(total_credit) / max(dte_exp,1)) if (not np.isnan(dte_exp) and not np.isnan(total_credit)) else np.nan
    ppd_per_margin = ((ppd_total * multiplier) / est_margin) if (est_margin > 0 and not np.isnan(ppd_total)) else np.nan

    mm1, mm2, mm3, mm4 = st.columns(4)
    with mm1: st.metric("Est. margin (1x)", f"{est_margin:,.0f}")
    with mm2: st.metric("# Contracts @budget", f"{n_contracts:,}")
    with mm3: st.metric("Credit (1x)", f"{tot_credit_cash:,.0f}")
    with mm4: st.metric("Credit / Margin", f"{credit_per_margin:.2f}" if not np.isnan(credit_per_margin) else "—")
    mm5, mm6 = st.columns(2)
    with mm5: st.metric("PPD (pts, 1x)", f"{ppd_total:,.2f}" if not np.isnan(ppd_total) else "—")
    with mm6: st.metric("PPD / Margin", f"{ppd_per_margin:.4f}" if not np.isnan(ppd_per_margin) else "—")

    if show_payoff:
        rng = 0.25
        S_grid = np.linspace(underlying_now*(1-rng), underlying_now*(1+rng), 400)
        pnl_grid = [strangle_payoff_at_expiry(s, float(target_put), float(target_call), float(total_credit), multiplier=multiplier) for s in S_grid]
        be_low  = float(target_put)  - float(total_credit)
        be_high = float(target_call) + float(total_credit)
        fig_pay = go.Figure()
        fig_pay.add_trace(go.Scatter(x=S_grid, y=pnl_grid, mode="lines", name="PNL @ expiry (1x)"))
        fig_pay.add_hline(y=0, line=dict(dash="dot"))
        fig_pay.add_vline(x=be_low,  line=dict(dash="dot"), annotation_text=f"BE low ≈ {be_low:.0f}")
        fig_pay.add_vline(x=be_high, line=dict(dash="dot"), annotation_text=f"BE high ≈ {be_high:.0f}")
        fig_pay.update_layout(title=f"Payoff @ Expiry (P {target_put:.0f} / C {target_call:.0f} | credit {total_credit:.2f} pts)",
                              xaxis_title="S (onderliggende)", yaxis_title=f"PNL (1x, multiplier {multiplier})",
                              height=420, dragmode="zoom")
        st.plotly_chart(fig_pay, use_container_width=True, config=PLOTLY_CONFIG)

# ─────────────────────────── K) Roll-sim ──────────────────────────
st.markdown("### 🔄 Roll-simulator (uitrollen / herpositioneren)")
if not ready_for_sizing or df_str.empty:
    st.info("Selecteer eerst een strangle in de *Strangle Helper*. Daarna kun je rollen simuleren.")
else:
    rr1, rr2, rr3 = st.columns([1.2, 1.0, 1.0])
    with rr1: roll_mode = st.radio("Rol-methode", ["σ-doel", "Δ-doel"], index=0, horizontal=True)
    with rr2: sigma_target_roll = st.slider("σ-doel (roll)", 0.5, 2.5, 1.2, step=0.1)
    with rr3: delta_target_roll = st.slider("Δ-doel (roll, abs)", 0.05, 0.30, 0.15, step=0.01)
    future_exps = [e for e in exps_all if e > exp_for_str] if 'exp_for_str' in locals() else []
    if not future_exps:
        st.info("Geen latere expiraties beschikbaar binnen de filters om naar uit te rollen.")
    else:
        new_exp = st.selectbox("Naar welke expiratie rollen?", options=future_exps, index=0)
        @st.cache_data(ttl=300, show_spinner=False)
        def load_slice_for_exp(expiration, snap_min):
            if expiration is None or snap_min is None: return pd.DataFrame()
            sql = f"""
              SELECT TIMESTAMP_TRUNC(snapshot_date, MINUTE) AS snap_m, snapshot_date, type, expiration,
                     days_to_exp, strike, underlying_price, implied_volatility, open_interest,
                     volume, last_price, mid_price
              FROM `{VIEW}`
              WHERE expiration = @exp AND DATE(snapshot_date) = DATE(@snap)
            """
            all_rows = run_query(sql, {"exp": expiration, "snap": pd.to_datetime(snap_min)})
            if all_rows.empty: return all_rows
            all_rows["snap_m"] = pd.to_datetime(all_rows["snap_m"])
            target = pd.to_datetime(snap_min)
            best_minute = all_rows.loc[(all_rows["snap_m"]-target).abs().idxmin(), "snap_m"]
            return all_rows[all_rows["snap_m"]==best_minute].copy()
        df_new = load_slice_for_exp(new_exp, sel_snapshot)
        if df_new.empty:
            st.info("Geen data voor de gekozen nieuwe expiratie op dit snapshot.")
        else:
            df_new["type"] = df_new["type"].str.lower()
            df_new["mny"] = df_new["strike"]/df_new["underlying_price"] - 1.0
            df_new = df_new[((df_new["open_interest"]>=min_oi) | (df_new["volume"]>=min_vol))]
            dte_new = int(pd.to_numeric(df_new["days_to_exp"], errors="coerce").median())
            iv_atm_new = float(df_new.loc[(df_new["days_to_exp"].between(20,60)) & (df_new["mny"].abs()<=0.01),"implied_volatility"].median())
            T_new = max(dte_new,1)/365.0 if not np.isnan(dte_new) else np.nan
            sigma_pts_new = underlying_now * iv_atm_new * math.sqrt(T_new) if (not np.isnan(underlying_now) and not np.isnan(iv_atm_new) and not np.isnan(T_new)) else np.nan
            smile_map_new = (df_new.groupby(["type","strike"], as_index=False)["implied_volatility"].median()
                                   .set_index(["type","strike"])["implied_volatility"].to_dict())

            # —— NIEUW: r/q voor nieuwe expiratie (yield-curve) ——
            _Tn = np.array([T_new if not np.isnan(T_new) else 30/365.0], dtype=float)
            r_T_new = get_r_curve_for_snapshot(
                snapshot_date=pd.to_datetime(sel_snapshot) if 'sel_snapshot' in locals() else pd.Timestamp(date.today()),
                T_years=_Tn,
                view="nth-pier-468314-p7.marketdata.yield_curve_analysis_wide",
                compounding_in="simple",
                extrapolate=True
            )[0]
            q_T_new = get_q_curve_const(_Tn, q_const=q_const_simple, to_continuous=True)[0]

            def get_iv_new(side: str, K: float) -> float:
                v = smile_map_new.get((side, K), np.nan); return float(v) if not np.isnan(v) else float(iv_atm_new)
            def nearest_strike_new(side: str, target_price: float) -> float:
                arr = sorted(df_new[df_new["type"]==side]["strike"].unique().tolist())
                return pick_closest_value(arr, target_price, fallback=(arr[len(arr)//2] if arr else 6000.0))

            if roll_mode.startswith("σ"):
                new_put  = nearest_strike_new("put",  underlying_now - sigma_target_roll * sigma_pts_new)
                new_call = nearest_strike_new("call", underlying_now + sigma_target_roll * sigma_pts_new)
            else:
                puts  = sorted(df_new[df_new["type"]=="put"]["strike"].unique().tolist())
                calls = sorted(df_new[df_new["type"]=="call"]["strike"].unique().tolist())
                best_p, best_c, err_p, err_c = np.nan, np.nan, 1e9, 1e9
                for K in puts:
                    d = bs_delta(underlying_now, K, get_iv_new("put", K), T_new, r_T_new, q_T_new, is_call=False)
                    e = abs(abs(d) - delta_target_roll)
                    if not np.isnan(d) and e < err_p: best_p, err_p = K, e
                for K in calls:
                    d = bs_delta(underlying_now, K, get_iv_new("call", K), T_new, r_T_new, q_T_new, is_call=True)
                    e = abs(d - delta_target_roll)
                    if not np.isnan(d) and e < err_c: best_c, err_c = K, e
                new_put, new_call = float(best_p), float(best_c)

            def _p(df_leg, typ, K):
                row = df_leg[(df_leg["type"]==typ) & (df_leg["strike"]==K)]
                return float(pd.to_numeric(row[price_source], errors="coerce").median()) if not row.empty else np.nan

            new_put_px, new_call_px = _p(df_new,"put",new_put), _p(df_new,"call",new_call)
            new_credit = (new_put_px + new_call_px) if (not np.isnan(new_put_px) and not np.isnan(new_call_px)) else np.nan
            close_cost = (float(put_px) if not np.isnan(put_px) else 0.0) + (float(call_px) if not np.isnan(call_px) else 0.0)
            net_roll_credit = (new_credit - close_cost) if (not np.isnan(new_credit)) else np.nan

            def sigma_dist(K, sp): return abs(K - underlying_now) / sp if (sp and sp>0 and not np.isnan(sp)) else np.nan
            old_sd_put, old_sd_call = sigma_dist(float(target_put), sigma_pts_new), sigma_dist(float(target_call), sigma_pts_new)
            new_sd_put, new_sd_call = sigma_dist(float(new_put),  sigma_pts_new), sigma_dist(float(new_call),  sigma_pts_new)

            r1, r2, r3, r4 = st.columns(4)
            with r1: st.metric("Nieuwe exp.", str(new_exp))
            with r2: st.metric("Nieuwe strikes", f"P {new_put:.0f} / C {new_call:.0f}")
            with r3: st.metric("Extra credit (roll)", f"{net_roll_credit:,.2f}" if not np.isnan(net_roll_credit) else "—")
            with r4: st.metric("DTE (nieuw)", f"{dte_new:.0f}")
            r5, r6 = st.columns(2)
            with r5: st.metric("σ-afstand PUT (oud → nieuw)", f"{old_sd_put:.2f}σ → {new_sd_put:.2f}σ" if not (np.isnan(old_sd_put) or np.isnan(new_sd_put)) else "—")
            with r6: st.metric("σ-afstand CALL (oud → nieuw)", f"{old_sd_call:.2f}σ → {new_sd_call:.2f}σ" if not (np.isnan(old_sd_call) or np.isnan(new_sd_call)) else "—")

# ─────────────────────────── L) VIX vs IV ─────────────────────────
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
    fig_vix.update_layout(height=620, title_text=f"VIX vs IV ({sel_type.upper()})", dragmode="zoom")
    st.plotly_chart(fig_vix, use_container_width=True, config=PLOTLY_CONFIG)

# ─────────────────────────── Footer ───────────────────────────────
st.caption("🔍 Navigatie: zoom met scroll/pinch, **double-click** om automatisch te rescalen. Filters voor **OI/Volume** helpen spikes te verwijderen en geven realistischer curves.")
