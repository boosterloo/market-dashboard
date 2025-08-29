# pages/3_SPX_Options.py

import streamlit as st
import pandas as pd
import numpy as np
import math
from datetime import datetime, timedelta, date
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from google.cloud import bigquery
from google.oauth2 import service_account

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

def bs_delta(S: float, K: float, iv: float, T_years: float, is_call: bool) -> float:
    if any(x is None or np.isnan(x) or x <= 0 for x in [S, K]) or np.isnan(iv) or T_years <= 0:
        return np.nan
    d1 = (math.log(S / K) + 0.5 * iv * iv * T_years) / (iv * math.sqrt(T_years))
    Nd1 = norm_cdf(d1)
    return Nd1 if is_call else (Nd1 - 1.0)

# ─────────────────────────── BigQuery ─────────────────────────────
def get_bq_client():
    sa_info = st.secrets["gcp_service_account"]
    creds = service_account.Credentials.from_service_account_info(sa_info)
    project_id = st.secrets.get("PROJECT_ID", sa_info.get("project_id"))
    return bigquery.Client(project=project_id, credentials=creds)

_bq_client = get_bq_client()

def _bq_param(name, value):
    if isinstance(value, bool):
        return bigquery.ScalarQueryParameter(name, "BOOL", value)
    if isinstance(value, (int, np.integer)):
        return bigquery.ScalarQueryParameter(name, "INT64", int(value))
    if isinstance(value, (float, np.floating)):
        return bigquery.ScalarQueryParameter(name, "FLOAT64", float(value))
    if isinstance(value, datetime):
        return bigquery.ScalarQueryParameter(name, "TIMESTAMP", value)
    if isinstance(value, (date, pd.Timestamp)):
        return bigquery.ScalarQueryParameter(name, "DATE", str(pd.to_datetime(value).date()))
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

PLOTLY_CONFIG = {
    "scrollZoom": True,
    "doubleClick": "reset",
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"]
}

# ─────────────────────────── Data boundaries ──────────────────────
@st.cache_data(ttl=600, show_spinner=False)
def load_bounds():
    df = run_query(f"SELECT MIN(DATE(snapshot_date)) mn, MAX(DATE(snapshot_date)) mx FROM `{VIEW}`")
    mn, mx = pd.to_datetime(df["mn"].iloc[0]).date(), pd.to_datetime(df["mx"].iloc[0]).date()
    return mn, mx

min_date, max_date = load_bounds()
default_start = max(min_date, max_date - timedelta(days=365))

start_date, end_date = st.date_input(
    "Periode (snapshot_date)",
    value=(default_start, max_date),
    min_value=min_date,
    max_value=max_date,
    format="YYYY-MM-DD"
)

# ─────────────────────────── Snapshots & Expirations ──────────────
@st.cache_data(ttl=600, show_spinner=False)
def load_snapshots():
    sql = f"SELECT DISTINCT TIMESTAMP_TRUNC(snapshot_date, HOUR) snap FROM `{VIEW}` ORDER BY snap"
    df = run_query(sql)
    return sorted(pd.to_datetime(df["snap"]).dt.to_pydatetime())

snapshots_all = load_snapshots()
default_snap = snapshots_all[-1] if snapshots_all else None
sel_snapshot = st.selectbox("Snapshot-datum", snapshots_all, index=len(snapshots_all)-1 if snapshots_all else 0)

@st.cache_data(ttl=600, show_spinner=False)
def load_expirations():
    sql = f"SELECT DISTINCT expiration FROM `{VIEW}` ORDER BY expiration"
    df = run_query(sql)
    return sorted(pd.to_datetime(df["expiration"]).dt.date.tolist())

exps_all = load_expirations()
target_exp = date.today() + timedelta(days=14)
default_exp = pick_first_on_or_after(exps_all, target_exp) or pick_closest_date(exps_all, target_exp)
sel_exp = st.selectbox("Expiratie", exps_all, index=exps_all.index(default_exp) if default_exp in exps_all else 0)

# ─────────────────────────── Filters ──────────────────────────────
sel_type = st.radio("Type", ["call", "put"], index=1, horizontal=True)
dte_range = st.slider("DTE (days)", 0, 365, (0, 60), step=1)
mny_range = st.slider("Moneyness (K/S − 1)", -0.20, 0.20, (-0.10, 0.10), step=0.01)

min_oi = st.slider("Min Open Interest", 0, 50, 1)
min_vol = st.slider("Min Volume", 0, 50, 1)
min_per_bin = st.slider("Min per bin (aggregatie)", 1, 10, 3)

# ─────────────────────────── Serie-selectie ──────────────────────
st.subheader("Serie-selectie — volg één optiereeks door de tijd")

@st.cache_data(ttl=600, show_spinner=True)
def load_series(snap: datetime, exp: date, typ: str, strike: float):
    sql = f"""
      SELECT snapshot_date, strike, type, expiration,
             last_price, mid_price, ppd, underlying_price, volume, open_interest
      FROM `{VIEW}`
      WHERE TIMESTAMP_TRUNC(snapshot_date, HOUR)=@snap
        AND expiration=@exp
        AND LOWER(type)=@typ
        AND strike=@strike
    """
    return run_query(sql, {"snap": snap, "exp": exp, "typ": typ, "strike": strike})

# default strike kiezen rond onderliggende prijs
@st.cache_data(ttl=600, show_spinner=True)
def pick_default_strike(exp: date, typ: str):
    sql = f"""
      SELECT APPROX_QUANTILES(strike, 100)[OFFSET(50)] AS median_strike
      FROM `{VIEW}`
      WHERE expiration=@exp AND LOWER(type)=@typ
    """
    df = run_query(sql, {"exp": exp, "typ": typ})
    return float(df["median_strike"].iloc[0]) if not df.empty else 6000.0

default_strike = pick_default_strike(sel_exp, sel_type)
sel_strike = st.number_input("Strike", value=default_strike, step=25.0)

serie = load_series(sel_snapshot, sel_exp, sel_type, sel_strike)
if serie.empty:
    st.info("Geen data voor deze serie.")
else:
    fig_price = make_subplots(specs=[[{"secondary_y": True}]])
    fig_price.add_trace(go.Scatter(x=serie["snapshot_date"], y=serie["last_price"],
                                   mode="lines+markers", name="Price"), secondary_y=False)
    fig_price.add_trace(go.Scatter(x=serie["snapshot_date"], y=serie["underlying_price"],
                                   mode="lines", line=dict(dash="dot"), name="SP500"), secondary_y=True)
    fig_price.update_layout(title=f"{sel_type.upper()} {sel_strike} exp {sel_exp} — Price vs SP500",
                            height=420, hovermode="x unified")
    st.plotly_chart(fig_price, use_container_width=True, config=PLOTLY_CONFIG)

    fig_ppd = make_subplots(specs=[[{"secondary_y": True}]])
    fig_ppd.add_trace(go.Scatter(x=serie["snapshot_date"], y=serie["ppd"],
                                 mode="lines+markers", name="PPD"), secondary_y=False)
    fig_ppd.add_trace(go.Scatter(x=serie["snapshot_date"], y=serie["underlying_price"],
                                 mode="lines", line=dict(dash="dot"), name="SP500"), secondary_y=True)
    fig_ppd.update_layout(title=f"{sel_type.upper()} {sel_strike} exp {sel_exp} — PPD vs SP500",
                          height=420, hovermode="x unified")
    st.plotly_chart(fig_ppd, use_container_width=True, config=PLOTLY_CONFIG)
# ─────────────────────────── PPD vs Afstand ──────────────────────
st.subheader("PPD vs Afstand tot Uitoefenprijs")

@st.cache_data(ttl=600, show_spinner=True)
def load_snapshot_data(snap: datetime, exp: date, typ: str):
    sql = f"""
      SELECT snapshot_date, strike, expiration, type, days_to_exp,
             underlying_price, ppd, open_interest, volume
      FROM `{VIEW}`
      WHERE TIMESTAMP_TRUNC(snapshot_date, HOUR)=@snap
        AND expiration=@exp
        AND LOWER(type)=@typ
    """
    return run_query(sql, {"snap": snap, "exp": exp, "typ": typ})

df_last = load_snapshot_data(sel_snapshot, sel_exp, sel_type)
if df_last.empty:
    st.info("Geen data voor dit snapshot.")
else:
    df_last["dist_pct"] = (df_last["strike"] - df_last["underlying_price"]) / df_last["underlying_price"] * 100
    bins = np.arange(-20, 21, 1)
    df_last["bin"] = pd.cut(df_last["dist_pct"], bins)
    g = df_last.groupby("bin")["ppd"].median().reset_index()
    g["mid"] = g["bin"].apply(lambda b: b.mid if pd.notna(b) else np.nan)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=g["mid"], y=g["ppd"], mode="lines+markers", name="PPD"))
    fig.add_vline(x=0, line=dict(dash="dot"), annotation_text="ATM")
    fig.update_layout(title="PPD vs Afstand (laatste snapshot)", xaxis_title="Afstand (%)", yaxis_title="PPD",
                      height=420)
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

# ─────────────────────────── PPD vs DTE ──────────────────────────
st.subheader("PPD vs DTE")

ppd_mode = st.radio("Bereik", ["ATM-band", "Rond strike"], index=0,
                    help="ATM-band = strikes binnen ±% van spot. Rond strike = venster rond gekozen strike.")
atm_band = st.slider("ATM-band (± %)", 0.0, 0.10, 0.02, 0.01)
strike_win = st.slider("Strike venster ± (punten)", 0, 500, 100, 25)
robust = st.checkbox("Robust scale (95e pct)", value=True)

if ppd_mode.startswith("ATM"):
    df_ppd = df_last[np.abs(df_last["strike"]/df_last["underlying_price"] - 1) <= atm_band].copy()
else:
    df_ppd = df_last[(df_last["strike"] >= sel_strike - strike_win) & (df_last["strike"] <= sel_strike + strike_win)].copy()

if df_ppd.empty:
    st.info("Geen data voor deze selectie.")
else:
    g = df_ppd.groupby("days_to_exp")["ppd"].median().reset_index().sort_values("days_to_exp")
    y = g["ppd"].to_numpy()
    if robust and len(y) > 3:
        lo, hi = np.nanpercentile(y, [5,95]); pad = (hi-lo)*0.2
        ymin, ymax = max(0,lo-pad), hi+pad
    else:
        ymin, ymax = None, None

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=g["days_to_exp"], y=g["ppd"], mode="lines+markers", name="PPD"))
    fig.update_layout(title="PPD vs Days To Expiration", xaxis_title="DTE", yaxis_title="PPD", height=420)
    if ymin is not None: fig.update_yaxes(range=[ymin,ymax])
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

# ─────────────────────────── Matrix meetmoment × strike ───────────
st.subheader("Matrix — meetmoment × strike")
max_rows = st.slider("Max. meetmomenten", 50, 500, 200, 50)

@st.cache_data(ttl=600, show_spinner=True)
def load_matrix(exp: date, typ: str, max_rows: int):
    sql = f"""
      SELECT TIMESTAMP_TRUNC(snapshot_date,HOUR) snap, strike, ppd
      FROM `{VIEW}`
      WHERE expiration=@exp AND LOWER(type)=@typ
      ORDER BY snap DESC
      LIMIT @lim
    """
    return run_query(sql, {"exp": exp, "typ": typ, "lim": max_rows})

mx = load_matrix(sel_exp, sel_type, max_rows)
if mx.empty:
    st.info("Geen matrix-data.")
else:
    pivot = mx.pivot_table(index="snap", columns="strike", values="ppd", aggfunc="median")
    fig_mx = go.Figure(go.Heatmap(z=pivot.values, x=pivot.columns, y=pivot.index, colorbar_title="PPD"))
    fig_mx.update_layout(height=520, title="Heatmap PPD", xaxis_title="Strike", yaxis_title="Snapshot")
    st.plotly_chart(fig_mx, use_container_width=True, config=PLOTLY_CONFIG)
    st.dataframe(pivot.round(3), use_container_width=True)

# ─────────────────────────── IV Term structure ────────────────────
st.subheader("IV Term Structure")
term = df_last.groupby("days_to_exp")["implied_volatility"].median().reset_index()
fig_term = go.Figure(go.Scatter(x=term["days_to_exp"], y=term["implied_volatility"],
                                mode="lines+markers", name="IV"))
fig_term.update_layout(height=380, title="Term Structure — median IV", xaxis_title="DTE", yaxis_title="IV")
st.plotly_chart(fig_term, use_container_width=True, config=PLOTLY_CONFIG)

# ─────────────────────────── IV Smile ─────────────────────────────
st.subheader("IV Smile")
smile = df_last.groupby("strike")["implied_volatility"].median().reset_index().sort_values("strike")
fig_sm = go.Figure(go.Scatter(x=smile["strike"], y=smile["implied_volatility"], mode="lines+markers"))
fig_sm.update_layout(height=420, title=f"IV Smile — {sel_type.upper()} {sel_exp}", xaxis_title="Strike", yaxis_title="IV")
st.plotly_chart(fig_sm, use_container_width=True, config=PLOTLY_CONFIG)

# ─────────────────────────── Put/Call Ratio ───────────────────────
st.subheader("Put/Call-ratio per expiratie")

@st.cache_data(ttl=600, show_spinner=True)
def load_pcr(start: date, end: date):
    sql = f"""
      SELECT expiration, type, SUM(volume) vol, SUM(open_interest) oi
      FROM `{VIEW}`
      WHERE DATE(snapshot_date) BETWEEN @s AND @e
      GROUP BY expiration, type
    """
    return run_query(sql, {"s": start, "e": end})

p = load_pcr(start_date, end_date)
if p.empty:
    st.info("Geen PCR-data.")
else:
    pv = p.pivot(index="expiration", columns="type", values="oi").fillna(0)
    pv["PCR"] = pv.get("put",0)/pv.get("call",1)
    fig_pcr = go.Figure(go.Bar(x=pv.index, y=pv["PCR"]))
    fig_pcr.update_layout(height=400, title="Put/Call Ratio (OI)", xaxis_title="Expiratie", yaxis_title="PCR")
    st.plotly_chart(fig_pcr, use_container_width=True, config=PLOTLY_CONFIG)
# ─────────────────────────── Vol & Risk ───────────────────────────
st.markdown("### 📊 Vol & Risk (ATM-IV, HV, VRP, IV-Rank, Expected Move)")

# Dagelijkse onderliggende (close per dag) voor HV
@st.cache_data(ttl=900, show_spinner=False)
def load_underlying_daily(start: date, end: date):
    sql = f"""
      WITH d AS (
        SELECT DATE(snapshot_date) d, ANY_VALUE(underlying_price) AS close
        FROM `{VIEW}`
        WHERE DATE(snapshot_date) BETWEEN @s AND @e
        GROUP BY d
      )
      SELECT d AS date, close FROM d ORDER BY date
    """
    return run_query(sql, {"s": start, "e": end})

u = load_underlying_daily(max(start_date, end_date - timedelta(days=365)), end_date)
if not u.empty:
    u["ret"] = pd.to_numeric(u["close"], errors="coerce").pct_change()
    hv20 = float(u["ret"].tail(21).std(ddof=0) * np.sqrt(252)) if u["ret"].notna().sum() >= 2 else np.nan
else:
    hv20 = np.nan

# ATM-IV rond 30D op het gekozen snapshot
if 'df_last' in locals() and not df_last.empty:
    df_last = df_last.copy()
    df_last["mny"] = df_last["strike"]/df_last["underlying_price"] - 1.0
    iv_atm = float(df_last.loc[(df_last["days_to_exp"].between(20, 40)) & (df_last["mny"].abs() <= 0.01),
                               "implied_volatility"].median())
    underlying_now = float(df_last["underlying_price"].median())
else:
    iv_atm, underlying_now = np.nan, np.nan

# IV-Rank (1y) gebaseerd op dag-medianen ATM-achtig (20-40D & |mny|<=1%)
@st.cache_data(ttl=900, show_spinner=False)
def load_iv_hist_for_rank():
    sql = f"""
      WITH base AS (
        SELECT DATE(snapshot_date) d, implied_volatility,
               SAFE_DIVIDE(strike, NULLIF(underlying_price,0)) - 1 AS mny,
               days_to_exp
        FROM `{VIEW}`
      ),
      filt AS (
        SELECT d, implied_volatility
        FROM base
        WHERE days_to_exp BETWEEN 20 AND 40 AND ABS(mny) <= 0.01
      )
      SELECT d AS date, MEDIAN(implied_volatility) AS iv
      FROM filt
      GROUP BY date
      ORDER BY date
    """
    return run_query(sql)

iv_hist = load_iv_hist_for_rank()
if not iv_hist.empty:
    iv_1y = pd.to_numeric(iv_hist["iv"], errors="coerce").tail(252).dropna()
    iv_rank = float((iv_1y <= iv_1y.iloc[-1]).mean()) if len(iv_1y) else np.nan
else:
    iv_rank = np.nan

# Expected move (σ) over median DTE in snapshot
if 'df_last' in locals() and not df_last.empty:
    dte_selected = int(pd.to_numeric(df_last["days_to_exp"], errors="coerce").median())
    em_sigma = (underlying_now * iv_atm * math.sqrt(max(dte_selected, 1)/365.0)
                if (not np.isnan(underlying_now) and not np.isnan(iv_atm)) else np.nan)
else:
    dte_selected, em_sigma = np.nan, np.nan

cv1, cv2, cv3, cv4, cv5 = st.columns(5)
with cv1: st.metric("ATM-IV (~30D)", f"{iv_atm:.2%}" if not np.isnan(iv_atm) else "—")
with cv2: st.metric("HV20", f"{hv20:.2%}" if not np.isnan(hv20) else "—")
with cv3: st.metric("VRP (IV−HV)", f"{(iv_atm - hv20):.2%}" if (not np.isnan(iv_atm) and not np.isnan(hv20)) else "—")
with cv4: st.metric("IV-Rank (1y)", f"{iv_rank*100:.0f}%" if not np.isnan(iv_rank) else "—")
with cv5:
    em_txt = f"±{em_sigma:,.0f} pts ({em_sigma/underlying_now:.2%})" if (not np.isnan(em_sigma) and not np.isnan(underlying_now)) else "—"
    st.metric("Expected Move (σ)", em_txt)
st.caption("**VRP**>0: IV boven gerealiseerde → gunstiger voor **short vol**. **IV-Rank** hoog → premie dikker (let op events).")

# ─────────────────────────── Strangle Helper ──────────────────────
st.markdown("### 🧠 Strangle Helper (σ- of Δ-doel / auto-pick)")

# Nuttige helpers en modellen
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

cm1, cm2, cm3, cm4 = st.columns([1.2, 1, 1, 1])
with cm1:  str_sel_mode = st.radio("Selectiemodus", ["σ-doel", "Δ-doel"], index=0,
                                   help="σ = afstand in standaarddeviaties; Δ = kans ITM.")
with cm2:  sigma_target = st.slider("σ-doel per zijde", 0.5, 2.5, 1.0, step=0.1)
with cm3:  delta_target = st.slider("Δ-doel (absoluut)", 0.05, 0.30, 0.15, step=0.01)
with cm4:  price_source = st.radio("Prijsbron", ["mid_price","last_price"], index=0, horizontal=True)

ce1, ce2, ce3 = st.columns([1.2, 1, 1])
# Expiratie voor strangle default = ~2w vanaf vandaag (nearest)
default_exp_idx = exps_all.index(default_exp) if ('default_exp' in locals() and default_exp in exps_all) else 0
with ce1:
    exp_for_str = st.selectbox("Expiratie voor strangle", options=exps_all or [], index=default_exp_idx if exps_all else 0)
with ce2:
    use_smile_iv = st.checkbox("Gebruik strike-IV (smile) voor Δ", value=False,
                               help="Gebruik per-strike IV i.p.v. ATM-IV voor delta-matching.")
with ce3:
    show_table = st.checkbox("Toon details tabel", value=False)

@st.cache_data(ttl=300, show_spinner=False)
def load_strangle_slice(expiration: date, snap_dt: datetime):
    if expiration is None or snap_dt is None: return pd.DataFrame()
    sql = f"""
      SELECT TIMESTAMP_TRUNC(snapshot_date, MINUTE) AS snap_m, snapshot_date, type, expiration,
             days_to_exp, strike, underlying_price, implied_volatility, open_interest,
             volume, last_price, mid_price
      FROM `{VIEW}`
      WHERE expiration = @exp AND DATE(snapshot_date) = DATE(@snap)
    """
    all_rows = run_query(sql, {"exp": expiration, "snap": snap_dt})
    if all_rows.empty: return all_rows
    all_rows["snap_m"] = pd.to_datetime(all_rows["snap_m"])
    target = pd.to_datetime(snap_dt)
    best_minute = all_rows.loc[(all_rows["snap_m"] - target).abs().idxmin(), "snap_m"]
    return all_rows[all_rows["snap_m"] == best_minute].copy()

df_str = load_strangle_slice(exp_for_str, sel_snapshot)
if not df_str.empty:
    df_str["type"] = df_str["type"].str.lower()
    df_str["mny"]  = df_str["strike"]/df_str["underlying_price"] - 1.0
    df_str = df_str[((df_str["open_interest"].fillna(0) >= min_oi) | (df_str["volume"].fillna(0) >= min_vol))]
    underlying_now = float(df_str["underlying_price"].median()) if np.isnan(underlying_now) else underlying_now

# IV basis voor exp_for_str
iv_atm_exp = float(df_str.loc[(df_str["days_to_exp"].between(20,60)) & (df_str["mny"].abs()<=0.01),
                              "implied_volatility"].median()) if not df_str.empty else np.nan
dte_exp = int(pd.to_numeric(df_str["days_to_exp"], errors="coerce").median()) if not df_str.empty else np.nan
T = max(dte_exp,1)/365.0 if not np.isnan(dte_exp) else np.nan
sigma_pts = (underlying_now * iv_atm_exp * math.sqrt(T)
             if (not np.isnan(underlying_now) and not np.isnan(iv_atm_exp) and not np.isnan(T)) else np.nan)

# Smile-IV map
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

def pick_by_sigma():
    if np.isnan(sigma_pts): return np.nan, np.nan
    return nearest_strike("put", underlying_now - sigma_target*sigma_pts), \
           nearest_strike("call", underlying_now + sigma_target*sigma_pts)

def pick_by_delta():
    if any(np.isnan(x) for x in [underlying_now, T]) or df_str.empty: return np.nan, np.nan
    puts  = sorted(df_str[df_str["type"]=="put"]["strike"].unique().tolist())
    calls = sorted(df_str[df_str["type"]=="call"]["strike"].unique().tolist())
    best_p, best_c, err_p, err_c = np.nan, np.nan, 1e9, 1e9
    for K in puts:
        d = bs_delta(underlying_now, K, get_iv_for("put", K), T, is_call=False); e = abs(abs(d) - delta_target)
        if not np.isnan(d) and e < err_p: best_p, err_p = K, e
    for K in calls:
        d = bs_delta(underlying_now, K, get_iv_for("call", K), T, is_call=True); e = abs(d - delta_target)
        if not np.isnan(d) and e < err_c: best_c, err_c = K, e
    return float(best_p), float(best_c)

ac1, ac2 = st.columns([1, 1])
with ac1:
    if st.button("🔮 Auto-pick (σ-doel)"): str_sel_mode = "σ-doel"
with ac2:
    if st.button("🎯 Auto-pick (Δ-doel)"): str_sel_mode = "Δ-doel"

target_put, target_call = (pick_by_sigma() if str_sel_mode.startswith("σ") else pick_by_delta())

def _val(row, col): 
    return float(pd.to_numeric(row[col], errors="coerce").median()) if (not row.empty and col in row) else np.nan

put_row  = df_str[(df_str["type"]=="put")  & (df_str["strike"]==target_put)].copy()  if not (np.isnan(target_put) or df_str.empty)  else pd.DataFrame()
call_row = df_str[(df_str["type"]=="call") & (df_str["strike"]==target_call)].copy() if not (np.isnan(target_call) or df_str.empty) else pd.DataFrame()
put_px, call_px = _val(put_row, price_source), _val(call_row, price_source)
total_credit = (put_px + call_px) if (not np.isnan(put_px) and not np.isnan(call_px)) else np.nan

# σ-afstand & ~P(touch) (benadering: 2×P(ITM))
def sigma_distance(K: float) -> float: 
    return abs(K - underlying_now) / sigma_pts if not np.isnan(sigma_pts) else np.nan
sd_put, sd_call = sigma_distance(target_put), sigma_distance(target_call)
def p_itm_at_exp(sd: float) -> float: return (1.0 - norm_cdf(sd)) if not np.isnan(sd) else np.nan
p_touch_put  = min(1.0, 2.0 * p_itm_at_exp(sd_put))  if not np.isnan(sd_put)  else np.nan
p_touch_call = min(1.0, 2.0 * p_itm_at_exp(sd_call)) if not np.isnan(sd_call) else np.nan
p_both_touch_approx = min(1.0, (p_touch_put if not np.isnan(p_touch_put) else 0.0) +
                                (p_touch_call if not np.isnan(p_touch_call) else 0.0))
ppd_total_pts = float(total_credit / max(dte_exp,1)) if not np.isnan(total_credit) and not np.isnan(dte_exp) else np.nan

km1, km2, km3, km4, km5, km6 = st.columns(6)
with km1: st.metric("Expiratie", str(exp_for_str) if exp_for_str else "—")
with km2: st.metric("DTE", f"{dte_exp:.0f}" if not np.isnan(dte_exp) else "—")
with km3: st.metric("Strikes", (f"P {target_put:.0f} / C {target_call:.0f}") if not (np.isnan(target_put) or np.isnan(target_call)) else "—")
with km4: st.metric("Credit", f"{total_credit:,.2f}" if not np.isnan(total_credit) else "—")
with km5: st.metric("PPD (tot.)", f"{ppd_total_pts:,.2f}" if not np.isnan(ppd_total_pts) else "—")
with km6: st.metric("~P(touch) max", f"{p_both_touch_approx*100:.0f}%" if not np.isnan(p_both_touch_approx) else "—")
if show_table and not df_str.empty:
    st.dataframe(df_str.sort_values(["type","strike"])[["type","strike","implied_volatility","open_interest","volume","last_price","mid_price"]],
                 use_container_width=True)

# ─────────────────────────── Margin & Sizing ──────────────────────
st.markdown("### 💳 Margin & Sizing")
ready_for_sizing = (not np.isnan(underlying_now)) and (not np.isnan(target_put)) and (not np.isnan(target_call))
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

# ─────────────────────────── Roll-simulator ───────────────────────
st.markdown("### 🔄 Roll-simulator (uitrollen / herpositioneren)")
if (('df_str' not in locals()) or df_str.empty) or np.isnan(target_put) or np.isnan(target_call):
    st.info("Selecteer eerst een strangle in de *Strangle Helper*. Daarna kun je rollen simuleren.")
else:
    future_exps = [e for e in exps_all if e > exp_for_str]
    if not future_exps:
        st.info("Geen latere expiraties beschikbaar binnen de filters om naar uit te rollen.")
    else:
        rr1, rr2, rr3 = st.columns([1.2, 1.0, 1.0])
        with rr1: roll_mode = st.radio("Rol-methode", ["σ-doel", "Δ-doel"], index=0, horizontal=True)
        with rr2: sigma_target_roll = st.slider("σ-doel (roll)", 0.5, 2.5, 1.2, step=0.1)
        with rr3: delta_target_roll = st.slider("Δ-doel (roll, abs)", 0.05, 0.30, 0.15, step=0.01)

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
            all_rows = run_query(sql, {"exp": expiration, "snap": snap_min})
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
            df_new = df_new[((df_new["open_interest"].fillna(0)>=min_oi) | (df_new["volume"].fillna(0)>=min_vol))]

            dte_new = int(pd.to_numeric(df_new["days_to_exp"], errors="coerce").median())
            iv_atm_new = float(df_new.loc[(df_new["days_to_exp"].between(20,60)) & (df_new["mny"].abs()<=0.01),
                                          "implied_volatility"].median())
            T_new = max(dte_new,1)/365.0 if not np.isnan(dte_new) else np.nan
            sigma_pts_new = underlying_now * iv_atm_new * math.sqrt(T_new) if (not np.isnan(underlying_now) and not np.isnan(iv_atm_new) and not np.isnan(T_new)) else np.nan

            smile_map_new = (df_new.groupby(["type","strike"], as_index=False)["implied_volatility"].median()
                                   .set_index(["type","strike"])["implied_volatility"].to_dict())
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
                    d = bs_delta(underlying_now, K, get_iv_new("put", K), T_new, is_call=False); e = abs(abs(d) - delta_target_roll)
                    if not np.isnan(d) and e < err_p: best_p, err_p = K, e
                for K in calls:
                    d = bs_delta(underlying_now, K, get_iv_new("call", K), T_new, is_call=True); e = abs(d - delta_target_roll)
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

# ─────────────────────────── VIX vs IV ────────────────────────────
st.subheader("VIX vs IV")

@st.cache_data(ttl=900, show_spinner=False)
def load_vix_iv_series(start: date, end: date):
    sql = f"""
      SELECT DATE(snapshot_date) d,
             MEDIAN(vix) AS vix,
             MEDIAN(implied_volatility) AS iv
      FROM `{VIEW}`
      WHERE DATE(snapshot_date) BETWEEN @s AND @e
      GROUP BY d
      ORDER BY d
    """
    return run_query(sql, {"s": start, "e": end})

vix_iv = load_vix_iv_series(max(start_date, end_date - timedelta(days=365)), end_date)
if vix_iv.empty:
    st.info("Geen VIX/IV data in de gekozen periode.")
else:
    cV1, cV2, cV3 = st.columns([1, 1, 1])
    with cV1: smooth_vix = st.checkbox("Smooth (7d)", value=False)
    with cV2: force_zero = st.checkbox("Forceer 0-baseline", value=False)
    with cV3: pad_pct    = st.slider("Y-pad (%)", 5, 30, 15, step=1)

    vix_iv = vix_iv.rename(columns={"d":"date"})
    if smooth_vix and len(vix_iv) >= 3:
        vix = vix_iv["vix"].rolling(7, min_periods=1, center=True).median()
        iv  = vix_iv["iv"].rolling(7, min_periods=1, center=True).median()
    else:
        vix, iv = vix_iv["vix"], vix_iv["iv"]

    def padded_range(series: pd.Series, pad_frac: float, floor_zero: bool):
        s = pd.to_numeric(series, errors="coerce").dropna()
        if s.empty: return None
        lo, hi = float(s.min()), float(s.max())
        if hi == lo:
            eps = max(0.1, abs(hi)*0.1)
            lo, hi = hi - eps, hi + eps
        pad = (hi - lo) * pad_frac
        lo2 = 0.0 if floor_zero else (lo - pad)
        hi2 = hi + pad
        return [lo2, hi2]

    vix_range = padded_range(vix.rename("vix"), pad_pct/100.0, force_zero)
    iv_range  = padded_range(iv.rename("iv"),  pad_pct/100.0, force_zero)

    fig_vix = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                            subplot_titles=("VIX", "Gemiddelde IV"))
    fig_vix.add_trace(go.Scatter(x=vix_iv["date"], y=vix, mode="lines+markers", name="VIX"), row=1, col=1)
    fig_vix.add_trace(go.Scatter(x=vix_iv["date"], y=iv,  mode="lines+markers", name="IV"),  row=2, col=1)
    if vix_range: fig_vix.update_yaxes(range=vix_range, row=1, col=1)
    if iv_range:  fig_vix.update_yaxes(range=iv_range,  row=2, col=1)
    fig_vix.update_layout(height=620, title_text=f"VIX vs IV ({sel_type.upper()})", dragmode="zoom")
    st.plotly_chart(fig_vix, use_container_width=True, config=PLOTLY_CONFIG)

# ─────────────────────────── Footer ───────────────────────────────
st.caption("🔍 Navigatie: scroll/pinch = zoom, **double-click** = autoscale. "
           "Gebruik **OI/Volume filters** om spikes te dempen. Tooltips bij schuivers leggen ATM, σ en Δ uit.")
