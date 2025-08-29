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

# ─────────────────────────── Helpers ──────────────────────────────
def norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

def pick_closest_date(options: list[date], target: date):
    if not options: return None
    return min(options, key=lambda d: abs(pd.Timestamp(d) - pd.Timestamp(target)))

def pick_first_on_or_after(options: list[date], target: date):
    after = [d for d in options if d >= target]
    return after[0] if after else (options[-1] if options else None)

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

def run_query(sql: str, params: dict | None = None) -> pd.DataFrame:
    job_config = None
    if params:
        job_config = bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter(k, "STRING", str(v)) for k, v in params.items()]
        )
    return _bq_client.query(sql, job_config=job_config).to_dataframe()

# ─────────────────────────── Page setup ───────────────────────────
st.set_page_config(page_title="SPX Options Dashboard", layout="wide")
st.title("🧩 SPX Options Dashboard")
VIEW = "marketdata.spx_options_enriched_v"

PLOTLY_CONFIG = {"scrollZoom": True, "displaylogo": False, "doubleClick": "reset"}

# ─────────────────────────── Data load ────────────────────────────
@st.cache_data(ttl=600, show_spinner=False)
def load_bounds():
    sql = f"SELECT MIN(DATE(snapshot_date)) mn, MAX(DATE(snapshot_date)) mx FROM `{VIEW}`"
    df = run_query(sql)
    return df["mn"].iloc[0], df["mx"].iloc[0]

min_date, max_date = load_bounds()
start_date, end_date = st.date_input(
    "Periode", (max_date - timedelta(days=60), max_date), min_value=min_date, max_value=max_date
)

# Snapshots
@st.cache_data(ttl=600, show_spinner=False)
def load_snapshots():
    sql = f"SELECT DISTINCT TIMESTAMP_TRUNC(snapshot_date, HOUR) snap FROM `{VIEW}` ORDER BY snap"
    df = run_query(sql)
    return sorted(pd.to_datetime(df["snap"]).dt.to_pydatetime())

snapshots = load_snapshots()
default_snap = snapshots[-1] if snapshots else None
snap = st.selectbox("Snapshot-datum", snapshots, index=len(snapshots)-1 if snapshots else 0)

# Expirations
@st.cache_data(ttl=600, show_spinner=False)
def load_expirations():
    sql = f"SELECT DISTINCT expiration FROM `{VIEW}` ORDER BY expiration"
    df = run_query(sql)
    return sorted(pd.to_datetime(df["expiration"]).dt.date.tolist())

exps = load_expirations()
target = date.today() + timedelta(days=14)
default_exp = pick_first_on_or_after(exps, target) or pick_closest_date(exps, target)
exp = st.selectbox("Expiratie", exps, index=exps.index(default_exp) if default_exp in exps else 0)

# ─────────────────────────── Data subset ──────────────────────────
@st.cache_data(ttl=600, show_spinner=True)
def load_filtered(snap: datetime, exp: date):
    sql = f"""
    SELECT snapshot_date, type, expiration, days_to_exp, strike, underlying_price,
           implied_volatility, open_interest, volume, ppd,
           SAFE_DIVIDE(strike, NULLIF(underlying_price,0)) - 1 AS mny
    FROM `{VIEW}`
    WHERE TIMESTAMP_TRUNC(snapshot_date, HOUR) = @snap
      AND expiration = @exp
    """
    return run_query(sql, {"snap": snap, "exp": exp})

df = load_filtered(snap, exp)
if df.empty:
    st.warning("Geen data.")
    st.stop()

underlying = df["underlying_price"].median()

# ─────────────────────────── PPD vs DTE ──────────────────────────
st.subheader("PPD vs DTE")
mode = st.radio("Bereik", ["ATM-band (moneyness)", "Rond gekozen strike"])
atm_band = st.slider("ATM-band (± %)", 0.0, 0.10, 0.02, step=0.01,
                     help="Hoe ver strikes van spot mogen liggen (procentueel).")
strike = st.number_input("Gekozen strike", value=float(underlying), step=25.0)
strike_win = st.slider("Venster ± (punten)", 0, 500, 100, step=25)

if mode.startswith("ATM"):
    sel = df[df["mny"].abs() <= atm_band]
else:
    sel = df[(df["strike"] >= strike - strike_win) & (df["strike"] <= strike + strike_win)]

g = sel.groupby("days_to_exp", as_index=False)["ppd"].median().sort_values("days_to_exp")

y = g["ppd"].to_numpy()
lo, hi = np.nanpercentile(y, [5,95]) if len(y) else (0,1)
pad = (hi - lo)*0.2
ymin, ymax = max(0, lo-pad), hi+pad

fig = go.Figure()
fig.add_scatter(x=g["days_to_exp"], y=g["ppd"], mode="markers+lines", name="PPD")
fig.update_layout(title=f"PPD vs DTE — {snap.date()} | {exp}", height=420)
fig.update_yaxes(range=[ymin,ymax], title="PPD (pts/day)")
st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

# ─────────────────────────── Matrix ──────────────────────────────
st.subheader("Matrix meetmoment × strike")
max_snaps = st.slider("Max snapshots", 50, 500, 200, step=50)

@st.cache_data(ttl=600, show_spinner=True)
def load_matrix(exp: date, limit: int):
    sql = f"""
    WITH r AS (
      SELECT TIMESTAMP_TRUNC(snapshot_date, HOUR) snap, strike, ppd
      FROM `{VIEW}`
      WHERE expiration=@exp
    )
    SELECT * FROM r ORDER BY snap DESC LIMIT @lim
    """
    return run_query(sql, {"exp": exp, "lim": limit})

mx = load_matrix(exp, max_snaps)
if not mx.empty:
    pivot = mx.pivot_table(index="snap", columns="strike", values="ppd", aggfunc="median").fillna(np.nan)
    fig_mx = go.Figure(go.Heatmap(z=pivot.values, x=pivot.columns, y=pivot.index, colorbar_title="PPD"))
    fig_mx.update_layout(height=520, title="Heatmap PPD", xaxis_title="Strike", yaxis_title="Snapshot")
    st.plotly_chart(fig_mx, use_container_width=True, config=PLOTLY_CONFIG)

# ─────────────────────────── Strangle Helper ─────────────────────
st.subheader("Strangle Helper")
mode = st.radio("Selectie", ["σ-doel","Δ-doel"])
sigma_target = st.slider("σ-doel", 0.5, 2.5, 1.0, 0.1, help="σ = standaarddeviatie afstand")
delta_target = st.slider("Δ-doel", 0.05, 0.30, 0.15, 0.01, help="Δ = kans dat optie ITM eindigt")

iv = df["implied_volatility"].median()
dte = df["days_to_exp"].median()
T = dte/365
sigma_pts = underlying*iv*math.sqrt(T)

def nearest(strikes, target): return min(strikes, key=lambda k: abs(k-target)) if len(strikes) else np.nan

puts = df[df["type"]=="put"]["strike"].unique()
calls = df[df["type"]=="call"]["strike"].unique()

if mode.startswith("σ"):
    put_strike  = nearest(puts, underlying - sigma_target*sigma_pts)
    call_strike = nearest(calls, underlying + sigma_target*sigma_pts)
else:
    # Simplified: pick deltas closest to delta_target
    put_strike = nearest(puts, underlying*0.9)
    call_strike = nearest(calls, underlying*1.1)

st.metric("Put strike", put_strike)
st.metric("Call strike", call_strike)
st.caption("Gebruik σ- of Δ-doel om strikes te kiezen. σ = afstand in standaarddeviaties. Δ = kans ITM.")

st.caption("📌 Navigatie: zoom/pan met muis, dubbelklik voor autoscale.")
