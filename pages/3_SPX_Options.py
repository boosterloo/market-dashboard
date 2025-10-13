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
from scipy.stats import norm

# ---------- Streamlit config ----------
st.set_page_config(page_title="SPX Options Dashboard", layout="wide")
st.title("🧰 SPX Options — Skew, Delta & PPD")

# ---------- BigQuery setup ----------
def get_bq_client():
    sa_info = st.secrets["gcp_service_account"]
    creds = service_account.Credentials.from_service_account_info(sa_info)
    project_id = sa_info.get("project_id")
    return bigquery.Client(project=project_id, credentials=creds)

_bq_client = get_bq_client()

def run_query(sql: str, params: dict | None = None) -> pd.DataFrame:
    job_config = None
    if params:
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter(k, "STRING", str(v))
                for k, v in params.items()
            ]
        )
    return _bq_client.query(sql, job_config=job_config).to_dataframe()

# Gebruik volledige tabelnaam
PROJECT_ID = st.secrets["gcp_service_account"]["project_id"]
VIEW = f"{PROJECT_ID}.marketdata.spx_options_enriched_v"

# ---------- Skew analyse (points-to-strike) ----------
st.sidebar.header("⚙️ Instellingen Skew")
center_mode = st.sidebar.radio("Centrering", ["Rounded (aanbevolen)", "ATM"], index=0)
round_base = st.sidebar.select_slider("Rond strikes op", options=[25, 50, 100], value=25)
max_pts = st.sidebar.slider("Afstand tot (gecentreerde) strike (± punten)", 50, 1000, 400, 50)
dte_pref = st.sidebar.selectbox("DTE-selectie voor skew", ["Nearest", "0–7", "8–21", "22–45", "46–90", "90+"])
r_input = st.sidebar.number_input("Risicovrije rente r", 0.00, 10.0, 0.00, 0.25)
q_input = st.sidebar.number_input("Dividend/Index carry q", 0.00, 10.0, 0.00, 0.25)

# ---------- Data ophalen (laatste snapshot) ----------
@st.cache_data(ttl=600, show_spinner=True)
def load_latest_snapshot() -> pd.DataFrame:
    sql = f"""
    WITH last AS (
      SELECT MAX(snapshot_date) AS snapshot_date FROM `{VIEW}`
    )
    SELECT * FROM `{VIEW}` WHERE snapshot_date = (SELECT snapshot_date FROM last)
    """
    return run_query(sql)

df = load_latest_snapshot()
if df.empty:
    st.warning("Geen data gevonden in view.")
    st.stop()

# ---------- Basis schoonmaak ----------
df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
df["underlying_price"] = pd.to_numeric(df["underlying_price"], errors="coerce")
df["implied_volatility"] = pd.to_numeric(df["implied_volatility"], errors="coerce")
df["days_to_exp"] = pd.to_numeric(df["days_to_exp"], errors="coerce")
df = df.dropna(subset=["strike", "underlying_price", "implied_volatility", "days_to_exp"])

# ---------- Centreren in punten ----------
S_now = float(np.nanmedian(df["underlying_price"]))
center = round_base * round(S_now / round_base) if center_mode.startswith("Rounded") else S_now
df["pts_to_strike"] = df["strike"] - center
df = df[df["pts_to_strike"].between(-max_pts, max_pts)]
if df.empty:
    st.warning("Geen rijen binnen ±afstand.")
    st.stop()

# ---------- Delta berekening (vectorized) ----------
def bs_delta_vectorized(S, K, IV, T, r, q, is_call):
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    IV = np.asarray(IV, dtype=float)
    T = np.asarray(T, dtype=float) / 365.0
    is_call = np.asarray(is_call, dtype=bool)
    n = len(S)
    r_arr = np.full(n, r, dtype=float)
    q_arr = np.full(n, q, dtype=float)
    eps = 1e-12
    sigma = np.maximum(IV, eps)
    sqrtT = np.sqrt(np.maximum(T, eps))
    d1 = (np.log(np.maximum(S, eps) / np.maximum(K, eps)) + (r_arr - q_arr + 0.5 * sigma**2) * T) / (sigma * sqrtT)
    disc = np.exp(-q_arr * T)
    return np.where(is_call, disc * norm.cdf(d1), -disc * norm.cdf(-d1)).astype(float)

df["is_call"] = df["type"].str.lower().eq("call")
df["delta"] = bs_delta_vectorized(
    df["underlying_price"], df["strike"], df["implied_volatility"], df["days_to_exp"],
    r_input, q_input, df["is_call"]
)

# ---------- Plot: IV & Delta vs points-to-strike ----------
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                    subplot_titles=("Implied Volatility vs Points", "Delta vs Points"))
for side in ["call", "put"]:
    sub = df[df["type"].str.lower() == side]
    if sub.empty:
        continue
    fig.add_trace(go.Scatter(x=sub["pts_to_strike"], y=sub["implied_volatility"],
                             mode="markers", name=f"IV {side}"), row=1, col=1)
    fig.add_trace(go.Scatter(x=sub["pts_to_strike"], y=sub["delta"],
                             mode="markers", name=f"Δ {side}"), row=2, col=1)

fig.update_xaxes(title_text="Points to strike (K − center)", row=2, col=1)
fig.update_yaxes(title_text="IV", tickformat=".1%", row=1, col=1)
fig.update_yaxes(title_text="Delta", row=2, col=1)
fig.update_layout(height=700, showlegend=True, margin=dict(t=60, b=40, l=40, r=20))
st.plotly_chart(fig, use_container_width=True)

