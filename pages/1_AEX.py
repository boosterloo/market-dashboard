# pages/2_AEX.py
# AEX - Market State Dashboard
# Focus op regime, trend, momentum, uitputting en forward returns.

import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =========================
# BigQuery helpers
# =========================
try:
    from utils.bq import run_query, bq_ping
except Exception:
    from google.cloud import bigquery
    from google.oauth2 import service_account

    @st.cache_resource(show_spinner=False)
    def _bq_client_from_secrets():
        creds = st.secrets["gcp_service_account"]
        credentials = service_account.Credentials.from_service_account_info(creds)
        return bigquery.Client(credentials=credentials, project=creds["project_id"])

    def run_query(sql: str) -> pd.DataFrame:
        client = _bq_client_from_secrets()
        return client.query(sql).to_dataframe()

    def bq_ping() -> bool:
        try:
            _ = run_query("SELECT 1 AS ok")
            return True
        except Exception:
            return False


# =========================
# App setup
# =========================
st.set_page_config(page_title="AEX - Market State Dashboard", layout="wide")
st.title("AEX - Market State Dashboard")

AEX_VIEW = st.secrets.get("tables", {}).get(
    "aex_view", "nth-pier-468314-p7.marketdata.aex_with_vix_v"
)

# =========================
# Health
# =========================
try:
    if not bq_ping():
        st.error("Geen BigQuery-verbinding.")
        st.stop()
except Exception as e:
    st.error("Geen BigQuery-verbinding.")
    st.caption(f"Details: {e}")
    st.stop()

# =========================
# Data
# =========================
@st.cache_data(ttl=1800, show_spinner=False)
def load_aex():
    return run_query(f"SELECT * FROM `{AEX_VIEW}` ORDER BY date")


with st.spinner("AEX data laden..."):
    df = load_aex()

if df.empty:
    st.warning("Geen data in view.")
    st.stop()

df["date"] = pd.to_datetime(df["date"])
df = df[df["date"].dt.weekday < 5].copy()

for c in ["open", "high", "low", "close", "vix_close", "delta_abs", "delta_pct"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

if "delta_abs" not in df.columns or df["delta_abs"].isna().all():
    df["delta_abs"] = df["close"].diff()

if "delta_pct" not in df.columns or df["delta_pct"].isna().all():
    df["delta_pct"] = df["close"].pct_change() * 100.0

# =========================
# Defaults
# =========================
DEFAULTS = {
    "ema_spans": [20, 50, 200],
    "macd": (12, 26, 9),
    "rsi_period": 14,
    "adx_length": 14,
    "donchian_n": 20,
    "corr_win_default": 20,
    "rsi_dyn_win": 252,
}