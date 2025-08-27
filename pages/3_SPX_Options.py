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

# ───────────────────────────── Helpers ─────────────────────────────
def norm_cdf(z: float) -> float:
    """Standaard-normale CDF zonder scipy (via error function)."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

def norm_pdf(z: float) -> float:
    """Standaard-normale PDF."""
    return (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * z * z)

def pick_closest_date(options: list[date], target: date):
    if not options: return None
    return min(options, key=lambda d: abs(pd.Timestamp(d) - pd.Timestamp(target)))

def pick_first_on_or_after(options: list[date], target: date):
    after = [d for d in options if d >= target]
    return after[0] if after else (options[-1] if options else None)

def pick_closest_value(options: list[float], target: float, fallback: float | None = None):
    if not options: return fallback
    return float(min(options, key=lambda x: abs(float(x) - float(target))))

def is_round(x: float) -> int:
    score = 0
    if round(x) % 100 == 0: score += 2
    if round(x) % 50 == 0:  score += 1
    if round(x) % 25 == 0:  score += 1
    return score

def smooth_series(y: pd.Series, window: int = 3) -> pd.Series:
    if len(y) < 3: return y
    return y.rolling(window, center=True, min_periods=1).median()

def annualize_std(s: pd.Series, periods_per_year: int = 252) -> float:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) < 2: return np.nan
    return float(s.std(ddof=0) * math.sqrt(periods_per_year))

# ── BigQuery client ────────────────────────────────
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

# ── Page ──────────────────────────────────────────
st.set_page_config(page_title="SPX Options Dashboard", layout="wide")
st.title("🧩 SPX Options Dashboard")
VIEW = "marketdata.spx_options_enriched_v"

# … ⚡️ de rest van de code is exact zoals ik je eerder stuurde ⚡️ …
# (inclusief alle secties: Serie-selectie, PPD vs strike, Exp-curve,
# Matrix, IV Smile, PCR, Volume heatmap, Vol & Risk, Strangle Helper, etc.)
# Alleen verschil: `norm.cdf(sd)` is vervangen door `norm_cdf(sd)`.

# Voorbeeld patch in Strangle Helper:
def p_itm_at_exp(sd: float) -> float:
    if np.isnan(sd): 
        return np.nan
    return 1.0 - norm_cdf(sd)
