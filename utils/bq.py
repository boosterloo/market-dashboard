# utils/bq.py
from __future__ import annotations
from typing import Any, Dict, List
import pandas as pd
import streamlit as st
from google.oauth2 import service_account
from google.cloud import bigquery

# ---- Client ----
@st.cache_resource(show_spinner=False)
def _bq_client_from_secrets() -> bigquery.Client:
    """Build one global BigQuery client from Streamlit secrets."""
    creds_dict = st.secrets["gcp_service_account"]
    credentials = service_account.Credentials.from_service_account_info(creds_dict)
    return bigquery.Client(credentials=credentials, project=creds_dict["project_id"])

# Backwards-compatible alias expected by older pages
def get_bq_client() -> bigquery.Client:
    return _bq_client_from_secrets()

# ---- Health check / ping ----
@st.cache_data(ttl=60, show_spinner=False)
def bq_ping() -> bool:
    """Returns True if a trivial query succeeds."""
    try:
        client = _bq_client_from_secrets()
        client.query("SELECT 1").result()
        return True
    except Exception:
        return False

# ---- Query helper ----
@st.cache_data(ttl=600, show_spinner=False)
def run_query(sql: str, params: Dict[str, Any] | None = None) -> pd.DataFrame:
    """
    Execute SQL and return pandas DataFrame.
    - Supports simple scalar parameters (STRING/INT64/FLOAT64/DATE/TIMESTAMP).
    - Normalizes 'date' and 'snapshot_date' columns.
    """
    client = _bq_client_from_secrets()

    job_config = None
    if params:
        job_params: List[bigquery.ScalarQueryParameter] = []
        for k, v in params.items():
            # best-effort type detection
            if isinstance(v, (int,)):
                tp = "INT64"
            elif isinstance(v, (float,)):
                tp = "FLOAT64"
            elif isinstance(v, pd.Timestamp):
                tp = "TIMESTAMP"
                v = v.to_pydatetime()
            elif isinstance(v, (pd.DateOffset,)):
                tp = "DATE"
            else:
                # try to parse common string date/timestamp
                try:
                    ts = pd.to_datetime(v)
                    if str(v).find("T") >= 0 or (hasattr(ts, "hour") and ts.hour is not pd.NaT):
                        tp = "TIMESTAMP"
                        v = ts.to_pydatetime()
                    else:
                        tp = "DATE"
                        v = ts.date()
                except Exception:
                    tp = "STRING"
            job_params.append(bigquery.ScalarQueryParameter(k, tp, v))
        job_config = bigquery.QueryJobConfig(query_parameters=job_params)

    df = client.query(sql, job_config=job_config).to_dataframe()

    # Normalize common columns
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.date
    if "snapshot_date" in df.columns:
        df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])

    return df
