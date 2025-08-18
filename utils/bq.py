# utils/bq.py
from __future__ import annotations
from typing import Any, Dict, List, Optional
import pandas as pd
import streamlit as st
from google.oauth2 import service_account
from google.cloud import bigquery

# ---------- Client ----------
@st.cache_resource(show_spinner=False)
def _bq_client_from_secrets() -> bigquery.Client:
    creds = st.secrets["gcp_service_account"]
    credentials = service_account.Credentials.from_service_account_info(creds)
    return bigquery.Client(credentials=credentials, project=creds["project_id"])

# Backwards-compatible alias
def get_bq_client() -> bigquery.Client:
    return _bq_client_from_secrets()

# ---------- Healthcheck ----------
@st.cache_data(ttl=60, show_spinner=False)
def bq_ping() -> bool:
    try:
        _bq_client_from_secrets().query("SELECT 1").result(timeout=10)
        return True
    except Exception:
        return False

# ---------- Query helper ----------
@st.cache_data(ttl=600, show_spinner=False)
def run_query(
    sql: str,
    params: Optional[Dict[str, Any]] = None,
    timeout: Optional[float] = None,  # <-- nieuw: seconds (float/int)
) -> pd.DataFrame:
    """
    Execute SQL and return a pandas DataFrame.
    - Supports simple scalar params (INT64, FLOAT64, STRING, DATE, TIMESTAMP).
    - Normalizes 'date' and 'snapshot_date' columns to pandas types.
    - `timeout` (sec) aborts waiting for the job if it runs too long.
    """
    client = _bq_client_from_secrets()

    job_config = None
    if params:
        bq_params: List[bigquery.ScalarQueryParameter] = []
        for k, v in params.items():
            # basic type inference
            if isinstance(v, int):
                typ = "INT64"
            elif isinstance(v, float):
                typ = "FLOAT64"
            elif isinstance(v, pd.Timestamp):
                typ = "TIMESTAMP"; v = v.to_pydatetime()
            else:
                # Try parse date/timestamp from string-like
                try:
                    ts = pd.to_datetime(v)
                    if " " in str(v) or "T" in str(v):
                        typ = "TIMESTAMP"; v = ts.to_pydatetime()
                    else:
                        typ = "DATE"; v = ts.date()
                except Exception:
                    typ = "STRING"
            bq_params.append(bigquery.ScalarQueryParameter(k, typ, v))
        job_config = bigquery.QueryJobConfig(query_parameters=bq_params)

    job = client.query(sql, job_config=job_config)

    # Respect timeout while waiting for completion
    job.result(timeout=timeout) if timeout else job.result()

    # Use default fetch path (no BigQuery Storage requirement)
    df = job.to_dataframe(create_bqstorage_client=False)

    # Normalize common columns
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.date
    if "snapshot_date" in df.columns:
        df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])

    return df
