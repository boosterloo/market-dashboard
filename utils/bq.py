# utils/bq.py
import streamlit as st
from google.cloud import bigquery
from google.oauth2 import service_account
import datetime as dt
import pandas as pd

@st.cache_resource(show_spinner="Verbinding met BigQuery…")
def get_bq_client() -> bigquery.Client:
    # Gebruik expliciete credentials uit Secrets
    sa = st.secrets.get("gcp_service_account", None)
    if not sa:
        raise RuntimeError(
            "Geen [gcp_service_account] in Secrets. "
            "Ga naar Settings → Secrets en voeg je service-account JSON toe."
        )
    creds = service_account.Credentials.from_service_account_info(sa)
    project = sa.get("project_id")
    return bigquery.Client(credentials=creds, project=project)

def _infer_bq_type(v):
    if isinstance(v, bool): return "BOOL"
    if isinstance(v, int): return "INT64"
    if isinstance(v, float): return "FLOAT64"
    if isinstance(v, dt.datetime): return "TIMESTAMP"
    if isinstance(v, dt.date): return "DATE"
    return "STRING"

@st.cache_data(ttl=300, show_spinner=False)
def run_query(sql: str, params: dict | None = None, timeout: int = 30) -> pd.DataFrame:
    client = get_bq_client()
    job_config = None
    if params:
        from google.cloud import bigquery as bq
        qparams = [bq.ScalarQueryParameter(k, _infer_bq_type(v), v) for k, v in params.items()]
        job_config = bq.QueryJobConfig(query_parameters=qparams)
    job = client.query(sql, job_config=job_config)
    return job.result(timeout=timeout).to_dataframe(create_bqstorage_client=False)

@st.cache_data(ttl=120)
def bq_ping() -> bool:
    client = get_bq_client()
    client.query("SELECT 1").result()
    return True
