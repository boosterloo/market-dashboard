import streamlit as st
from google.cloud import bigquery
import datetime as dt
import pandas as pd

@st.cache_resource
def get_bq_client() -> bigquery.Client:
    # Authenticatie via Streamlit Cloud secrets of gcloud default creds
    return bigquery.Client()

def _infer_bq_type(v):
    if isinstance(v, bool): return "BOOL"
    if isinstance(v, int): return "INT64"
    if isinstance(v, float): return "FLOAT64"
    if isinstance(v, dt.datetime): return "TIMESTAMP"
    if isinstance(v, dt.date): return "DATE"
    return "STRING"

@st.cache_data(ttl=300, show_spinner=False)
def run_query(sql: str, params: dict | None = None, timeout: int = 30) -> pd.DataFrame:
    """Kleine, veilige wrapper om query’s te cachen en parameters te typen."""
    client = get_bq_client()
    job_config = None
    if params:
        qparams = []
        for k, v in params.items():
            qparams.append(bigquery.ScalarQueryParameter(k, _infer_bq_type(v), v))
        job_config = bigquery.QueryJobConfig(query_parameters=qparams)
    job = client.query(sql, job_config=job_config)
    df = job.result(timeout=timeout).to_dataframe(create_bqstorage_client=False)
    return df
