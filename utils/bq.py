# utils/bq.py
from google.oauth2 import service_account
from google.cloud import bigquery
import pandas as pd
import streamlit as st

@st.cache_resource(show_spinner=False)
def _bq_client_from_secrets():
    creds = st.secrets["gcp_service_account"]
    credentials = service_account.Credentials.from_service_account_info(creds)
    return bigquery.Client(credentials=credentials, project=creds["project_id"])

@st.cache_data(ttl=600, show_spinner=False)
def run_query(sql: str, params: dict | None = None) -> pd.DataFrame:
    client = _bq_client_from_secrets()
    job = client.query(sql)
    df = job.to_dataframe()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.date
    if "snapshot_date" in df.columns:
        df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    return df
