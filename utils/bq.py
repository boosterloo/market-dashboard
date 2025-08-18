
import pandas as pd
import streamlit as st
from google.cloud import bigquery

# Hardcoded project + dataset + tables
PROJECT = "nth-pier-468314-p7"
DATASET = "marketdata"
TABLE_SP500 = "sp500_prices"
TABLE_VIX = "vix_prices"
TABLE_SPX_OPTIONS = "spx_options"

@st.cache_resource(show_spinner=False)
def _get_bq_client():
    return bigquery.Client(project=PROJECT)

def _to_bq_param(name, value):
    import datetime as dt
    if isinstance(value, pd.Timestamp):
        value = value.date()
    if isinstance(value, dt.datetime):
        return bigquery.ScalarQueryParameter(name, "TIMESTAMP", value)
    if isinstance(value, dt.date):
        return bigquery.ScalarQueryParameter(name, "DATE", value)
    if isinstance(value, bool):
        return bigquery.ScalarQueryParameter(name, "BOOL", value)
    if isinstance(value, int):
        return bigquery.ScalarQueryParameter(name, "INT64", value)
    if isinstance(value, float):
        return bigquery.ScalarQueryParameter(name, "FLOAT64", value)
    return bigquery.ScalarQueryParameter(name, "STRING", str(value))

@st.cache_data(ttl=900, show_spinner=False)
def _cached_query(sql: str, params_frozen):
    client = _get_bq_client()
    if params_frozen:
        params_dict = dict(params_frozen)
        job_config = bigquery.QueryJobConfig(
            query_parameters=[_to_bq_param(k, v) for k, v in params_dict.items()]
        )
    else:
        job_config = None
    df = client.query(sql, job_config=job_config).result().to_dataframe()
    return df

def run_query(sql: str, params: dict | None = None) -> pd.DataFrame:
    frozen = frozenset(params.items()) if params else None
    return _cached_query(sql, frozen)

def load_sp500(start_date, end_date) -> pd.DataFrame:
    sql = f"""
    SELECT
      DATE(date) AS date,
      CAST(open AS NUMERIC)  AS open,
      CAST(high AS NUMERIC)  AS high,
      CAST(low  AS NUMERIC)  AS low,
      CAST(close AS NUMERIC) AS close,
      CAST(volume AS INT64)  AS volume
    FROM `{PROJECT}.{DATASET}.{TABLE_SP500}`
    WHERE DATE(date) BETWEEN @start_date AND @end_date
    ORDER BY date
    """
    return run_query(sql, {"start_date": start_date, "end_date": end_date})

def load_vix(start_date, end_date) -> pd.DataFrame:
    sql = f"""
    SELECT
      DATE(date) AS date,
      CAST(close AS NUMERIC) AS close
    FROM `{PROJECT}.{DATASET}.{TABLE_VIX}`
    WHERE DATE(date) BETWEEN @start_date AND @end_date
    ORDER BY date
    """
    return run_query(sql, {"start_date": start_date, "end_date": end_date})

def load_spx_options(start_date, end_date) -> pd.DataFrame:
    sql = f"""
    SELECT
      TIMESTAMP(snapshot_date) AS snapshot_date,
      DATE(expiration) AS expiration,
      SAFE_CAST(days_to_exp AS INT64) AS days_to_exp,
      SAFE_CAST(strike AS NUMERIC) AS strike,
      UPPER(CAST(type AS STRING)) AS type,
      SAFE_CAST(last_price AS NUMERIC) AS last_price,
      SAFE_CAST(bid AS NUMERIC) AS bid,
      SAFE_CAST(ask AS NUMERIC) AS ask,
      SAFE_CAST(implied_volatility AS NUMERIC) AS implied_volatility,
      SAFE_CAST(open_interest AS INT64) AS open_interest,
      SAFE_CAST(volume AS INT64) AS volume,
      SAFE_CAST(underlying_price AS NUMERIC) AS underlying_price,
      SAFE_CAST(ppd AS NUMERIC) AS ppd,
      SAFE_CAST(vix AS NUMERIC) AS vix
    FROM `{PROJECT}.{DATASET}.{TABLE_SPX_OPTIONS}`
    WHERE DATE(snapshot_date) BETWEEN @start_date AND @end_date
    ORDER BY snapshot_date DESC, expiration, strike
    """
    return run_query(sql, {"start_date": start_date, "end_date": end_date})
