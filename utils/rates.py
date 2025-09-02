# utils/rates.py
import re
import math
import numpy as np
import pandas as pd
from typing import Iterable, Tuple, List, Optional
from google.cloud import bigquery
from google.oauth2 import service_account

# --------- BigQuery client helper (zoals elders) ----------
def _bq_client_from_st():
    import streamlit as st
    sa_info = st.secrets["gcp_service_account"]
    creds = service_account.Credentials.from_service_account_info(sa_info)
    project_id = st.secrets.get("PROJECT_ID", sa_info.get("project_id"))
    return bigquery.Client(project=project_id, credentials=creds)

# --------- Kolom-detectie: r_<tenor>  ----------
# Herkent o.a.: r_1d r_7d r_1w r_1m r_3m r_6m r_9m r_1y r_2y r_5y r_10y r_30y
# en varianten als usd_1m, rate_2y, y_10y, etc.
_TENOR_RE = re.compile(
    r"(?P<prefix>.*?)(?P<num>\d+)\s*(?P<unit>d|day|days|w|week|weeks|m|mon|month|months|y|yr|year|years)\b",
    flags=re.IGNORECASE
)

def _tenor_to_years(num: int, unit: str) -> float:
    u = unit.lower()
    if u in ("d","day","days"):   return num / 365.0
    if u in ("w","week","weeks"): return num * 7 / 365.0
    if u in ("m","mon","month","months"): return num / 12.0
    if u in ("y","yr","year","years"):    return float(num)
    return float("nan")

def _detect_tenor_columns(cols: Iterable[str]) -> List[Tuple[str,float]]:
    """
    Return [(colname, T_years), ...] gesorteerd op T_years
    We accepteren alles wat _TENOR_RE matched en kolomtype ‘rate’ lijkt te zijn.
    """
    out = []
    for c in cols:
        m = _TENOR_RE.search(str(c))
        if not m:
            continue
        num = int(m.group("num"))
        unit = m.group("unit")
        T = _tenor_to_years(num, unit)
        if T and T > 0:
            out.append((c, T))
    out = sorted(out, key=lambda x: x[1])
    return out

def _interp_rate(T_query: np.ndarray, T_nodes: np.ndarray, r_nodes: np.ndarray, extrapolate: bool) -> np.ndarray:
    if len(T_nodes) == 0:
        return np.full_like(T_query, np.nan, dtype=float)
    if len(T_nodes) == 1:
        return np.full_like(T_query, float(r_nodes[0]), dtype=float)
    # lineaire interp (simple)
    def _one(t):
        if t <= T_nodes[0]:
            return r_nodes[0] if extrapolate else np.nan
        if t >= T_nodes[-1]:
            return r_nodes[-1] if extrapolate else np.nan
        i = np.searchsorted(T_nodes, t)
        x0, x1 = T_nodes[i-1], T_nodes[i]
        y0, y1 = r_nodes[i-1], r_nodes[i]
        w = (t - x0) / (x1 - x0) if x1 != x0 else 0.0
        return float(y0 + w*(y1 - y0))
    return np.array([_one(float(t)) for t in T_query], dtype=float)

def _to_continuous(r_simple: np.ndarray) -> np.ndarray:
    # simple (apr) -> continuous
    return np.array([math.log1p(float(x)) for x in r_simple], dtype=float)

def get_r_curve_for_snapshot(
    snapshot_date,
    T_years: np.ndarray,
    view: str,
    date_col: str = "date",
    output: str = "continuous",   # "continuous" | "simple"
    extrapolate: bool = True,
) -> np.ndarray:
    """
    Leest ÉÉN rij uit je yield-view (dichtst bij snapshot_date) en detecteert automatisch tenor-kolommen.
    De kolomnamen mogen variëren zolang ze een <num><unit> bevatten (b.v. 'r_3m', 'usd_10y', 'rate_2y').

    Parameters:
      - snapshot_date: pd.Timestamp / str / datetime
      - T_years: array van looptijden in jaren (bv. [0.08, 0.5, 1, 2, 5])
      - view: fully-qualified tabel/view naam in BigQuery
      - date_col: kolom in de view die de datum/timestamp bevat (default 'date')
      - output: "continuous" (default) of "simple"
      - extrapolate: True = buiten bereik vlak door trekken; False = NaN buiten bereik

    Returns: np.ndarray met r(T) in 'output' compounding.
    """
    if not isinstance(T_years, np.ndarray):
        T_years = np.array(T_years, dtype=float)

    client = _bq_client_from_st()
    # Pak de dichtstbijzijnde dag t.o.v. snapshot_date (dagresolutie is meestal genoeg)
    snap = pd.to_datetime(snapshot_date).date()

    sql = f"""
    WITH base AS (
      SELECT * FROM `{view}`
    )
    SELECT * FROM base
    WHERE DATE({date_col}) <= @snap
    ORDER BY DATE({date_col}) DESC
    LIMIT 1
    """
    params = [bigquery.ScalarQueryParameter("snap","DATE",str(snap))]
    df = client.query(sql, job_config=bigquery.QueryJobConfig(query_parameters=params)).to_dataframe()

    if df.empty:
        # als er niets <= snap is, probeer >= snap
        sql2 = f"""
        WITH base AS (
          SELECT * FROM `{view}`
        )
        SELECT * FROM base
        WHERE DATE({date_col}) >= @snap
        ORDER BY DATE({date_col}) ASC
        LIMIT 1
        """
        df = client.query(sql2, job_config=bigquery.QueryJobConfig(query_parameters=params)).to_dataframe()

    if df.empty:
        raise ValueError("Geen r-row gevonden in yield-view rond de gevraagde datum.")

    row = df.iloc[0]
    cols = list(df.columns)
    cand = _detect_tenor_columns(cols)
    if not cand:
        raise ValueError("Geen tenor-kolommen gevonden in yield-rij. Controleer kolomnamen in je view.")

    # Lees rates uit de gedetecteerde kolommen
    T_nodes, r_nodes = [], []
    for col, T in cand:
        try:
            v = float(row[col])
        except Exception:
            continue
        if not (v is None or np.isnan(v)):
            T_nodes.append(float(T))
            r_nodes.append(float(v))
    if not T_nodes:
        raise ValueError("Tenor-kolommen gedetecteerd, maar geen numerieke r-waarden in de geselecteerde rij.")

    T_nodes = np.array(T_nodes, dtype=float)
    r_nodes = np.array(r_nodes, dtype=float)

    # Interpoleer (in simple apr)
    r_simple = _interp_rate(T_years, T_nodes, r_nodes, extrapolate=extrapolate)

    if output == "simple":
        return r_simple
    # default: continuous
    return _to_continuous(r_simple)

def get_q_curve_const(T_years: np.ndarray, q_const: float = 0.015, to_continuous: bool = True) -> np.ndarray:
    if not isinstance(T_years, np.ndarray):
        T_years = np.array(T_years, dtype=float)
    if to_continuous:
        return np.array([math.log1p(q_const)]*len(T_years), dtype=float)
    return np.array([q_const]*len(T_years), dtype=float)
