# utils/rates.py
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Literal

# ---- Config ----
DEFAULT_YIELD_VIEW = "nth-pier-468314-p7.marketdata.yield_curve_analysis_wide"

# Mapping van kolomnamen naar looptijd in jaren
TENOR_MAP_YEARS: Dict[str, float] = {
    "y_1w":  7/365.0,
    "y_2w": 14/365.0,
    "y_3w": 21/365.0,
    "y_1m":  1/12.0,
    "y_2m":  2/12.0,
    "y_3m":  3/12.0,
    "y_6m":  6/12.0,
    "y_9m":  9/12.0,
    "y_1y":  1.0,
    "y_2y":  2.0,
    "y_3y":  3.0,
    "y_5y":  5.0,
    "y_7y":  7.0,
    "y_10y": 10.0,
    "y_20y": 20.0,
    "y_30y": 30.0,
}

# ---- BigQuery helpers: probeer eerst jullie utils/bq, anders fallback ----
try:
    from utils.bq import run_query, bq_ping  # type: ignore
    _HAS_UTILS_BQ = True
except Exception:
    _HAS_UTILS_BQ = False
    import streamlit as st
    import google.cloud.bigquery as bq
    from google.oauth2 import service_account

    @st.cache_resource(show_spinner=False)
    def _bq_client_from_secrets():
        creds = st.secrets["gcp_service_account"]
        credentials = service_account.Credentials.from_service_account_info(creds)
        return bq.Client(credentials=credentials, project=creds["project_id"])

    def bq_ping() -> bool:
        try:
            _bq_client_from_secrets().query("SELECT 1").result(timeout=10)
            return True
        except Exception:
            return False

    def run_query(sql: str) -> pd.DataFrame:
        client = _bq_client_from_secrets()
        job = client.query(sql)
        return job.result().to_dataframe(create_bqstorage_client=False)


# ---------- Intern: data uit view halen ----------
def _load_yield_row_for_snapshot(
    snapshot_date: pd.Timestamp,
    view: str = DEFAULT_YIELD_VIEW,
) -> pd.Series:
    """
    Haalt de dichtstbijzijnde (<= snapshot_date) rij op uit de yield-view.
    Valt terug op exact- of laatst-beschikbare datum ≤ snapshot.
    """
    snap = pd.to_datetime(snapshot_date).tz_localize(None)

    sql = f"""
    WITH src AS (
      SELECT *
      FROM `{view}`
      WHERE date <= DATE('{snap.date()}')
    )
    SELECT *
    FROM src
    WHERE date = (SELECT MAX(date) FROM src)
    LIMIT 1
    """
    df = run_query(sql)
    if df.empty:
        # Probeer eventueel >= snapshot (toekomst) als niets <= snapshot is
        sql2 = f"""
        WITH src AS (
          SELECT *
          FROM `{view}`
          WHERE date >= DATE('{snap.date()}')
        )
        SELECT *
        FROM src
        WHERE date = (SELECT MIN(date) FROM src)
        LIMIT 1
        """
        df = run_query(sql2)

    if df.empty:
        raise ValueError(f"Geen yield data gevonden in view {view} rond {snap.date()}")

    row = df.iloc[0]
    return row


def _detect_tenors_in_row(row: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    """
    Leest beschikbare tenor-kolommen en geeft (tenor_years, rates) arrays terug.
    Rates worden als float (decimaal) verwacht.
    """
    tenors: List[float] = []
    rates: List[float] = []

    for col, yrs in TENOR_MAP_YEARS.items():
        if col in row.index and pd.notna(row[col]):
            tenors.append(yrs)
            rates.append(float(row[col]))

    if not tenors:
        raise ValueError("Geen tenor-kolommen gevonden in yield-rij. Controleer kolomnamen in je view.")

    tenors_arr = np.array(tenors, dtype=float)
    rates_arr = np.array(rates, dtype=float)

    # Sorteer op looptijd
    order = np.argsort(tenors_arr)
    return tenors_arr[order], rates_arr[order]


# ---------- Rente conversies ----------
def _to_continuous_rate(y: np.ndarray, compounding: Literal["cont", "simple", "act365"] = "cont") -> np.ndarray:
    """
    Zet ingevoerde jaarrentes om naar continue samengestelde r.
    - 'cont'   : veronderstelt dat y reeds continue is -> retourneer y ongewijzigd
    - 'simple' : y_simple -> r_cont = ln(1 + y)
    - 'act365' : zelfde als 'simple' voor jaarrente; dagbasis speelt bij looptijd, niet hier
    """
    if compounding == "cont":
        return y
    elif compounding in ("simple", "act365"):
        return np.log1p(y)
    else:
        raise ValueError(f"Onbekende compounding: {compounding}")


# ---------- Publieke API ----------
def get_r_curve_for_snapshot(
    snapshot_date: pd.Timestamp,
    T_years: np.ndarray,
    *,
    view: str = DEFAULT_YIELD_VIEW,
    compounding_in: Literal["cont", "simple", "act365"] = "simple",
    extrapolate: bool = True,
) -> np.ndarray:
    """
    Haal de yield-curve op voor snapshot_date en interpoleer naar de opgegeven looptijden (in jaren).
    Retourneert continue samengestelde r(T) als numpy-array (zelfde shape als T_years).
    """
    row = _load_yield_row_for_snapshot(snapshot_date, view=view)
    tenors, rates_in = _detect_tenors_in_row(row)

    # Zet naar continue samengestelde r
    r_cont = _to_continuous_rate(rates_in, compounding=compounding_in)

    T_years = np.asarray(T_years, dtype=float)
    if T_years.ndim != 1:
        T_flat = T_years.reshape(-1)
    else:
        T_flat = T_years

    # Interpolatie (lineair). Extrapolatie indien gevraagd, anders clip
    x = tenors
    y = r_cont

    if extrapolate:
        # np.interp extrapoleert niet, dus doen we het handmatig met endpoints
        r_interp = np.interp(T_flat, x, y)
        below = T_flat < x[0]
        above = T_flat > x[-1]
        if below.any():
            # lineair richting origin met eerste segment (of constant)
            # Hier: constant extrapolatie met kortste tenor is meestal prima voor DTE<1m
            r_interp[below] = y[0]
        if above.any():
            # constant extrapolatie met langste tenor
            r_interp[above] = y[-1]
    else:
        # Clip naar bereik en interpoleer
        Tc = np.clip(T_flat, x[0], x[-1])
        r_interp = np.interp(Tc, x, y)

    return r_interp.reshape(T_years.shape)


def get_r_grid_for_surface(
    snapshot_date: pd.Timestamp,
    T_vals: np.ndarray,
    K_vals: np.ndarray,
    *,
    view: str = DEFAULT_YIELD_VIEW,
    compounding_in: Literal["cont", "simple", "act365"] = "simple",
) -> np.ndarray:
    """
    Maakt een 2D-grid R(T,K) door r(T) over de strike-as te broadcasten.
    - T_vals: 1D array (bijv. unieke T in jaren uit je surface grid)
    - K_vals: 1D array (strikes)
    """
    r_T = get_r_curve_for_snapshot(
        snapshot_date=snapshot_date,
        T_years=T_vals,
        view=view,
        compounding_in=compounding_in,
        extrapolate=True,
    )
    RR = np.tile(r_T.reshape(-1, 1), (1, len(K_vals)))
    return RR


def get_q_curve_const(
    T_years: np.ndarray,
    q_const: float = 0.0,
    *,
    to_continuous: bool = True,
) -> np.ndarray:
    """
    Eenvoudige dividend-curve: constante q per jaar.
    - Als to_continuous=True, zet y_simple -> ln(1+y) om naar continue q.
    """
    if to_continuous:
        q_cont = np.log1p(q_const)
    else:
        q_cont = q_const
    T_years = np.asarray(T_years, dtype=float)
    q_T = np.full_like(T_years, q_cont, dtype=float)
    return q_T


def get_rq_grids_for_surface(
    snapshot_date: pd.Timestamp,
    T_vals_years: np.ndarray,
    K_vals: np.ndarray,
    *,
    view: str = DEFAULT_YIELD_VIEW,
    r_compounding_in: Literal["cont", "simple", "act365"] = "simple",
    q_const: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Handige combi: geeft (RR, QQ) terug voor je Greeks-surface.
    - RR: r(T) gebroadcast naar (len(T_vals) x len(K_vals))
    - QQ: q(T) idem (hier constant)
    """
    RR = get_r_grid_for_surface(
        snapshot_date=snapshot_date,
        T_vals=T_vals_years,
        K_vals=K_vals,
        view=view,
        compounding_in=r_compounding_in,
    )
    q_T = get_q_curve_const(T_vals_years, q_const=q_const, to_continuous=True)
    QQ = np.tile(q_T.reshape(-1, 1), (1, len(K_vals)))
    return RR, QQ
