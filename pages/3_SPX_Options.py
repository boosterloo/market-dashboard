import math
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from google.cloud import bigquery
from google.oauth2 import service_account
from plotly.subplots import make_subplots

from utils.rates import get_q_curve_const


# Helpers
def norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def bs_delta(S, K, iv, T, r, q, is_call):
    if any(x is None or np.isnan(x) or x <= 0 for x in [S, K]) or np.isnan(iv) or iv <= 0 or T <= 0:
        return np.nan
    d1 = (math.log(S / K) + (r - q + 0.5 * iv**2) * T) / (iv * math.sqrt(T))
    nd1 = norm_cdf(d1)
    disc_q = math.exp(-q * T)
    return disc_q * nd1 if is_call else disc_q * (nd1 - 1.0)


def annualize_std(s: pd.Series, periods_per_year: int = 252) -> float:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) < 2:
        return np.nan
    return float(s.std(ddof=0) * math.sqrt(periods_per_year))


def pick_first_on_or_after(options, target):
    after = [d for d in options if d >= target]
    return after[0] if after else (options[-1] if options else None)


def pick_closest_value(options, target, fallback=None):
    if not options or target is None or np.isnan(target):
        return fallback
    return float(min(options, key=lambda x: abs(float(x) - float(target))))


def smooth_series(y: pd.Series, window: int = 3) -> pd.Series:
    if len(y) < 3:
        return y
    return y.rolling(window, center=True, min_periods=1).median()


def pct_rank_last(s: pd.Series) -> float:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return np.nan
    return float((s <= s.iloc[-1]).mean())


def safe_float(v, default=np.nan):
    try:
        if v is None:
            return default
        return float(v)
    except Exception:
        return default


def fmt_num(v, digits=2):
    return f"{v:.{digits}f}" if not np.isnan(safe_float(v)) else "-"


def fmt_pct(v, digits=2):
    return f"{v:.{digits}%}" if not np.isnan(safe_float(v)) else "-"


# BigQuery
MAX_BYTES_BILLED = 2 * 1024**3
BQ_LOCATION = "europe-west1"
PROJECT_ID = "nth-pier-468314-p7"
VIEW = "nth-pier-468314-p7.marketdata.spx_options_enriched_v"


@st.cache_resource(show_spinner=False)
def get_bq_client():
    sa_info = st.secrets["gcp_service_account"]
    creds = service_account.Credentials.from_service_account_info(sa_info)
    return bigquery.Client(project=PROJECT_ID, credentials=creds)


_bq_client = get_bq_client()


def _bq_param(name, value):
    if isinstance(value, (list, tuple)):
        if not value:
            return bigquery.ArrayQueryParameter(name, "STRING", [])
        e = value[0]
        if isinstance(e, int):
            return bigquery.ArrayQueryParameter(name, "INT64", list(value))
        if isinstance(e, float):
            return bigquery.ArrayQueryParameter(name, "FLOAT64", list(value))
        if isinstance(e, (date, pd.Timestamp, datetime)):
            vals = [str(pd.to_datetime(v).date()) for v in value]
            return bigquery.ArrayQueryParameter(name, "DATE", vals)
        return bigquery.ArrayQueryParameter(name, "STRING", [str(v) for v in value])

    if isinstance(value, bool):
        return bigquery.ScalarQueryParameter(name, "BOOL", value)
    if isinstance(value, (int, np.integer)):
        return bigquery.ScalarQueryParameter(name, "INT64", int(value))
    if isinstance(value, (float, np.floating)):
        return bigquery.ScalarQueryParameter(name, "FLOAT64", float(value))
    if isinstance(value, datetime):
        return bigquery.ScalarQueryParameter(name, "TIMESTAMP", value)
    if isinstance(value, (date, pd.Timestamp)):
        return bigquery.ScalarQueryParameter(name, "DATE", str(pd.to_datetime(value).date()))
    return bigquery.ScalarQueryParameter(name, "STRING", str(value))


def run_query(sql: str, params: dict | None = None) -> pd.DataFrame:
    job_config = bigquery.QueryJobConfig(
        maximum_bytes_billed=MAX_BYTES_BILLED
    )

    if params:
        job_config.query_parameters = [_bq_param(k, v) for k, v in params.items()]

    job = _bq_client.query(
        sql,
        job_config=job_config,
        location=BQ_LOCATION,
    )

    return job.to_dataframe(create_bqstorage_client=False)


# Styling
st.set_page_config(page_title="SPX Options Dashboard", layout="wide")
st.title("SPX Options Dashboard")

PLOTLY_CONFIG = {
    "scrollZoom": True,
    "doubleClick": "reset",
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
}

_BASE = 12
_DEF_MARGIN = dict(l=92, r=62, t=68, b=72)


def _axis_size():
    mult = st.session_state.get("_axis_mult", 1.5)
    return int(_BASE * mult)


def amplify_axes(fig, height: int | None = None, legend_top: bool = True):
    sz = _axis_size()
    fig.update_xaxes(
        tickfont=dict(size=sz),
        title_font=dict(size=sz),
        automargin=True,
        showgrid=True,
        gridwidth=1,
        gridcolor="rgba(0,0,0,0.10)",
        ticks="outside",
    )
    fig.update_yaxes(
        tickfont=dict(size=sz),
        title_font=dict(size=sz),
        automargin=True,
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor="rgba(0,0,0,0.25)",
        showgrid=True,
        gridwidth=1,
        gridcolor="rgba(0,0,0,0.10)",
    )
    fig.update_layout(
        margin=_DEF_MARGIN,
        font=dict(size=max(sz - 6, 13)),
        legend=dict(
            font=dict(size=max(sz - 8, 12)),
            orientation="h",
            yanchor="bottom",
            y=1.02 if legend_top else -0.22,
            xanchor="left",
            x=0.0,
        ),
        title=dict(font=dict(size=sz)),
        hoverlabel=dict(font_size=max(sz - 8, 12)),
        dragmode="zoom",
        autosize=True,
        plot_bgcolor="white",
    )
    if height:
        fig.update_layout(height=height)
    return fig


def show_fig(fig, height: int | None = None, legend_top: bool = True):
    amplify_axes(fig, height=height, legend_top=legend_top)
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG, theme=None)


def section_title(title: str, subtitle: str):
    st.subheader(title)
    st.caption(subtitle)


def add_pcr_midline(fig, row: int, col: int):
    fig.add_hline(y=1.0, line=dict(dash="dot", width=1), row=row, col=col)


# Data loaders
@st.cache_data(ttl=600, show_spinner=False)
def load_date_bounds():
    df = run_query(
        f"""
        SELECT
          MIN(DATE(snapshot_date)) AS min_date,
          MAX(DATE(snapshot_date)) AS max_date
        FROM `{VIEW}`
        WHERE DATE(snapshot_date) >= DATE_SUB(CURRENT_DATE(), INTERVAL 400 DAY)
        """
    )
    return df["min_date"].iloc[0], df["max_date"].iloc[0]


@st.cache_data(ttl=600, show_spinner=False)
def load_snapshot_freshness():
    df = run_query(
        f"""
        SELECT
          MAX(CAST(snapshot_date AS TIMESTAMP)) AS latest_snapshot,
          COUNT(*) AS rows_total,
          COUNTIF(DATE(snapshot_date) = (
            SELECT MAX(DATE(snapshot_date))
            FROM `{VIEW}`
            WHERE DATE(snapshot_date) >= DATE_SUB(CURRENT_DATE(), INTERVAL 400 DAY)
          )) AS rows_latest_day
        FROM `{VIEW}`
        WHERE DATE(snapshot_date) >= DATE_SUB(CURRENT_DATE(), INTERVAL 400 DAY)
        """
    )
    if df.empty:
        return None, 0, 0
    return (
        pd.to_datetime(df["latest_snapshot"].iloc[0]) if pd.notna(df["latest_snapshot"].iloc[0]) else None,
        int(df["rows_total"].iloc[0] or 0),
        int(df["rows_latest_day"].iloc[0] or 0),
    )


@st.cache_data(ttl=600, show_spinner=False)
def load_snapshots(start_date: date, end_date: date):
    df = run_query(
        f"""
        SELECT DISTINCT TIMESTAMP_TRUNC(CAST(snapshot_date AS TIMESTAMP), MINUTE) AS snap_min
        FROM `{VIEW}`
        WHERE DATE(snapshot_date) BETWEEN @start AND @end
        ORDER BY snap_min
        """,
        {"start": start_date, "end": end_date},
    )
    if df.empty:
        return []
    return sorted(pd.to_datetime(df["snap_min"]).dt.to_pydatetime().tolist())


@st.cache_data(ttl=600, show_spinner=False)
def load_snapshot_context(sel_snapshot):
    df = run_query(
        f"""
        SELECT
          AVG(CAST(underlying_price AS FLOAT64)) AS spx,
          AVG(CAST(vix AS FLOAT64)) AS vix
        FROM `{VIEW}`
        WHERE DATE(snapshot_date) = DATE(@snap_min)
          AND TIMESTAMP_TRUNC(CAST(snapshot_date AS TIMESTAMP), MINUTE) = @snap_min
        """,
        {"snap_min": pd.to_datetime(sel_snapshot).to_pydatetime()},
    )
    if df.empty:
        return np.nan, np.nan
    return safe_float(df["spx"].iloc[0]), safe_float(df["vix"].iloc[0])


@st.cache_data(ttl=600, show_spinner=False)
def load_strikes_for_snapshot(sel_snapshot, sel_type, min_oi, min_vol):
    df = run_query(
        f"""
        SELECT DISTINCT CAST(strike AS FLOAT64) AS strike
        FROM `{VIEW}`
        WHERE DATE(snapshot_date) = DATE(@snap_min)
          AND TIMESTAMP_TRUNC(CAST(snapshot_date AS TIMESTAMP), MINUTE) = @snap_min
          AND LOWER(type) = @t
          AND (COALESCE(open_interest, 0) >= @min_oi OR COALESCE(volume, 0) >= @min_vol)
        ORDER BY strike
        """,
        {
            "snap_min": pd.to_datetime(sel_snapshot).to_pydatetime(),
            "t": sel_type,
            "min_oi": int(min_oi),
            "min_vol": int(min_vol),
        },
    )
    return sorted([float(x) for x in df["strike"].dropna().tolist()])


@st.cache_data(ttl=600, show_spinner=False)
def load_expirations_for_option(sel_snapshot, sel_type, strike, min_oi, min_vol):
    df = run_query(
        f"""
        SELECT DISTINCT expiration
        FROM `{VIEW}`
        WHERE DATE(snapshot_date) = DATE(@snap_min)
          AND TIMESTAMP_TRUNC(CAST(snapshot_date AS TIMESTAMP), MINUTE) = @snap_min
          AND LOWER(type) = @t
          AND ABS(CAST(strike AS FLOAT64) - @strike) < 0.0001
          AND (COALESCE(open_interest, 0) >= @min_oi OR COALESCE(volume, 0) >= @min_vol)
        ORDER BY expiration
        """,
        {
            "snap_min": pd.to_datetime(sel_snapshot).to_pydatetime(),
            "t": sel_type,
            "strike": float(strike),
            "min_oi": int(min_oi),
            "min_vol": int(min_vol),
        },
    )
    if df.empty:
        return []
    return sorted(pd.to_datetime(df["expiration"]).dt.date.unique())


@st.cache_data(ttl=600, show_spinner=False)
def load_daily_market_context(start_date, end_date, dte_min, dte_max, mny_min, mny_max, min_oi, min_vol):
    sql = f"""
    WITH base AS (
      SELECT
        DATE(snapshot_date) AS day,
        LOWER(type) AS type,
        CAST(underlying_price AS FLOAT64) AS underlying_price,
        CAST(open_interest AS FLOAT64) AS open_interest,
        CAST(volume AS FLOAT64) AS volume,
        CAST(implied_volatility AS FLOAT64) AS implied_volatility,
        CAST(vix AS FLOAT64) AS vix,
        CAST(days_to_exp AS INT64) AS days_to_exp,
        SAFE_DIVIDE(CAST(strike AS FLOAT64), NULLIF(CAST(underlying_price AS FLOAT64), 0)) - 1.0 AS moneyness
      FROM `{VIEW}`
      WHERE DATE(snapshot_date) BETWEEN @start AND @end
        AND days_to_exp BETWEEN @dte_min AND @dte_max
        AND SAFE_DIVIDE(CAST(strike AS FLOAT64), NULLIF(CAST(underlying_price AS FLOAT64), 0)) - 1.0
            BETWEEN @mny_min AND @mny_max
        AND (COALESCE(open_interest, 0) >= @min_oi OR COALESCE(volume, 0) >= @min_vol)
    )
    SELECT
      day,
      AVG(underlying_price) AS spx,
      AVG(vix) AS vix,
      AVG(implied_volatility) AS avg_iv,
      SUM(CASE WHEN type = 'put' THEN volume ELSE 0 END) AS vol_put,
      SUM(CASE WHEN type = 'call' THEN volume ELSE 0 END) AS vol_call,
      SUM(CASE WHEN type = 'put' THEN open_interest ELSE 0 END) AS oi_put,
      SUM(CASE WHEN type = 'call' THEN open_interest ELSE 0 END) AS oi_call
    FROM base
    GROUP BY day
    ORDER BY day
    """
    df = run_query(
        sql,
        {
            "start": start_date,
            "end": end_date,
            "dte_min": int(dte_min),
            "dte_max": int(dte_max),
            "mny_min": float(mny_min),
            "mny_max": float(mny_max),
            "min_oi": int(min_oi),
            "min_vol": int(min_vol),
        },
    )
    if not df.empty:
        df["day"] = pd.to_datetime(df["day"])
        for col in ["spx", "vix", "avg_iv", "vol_put", "vol_call", "oi_put", "oi_call"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


@st.cache_data(ttl=600, show_spinner=False)
def load_skew_source(start_date, end_date, dte_lo, dte_hi, min_oi, min_vol):
    df = run_query(
        f"""
        SELECT
          snapshot_date,
          expiration,
          LOWER(type) AS type,
          CAST(days_to_exp AS INT64) AS days_to_exp,
          CAST(strike AS FLOAT64) AS strike,
          CAST(underlying_price AS FLOAT64) AS underlying_price,
          CAST(implied_volatility AS FLOAT64) AS implied_volatility,
          CAST(open_interest AS FLOAT64) AS open_interest,
          CAST(volume AS FLOAT64) AS volume
        FROM `{VIEW}`
        WHERE DATE(snapshot_date) BETWEEN @start AND @end
          AND days_to_exp BETWEEN @dte_lo AND @dte_hi
          AND (COALESCE(open_interest, 0) >= @min_oi OR COALESCE(volume, 0) >= @min_vol)
        """,
        {
            "start": start_date,
            "end": end_date,
            "dte_lo": int(dte_lo),
            "dte_hi": int(dte_hi),
            "min_oi": int(min_oi),
            "min_vol": int(min_vol),
        },
    )
    if not df.empty:
        df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
        df["expiration"] = pd.to_datetime(df["expiration"]).dt.date
    return df


@st.cache_data(ttl=600, show_spinner=False)
def load_ppd_expiry_ladder_for_strike(sel_snapshot, sel_type, strike, dte_min, dte_max, min_oi, min_vol):
    df = run_query(
        f"""
        SELECT
          snapshot_date,
          expiration,
          LOWER(type) AS type,
          CAST(days_to_exp AS INT64) AS days_to_exp,
          CAST(strike AS FLOAT64) AS strike,
          CAST(underlying_price AS FLOAT64) AS underlying_price,
          SAFE_DIVIDE(CAST(strike AS FLOAT64), NULLIF(CAST(underlying_price AS FLOAT64), 0)) - 1.0 AS moneyness,
          CAST(last_price AS FLOAT64) AS last_price,
          CAST(mid_price AS FLOAT64) AS mid_price,
          CAST(bid AS FLOAT64) AS bid,
          CAST(ask AS FLOAT64) AS ask,
          CAST(implied_volatility AS FLOAT64) AS implied_volatility,
          CAST(open_interest AS FLOAT64) AS open_interest,
          CAST(volume AS FLOAT64) AS volume,
          CAST(ppd AS FLOAT64) AS ppd
        FROM `{VIEW}`
        WHERE DATE(snapshot_date) = DATE(@snap_min)
          AND TIMESTAMP_TRUNC(CAST(snapshot_date AS TIMESTAMP), MINUTE) = @snap_min
          AND LOWER(type) = @t
          AND ABS(CAST(strike AS FLOAT64) - @strike) < 0.0001
          AND days_to_exp BETWEEN @dte_min AND @dte_max
          AND (COALESCE(open_interest, 0) >= @min_oi OR COALESCE(volume, 0) >= @min_vol)
        ORDER BY expiration
        """,
        {
            "snap_min": pd.to_datetime(sel_snapshot).to_pydatetime(),
            "t": sel_type,
            "strike": float(strike),
            "dte_min": int(dte_min),
            "dte_max": int(dte_max),
            "min_oi": int(min_oi),
            "min_vol": int(min_vol),
        },
    )
    if not df.empty:
        df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
        df["expiration"] = pd.to_datetime(df["expiration"]).dt.date
        for col in [
            "days_to_exp",
            "strike",
            "underlying_price",
            "moneyness",
            "last_price",
            "mid_price",
            "bid",
            "ask",
            "implied_volatility",
            "open_interest",
            "volume",
            "ppd",
        ]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["spread"] = df["ask"] - df["bid"]
        df["spread_pct_mid"] = df["spread"] / df["mid_price"].replace(0, np.nan)
    return df


@st.cache_data(ttl=600, show_spinner=False)
def load_option_history(start_date, end_date, sel_type, strike, expiration, min_oi, min_vol):
    df = run_query(
        f"""
        SELECT
          snapshot_date,
          expiration,
          LOWER(type) AS type,
          CAST(days_to_exp AS INT64) AS days_to_exp,
          CAST(strike AS FLOAT64) AS strike,
          CAST(underlying_price AS FLOAT64) AS underlying_price,
          CAST(last_price AS FLOAT64) AS last_price,
          CAST(mid_price AS FLOAT64) AS mid_price,
          CAST(bid AS FLOAT64) AS bid,
          CAST(ask AS FLOAT64) AS ask,
          CAST(implied_volatility AS FLOAT64) AS implied_volatility,
          CAST(open_interest AS FLOAT64) AS open_interest,
          CAST(volume AS FLOAT64) AS volume,
          CAST(ppd AS FLOAT64) AS ppd
        FROM `{VIEW}`
        WHERE DATE(snapshot_date) BETWEEN @start AND @end
          AND LOWER(type) = @t
          AND ABS(CAST(strike AS FLOAT64) - @strike) < 0.0001
          AND expiration = @expiration
          AND (COALESCE(open_interest, 0) >= @min_oi OR COALESCE(volume, 0) >= @min_vol)
        ORDER BY snapshot_date
        """,
        {
            "start": start_date,
            "end": end_date,
            "t": sel_type,
            "strike": float(strike),
            "expiration": expiration,
            "min_oi": int(min_oi),
            "min_vol": int(min_vol),
        },
    )
    if not df.empty:
        df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
        df["expiration"] = pd.to_datetime(df["expiration"]).dt.date
        for col in ["days_to_exp", "strike", "underlying_price", "last_price", "mid_price", "bid", "ask", "implied_volatility", "open_interest", "volume", "ppd"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


@st.cache_data(ttl=600, show_spinner=False)
def load_snapshot_chain(sel_snapshot, sel_type, dte_min, dte_max, min_oi, min_vol):
    df = run_query(
        f"""
        SELECT
          snapshot_date,
          expiration,
          LOWER(type) AS type,
          CAST(days_to_exp AS INT64) AS days_to_exp,
          CAST(strike AS FLOAT64) AS strike,
          CAST(underlying_price AS FLOAT64) AS underlying_price,
          SAFE_DIVIDE(CAST(strike AS FLOAT64), NULLIF(CAST(underlying_price AS FLOAT64), 0)) - 1.0 AS moneyness,
          CAST(strike AS FLOAT64) - CAST(underlying_price AS FLOAT64) AS dist_points,
          CAST(mid_price AS FLOAT64) AS mid_price,
          CAST(bid AS FLOAT64) AS bid,
          CAST(ask AS FLOAT64) AS ask,
          CAST(implied_volatility AS FLOAT64) AS implied_volatility,
          CAST(open_interest AS FLOAT64) AS open_interest,
          CAST(volume AS FLOAT64) AS volume,
          CAST(ppd AS FLOAT64) AS ppd
        FROM `{VIEW}`
        WHERE DATE(snapshot_date) = DATE(@snap_min)
          AND TIMESTAMP_TRUNC(CAST(snapshot_date AS TIMESTAMP), MINUTE) = @snap_min
          AND LOWER(type) = @t
          AND days_to_exp BETWEEN @dte_min AND @dte_max
          AND (COALESCE(open_interest, 0) >= @min_oi OR COALESCE(volume, 0) >= @min_vol)
        """,
        {
            "snap_min": pd.to_datetime(sel_snapshot).to_pydatetime(),
            "t": sel_type,
            "dte_min": int(dte_min),
            "dte_max": int(dte_max),
            "min_oi": int(min_oi),
            "min_vol": int(min_vol),
        },
    )
    if not df.empty:
        df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
        df["expiration"] = pd.to_datetime(df["expiration"]).dt.date
        for col in ["days_to_exp", "strike", "underlying_price", "moneyness", "dist_points", "mid_price", "bid", "ask", "implied_volatility", "open_interest", "volume", "ppd"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["abs_dist_pts"] = df["dist_points"].abs()
        df["abs_dist_pct"] = 100.0 * df["abs_dist_pts"] / df["underlying_price"].replace(0, np.nan)
    return df


def clear_option_caches():
    st.cache_data.clear()


def as_utc_naive(ts) -> pd.Timestamp:
    t = pd.to_datetime(ts)
    if t.tzinfo is not None:
        return t.tz_convert("UTC").tz_localize(None)
    return t


# Global controls
if st.button("Ververs SPX option data uit BigQuery"):
    clear_option_caches()
    st.rerun()

min_date, max_date = load_date_bounds()
latest_snapshot, rows_total, rows_latest_day = load_snapshot_freshness()
default_start = max(min_date, max_date - timedelta(days=14))

if latest_snapshot is not None:
    latest_snapshot_utc = as_utc_naive(latest_snapshot)
    age_hours = (pd.Timestamp.utcnow().tz_localize(None) - latest_snapshot_utc).total_seconds() / 3600
    st.caption(
        "SPX option view: "
        f"laatste snapshot {latest_snapshot_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC "
        f"({age_hours:.1f} uur oud) | rows op laatste dag: {rows_latest_day:,} | view: `{VIEW}`"
    )
    if age_hours > 36:
        st.warning(
            "De BigQuery-view zelf lijkt achter te lopen. "
            "Als de scraper wel groen is, schrijft hij waarschijnlijk niet door naar deze enriched view."
        )
else:
    st.warning(f"Geen snapshots gevonden in `{VIEW}`.")

with st.expander("Algemene filters", expanded=True):
    c0, c1, c2 = st.columns([1.1, 1.8, 1.1])
    with c0:
        axis_scale = st.slider("As-lettergrootte", 1.2, 2.2, 1.5, 0.1)
        st.session_state["_axis_mult"] = axis_scale
    with c1:
        range_choice = st.radio("Periode voor marktcontext", ["14d", "30d", "90d", "1y", "Custom"], index=0, horizontal=True)
    with c2:
        q_const_simple = st.number_input("Dividendrendement q", 0.0, 0.10, 0.016, 0.001, format="%.3f")

    if range_choice != "Custom":
        days_map = {"14d": 14, "30d": 30, "90d": 90, "1y": 365}
        start_date = max(min_date, max_date - timedelta(days=days_map[range_choice]))
        end_date = max_date
        st.caption(f"Periode: {start_date} t/m {end_date}")
    else:
        start_date, end_date = st.date_input(
            "Periode",
            value=(default_start, max_date),
            min_value=min_date,
            max_value=max_date,
            format="YYYY-MM-DD",
        )

    f1, f2, f3, f4 = st.columns([1.2, 1.2, 1, 1])
    with f1:
        context_dte_range = st.slider("DTE voor marktcontext", 0, 365, (0, 60), step=1)
    with f2:
        context_mny_range = st.slider("Moneyness voor marktcontext", -0.30, 0.30, (-0.15, 0.15), step=0.01)
    with f3:
        min_oi = st.slider("Min OI", 0, 100, 1, step=1)
    with f4:
        min_vol = st.slider("Min Volume", 0, 100, 1, step=1)


snapshots_all = load_snapshots(start_date, end_date)
if not snapshots_all:
    st.warning("Geen snapshots gevonden voor deze periode.")
    st.stop()

default_snapshot = snapshots_all[-1]

market = load_daily_market_context(
    start_date,
    end_date,
    context_dte_range[0],
    context_dte_range[1],
    context_mny_range[0],
    context_mny_range[1],
    min_oi,
    min_vol,
)

if market.empty:
    st.warning("Geen marktdata voor de huidige algemene filters.")
    st.stop()

market["PCR_vol"] = market["vol_put"] / market["vol_call"].replace(0, np.nan)
market["PCR_oi"] = market["oi_put"] / market["oi_call"].replace(0, np.nan)
market["PCR_gap"] = market["PCR_vol"] - market["PCR_oi"]

u_daily = market[["day", "spx"]].dropna().copy()
u_daily["ret"] = u_daily["spx"].pct_change()
hv20 = annualize_std(u_daily["ret"].tail(21).dropna())

iv_rank = pct_rank_last(market["avg_iv"].tail(252))
last_pcr_vol = float(market["PCR_vol"].dropna().iloc[-1]) if market["PCR_vol"].notna().any() else np.nan
last_pcr_oi = float(market["PCR_oi"].dropna().iloc[-1]) if market["PCR_oi"].notna().any() else np.nan
last_pcr_gap = float(market["PCR_gap"].dropna().iloc[-1]) if market["PCR_gap"].notna().any() else np.nan
last_iv = float(market["avg_iv"].dropna().iloc[-1]) if market["avg_iv"].notna().any() else np.nan
last_vix = float(market["vix"].dropna().iloc[-1]) if market["vix"].notna().any() else np.nan
last_spx = float(market["spx"].dropna().iloc[-1]) if market["spx"].notna().any() else np.nan
vrp = last_iv - hv20 if not np.isnan(last_iv) and not np.isnan(hv20) else np.nan

# Header KPIs
k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.metric("SPX", fmt_num(last_spx, 1))
with k2:
    st.metric("VIX", fmt_num(last_vix, 2))
with k3:
    st.metric("PCR Volume", fmt_num(last_pcr_vol, 2))
with k4:
    st.metric("PCR OI", fmt_num(last_pcr_oi, 2))
with k5:
    st.metric("VRP IV-HV20", fmt_pct(vrp, 2))

st.caption(
    "Tip: gebruik vooral de tab 'Beste expiratie voor short optie' om te kiezen welke maturity het meeste PPD oplevert voor een gekozen far OTM strike."
)


def snapshot_selectbox(label, key):
    return st.selectbox(
        label,
        options=snapshots_all,
        index=len(snapshots_all) - 1,
        format_func=lambda x: pd.to_datetime(x).strftime("%Y-%m-%d %H:%M"),
        key=key,
    )


def type_radio(label, key, default="put"):
    options = ["put", "call"]
    index = options.index(default)
    return st.radio(label, options, index=index, horizontal=True, key=key)


def strike_selectbox(label, sel_snapshot, sel_type, key, default_offset_points=400.0):
    spot, _ = load_snapshot_context(sel_snapshot)
    strikes = load_strikes_for_snapshot(sel_snapshot, sel_type, min_oi, min_vol)
    if not strikes:
        st.warning("Geen strikes gevonden voor deze selectie.")
        return None, spot, []

    target = spot - default_offset_points if sel_type == "put" else spot + default_offset_points
    default_strike = pick_closest_value(strikes, target, fallback=strikes[0])
    default_index = strikes.index(default_strike) if default_strike in strikes else 0

    strike = st.selectbox(
        label,
        options=strikes,
        index=default_index,
        key=key,
        format_func=lambda x: f"{x:.0f}",
    )
    st.caption(f"Spot op peildatum: {fmt_num(spot, 1)} | standaard rond {'SPX - 400' if sel_type == 'put' else 'SPX + 400'}")
    return strike, spot, strikes


st.markdown(
    """
    <style>
    div[data-testid="stRadio"] > label {
        font-weight: 700;
        font-size: 1.05rem;
        margin-bottom: 0.45rem;
    }

    div[data-testid="stRadio"] div[role="radiogroup"] {
        display: flex;
        gap: 0.55rem;
        flex-wrap: wrap;
    }

    div[data-testid="stRadio"] div[role="radiogroup"] label {
        background: #f7f8fb;
        border: 1px solid rgba(49, 51, 63, 0.20);
        border-radius: 999px;
        padding: 0.65rem 0.95rem;
        min-height: 42px;
        cursor: pointer;
        transition: all 0.15s ease;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04);
    }

    div[data-testid="stRadio"] div[role="radiogroup"] label:hover {
        background: #fff5f5;
        border-color: #ff4b4b;
        box-shadow: 0 2px 6px rgba(255, 75, 75, 0.15);
    }

    div[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) {
        background: #ff4b4b;
        border-color: #ff4b4b;
        color: white;
        font-weight: 700;
        box-shadow: 0 3px 10px rgba(255, 75, 75, 0.28);
    }

    div[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) p,
    div[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) div {
        color: white;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

page = st.radio(
    "Onderdeel",
    [
        "📊 Marktbeeld",
        "🎯 Beste expiratie",
        "🕰️ Optiehistoriek",
        "📏 PPD naar afstand",
        "🌊 Term structure en smile",
    ],
    horizontal=True,
)


if page == "📊 Marktbeeld":
    section_title(
        "Marktbeeld",
        "Algemene context: SPX, VIX, put/call-ratio, implied volatility en volatiliteitspremie.",
    )

    fig_pcr = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=("PCR Volume: actuele flow", "PCR OI: bestaande positioning", "PCR Gap: flow minus positioning"),
    )
    fig_pcr.add_trace(go.Scatter(x=market["day"], y=market["PCR_vol"], mode="lines", name="PCR Volume"), row=1, col=1)
    fig_pcr.add_trace(go.Scatter(x=market["day"], y=market["PCR_oi"], mode="lines", name="PCR OI"), row=2, col=1)
    fig_pcr.add_trace(go.Bar(x=market["day"], y=market["PCR_gap"], name="PCR Gap"), row=3, col=1)
    add_pcr_midline(fig_pcr, 1, 1)
    add_pcr_midline(fig_pcr, 2, 1)
    show_fig(fig_pcr, height=740)

    fig_ctx = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=("SPX", "VIX", "Gemiddelde IV"),
    )
    fig_ctx.add_trace(go.Scatter(x=market["day"], y=market["spx"], mode="lines", name="SPX"), row=1, col=1)
    fig_ctx.add_trace(go.Scatter(x=market["day"], y=market["vix"], mode="lines", name="VIX"), row=2, col=1)
    fig_ctx.add_trace(go.Scatter(x=market["day"], y=market["avg_iv"], mode="lines", name="IV"), row=3, col=1)
    show_fig(fig_ctx, height=740)

    st.info(
        f"IV-rank: {fmt_pct(iv_rank, 0)} | HV20: {fmt_pct(hv20, 2)} | "
        f"VRP: {fmt_pct(vrp, 2)} | PCR-gap: {fmt_num(last_pcr_gap, 2)}"
    )


elif page == "🎯 Beste expiratie":
    section_title(
        "Beste expiratie voor short optie",
        "Kies een peildatum, put/call en strike. De grafiek vergelijkt alle expiraties voor die strike. Links staat PPD, rechts de optieprijs.",
    )

    c1, c2, c3 = st.columns([1.2, 0.8, 1.2])
    with c1:
        ladder_snapshot = snapshot_selectbox("Peildatum", "ladder_snapshot")
    with c2:
        ladder_type = type_radio("Type", "ladder_type", default="put")
    with c3:
        ladder_strike, ladder_spot, _ = strike_selectbox("Strike", ladder_snapshot, ladder_type, "ladder_strike")

    c4, c5, c6 = st.columns([1.2, 1, 1])
    with c4:
        ladder_dte = st.slider("DTE-bereik", 1, 365, (1, 180), step=1, key="ladder_dte")
    with c5:
        ladder_price_col = st.radio("Prijs op rechteras", ["mid_price", "bid", "last_price"], index=0, horizontal=True, key="ladder_price")
    with c6:
        ladder_smooth = st.toggle("Smooth PPD", value=True, key="ladder_smooth")

    if ladder_strike is not None:
        ladder = load_ppd_expiry_ladder_for_strike(
            ladder_snapshot,
            ladder_type,
            ladder_strike,
            ladder_dte[0],
            ladder_dte[1],
            min_oi,
            min_vol,
        )

        if ladder.empty:
            st.info("Geen expiry ladder gevonden voor deze peildatum, type en strike.")
        else:
            ladder = ladder.dropna(subset=["expiration", "days_to_exp", "ppd"]).sort_values("expiration").copy()
            ladder["ppd_s"] = smooth_series(ladder["ppd"], 3)
            ladder["T"] = ladder["days_to_exp"] / 365.0
            ladder["q"] = get_q_curve_const(ladder["T"].to_numpy(dtype=float), q_const=q_const_simple, to_continuous=True)
            ladder["delta"] = ladder.apply(
                lambda r: bs_delta(
                    float(r["underlying_price"]),
                    float(r["strike"]),
                    float(r["implied_volatility"]),
                    float(r["T"]),
                    0.0,
                    float(r["q"]),
                    is_call=(ladder_type == "call"),
                ),
                axis=1,
            )

            y_for_best = "ppd_s" if ladder_smooth and ladder["ppd_s"].notna().any() else "ppd"
            best = ladder.loc[ladder[y_for_best].idxmax()] if ladder[y_for_best].notna().any() else None

            fig = make_subplots(specs=[[{"secondary_y": True}]])

            fig.add_trace(
                go.Scatter(
                    x=ladder["expiration"],
                    y=ladder["ppd"],
                    mode="markers",
                    name="PPD",
                    customdata=np.stack(
                        [
                            ladder["days_to_exp"],
                            ladder[ladder_price_col],
                            ladder["bid"],
                            ladder["ask"],
                            ladder["spread_pct_mid"],
                            ladder["implied_volatility"],
                            ladder["delta"],
                            ladder["moneyness"] * 100.0,
                            ladder["open_interest"],
                            ladder["volume"],
                        ],
                        axis=-1,
                    ),
                    hovertemplate=(
                        "Expiratie=%{x}<br>"
                        "DTE=%{customdata[0]:.0f}<br>"
                        "PPD=%{y:.2f}<br>"
                        "Prijs=%{customdata[1]:.2f}<br>"
                        "Bid/Ask=%{customdata[2]:.2f} / %{customdata[3]:.2f}<br>"
                        "Spread/mid=%{customdata[4]:.1%}<br>"
                        "IV=%{customdata[5]:.2%}<br>"
                        "Delta=%{customdata[6]:.3f}<br>"
                        "Moneyness=%{customdata[7]:.1f}%<br>"
                        "OI=%{customdata[8]:.0f}<br>"
                        "Volume=%{customdata[9]:.0f}<extra></extra>"
                    ),
                ),
                secondary_y=False,
            )

            if ladder_smooth:
                fig.add_trace(
                    go.Scatter(x=ladder["expiration"], y=ladder["ppd_s"], mode="lines", name="PPD smooth"),
                    secondary_y=False,
                )

            fig.add_trace(
                go.Scatter(
                    x=ladder["expiration"],
                    y=ladder[ladder_price_col],
                    mode="lines+markers",
                    name=ladder_price_col,
                ),
                secondary_y=True,
            )

            if best is not None:
                fig.add_vline(x=best["expiration"], line=dict(dash="dot", width=2))

            fig.update_xaxes(title_text="Expiratiedatum", type="date", tickformat="%d-%m-%Y")
            fig.update_yaxes(title_text="PPD", secondary_y=False)
            fig.update_yaxes(title_text="Optieprijs", secondary_y=True)
            fig.update_layout(
                title=(
                    f"{ladder_type.upper()} {ladder_strike:.0f}: PPD en prijs per expiratie "
                    f"op {pd.to_datetime(ladder_snapshot).strftime('%Y-%m-%d %H:%M')}"
                )
            )
            show_fig(fig, height=520)

            if best is not None:
                st.success(
                    f"Hoogste PPD ligt bij expiratie {best['expiration']} "
                    f"({int(best['days_to_exp'])} DTE): PPD {best[y_for_best]:.2f}, "
                    f"{ladder_price_col} {best[ladder_price_col]:.2f}, "
                    f"delta {best['delta']:.3f}, spread/mid {best['spread_pct_mid']:.1%}."
                )

            st.dataframe(
                ladder[
                    [
                        "expiration",
                        "days_to_exp",
                        "ppd",
                        ladder_price_col,
                        "bid",
                        "ask",
                        "spread_pct_mid",
                        "implied_volatility",
                        "delta",
                        "moneyness",
                        "open_interest",
                        "volume",
                    ]
                ].rename(
                    columns={
                        "expiration": "Expiratie",
                        "days_to_exp": "DTE",
                        "ppd": "PPD",
                        ladder_price_col: "Prijs",
                        "bid": "Bid",
                        "ask": "Ask",
                        "spread_pct_mid": "Spread/mid",
                        "implied_volatility": "IV",
                        "delta": "Delta",
                        "moneyness": "Moneyness",
                        "open_interest": "OI",
                        "volume": "Volume",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )


elif page == "🕰️ Optiehistoriek":
    section_title(
        "Historiek van een gekozen optie",
        "Volg een specifieke optie door de tijd: peildatum voor defaults, put/call, strike en expiratie.",
    )

    h1, h2, h3 = st.columns([1.2, 0.8, 1.2])
    with h1:
        hist_snapshot = snapshot_selectbox("Peildatum voor standaardkeuze", "hist_snapshot")
    with h2:
        hist_type = type_radio("Type", "hist_type", default="put")
    with h3:
        hist_strike, _, _ = strike_selectbox("Strike", hist_snapshot, hist_type, "hist_strike")

    if hist_strike is not None:
        hist_exps = load_expirations_for_option(hist_snapshot, hist_type, hist_strike, min_oi, min_vol)
        if not hist_exps:
            st.info("Geen expiraties gevonden voor deze optie.")
        else:
            target_exp = date.today() + timedelta(days=14)
            default_exp = pick_first_on_or_after(hist_exps, target_exp)
            h4, h5 = st.columns([1.2, 1])
            with h4:
                hist_exp = st.selectbox(
                    "Expiratie",
                    options=hist_exps,
                    index=hist_exps.index(default_exp) if default_exp in hist_exps else 0,
                    key="hist_exp",
                )
            with h5:
                hist_price_col = st.radio("Prijsbron", ["mid_price", "last_price", "bid"], index=0, horizontal=True, key="hist_price")

            hist = load_option_history(start_date, end_date, hist_type, hist_strike, hist_exp, min_oi, min_vol)

            if hist.empty:
                st.info("Geen historische data voor deze optie.")
            else:
                s1, s2 = st.columns(2)
                with s1:
                    fig_price = make_subplots(specs=[[{"secondary_y": True}]])
                    fig_price.add_trace(go.Scatter(x=hist["snapshot_date"], y=hist[hist_price_col], mode="lines+markers", name="Optieprijs"), secondary_y=False)
                    fig_price.add_trace(go.Scatter(x=hist["snapshot_date"], y=hist["underlying_price"], mode="lines", line=dict(dash="dot"), name="SPX"), secondary_y=True)
                    fig_price.update_yaxes(title_text="Optieprijs", secondary_y=False)
                    fig_price.update_yaxes(title_text="SPX", secondary_y=True)
                    fig_price.update_layout(title=f"{hist_type.upper()} {hist_strike:.0f} exp {hist_exp}: prijs vs SPX")
                    show_fig(fig_price, height=430)

                with s2:
                    fig_iv = make_subplots(specs=[[{"secondary_y": True}]])
                    fig_iv.add_trace(go.Scatter(x=hist["snapshot_date"], y=hist["implied_volatility"], mode="lines+markers", name="IV"), secondary_y=False)
                    fig_iv.add_trace(go.Scatter(x=hist["snapshot_date"], y=hist["ppd"], mode="lines+markers", name="PPD"), secondary_y=True)
                    fig_iv.update_yaxes(title_text="IV", secondary_y=False)
                    fig_iv.update_yaxes(title_text="PPD", secondary_y=True)
                    fig_iv.update_layout(title="IV en PPD door de tijd")
                    show_fig(fig_iv, height=430)


elif page == "📏 PPD naar afstand":
    section_title(
        "PPD naar afstand tot spot",
        "Bekijk hoeveel PPD de markt geeft bij verschillende afstanden vanaf SPX op een gekozen peildatum.",
    )

    d1, d2, d3 = st.columns([1.2, 0.8, 1.2])
    with d1:
        dist_snapshot = snapshot_selectbox("Peildatum", "dist_snapshot")
    with d2:
        dist_type = type_radio("Type", "dist_type", default="put")
    with d3:
        dist_dte = st.slider("DTE-bereik", 1, 365, (1, 60), step=1, key="dist_dte")

    dist_mode = st.radio("Afstandseenheid", ["punten", "% van spot"], index=0, horizontal=True, key="dist_mode")
    chain = load_snapshot_chain(dist_snapshot, dist_type, dist_dte[0], dist_dte[1], min_oi, min_vol)

    if chain.empty:
        st.info("Geen chain-data voor deze selectie.")
    else:
        if dist_mode == "punten":
            x_col = "abs_dist_pts"
            max_x = max(50, int(np.nanpercentile(chain[x_col], 98)) + 25)
            bins = np.arange(0, max_x, 25)
            x_title = "|K-S| in punten"
        else:
            x_col = "abs_dist_pct"
            bins = np.arange(0, 30.5, 0.5)
            x_title = "|K-S| / S in procent"

        chain["dist_bin"] = pd.cut(chain[x_col], bins=bins, include_lowest=True)
        g = (
            chain.groupby("dist_bin")
            .agg(ppd=("ppd", "median"), iv=("implied_volatility", "median"), price=("mid_price", "median"), n=("ppd", "count"))
            .reset_index()
        )
        g = g[g["n"] >= 3].copy()
        g["bin_mid"] = g["dist_bin"].apply(lambda iv: iv.mid if pd.notna(iv) else np.nan)
        g["ppd_s"] = smooth_series(g["ppd"], 3)

        fig_dist = make_subplots(specs=[[{"secondary_y": True}]])
        fig_dist.add_trace(go.Scatter(x=g["bin_mid"], y=g["ppd"], mode="markers", name="PPD"), secondary_y=False)
        fig_dist.add_trace(go.Scatter(x=g["bin_mid"], y=g["ppd_s"], mode="lines", name="PPD smooth"), secondary_y=False)
        fig_dist.add_trace(go.Scatter(x=g["bin_mid"], y=g["price"], mode="lines", name="Mid prijs"), secondary_y=True)
        fig_dist.update_xaxes(title_text=x_title)
        fig_dist.update_yaxes(title_text="PPD", secondary_y=False)
        fig_dist.update_yaxes(title_text="Mid prijs", secondary_y=True)
        fig_dist.update_layout(title=f"{dist_type.upper()}: PPD en prijs naar afstand tot spot")
        show_fig(fig_dist, height=500)


elif page == "🌊 Term structure en smile":
    section_title(
        "Term structure en smile",
        "Bekijk implied volatility per looptijd en per strike. Dit is context voor je trade-keuze.",
    )

    t1, t2, t3 = st.columns([1.2, 0.8, 1.2])
    with t1:
        term_snapshot = snapshot_selectbox("Peildatum", "term_snapshot")
    with t2:
        term_type = type_radio("Type", "term_type", default="put")
    with t3:
        term_strike, _, _ = strike_selectbox("Strike voor smile-defaults", term_snapshot, term_type, "term_strike")

    term_chain = load_snapshot_chain(term_snapshot, term_type, 1, 365, min_oi, min_vol)

    if term_chain.empty:
        st.info("Geen term structure data voor deze selectie.")
    else:
        c1, c2 = st.columns(2)

        with c1:
            term = term_chain.groupby("days_to_exp", as_index=False)["implied_volatility"].median().sort_values("days_to_exp")
            fig_term = go.Figure(
                go.Scatter(
                    x=term["days_to_exp"],
                    y=term["implied_volatility"],
                    mode="lines+markers",
                    name="IV",
                )
            )
            fig_term.update_layout(title="IV term structure", xaxis_title="DTE", yaxis_title="IV")
            show_fig(fig_term, height=410)

        with c2:
            exps = sorted(term_chain["expiration"].dropna().unique())
            if not exps:
                st.info("Geen expiraties voor smile.")
            else:
                default_exp = pick_first_on_or_after(exps, date.today() + timedelta(days=14))
                smile_exp = st.selectbox(
                    "Expiratie voor smile",
                    options=exps,
                    index=exps.index(default_exp) if default_exp in exps else 0,
                    key="smile_exp",
                )
                sm = (
                    term_chain[term_chain["expiration"] == smile_exp]
                    .groupby("strike", as_index=False)["implied_volatility"]
                    .median()
                    .sort_values("strike")
                )
                fig_sm = go.Figure(go.Scatter(x=sm["strike"], y=sm["implied_volatility"], mode="lines+markers", name="IV Smile"))
                if term_strike is not None:
                    fig_sm.add_vline(x=float(term_strike), line=dict(dash="dot"))
                fig_sm.update_layout(title=f"IV smile exp {smile_exp}", xaxis_title="Strike", yaxis_title="IV")
                show_fig(fig_sm, height=410)

st.caption(
    "Gebruik dit dashboard als contextlaag. Voor concrete trades blijft bevestiging via price action, macro-events, expiratiekalender en executionkwaliteit nodig."
)
