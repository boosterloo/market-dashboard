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


# ────────────────────────────── Helpers ──────────────────────────────
def norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def norm_pdf(z: float) -> float:
    return (1.0 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * z * z)


def bs_delta(S, K, iv, T, r, q, is_call):
    if any(x is None or np.isnan(x) or x <= 0 for x in [S, K]) or np.isnan(iv) or iv <= 0 or T <= 0:
        return np.nan
    d1 = (math.log(S / K) + (r - q + 0.5 * iv**2) * T) / (iv * math.sqrt(T))
    Nd1 = norm_cdf(d1)
    disc_q = math.exp(-q * T)
    return disc_q * Nd1 if is_call else disc_q * (Nd1 - 1.0)


def annualize_std(s: pd.Series, periods_per_year: int = 252) -> float:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) < 2:
        return np.nan
    return float(s.std(ddof=0) * math.sqrt(periods_per_year))


def pick_first_on_or_after(options, target):
    after = [d for d in options if d >= target]
    return after[0] if after else (options[-1] if options else None)


def pick_closest_value(options, target, fallback=None):
    if not options:
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


def zscore_last(s: pd.Series, window: int = 60) -> float:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) < max(20, window // 3):
        return np.nan
    roll = s.tail(window)
    mu = roll.mean()
    sd = roll.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return np.nan
    return float((roll.iloc[-1] - mu) / sd)


def safe_ratio(a, b):
    return a / b if (b is not None and not np.isnan(b) and b != 0) else np.nan


# ────────────────────────────── BigQuery ──────────────────────────────
def get_bq_client():
    sa_info = st.secrets["gcp_service_account"]
    creds = service_account.Credentials.from_service_account_info(sa_info)
    project_id = st.secrets.get("PROJECT_ID", sa_info.get("project_id"))
    return bigquery.Client(project=project_id, credentials=creds)


_bq_client = get_bq_client()
VIEW = "marketdata.spx_options_enriched_v"


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
    job_config = None
    if params:
        job_config = bigquery.QueryJobConfig(
            query_parameters=[_bq_param(k, v) for k, v in params.items()]
        )
    return _bq_client.query(sql, job_config=job_config).to_dataframe()


# ────────────────────────────── Styling ──────────────────────────────
st.set_page_config(page_title="SPX Options Dashboard", layout="wide")
st.title("SPX Options Dashboard")

PLOTLY_CONFIG = {
    "scrollZoom": True,
    "doubleClick": "reset",
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
}

_BASE = 12
_DEF_MARGIN = dict(l=96, r=48, t=64, b=56)


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
            y=1.02 if legend_top else -0.2,
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


def add_pcr_midline(fig, row: int, col: int):
    fig.add_hline(y=1.0, line=dict(dash="dot", width=1), row=row, col=col)


# ────────────────────────────── Filters ──────────────────────────────
@st.cache_data(ttl=600, show_spinner=False)
def load_date_bounds():
    df = run_query(
        f"""
        SELECT
          MIN(CAST(snapshot_date AS DATE)) AS min_date,
          MAX(CAST(snapshot_date AS DATE)) AS max_date
        FROM `{VIEW}`
        """
    )
    return df["min_date"].iloc[0], df["max_date"].iloc[0]


min_date, max_date = load_date_bounds()
default_start = max(min_date, max_date - timedelta(days=14))

c0, c1 = st.columns([1.2, 1.8])
with c0:
    axis_scale = st.slider("As-lettergrootte", 1.2, 2.2, 1.5, 0.1)
    st.session_state["_axis_mult"] = axis_scale
with c1:
    range_choice = st.radio("Snelle periode", ["14d", "30d", "90d", "1y", "Custom"], index=0, horizontal=True)

if range_choice != "Custom":
    days_map = {"14d": 14, "30d": 30, "90d": 90, "1y": 365}
    start_date = max(min_date, max_date - timedelta(days=days_map[range_choice]))
    end_date = max_date
    st.caption(f"Periode: {start_date} t/m {end_date}")
else:
    start_date, end_date = st.date_input(
        "Periode (snapshot_date)",
        value=(default_start, max_date),
        min_value=min_date,
        max_value=max_date,
        format="YYYY-MM-DD",
    )

f1, f2, f3, f4, f5 = st.columns([1, 1, 1, 1, 1])
with f1:
    sel_type = st.radio("Type", ["put", "call"], index=0, horizontal=True)
with f2:
    dte_range = st.slider("DTE", 0, 365, (0, 60), step=1)
with f3:
    mny_range = st.slider("Moneyness (K/S - 1)", -0.20, 0.20, (-0.10, 0.10), step=0.01)
with f4:
    min_oi = st.slider("Min OI", 0, 100, 1, step=1)
with f5:
    min_vol = st.slider("Min Volume", 0, 100, 1, step=1)

g1, g2, g3 = st.columns([1, 1, 1])
with g1:
    load_deep_analysis = st.toggle("Laad detailanalyse", value=True)
with g2:
    show_underlying = st.toggle("Overlay SPX", value=True)
with g3:
    q_const_simple = st.number_input("Dividendrendement q", 0.0, 0.10, 0.016, 0.001, format="%.3f")


# ────────────────────────────── Data loaders ──────────────────────────────
@st.cache_data(ttl=600, show_spinner=False)
def load_expirations(start_date: date, end_date: date):
    df = run_query(
        f"""
        SELECT DISTINCT expiration
        FROM `{VIEW}`
        WHERE DATE(snapshot_date) BETWEEN @start AND @end
        ORDER BY expiration
        """,
        {"start": start_date, "end": end_date},
    )
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
def load_detail_filtered(start_date, end_date, sel_type, dte_min, dte_max, mny_min, mny_max, min_oi, min_vol):
    sql = f"""
    SELECT
      snapshot_date,
      expiration,
      LOWER(type) AS type,
      CAST(days_to_exp AS INT64) AS days_to_exp,
      CAST(strike AS FLOAT64) AS strike,
      CAST(underlying_price AS FLOAT64) AS underlying_price,
      SAFE_DIVIDE(CAST(strike AS FLOAT64), NULLIF(CAST(underlying_price AS FLOAT64), 0)) - 1.0 AS moneyness,
      CAST(strike AS FLOAT64) - CAST(underlying_price AS FLOAT64) AS dist_points,
      CAST(last_price AS FLOAT64) AS last_price,
      CAST(mid_price AS FLOAT64) AS mid_price,
      CAST(bid AS FLOAT64) AS bid,
      CAST(ask AS FLOAT64) AS ask,
      CAST(implied_volatility AS FLOAT64) AS implied_volatility,
      CAST(open_interest AS FLOAT64) AS open_interest,
      CAST(volume AS FLOAT64) AS volume,
      CAST(ppd AS FLOAT64) AS ppd,
      CAST(vix AS FLOAT64) AS vix
    FROM `{VIEW}`
    WHERE DATE(snapshot_date) BETWEEN @start AND @end
      AND LOWER(type) = @t
      AND days_to_exp BETWEEN @dte_min AND @dte_max
      AND SAFE_DIVIDE(CAST(strike AS FLOAT64), NULLIF(CAST(underlying_price AS FLOAT64), 0)) - 1.0
          BETWEEN @mny_min AND @mny_max
      AND (COALESCE(open_interest, 0) >= @min_oi OR COALESCE(volume, 0) >= @min_vol)
    """
    df = run_query(
        sql,
        {
            "start": start_date,
            "end": end_date,
            "t": sel_type,
            "dte_min": int(dte_min),
            "dte_max": int(dte_max),
            "mny_min": float(mny_min),
            "mny_max": float(mny_max),
            "min_oi": int(min_oi),
            "min_vol": int(min_vol),
        },
    )
    if not df.empty:
        df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
        df["expiration"] = pd.to_datetime(df["expiration"]).dt.date
        df["moneyness_pct"] = 100.0 * df["moneyness"]
        df["abs_dist_pts"] = df["dist_points"].abs()
        df["abs_dist_pct"] = 100.0 * df["abs_dist_pts"] / df["underlying_price"]
        df["snap_min"] = df["snapshot_date"].dt.floor("min")
    return df


@st.cache_data(ttl=600, show_spinner=False)
def load_skew_source(start_date, end_date, dte_lo, dte_hi, min_oi, min_vol):
    sql = f"""
    SELECT
      snapshot_date,
      expiration,
      LOWER(type) AS type,
      CAST(days_to_exp AS INT64) AS days_to_exp,
      CAST(strike AS FLOAT64) AS strike,
      CAST(underlying_price AS FLOAT64) AS underlying_price,
      CAST(implied_volatility AS FLOAT64) AS implied_volatility,
      CAST(open_interest AS FLOAT64) AS open_interest,
      CAST(volume AS FLOAT64) AS volume,
      SAFE_DIVIDE(CAST(strike AS FLOAT64), NULLIF(CAST(underlying_price AS FLOAT64), 0)) - 1.0 AS moneyness
    FROM `{VIEW}`
    WHERE DATE(snapshot_date) BETWEEN @start AND @end
      AND days_to_exp BETWEEN @dte_lo AND @dte_hi
      AND (COALESCE(open_interest, 0) >= @min_oi OR COALESCE(volume, 0) >= @min_vol)
    """
    df = run_query(
        sql,
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


# ────────────────────────────── Main data ──────────────────────────────
market = load_daily_market_context(
    start_date,
    end_date,
    dte_range[0],
    dte_range[1],
    mny_range[0],
    mny_range[1],
    min_oi,
    min_vol,
)

if market.empty:
    st.warning("Geen data voor de huidige filters.")
    st.stop()

market["PCR_vol"] = market["vol_put"] / market["vol_call"].replace(0, np.nan)
market["PCR_oi"] = market["oi_put"] / market["oi_call"].replace(0, np.nan)
market["PCR_gap"] = market["PCR_vol"] - market["PCR_oi"]
market["PCR_vol_pctrank"] = market["PCR_vol"].expanding().apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
market["PCR_oi_pctrank"] = market["PCR_oi"].expanding().apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)

df = pd.DataFrame()
if load_deep_analysis:
    df = load_detail_filtered(
        start_date,
        end_date,
        sel_type,
        dte_range[0],
        dte_range[1],
        mny_range[0],
        mny_range[1],
        min_oi,
        min_vol,
    )

exps_all = load_expirations(start_date, end_date)


# ────────────────────────────── Context metrics ──────────────────────────────
u_daily = market[["day", "spx"]].dropna().copy()
u_daily["ret"] = u_daily["spx"].pct_change()
hv20 = annualize_std(u_daily["ret"].tail(21).dropna())

iv_rank = pct_rank_last(market["avg_iv"].tail(252))
vix_rank = pct_rank_last(market["vix"].tail(252))

last_pcr_vol = float(market["PCR_vol"].dropna().iloc[-1]) if market["PCR_vol"].notna().any() else np.nan
last_pcr_oi = float(market["PCR_oi"].dropna().iloc[-1]) if market["PCR_oi"].notna().any() else np.nan
last_pcr_gap = float(market["PCR_gap"].dropna().iloc[-1]) if market["PCR_gap"].notna().any() else np.nan
last_iv = float(market["avg_iv"].dropna().iloc[-1]) if market["avg_iv"].notna().any() else np.nan
last_vix = float(market["vix"].dropna().iloc[-1]) if market["vix"].notna().any() else np.nan
last_spx = float(market["spx"].dropna().iloc[-1]) if market["spx"].notna().any() else np.nan
vrp = last_iv - hv20 if not np.isnan(last_iv) and not np.isnan(hv20) else np.nan

dte_for_skew = (20, 40)
skew_src = load_skew_source(start_date, end_date, dte_for_skew[0], dte_for_skew[1], min_oi, min_vol)

skew_25d = np.nan
if not skew_src.empty:
    latest_day = skew_src["snapshot_date"].max()
    z = skew_src[skew_src["snapshot_date"] == latest_day].copy()
    z["T"] = pd.to_numeric(z["days_to_exp"], errors="coerce").fillna(0) / 365.0
    z["delta"] = z.apply(
        lambda r: bs_delta(
            float(r["underlying_price"]),
            float(r["strike"]),
            float(r["implied_volatility"]),
            float(r["T"]),
            0.0,
            float(
                get_q_curve_const(
                    np.array([float(r["T"])], dtype=float),
                    q_const=q_const_simple,
                    to_continuous=True,
                )[0]
            ),
            is_call=(str(r["type"]).lower() == "call"),
        ),
        axis=1,
    )
    rows = []
    for _, g in z.groupby("expiration"):
        gp = g[g["type"] == "put"].copy()
        gc = g[g["type"] == "call"].copy()
        if gp.empty or gc.empty:
            continue
        gp["dist"] = (gp["delta"] + 0.25).abs()
        gc["dist"] = (gc["delta"] - 0.25).abs()
        rowp = gp.loc[gp["dist"].idxmin()] if gp["dist"].notna().any() else None
        rowc = gc.loc[gc["dist"].idxmin()] if gc["dist"].notna().any() else None
        if rowp is not None and rowc is not None:
            rows.append(float(rowp["implied_volatility"]) - float(rowc["implied_volatility"]))
    if rows:
        skew_25d = float(np.nanmedian(rows))

iv_short = np.nan
iv_mid = np.nan
if load_deep_analysis and not df.empty:
    latest_snap = df["snap_min"].max()
    last_df = df[df["snap_min"] == latest_snap]
    iv_short = float(last_df[last_df["days_to_exp"].between(7, 15)]["implied_volatility"].median())
    iv_mid = float(last_df[last_df["days_to_exp"].between(30, 60)]["implied_volatility"].median())

iv_slope = iv_short - iv_mid if not np.isnan(iv_short) and not np.isnan(iv_mid) else np.nan


# ────────────────────────────── Regime classification ──────────────────────────────
def classify_regime(pcr_vol, pcr_oi, pcr_gap, skew, slope, vrp):
    items = []

    if not np.isnan(pcr_vol):
        if pcr_vol > 1.15:
            items.append("flow defensief")
        elif pcr_vol < 0.85:
            items.append("flow risk-on")
        else:
            items.append("flow neutraal")

    if not np.isnan(pcr_oi):
        if pcr_oi > 1.10:
            items.append("positionering defensief")
        elif pcr_oi < 0.90:
            items.append("positionering call/risk-on")
        else:
            items.append("positionering gemengd")

    if not np.isnan(pcr_gap):
        if pcr_gap > 0.15:
            items.append("actuele flow bearish dan bestaande positioning")
        elif pcr_gap < -0.15:
            items.append("actuele flow bullish/speculatiever dan bestaande positioning")

    if not np.isnan(skew):
        if skew > 0.02:
            items.append("downside hedge-vraag verhoogd")
        elif skew < 0.0:
            items.append("skew vlak / weinig downside-premie")

    if not np.isnan(slope):
        if slope > 0.01:
            items.append("korte termijn event/stress geprijsd")
        elif slope < -0.01:
            items.append("korte termijn rustiger dan mid-curve")

    if not np.isnan(vrp):
        if vrp > 0.03:
            items.append("vol-premie gunstig voor short vol")
        elif vrp < 0.0:
            items.append("weinig of negatieve vol-premie")

    return items


regime_flags = classify_regime(last_pcr_vol, last_pcr_oi, last_pcr_gap, skew_25d, iv_slope, vrp)


def trade_context_label(vrp, skew, slope):
    if np.isnan(vrp):
        short_vol = "Onbekend"
    elif vrp > 0.03 and (np.isnan(slope) or slope < 0.02):
        short_vol = "Relatief gunstig"
    elif vrp > 0.0:
        short_vol = "Voorzichtig gunstig"
    else:
        short_vol = "Ongunstig"

    if np.isnan(skew):
        downside = "Onbekend"
    elif skew > 0.03:
        downside = "Hoog"
    elif skew > 0.01:
        downside = "Middel"
    else:
        downside = "Laag"

    if np.isnan(slope):
        event_risk = "Onbekend"
    elif slope > 0.02:
        event_risk = "Hoog"
    elif slope > 0.005:
        event_risk = "Middel"
    else:
        event_risk = "Laag"

    return short_vol, downside, event_risk


short_vol_label, downside_label, event_label = trade_context_label(vrp, skew_25d, iv_slope)


# ────────────────────────────── Header KPIs ──────────────────────────────
k1, k2, k3, k4, k5, k6 = st.columns(6)
with k1:
    st.metric("PCR Volume", f"{last_pcr_vol:.2f}" if not np.isnan(last_pcr_vol) else "—")
with k2:
    st.metric("PCR OI", f"{last_pcr_oi:.2f}" if not np.isnan(last_pcr_oi) else "—")
with k3:
    st.metric("25Δ Skew", f"{skew_25d:.2%}" if not np.isnan(skew_25d) else "—")
with k4:
    st.metric("IV Term Slope", f"{iv_slope:.2%}" if not np.isnan(iv_slope) else "—")
with k5:
    st.metric("VRP (IV-HV20)", f"{vrp:.2%}" if not np.isnan(vrp) else "—")
with k6:
    st.metric("IV Rank", f"{iv_rank * 100:.0f}%" if not np.isnan(iv_rank) else "—")

if regime_flags:
    st.markdown("**Marktregime:** " + " | ".join(regime_flags))

cx1, cx2, cx3 = st.columns(3)
with cx1:
    st.info(f"**Short vol context:** {short_vol_label}")
with cx2:
    st.info(f"**Downside hedge-vraag:** {downside_label}")
with cx3:
    st.info(f"**Event risk korte termijn:** {event_label}")

st.caption(
    "Interpretatie: `Volume PCR` meet actuele flow. `OI PCR` meet bestaande positioning. "
    "Als die twee niet synchroon lopen is dat vaak informatief, niet fout."
)


# ────────────────────────────── Regime charts ──────────────────────────────
st.header("Regime & Risicoperceptie")

fig_pcr = make_subplots(
    rows=3,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.05,
    subplot_titles=("PCR Volume (flow)", "PCR OI (positioning)", "PCR Gap (flow - positioning)"),
)
fig_pcr.add_trace(go.Scatter(x=market["day"], y=market["PCR_vol"], mode="lines", name="PCR Volume"), row=1, col=1)
fig_pcr.add_trace(go.Scatter(x=market["day"], y=market["PCR_oi"], mode="lines", name="PCR OI"), row=2, col=1)
fig_pcr.add_trace(go.Bar(x=market["day"], y=market["PCR_gap"], name="PCR Gap"), row=3, col=1)
add_pcr_midline(fig_pcr, 1, 1)
add_pcr_midline(fig_pcr, 2, 1)
show_fig(fig_pcr, height=760)
st.caption(
    "Leeswijzer: PCR > 1 betekent relatief meer puts dan calls. `PCR Gap > 0` betekent dat de actuele flow defensiever is dan de bestaande positioning."
)

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
show_fig(fig_ctx, height=760)


# ────────────────────────────── Execution analysis ──────────────────────────────
if not load_deep_analysis:
    st.warning("Detailanalyse staat uit. Zet `Laad detailanalyse` aan voor series, smile en execution-secties.")
    st.stop()

if df.empty:
    st.warning("Geen detaildata voor de huidige filters.")
    st.stop()

st.header("Execution & Trade Selectie")

snapshots_all = sorted(df["snap_min"].unique())
default_snapshot = snapshots_all[-1] if snapshots_all else None
underlying_now = float(df[df["snap_min"] == default_snapshot]["underlying_price"].median()) if default_snapshot is not None else np.nan
strikes_all = sorted([float(x) for x in df["strike"].dropna().unique().tolist()])
target_exp = date.today() + timedelta(days=14)
default_exp = pick_first_on_or_after(exps_all, target_exp)

target_strike = underlying_now - 300.0 if sel_type == "put" else underlying_now + 200.0
default_series_strike = pick_closest_value(strikes_all, target_strike, fallback=(strikes_all[0] if strikes_all else 0.0))

e1, e2, e3 = st.columns([1, 1, 1.5])
with e1:
    sel_snapshot = st.selectbox(
        "Peildatum",
        options=snapshots_all,
        index=max(len(snapshots_all) - 1, 0),
        format_func=lambda x: pd.to_datetime(x).strftime("%Y-%m-%d %H:%M"),
    )
with e2:
    series_strike = st.selectbox(
        "Strike",
        options=strikes_all,
        index=(strikes_all.index(default_series_strike) if default_series_strike in strikes_all else 0),
    )
with e3:
    series_exp = st.selectbox(
        "Expiratie",
        options=exps_all,
        index=(exps_all.index(default_exp) if default_exp in exps_all else 0) if exps_all else 0,
    )

series_price_col = st.radio("Prijsbron", ["last_price", "mid_price"], index=1, horizontal=True)

df_last = df[df["snap_min"] == sel_snapshot].copy()
serie = df[(df["strike"] == series_strike) & (df["expiration"] == series_exp)].copy().sort_values("snapshot_date")

if serie.empty:
    st.info("Geen data voor deze serie.")
else:
    s1, s2 = st.columns(2)
    with s1:
        fig_price = make_subplots(specs=[[{"secondary_y": True}]])
        fig_price.add_trace(
            go.Scatter(x=serie["snapshot_date"], y=serie[series_price_col], mode="lines+markers", name="Optieprijs"),
            secondary_y=False,
        )
        if show_underlying:
            fig_price.add_trace(
                go.Scatter(
                    x=serie["snapshot_date"],
                    y=serie["underlying_price"],
                    mode="lines",
                    line=dict(dash="dot"),
                    name="SPX",
                ),
                secondary_y=True,
            )
        fig_price.update_yaxes(title_text="Optieprijs", secondary_y=False)
        fig_price.update_yaxes(title_text="SPX", secondary_y=True)
        fig_price.update_layout(title=f"{sel_type.upper()} {series_strike} exp {series_exp}")
        show_fig(fig_price, height=420)

    with s2:
        fig_iv = make_subplots(specs=[[{"secondary_y": True}]])
        fig_iv.add_trace(
            go.Scatter(x=serie["snapshot_date"], y=serie["implied_volatility"], mode="lines+markers", name="IV"),
            secondary_y=False,
        )
        if show_underlying:
            fig_iv.add_trace(
                go.Scatter(
                    x=serie["snapshot_date"],
                    y=serie["underlying_price"],
                    mode="lines",
                    line=dict(dash="dot"),
                    name="SPX",
                ),
                secondary_y=True,
            )
        fig_iv.update_yaxes(title_text="IV", secondary_y=False)
        fig_iv.update_yaxes(title_text="SPX", secondary_y=True)
        fig_iv.update_layout(title="IV vs SPX")
        show_fig(fig_iv, height=420)

st.subheader("PPD vs Afstand")
dist_mode = st.radio("Afstandseenheid", ["punten", "% van spot"], index=0, horizontal=True)

if not df_last.empty:
    df_last["abs_dist_pts"] = df_last["dist_points"].abs()
    df_last["abs_dist_pct"] = 100.0 * df_last["abs_dist_pts"] / df_last["underlying_price"]

    if dist_mode == "punten":
        x_col = "abs_dist_pts"
        bins = np.arange(0, max(50, int(np.nanpercentile(df_last[x_col], 98)) + 25), 25)
        x_title = "|K-S| (punten)"
        cur_x = abs(series_strike - underlying_now)
    else:
        x_col = "abs_dist_pct"
        bins = np.arange(0, 20.5, 0.5)
        x_title = "|K-S| / S (%)"
        cur_x = 100.0 * abs(series_strike - underlying_now) / underlying_now if underlying_now else np.nan

    df_last["dist_bin"] = pd.cut(df_last[x_col], bins=bins, include_lowest=True)
    g = (
        df_last.groupby("dist_bin")
        .agg(ppd=("ppd", "median"), iv=("implied_volatility", "median"), n=("ppd", "count"))
        .reset_index()
    )
    g = g[g["n"] >= 3].copy()
    g["bin_mid"] = g["dist_bin"].apply(lambda iv: iv.mid if pd.notna(iv) else np.nan)
    g["ppd_s"] = smooth_series(g["ppd"], 3)

    fig_dist = make_subplots(specs=[[{"secondary_y": True}]])
    fig_dist.add_trace(go.Scatter(x=g["bin_mid"], y=g["ppd"], mode="markers", name="PPD"), secondary_y=False)
    fig_dist.add_trace(go.Scatter(x=g["bin_mid"], y=g["ppd_s"], mode="lines", name="PPD smooth"), secondary_y=False)
    fig_dist.add_trace(go.Scatter(x=g["bin_mid"], y=g["iv"], mode="lines", name="IV"), secondary_y=True)
    if not np.isnan(cur_x):
        fig_dist.add_vline(x=cur_x, line=dict(dash="dot"))
    fig_dist.update_xaxes(title_text=x_title)
    fig_dist.update_yaxes(title_text="PPD", secondary_y=False)
    fig_dist.update_yaxes(title_text="IV", secondary_y=True)
    fig_dist.update_layout(title="PPD en IV versus afstand")
    show_fig(fig_dist, height=440)

st.subheader("PPD vs DTE")
dte_scope = df_last[np.abs(df_last["moneyness"]) <= 0.02].copy() if not df_last.empty else pd.DataFrame()
if dte_scope.empty:
    st.info("Geen voldoende ATM-data voor PPD vs DTE.")
else:
    d_curve = (
        dte_scope.groupby("days_to_exp", as_index=False)
        .agg(ppd=("ppd", "median"), iv=("implied_volatility", "median"), n=("ppd", "count"))
        .query("n >= 3")
        .sort_values("days_to_exp")
    )
    d_curve["ppd_s"] = smooth_series(d_curve["ppd"], 3)

    fig_dte = make_subplots(specs=[[{"secondary_y": True}]])
    fig_dte.add_trace(go.Scatter(x=d_curve["days_to_exp"], y=d_curve["ppd"], mode="markers", name="PPD"), secondary_y=False)
    fig_dte.add_trace(go.Scatter(x=d_curve["days_to_exp"], y=d_curve["ppd_s"], mode="lines", name="PPD smooth"), secondary_y=False)
    fig_dte.add_trace(go.Scatter(x=d_curve["days_to_exp"], y=d_curve["iv"], mode="lines", name="IV"), secondary_y=True)
    fig_dte.update_xaxes(title_text="DTE")
    fig_dte.update_yaxes(title_text="PPD", secondary_y=False)
    fig_dte.update_yaxes(title_text="IV", secondary_y=True)
    fig_dte.update_layout(title="PPD en IV versus DTE")
    show_fig(fig_dte, height=440)

    if d_curve["ppd_s"].notna().any():
        sweet = d_curve.loc[d_curve["ppd_s"].idxmax()]
        st.info(
            f"PPD-sweet spot ligt rond **{int(sweet['days_to_exp'])} DTE** met mediane **PPD ≈ {sweet['ppd_s']:.2f}**."
        )

st.subheader("Term Structure & Smile")
t1, t2 = st.columns(2)

with t1:
    term = df_last.groupby("days_to_exp", as_index=False)["implied_volatility"].median().sort_values("days_to_exp")
    if term.empty:
        st.info("Geen term structure data.")
    else:
        fig_term = go.Figure(go.Scatter(x=term["days_to_exp"], y=term["implied_volatility"], mode="lines+markers", name="IV"))
        fig_term.update_layout(title="Term Structure", xaxis_title="DTE", yaxis_title="IV")
        show_fig(fig_term, height=380)

with t2:
    sm = df_last[df_last["expiration"] == series_exp].groupby("strike", as_index=False)["implied_volatility"].median().sort_values("strike")
    if sm.empty:
        st.info("Geen smile data.")
    else:
        fig_sm = go.Figure(go.Scatter(x=sm["strike"], y=sm["implied_volatility"], mode="lines+markers", name="IV Smile"))
        fig_sm.update_layout(title=f"Smile exp {series_exp}", xaxis_title="Strike", yaxis_title="IV")
        show_fig(fig_sm, height=380)

st.header("Trading Context")
tc1, tc2, tc3 = st.columns(3)
with tc1:
    st.markdown(
        "**Short vol**\n\n"
        f"- Context: `{short_vol_label}`\n"
        f"- Let op: VRP={vrp:.2%} | IV Rank={iv_rank:.0%}" if not np.isnan(vrp) and not np.isnan(iv_rank) else "**Short vol**\n\n- Onvoldoende data"
    )
with tc2:
    st.markdown(
        "**Downside hedge demand**\n\n"
        f"- Niveau: `{downside_label}`\n"
        f"- Inputs: 25Δ skew={skew_25d:.2%} | PCR gap={last_pcr_gap:.2f}" if not np.isnan(skew_25d) and not np.isnan(last_pcr_gap) else "**Downside hedge demand**\n\n- Onvoldoende data"
    )
with tc3:
    st.markdown(
        "**Korte-termijn event risk**\n\n"
        f"- Niveau: `{event_label}`\n"
        f"- Input: term slope={iv_slope:.2%}" if not np.isnan(iv_slope) else "**Korte-termijn event risk**\n\n- Onvoldoende data"
    )

st.caption(
    "Gebruik dit dashboard als contextlaag. Voor concrete trades blijft bevestiging via price action, macro-events, expiratiekalender en executionkwaliteit nodig."
)
