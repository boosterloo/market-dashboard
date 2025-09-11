# pages/Yield_US_vs_EU.py
# 🇺🇸 vs 🇪🇺 — vergelijking met 90d μ±1σ-band & Z-scores (compatible met jouw secrets-setup)

import streamlit as st
import pandas as pd
import numpy as np
import re
import plotly.graph_objects as go
from google.api_core.exceptions import NotFound, BadRequest
from utils.bq import run_query, bq_ping

st.set_page_config(page_title="🇺🇸 vs 🇪🇺 Yield Curve", layout="wide")
st.title("🇺🇸 vs 🇪🇺 Yield Curve Vergelijking")

# ================== SECRETS / DEFAULTS ==================
PROJECT_ID = st.secrets["gcp_service_account"]["project_id"]
TABLES     = st.secrets.get("tables", {})

US_VIEW   = TABLES.get("us_yield_view", f"{PROJECT_ID}.marketdata.us_yield_curve_enriched_v")
EU_VIEW   = TABLES.get("eu_yield_view", f"{PROJECT_ID}.marketdata.eu_yield_curve_enriched_v")
WIDE_VIEW = TABLES.get("yield_wide_view", f"{PROJECT_ID}.marketdata.yield_curve_analysis_wide")

with st.expander("🔎 Debug: opgegeven bronnen"):
    st.write({"US_VIEW": US_VIEW, "EU_VIEW": EU_VIEW, "WIDE_VIEW": WIDE_VIEW})

# ================== HEALTH ==================
try:
    if not bq_ping():
        st.error("Geen BigQuery-verbinding."); st.stop()
except Exception as e:
    st.error("Geen BigQuery-verbinding.")
    st.caption(f"Details: {e}")
    st.stop()

# ================== HELPERS ==================
def list_columns(fqtn: str) -> set[str]:
    proj, dset, tbl = fqtn.split(".")
    sql = f"""
    SELECT column_name
    FROM `{proj}.{dset}.INFORMATION_SCHEMA.COLUMNS`
    WHERE table_name = @tbl
    """
    dfc = run_query(sql, params={"tbl": tbl}, timeout=30)
    return set([c.lower() for c in dfc["column_name"].astype(str)])

def choose_col(cols: set[str], preferred: list[str]) -> str | None:
    for c in preferred:
        if c.lower() in cols:
            return c
    return None

def add_roll_stats(frame: pd.DataFrame, col: str, window: int = 90, minp: int = 30):
    mu = frame[col].rolling(window, min_periods=minp).mean()
    sd = frame[col].rolling(window, min_periods=minp).std()
    frame[f"{col}_mu"] = mu
    frame[f"{col}_sd"] = sd
    frame[f"{col}_z"]  = (frame[col] - mu) / sd
    return frame

# ================== LOADERS ==================
def load_from_two_enriched_views(us_fqn: str, eu_fqn: str) -> pd.DataFrame:
    # autodetect kolomnamen per view (y_2y_synth/y_2y, y_10y_synth/y_10y)
    us_cols = list_columns(us_fqn)
    eu_cols = list_columns(eu_fqn)

    us_2y  = choose_col(us_cols, ["y_2y_synth", "y_2y"])
    us_10y = choose_col(us_cols, ["y_10y_synth", "y_10y"])
    eu_2y  = choose_col(eu_cols, ["y_2y_synth", "y_2y"])
    eu_10y = choose_col(eu_cols, ["y_10y_synth", "y_10y"])

    missing = [n for n,v in {
        "US 2Y": us_2y, "US 10Y": us_10y, "EU 2Y": eu_2y, "EU 10Y": eu_10y
    }.items() if v is None]
    if missing:
        raise BadRequest(f"Ontbrekende kolommen in enriched views: {missing}")

    q = f"""
    WITH us AS (
      SELECT date, SAFE_CAST({us_2y} AS FLOAT64) AS us_2y,
                   SAFE_CAST({us_10y} AS FLOAT64) AS us_10y,
             SAFE_CAST({us_10y} AS FLOAT64) - SAFE_CAST({us_2y} AS FLOAT64) AS us_spread
      FROM `{us_fqn}`
    ),
    eu AS (
      SELECT date, SAFE_CAST({eu_2y} AS FLOAT64) AS eu_2y,
                   SAFE_CAST({eu_10y} AS FLOAT64) AS eu_10y,
             SAFE_CAST({eu_10y} AS FLOAT64) - SAFE_CAST({eu_2y} AS FLOAT64) AS eu_spread
      FROM `{eu_fqn}`
    )
    SELECT
      us.date,
      us.us_2y, us.us_10y, us.us_spread,
      eu.eu_2y, eu.eu_10y, eu.eu_spread,
      (us.us_2y - eu.eu_2y)                         AS diff_2y,
      (us.us_10y - eu.eu_10y)                       AS diff_10y,
      (us.us_spread - eu.eu_spread)                 AS diff_spread
    FROM us
    JOIN eu USING(date)
    ORDER BY date
    """
    return run_query(q, timeout=60)

def load_from_wide_autodetect(wide_fqn: str) -> pd.DataFrame:
    # brede bron met 'region' + yields (varianten)
    cols = list_columns(wide_fqn)
    date_col   = "date"   if "date"   in cols else ("datum" if "datum" in cols else None)
    region_col = "region" if "region" in cols else ( "regio" if "regio" in cols else None)
    y2_col     = choose_col(cols, ["y_2y_synth","y_2y","yield_2y","r_2y","y2y"])
    y10_col    = choose_col(cols, ["y_10y_synth","y_10y","yield_10y","r_10y","y10y"])
    missing = [n for n,v in {"date":date_col,"region":region_col,"2Y":y2_col,"10Y":y10_col}.items() if v is None]
    if missing:
        raise BadRequest(f"Brede bron mist kolommen: {missing}")

    # detecteer labels voor US / EU
    q_labels = f"SELECT DISTINCT {region_col} AS region FROM `{wide_fqn}` ORDER BY 1"
    rgn = run_query(q_labels, timeout=30)
    regs = [str(x) for x in rgn["region"].tolist()]
    regs_l = [x.lower() for x in regs]
    def pick(cands: list[str]) -> str | None:
        for c in cands:
            if c in regs_l:
                return regs[regs_l.index(c)]
        return None
    us_label = pick(["us","usa","united states","u.s.","u.s.a."])
    eu_label = pick(["eu","euro area","eurozone","ea","euro area (19)","euro area (20)"])
    if not us_label or not eu_label:
        raise BadRequest(f"Kon US/EU labels niet afleiden uit {wide_fqn}. Gevonden: {regs}")

    q = f"""
    WITH base AS (
      SELECT
        CAST({date_col} AS DATE) AS date,
        {region_col} AS region,
        SAFE_CAST({y2_col}  AS FLOAT64) AS y2,
        SAFE_CAST({y10_col} AS FLOAT64) AS y10
      FROM `{wide_fqn}`
      WHERE {region_col} IN ('{us_label}','{eu_label}')
    ),
    pivot AS (
      SELECT
        date,
        MAX(IF(region='{us_label}', y2, NULL))  AS us_2y,
        MAX(IF(region='{us_label}', y10, NULL)) AS us_10y,
        MAX(IF(region='{eu_label}', y2, NULL))  AS eu_2y,
        MAX(IF(region='{eu_label}', y10, NULL)) AS eu_10y
      FROM base
      GROUP BY date
    )
    SELECT
      date,
      us_2y, us_10y,
      (us_10y - us_2y) AS us_spread,
      eu_2y, eu_10y,
      (eu_10y - eu_2y) AS eu_spread,
      (us_2y - eu_2y)  AS diff_2y,
      (us_10y - eu_10y) AS diff_10y,
      ((us_10y - us_2y) - (eu_10y - eu_2y)) AS diff_spread
    FROM pivot
    WHERE us_2y IS NOT NULL AND eu_2y IS NOT NULL
    ORDER BY date
    """
    return run_query(q, timeout=60)

@st.cache_data(ttl=1800, show_spinner=False)
def load_data_resilient():
    # 1) enriched views
    try:
        return load_from_two_enriched_views(US_VIEW, EU_VIEW), "enriched_views"
    except Exception as e1:
        st.info("Kon enriched US/EU views niet laden — fallback via brede bron.")
        with st.expander("Technische foutmelding (enriched views)"):
            st.code(repr(e1))
    # 2) brede bron
    try:
        return load_from_wide_autodetect(WIDE_VIEW), "wide_autodetect"
    except Exception as e2:
        with st.expander("Technische foutmelding (brede bron)"):
            st.code(repr(e2))
        raise

# ================== LOAD ==================
with st.spinner("Data laden…"):
    try:
        df, mode = load_data_resilient()
    except Exception:
        st.error("Kon geen yield-data laden uit BigQuery.")
        st.caption("Controleer of de opgegeven views/tables bestaan en of er kolommen voor 2Y en 10Y zijn.")
        st.stop()

st.caption(f"Bronmodus: **{mode}**")
if df.empty:
    st.warning("Geen rijen (join of filter levert leeg resultaat)."); st.stop()

# ================== ROLLING STATS (90d) ==================
for c in ["diff_spread", "diff_2y", "diff_10y"]:
    df = add_roll_stats(df, c)

# ================== CURVES (2Y/10Y, dubbele y-as) ==================
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=df["date"], y=df["us_2y"],  mode="lines", name="US 2Y"))
fig1.add_trace(go.Scatter(x=df["date"], y=df["us_10y"], mode="lines", name="US 10Y"))
fig1.add_trace(go.Scatter(x=df["date"], y=df["eu_2y"],  mode="lines", name="EU 2Y", yaxis="y2"))
fig1.add_trace(go.Scatter(x=df["date"], y=df["eu_10y"], mode="lines", name="EU 10Y", yaxis="y2"))
fig1.update_layout(
    title="Yield Curves US vs EU (2Y & 10Y)",
    xaxis_title="Date",
    yaxis=dict(title="US Yield (%)"),
    yaxis2=dict(title="EU Yield (%)", overlaying="y", side="right"),
    legend=dict(orientation="h"),
    margin=dict(l=10,r=10,t=10,b=10),
)
st.plotly_chart(fig1, use_container_width=True)

# ================== SPREADS (10Y–2Y) + DIFFERENTIAL ==================
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=df["date"], y=df["us_spread"], mode="lines", name="US 10Y–2Y"))
fig2.add_trace(go.Scatter(x=df["date"], y=df["eu_spread"], mode="lines", name="EU 10Y–2Y"))
fig2.add_trace(go.Bar(x=df["date"], y=df["diff_spread"], name="US–EU diff (bp)", opacity=0.35))
fig2.update_layout(
    title="10Y–2Y Spreads (US, EU) + US–EU differential",
    xaxis_title="Date",
    yaxis_title="Basis points",
    legend=dict(orientation="h"),
    margin=dict(l=10,r=10,t=10,b=10),
)
st.plotly_chart(fig2, use_container_width=True)

# ================== DIFFERENTIALS — BAND & Z-SCORES ==================
st.subheader("US–EU differentials — 90d μ ± 1σ & Z-scores")
show_band = st.checkbox("Toon 90d μ ± 1σ band", value=True)
as_z      = st.checkbox("Toon als Z-scores (rolling, 90d)", value=False)
tabs      = st.tabs(["Spread (10Y–2Y)", "2Y", "10Y"])

def plot_diff(tab, df_in: pd.DataFrame, base_col: str, title_txt: str, unit: str):
    with tab:
        series = df_in[f"{base_col}_z"] if as_z else df_in[base_col]
        mu = df_in[f"{base_col}_mu"]; sd = df_in[f"{base_col}_sd"]

        if as_z:
            y_title = "Z-score"
            upper = (mu + sd - mu) / sd  # +1
            lower = (mu - sd - mu) / sd  # -1
            mu_plot = (mu - mu) / sd     # 0
        else:
            y_title = unit
            upper = mu + sd; lower = mu - sd; mu_plot = mu

        fig = go.Figure()
        if show_band:
            fig.add_trace(go.Scatter(x=df_in["date"], y=upper, name="μ+1σ", mode="lines",
                                     line=dict(width=0.5), showlegend=False))
            fig.add_trace(go.Scatter(x=df_in["date"], y=lower, name="μ-1σ", mode="lines",
                                     fill="tonexty", line=dict(width=0.5), opacity=0.20, showlegend=False))
        fig.add_trace(go.Scatter(x=df_in["date"], y=mu_plot, name="μ (90d)", mode="lines",
                                 line=dict(width=1, dash="dot")))
        fig.add_trace(go.Scatter(x=df_in["date"], y=series, name="US–EU differential", mode="lines"))
        fig.update_layout(
            title=title_txt + (" — Z-scores" if as_z else ""),
            xaxis_title="Date", yaxis_title=y_title, legend=dict(orientation="h"),
            margin=dict(l=10,r=10,t=10,b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

        # KPI’s
        last = df_in.dropna(subset=[base_col, f"{base_col}_z"]).iloc[-1]
        c1, c2 = st.columns(2)
        c1.metric("Laatste differential", f"{last[base_col]:.1f} {unit}")
        c2.metric("Laatste Z-score (90d)", f"{last[f'{base_col}_z']:.2f}")

plot_diff(tabs[0], df, "diff_spread", "US–EU differential: (10Y–2Y)", "bp")
plot_diff(tabs[1], df, "diff_2y",     "US–EU differential: 2Y",        "bp")
plot_diff(tabs[2], df, "diff_10y",    "US–EU differential: 10Y",       "bp")
