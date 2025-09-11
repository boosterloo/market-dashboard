# pages/Yield_US_vs_EU.py
# 🇺🇸 vs 🇪🇺 — robuuste autodetect + vergelijking + 90d μ±1σ & z-scores

import streamlit as st
import pandas as pd
import numpy as np
import re
import plotly.graph_objects as go
from google.api_core.exceptions import NotFound, BadRequest
from utils.bq import run_query, bq_ping

st.set_page_config(page_title="🇺🇸 vs 🇪🇺 Yield Curve", layout="wide")
st.title("🇺🇸 vs 🇪🇺 Yield Curve Vergelijking")

# -------------------- Health --------------------
try:
    if not bq_ping():
        st.error("Geen BigQuery-verbinding.")
        st.stop()
except Exception as e:
    st.error("Geen BigQuery-verbinding.")
    st.caption(f"Details: {e}")
    st.stop()

# -------------------- Secrets / Defaults --------------------
SECRETS = st.secrets.get("tables", {})
US_VIEW = SECRETS.get("us_yield_view", "nth-pier-468314-p7.marketdata.us_yield_v")
EU_VIEW = SECRETS.get("eu_yield_view", "nth-pier-468314-p7.marketdata.eu_yield_v")
WIDE_VIEW = SECRETS.get("yield_wide_view", "nth-pier-468314-p7.marketdata.yield_curve_analysis_wide")

with st.expander("🔎 Debug: opgegeven bronnen"):
    st.write({"US_VIEW": US_VIEW, "EU_VIEW": EU_VIEW, "WIDE_VIEW": WIDE_VIEW})

# -------------------- Helpers --------------------
def list_candidate_tables(project_dataset: str) -> pd.DataFrame:
    # project_dataset: "project.dataset"
    q = f"""
    SELECT table_schema, table_name
    FROM `{project_dataset}.INFORMATION_SCHEMA.TABLES`
    ORDER BY table_name
    """
    return run_query(q)

def describe_table(project_dataset: str, table: str) -> pd.DataFrame:
    q = f"""
    SELECT column_name, data_type
    FROM `{project_dataset}.INFORMATION_SCHEMA.COLUMNS`
    WHERE table_name = '{table}'
    ORDER BY ordinal_position
    """
    return run_query(q)

def split_fqn(fqn: str):
    # "proj.dataset.table" -> proj, dataset, table
    parts = fqn.split(".")
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    raise ValueError(f"FQN verwacht 'project.dataset.table' maar kreeg: {fqn}")

def find_col(cols, patterns):
    """Zoek een kolomnaam die matcht op 1 van de regex patterns (case-insensitive)."""
    for p in patterns:
        rx = re.compile(p, re.IGNORECASE)
        for c in cols:
            if rx.fullmatch(c) or rx.search(c):
                return c
    return None

def safe_cols_for_yields(df_columns: list[str]):
    # brede tabel: verwacht date, region, y_2y, y_10y in welke naamvariant dan ook
    date_col   = find_col(df_columns, [r"^date$", r"^datum$"])
    region_col = find_col(df_columns, [r"^region$", r"^regio$", r"^area$", r"^zone$"])
    y2_col     = find_col(df_columns, [r"^y[_]?2y$", r"^y2y$", r"^yield[_]?(2y|2yr)$", r"^r[_]?2y$"])
    y10_col    = find_col(df_columns,[r"^y[_]?10y$", r"^y10y$", r"^yield[_]?(10y|10yr)$", r"^r[_]?10y$"])
    return date_col, region_col, y2_col, y10_col

def add_roll_stats(frame: pd.DataFrame, col: str, window: int = 90, minp: int = 30):
    mu = frame[col].rolling(window, min_periods=minp).mean()
    sd = frame[col].rolling(window, min_periods=minp).std()
    frame[f"{col}_mu"] = mu
    frame[f"{col}_sd"] = sd
    frame[f"{col}_z"]  = (frame[col] - mu) / sd
    return frame

# -------------------- Auto-detect modus --------------------
# 1) Probeer losse US/EU views
def load_from_two_views(us_fqn: str, eu_fqn: str) -> pd.DataFrame:
    q = f"""
    WITH us AS (
      SELECT date, y_2y AS us_2y, y_10y AS us_10y, (y_10y - y_2y) AS us_spread
      FROM `{us_fqn}`
    ),
    eu AS (
      SELECT date, y_2y AS eu_2y, y_10y AS eu_10y, (y_10y - y_2y) AS eu_spread
      FROM `{eu_fqn}`
    )
    SELECT
      us.date,
      us.us_2y, us.us_10y, us.us_spread,
      eu.eu_2y, eu.eu_10y, eu.eu_spread,
      (us.us_2y - eu.eu_2y)                AS diff_2y,
      (us.us_10y - eu.eu_10y)              AS diff_10y,
      (us.us_spread - eu.eu_spread)        AS diff_spread
    FROM us
    JOIN eu USING(date)
    ORDER BY date
    """
    return run_query(q)

# 2) Zo niet, probeer brede view: detecteer kolommen dynamisch
def load_from_wide_autodetect(wide_fqn: str) -> pd.DataFrame:
    proj, dset, tbl = split_fqn(wide_fqn)
    cols_df = describe_table(f"{proj}.{dset}", tbl)
    if cols_df.empty:
        raise NotFound(f"Geen schema gevonden voor {wide_fqn}")

    cols = cols_df["column_name"].str.lower().tolist()
    date_col, region_col, y2_col, y10_col = safe_cols_for_yields(cols)

    with st.expander("📋 Gevonden kolommen (brede bron)"):
        st.dataframe(cols_df)

    missing = [n for n, v in {
        "date": date_col, "region": region_col, "2Y": y2_col, "10Y": y10_col
    }.items() if v is None]
    if missing:
        raise BadRequest(f"Kolommen niet gevonden in {wide_fqn}: {missing}")

    # Regionwaarden bepalen (US/EU)
    q_regions = f"""
    SELECT DISTINCT {region_col} AS region
    FROM `{wide_fqn}`
    ORDER BY 1
    """
    rgn = run_query(q_regions)
    # probeer intelligent de juiste labels te kiezen
    # we accepteren 'US','USA','United States' etc. en 'EU','EA','Euro Area','Eurozone'
    def pick(regs, candidates):
        regs_lower = [str(x).lower() for x in regs]
        for cands in candidates:
            for c in cands:
                if c in regs_lower:
                    # return originele string met zelfde index
                    return regs[regs_lower.index(c)]
        return None

    regs = rgn["region"].astype(str).tolist()
    us_label = pick(regs, [["us"], ["usa","united states","u.s.","u.s.a."]])
    eu_label = pick(regs, [["eu","euro area","eurozone","ea","euro area (19)","euro area (20)"]])

    with st.expander("🌍 Gevonden regio labels"):
        st.write({"alle_labels": regs, "gekozen_US": us_label, "gekozen_EU": eu_label})

    if not us_label or not eu_label:
        raise BadRequest(f"Kon US/EU labels niet afleiden uit {wide_fqn}. Gevonden: {regs}")

    q = f"""
    WITH base AS (
      SELECT {date_col} AS date, {region_col} AS region, {y2_col} AS y2, {y10_col} AS y10
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
    return run_query(q)

@st.cache_data(ttl=1800, show_spinner=False)
def load_data_resilient():
    # 1) losse views
    try:
        df = load_from_two_views(US_VIEW, EU_VIEW)
        return df, "two_views"
    except Exception as e1:
        st.info("Kon US/EU losse views niet laden — probeer fallback via brede bron.")
        with st.expander("Technische foutmelding (losse views)"):
            st.code(repr(e1))

    # 2) brede view/tabel autodetect
    try:
        df = load_from_wide_autodetect(WIDE_VIEW)
        return df, "wide_autodetect"
    except Exception as e2:
        with st.expander("Technische foutmelding (brede bron)"):
            st.code(repr(e2))
        raise

# -------------------- Load --------------------
with st.spinner("Data laden…"):
    try:
        df, mode = load_data_resilient()
    except Exception:
        st.error("Kon geen yield-data laden uit BigQuery.")
        st.caption("Controleer of de opgegeven views/tables bestaan en of er kolommen voor 2Y en 10Y zijn.")
        # Handige tips:
        st.markdown("""
        **Snelle checks (voer in BigQuery uit):**
        ```sql
        -- Alles in je dataset
        SELECT table_schema, table_name
        FROM `YOUR_PROJECT.YOUR_DATASET.INFORMATION_SCHEMA.TABLES`
        ORDER BY table_name;

        -- Kolommen van je brede bron
        SELECT column_name, data_type
        FROM `YOUR_PROJECT.YOUR_DATASET.INFORMATION_SCHEMA.COLUMNS`
        WHERE table_name = 'YOUR_TABLE';
        ```
        """)
        st.stop()

st.caption(f"Bronmodus: **{mode}**")
if df.empty:
    st.warning("Geen rijen (join of filter levert leeg resultaat).")
    st.stop()

# -------------------- Charts: Term structure + spreads --------------------
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
    legend=dict(orientation="h")
)
st.plotly_chart(fig1, use_container_width=True)

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=df["date"], y=df["us_spread"], mode="lines", name="US 10Y–2Y"))
fig2.add_trace(go.Scatter(x=df["date"], y=df["eu_spread"], mode="lines", name="EU 10Y–2Y"))
fig2.add_trace(go.Bar(x=df["date"], y=df["diff_spread"], name="US–EU diff (bp)", opacity=0.35))
fig2.update_layout(
    title="10Y–2Y Spreads (US, EU) + US–EU differential",
    xaxis_title="Date",
    yaxis_title="Basis points",
    legend=dict(orientation="h")
)
st.plotly_chart(fig2, use_container_width=True)

# -------------------- Rolling band & Z-scores --------------------
st.subheader("US–EU differentials — 90d μ ± 1σ & Z-scores")

show_band = st.checkbox("Toon 90d μ ± 1σ band", value=True)
as_z = st.checkbox("Toon als Z-scores (rolling, 90d)", value=False)
tabs = st.tabs(["Spread (10Y–2Y)", "2Y", "10Y"])

def plot_diff(tab, df: pd.DataFrame, base_col: str, title_txt: str, unit: str):
    with tab:
        series = df[f"{base_col}_z"] if as_z else df[base_col]
        mu = df[f"{base_col}_mu"]
        sd = df[f"{base_col}_sd"]

        if as_z:
            y_title = "Z-score"
            upper = (mu + sd - mu) / sd  # +1
            lower = (mu - sd - mu) / sd  # -1
            mu_plot = (mu - mu) / sd     # 0
        else:
            y_title = unit
            upper = mu + sd
            lower = mu - sd
            mu_plot = mu

        fig = go.Figure()
        if show_band:
            fig.add_trace(go.Scatter(x=df["date"], y=upper, name="μ+1σ", mode="lines", line=dict(width=0.5), showlegend=False))
            fig.add_trace(go.Scatter(x=df["date"], y=lower, name="μ-1σ", mode="lines", fill="tonexty", line=dict(width=0.5), opacity=0.20, showlegend=False))
        fig.add_trace(go.Scatter(x=df["date"], y=mu_plot, name="μ (90d)", mode="lines", line=dict(width=1, dash="dot")))
        fig.add_trace(go.Scatter(x=df["date"], y=series, name="US–EU differential", mode="lines"))
        fig.update_layout(title=title_txt + (" — Z-scores" if as_z else ""), xaxis_title="Date", yaxis_title=y_title, legend=dict(orientation="h"))
        st.plotly_chart(fig, use_container_width=True)

        last = df.dropna(subset=[base_col, f"{base_col}_z"]).iloc[-1]
        c1, c2 = st.columns(2)
        c1.metric("Laatste differential", f"{last[base_col]:.1f} {unit}")
        c2.metric("Laatste Z-score (90d)", f"{last[f'{base_col}_z']:.2f}")

for spec in [
    ("diff_spread", "US–EU differential: (10Y–2Y)", "bp"),
    ("diff_2y",     "US–EU differential: 2Y",        "bp"),
    ("diff_10y",    "US–EU differential: 10Y",       "bp"),
]:
    plot_diff(tabs[["diff_spread","diff_2y","diff_10y"].index(spec[0])], df, spec[0], spec[1], spec[2])
