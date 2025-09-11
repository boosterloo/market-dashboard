# pages/Yield_US_vs_EU.py
# 🇺🇸 vs 🇪🇺 — vergelijking met 90d μ±1σ-band & Z-scores
# + globale periode-slider, KPI-balk en |Z|>2 alert

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from google.api_core.exceptions import BadRequest
from utils.bq import run_query, bq_ping

st.set_page_config(page_title="🇺🇸 vs 🇪🇺 Yield Curve", layout="wide")
st.title("🇺🇸 vs 🇪🇺 Yield Curve Vergelijking")

# ================== SECRETS / DEFAULTS ==================
PROJECT_ID = st.secrets["gcp_service_account"]["project_id"]
TABLES     = st.secrets.get("tables", {})

US_VIEW   = TABLES.get("us_yield_view",   f"{PROJECT_ID}.marketdata.us_yield_curve_enriched_v")
EU_VIEW   = TABLES.get("eu_yield_view",   f"{PROJECT_ID}.marketdata.eu_yield_curve_enriched_v")
WIDE_VIEW = TABLES.get("yield_wide_view", f"{PROJECT_ID}.marketdata.yield_curve_analysis_wide")  # momenteel niet gebruikt

with st.expander("🔎 Debug: opgegeven bronnen"):
    st.write({"US_VIEW": US_VIEW, "EU_VIEW": EU_VIEW, "WIDE_VIEW": WIDE_VIEW})

# ================== HEALTH ==================
try:
    if not bq_ping():
        st.error("Geen BigQuery-verbinding."); st.stop()
except Exception as e:
    st.error("Geen BigQuery-verbinding."); st.caption(f"Details: {e}"); st.stop()

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
        if c.lower() in cols: return c
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
    us_cols = list_columns(us_fqn)
    eu_cols = list_columns(eu_fqn)

    us_2y  = choose_col(us_cols, ["y_2y_synth", "y_2y"])
    us_10y = choose_col(us_cols, ["y_10y_synth", "y_10y"])
    eu_2y  = choose_col(eu_cols, ["y_2y_synth", "y_2y"])
    eu_10y = choose_col(eu_cols, ["y_10y_synth", "y_10y"])

    missing = [n for n,v in {"US 2Y":us_2y, "US 10Y":us_10y, "EU 2Y":eu_2y, "EU 10Y":eu_10y}.items() if v is None]
    if missing:
        raise BadRequest(f"Ontbrekende kolommen in enriched views: {missing}")

    q = f"""
    WITH us AS (
      SELECT date,
             SAFE_CAST({us_2y}  AS FLOAT64) AS us_2y,
             SAFE_CAST({us_10y} AS FLOAT64) AS us_10y,
             SAFE_CAST({us_10y} AS FLOAT64) - SAFE_CAST({us_2y} AS FLOAT64) AS us_spread
      FROM `{us_fqn}`
    ),
    eu AS (
      SELECT date,
             SAFE_CAST({eu_2y}  AS FLOAT64) AS eu_2y,
             SAFE_CAST({eu_10y} AS FLOAT64) AS eu_10y,
             SAFE_CAST({eu_10y} AS FLOAT64) - SAFE_CAST({eu_2y} AS FLOAT64) AS eu_spread
      FROM `{eu_fqn}`
    )
    SELECT
      us.date,
      us.us_2y, us.us_10y, us.us_spread,
      eu.eu_2y, eu.eu_10y, eu.eu_spread,
      (us.us_2y - eu.eu_2y)                   AS diff_2y,
      (us.us_10y - eu.eu_10y)                 AS diff_10y,
      (us.us_spread - eu.eu_spread)           AS diff_spread
    FROM us
    JOIN eu USING(date)
    ORDER BY date
    """
    return run_query(q, timeout=60)

def load_data_resilient():
    df = load_from_two_enriched_views(US_VIEW, EU_VIEW)
    return df, "enriched_views"

# ================== LOAD ==================
with st.spinner("Data laden…"):
    df, mode = load_data_resilient()

st.caption(f"Bronmodus: **{mode}**")
if df.empty:
    st.warning("Geen rijen."); st.stop()

df["date"] = pd.to_datetime(df["date"])

# Rolling stats (op volledige reeks, daarna filteren met periode)
for c in ["diff_spread", "diff_2y", "diff_10y"]:
    df = add_roll_stats(df, c)

# ================== GLOBALE PERIODE ==================
st.subheader("Periode")
dmin, dmax = df["date"].min(), df["date"].max()

left, right = st.columns([1.8, 1])
with left:
    preset = st.radio(
        "Preset",
        ["6M","1Y","3Y","5Y","10Y","YTD","Max","Custom"],
        horizontal=True, index=1
    )

def clamp(ts: pd.Timestamp) -> pd.Timestamp: return max(dmin, ts)

if preset == "6M":
    start_date, end_date = clamp(dmax - pd.DateOffset(months=6)), dmax
elif preset == "1Y":
    start_date, end_date = clamp(dmax - pd.DateOffset(years=1)), dmax
elif preset == "3Y":
    start_date, end_date = clamp(dmax - pd.DateOffset(years=3)), dmax
elif preset == "5Y":
    start_date, end_date = clamp(dmax - pd.DateOffset(years=5)), dmax
elif preset == "10Y":
    start_date, end_date = clamp(dmax - pd.DateOffset(years=10)), dmax
elif preset == "YTD":
    start_date, end_date = clamp(pd.Timestamp(dmax.year,1,1)), dmax
elif preset == "Max":
    start_date, end_date = dmin, dmax
else:
    date_range = st.slider(
        "Selecteer periode (Custom)",
        min_value=dmin.date(), max_value=dmax.date(),
        value=(clamp(dmax - pd.DateOffset(years=1)).date(), dmax.date()),
    )
    start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])

mask = (df["date"] >= start_date) & (df["date"] <= end_date)
dfp = df.loc[mask].copy()
if dfp.empty:
    st.info("Geen data in de gekozen periode."); st.stop()

# ================== KPI-BALK (gesynchroniseerd met periode) ==================
st.subheader("KPI’s (laatste in periode)")
try:
    last = dfp.dropna(subset=["us_2y","us_10y","eu_2y","eu_10y"]).iloc[-1]
except IndexError:
    st.info("Niet genoeg data voor KPI’s in de geselecteerde periode."); last = None

def fmt_pct(p): 
    return "—" if pd.isna(p) else f"{float(p):.2f}%"

def fmt_bp(x):
    return "—" if pd.isna(x) else f"{float(x):.1f} bp"

if last is not None:
    # differentials in bp
    diff2_bp   = (last["diff_2y"]   * 100.0) if pd.notna(last["diff_2y"])   else np.nan
    diff10_bp  = (last["diff_10y"]  * 100.0) if pd.notna(last["diff_10y"])  else np.nan
    diffsp_bp  = (last["diff_spread"] * 100.0) if pd.notna(last["diff_spread"]) else np.nan

    r1 = st.columns(4)
    r1[0].metric("US 2Y", fmt_pct(last["us_2y"]))
    r1[1].metric("EU 2Y", fmt_pct(last["eu_2y"]))
    r1[2].metric("Δ2Y (US–EU)", fmt_bp(diff2_bp), delta=(f"z={last['diff_2y_z']:.2f}" if pd.notna(last["diff_2y_z"]) else None))
    r1[3].metric("Δ(10Y–2Y) (US–EU)", fmt_bp(diffsp_bp), delta=(f"z={last['diff_spread_z']:.2f}" if pd.notna(last["diff_spread_z"]) else None))

    r2 = st.columns(3)
    r2[0].metric("US 10Y", fmt_pct(last["us_10y"]))
    r2[1].metric("EU 10Y", fmt_pct(last["eu_10y"]))
    r2[2].metric("Δ10Y (US–EU)", fmt_bp(diff10_bp), delta=(f"z={last['diff_10y_z']:.2f}" if pd.notna(last["diff_10y_z"]) else None))

    # ===== Alert bij |Z| > 2 =====
    alerts = []
    def add_alert(label, base):
        z = last.get(f"{base}_z", np.nan)
        val = last.get(base, np.nan)
        if pd.notna(z) and abs(z) > 2:
            dir_txt = "US > EU" if (pd.notna(val) and val > 0) else ("US < EU" if pd.notna(val) else "")
            alerts.append((label, z, dir_txt))
    add_alert("Δ(10Y–2Y)", "diff_spread")
    add_alert("Δ2Y",       "diff_2y")
    add_alert("Δ10Y",      "diff_10y")

    if alerts:
        # escalatie bij |z|>=3
        sev = "warning"
        if any(abs(z) >= 3 for _, z, _ in alerts): sev = "error"
        msg = " • ".join([f"{lbl}: z={z:.2f} ({dir_})" for lbl, z, dir_ in alerts])
        if sev == "error":
            st.error(f"📣 Extreem: {msg}")
        else:
            st.warning(f"Let op: {msg}")

# ================== CURVES (2Y/10Y) ==================
st.subheader("Yield Curves US vs EU (2Y & 10Y)")
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=dfp["date"], y=dfp["us_2y"],  mode="lines", name="US 2Y"))
fig1.add_trace(go.Scatter(x=dfp["date"], y=dfp["us_10y"], mode="lines", name="US 10Y"))
fig1.add_trace(go.Scatter(x=dfp["date"], y=dfp["eu_2y"],  mode="lines", name="EU 2Y", yaxis="y2"))
fig1.add_trace(go.Scatter(x=dfp["date"], y=dfp["eu_10y"], mode="lines", name="EU 10Y", yaxis="y2"))
fig1.update_layout(
    margin=dict(l=10,r=10,t=6,b=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    yaxis=dict(title="US Yield (%)"),
    yaxis2=dict(title="EU Yield (%)", overlaying="y", side="right"),
)
fig1.update_xaxes(range=[start_date, end_date])
st.plotly_chart(fig1, use_container_width=True)

# ================== SPREADS (10Y–2Y) + DIFFERENTIAL ==================
st.subheader("10Y–2Y Spreads (US & EU) + US–EU differential")
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=dfp["date"], y=dfp["us_spread"], mode="lines", name="US 10Y–2Y"))
fig2.add_trace(go.Scatter(x=dfp["date"], y=dfp["eu_spread"], mode="lines", name="EU 10Y–2Y"))
fig2.add_trace(go.Bar(x=dfp["date"], y=dfp["diff_spread"], name="US–EU diff (bp)", opacity=0.35))
fig2.update_layout(
    margin=dict(l=10,r=10,t=6,b=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    yaxis_title="Basis points",
    barmode="overlay",
)
fig2.update_xaxes(range=[start_date, end_date])
st.plotly_chart(fig2, use_container_width=True)

# ================== DIFFERENTIALS — BAND & Z-SCORES ==================
st.subheader("US–EU differentials — 90d μ ± 1σ & Z-scores")
show_band = st.checkbox("Toon 90d μ ± 1σ band", value=True)
as_z      = st.checkbox("Toon als Z-scores (rolling, 90d)", value=False)
tabs      = st.tabs(["Spread (10Y–2Y)", "2Y", "10Y"])

def plot_diff(tab, df_in: pd.DataFrame, base_col: str, title_txt: str, unit: str):
    with tab:
        loc = df_in.loc[(df_in["date"] >= start_date) & (df_in["date"] <= end_date)].copy()
        if loc.empty:
            st.info("Geen data in de gekozen periode."); return

        series = loc[f"{base_col}_z"] if as_z else loc[base_col]
        mu = loc[f"{base_col}_mu"]; sd = loc[f"{base_col}_sd"]

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
            fig.add_trace(go.Scatter(x=loc["date"], y=upper, name="μ+1σ", mode="lines",
                                     line=dict(width=0.5), showlegend=False))
            fig.add_trace(go.Scatter(x=loc["date"], y=lower, name="μ-1σ", mode="lines",
                                     fill="tonexty", line=dict(width=0.5), opacity=0.20, showlegend=False))
        fig.add_trace(go.Scatter(x=loc["date"], y=mu_plot, name="μ (90d)", mode="lines",
                                 line=dict(width=1, dash="dot")))
        fig.add_trace(go.Scatter(x=loc["date"], y=series, name="US–EU differential", mode="lines"))
        fig.update_layout(
            margin=dict(l=10,r=10,t=6,b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            yaxis_title=y_title,
        )
        fig.update_xaxes(range=[start_date, end_date])
        st.caption(title_txt)
        st.plotly_chart(fig, use_container_width=True)

        last_loc = loc.dropna(subset=[base_col, f"{base_col}_z"]).iloc[-1]
        c1, c2 = st.columns(2)
        c1.metric("Laatste differential", f"{(last_loc[base_col]*100):.1f} {unit}" if unit=="bp" else f"{last_loc[base_col]:.2f}")
        c2.metric("Laatste Z-score (90d)", f"{last_loc[f'{base_col}_z']:.2f}")

plot_diff(tabs[0], dfp, "diff_spread", "Differential: (10Y–2Y)", "bp")
plot_diff(tabs[1], dfp, "diff_2y",     "Differential: 2Y",       "bp")
plot_diff(tabs[2], dfp, "diff_10y",    "Differential: 10Y",      "bp")
