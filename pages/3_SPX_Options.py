# pages/3_SPX_Options.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from google.cloud import bigquery
from google.oauth2 import service_account

# ────────────────────────────────────────────────────────────────────────────────
# BigQuery client via st.secrets
# ────────────────────────────────────────────────────────────────────────────────
def get_bq_client():
    sa_info = st.secrets["gcp_service_account"]
    creds = service_account.Credentials.from_service_account_info(sa_info)
    project_id = st.secrets.get("PROJECT_ID", sa_info.get("project_id"))
    return bigquery.Client(project=project_id, credentials=creds)

_bq_client = get_bq_client()

def _bq_param(name, value):
    """Juiste BigQuery query-parameter (scalar/array) met correcte type."""
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return bigquery.ArrayQueryParameter(name, "STRING", [])
        elem = value[0]
        if isinstance(elem, int):
            return bigquery.ArrayQueryParameter(name, "INT64", list(value))
        if isinstance(elem, float):
            return bigquery.ArrayQueryParameter(name, "FLOAT64", list(value))
        if isinstance(elem, (date, pd.Timestamp, datetime)):
            return bigquery.ArrayQueryParameter(name, "DATE", [str(pd.to_datetime(v).date()) for v in value])
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

# ────────────────────────────────────────────────────────────────────────────────
# Streamlit page settings
# ────────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="SPX Options Dashboard", layout="wide")
st.title("🧩 SPX Options Dashboard")

VIEW = "marketdata.spx_options_enriched_v"

# ────────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=600, show_spinner=False)
def load_date_bounds():
    sql = f"""
    SELECT
      MIN(CAST(snapshot_date AS DATE)) AS min_date,
      MAX(CAST(snapshot_date AS DATE)) AS max_date
    FROM `{VIEW}`
    """
    df = run_query(sql)
    return df["min_date"].iloc[0], df["max_date"].iloc[0]

min_date, max_date = load_date_bounds()
default_start = max(min_date, max_date - timedelta(days=365))

colA, colB, colC, colD = st.columns([1.2, 1, 1, 1])
with colA:
    daterange = st.date_input(
        "Periode (snapshot_date)",
        value=(default_start, max_date),
        min_value=min_date,
        max_value=max_date,
        format="YYYY-MM-DD"
    )
    start_date, end_date = daterange  # tuple(date, date)
with colB:
    opt_types = st.multiselect("Type", ["call", "put"], default=["call", "put"])
with colC:
    dte_range = st.slider("Days to Expiration (DTE)", 0, 365, (0, 60), step=1)
with colD:
    moneyness_range = st.slider("Moneyness (strike/underlying − 1)", -0.20, 0.20, (-0.10, 0.10), step=0.01)

@st.cache_data(ttl=600, show_spinner=False)
def load_expirations(start_date: date, end_date: date):
    sql = f"""
    SELECT DISTINCT expiration
    FROM `{VIEW}`
    WHERE DATE(snapshot_date) BETWEEN @start AND @end
    ORDER BY expiration
    """
    df = run_query(sql, params={"start": start_date, "end": end_date})
    return sorted(pd.to_datetime(df["expiration"]).dt.date.unique())

exps = load_expirations(start_date, end_date)
exp_default = [x for x in exps[:5]] if len(exps) > 0 else []
selected_exps = st.multiselect("Expiraties (optioneel, voor strike- en OI-grafiek)", exps, default=exp_default)

# ────────────────────────────────────────────────────────────────────────────────
# Data laden met filters
# ────────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=600, show_spinner=True)
def load_filtered(start_date, end_date, types, dte_min, dte_max, mny_min, mny_max):
    sql = f"""
    WITH base AS (
      SELECT
        snapshot_date,
        contract_symbol,
        type,
        expiration,
        days_to_exp,
        strike,
        underlying_price,
        SAFE_DIVIDE(CAST(strike AS FLOAT64), NULLIF(underlying_price, 0)) - 1.0 AS moneyness,
        in_the_money,
        last_price,
        bid, ask, mid_price,
        implied_volatility,
        open_interest,
        volume,
        vix,
        ppd
      FROM `{VIEW}`
      WHERE DATE(snapshot_date) BETWEEN @start AND @end
        AND days_to_exp BETWEEN @dte_min AND @dte_max
        AND SAFE_DIVIDE(CAST(strike AS FLOAT64), NULLIF(underlying_price, 0)) - 1.0 BETWEEN @mny_min AND @mny_max
        AND LOWER(type) IN UNNEST(@types)
    )
    SELECT * FROM base
    """
    params = {
        "start": start_date,
        "end": end_date,
        "dte_min": int(dte_min),
        "dte_max": int(dte_max),
        "mny_min": float(mny_min),
        "mny_max": float(mny_max),
        "types": [t.lower() for t in types] if types else ["call", "put"],
    }
    df = run_query(sql, params=params)
    if not df.empty:
        df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
        df["expiration"] = pd.to_datetime(df["expiration"]).dt.date
        df["days_to_exp"] = pd.to_numeric(df["days_to_exp"], errors="coerce")
        df["implied_volatility"] = pd.to_numeric(df["implied_volatility"], errors="coerce")
        df["open_interest"] = pd.to_numeric(df["open_interest"], errors="coerce")
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
        df["ppd"] = pd.to_numeric(df["ppd"], errors="coerce")
        df["moneyness_pct"] = 100 * df["moneyness"]
    return df

df = load_filtered(
    start_date, end_date, opt_types,
    dte_range[0], dte_range[1],
    moneyness_range[0], moneyness_range[1]
)

if df.empty:
    st.warning("Geen data voor de huidige filters.")
    st.stop()

# ────────────────────────────────────────────────────────────────────────────────
# KPI’s
# ────────────────────────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
with col1: st.metric("Records", f"{len(df):,}")
with col2: st.metric("Gem. IV", f"{df['implied_volatility'].mean():.2%}")
with col3: st.metric("Som Volume", f"{int(df['volume'].sum()):,}")
with col4: st.metric("Som Open Interest", f"{int(df['open_interest'].sum()):,}")

st.markdown("---")

# 1) Term structure IV
term = df.groupby(["days_to_exp", "type"], as_index=False)["implied_volatility"].mean().sort_values("days_to_exp")
fig_term = go.Figure()
for t in sorted(term["type"].unique()):
    sub = term[term["type"] == t]
    fig_term.add_trace(go.Scatter(x=sub["days_to_exp"], y=sub["implied_volatility"],
                                  mode="lines+markers", name=f"IV {t.upper()}"))
fig_term.update_layout(title="Term Structure — Gemiddelde IV", xaxis_title="DTE", yaxis_title="Implied Volatility", height=420)
st.plotly_chart(fig_term, use_container_width=True)

# 2) Calls vs Puts — volume en OI
agg_cp = df.groupby("type", as_index=False)[["volume", "open_interest"]].sum().sort_values("type")
fig_cp = make_subplots(rows=1, cols=2, subplot_titles=("Volume", "Open Interest"))
fig_cp.add_trace(go.Bar(x=agg_cp["type"].str.upper(), y=agg_cp["volume"]), row=1, col=1)
fig_cp.add_trace(go.Bar(x=agg_cp["type"].str.upper(), y=agg_cp["open_interest"]), row=1, col=2)
fig_cp.update_layout(height=420, title_text="Calls vs Puts — Volume & Open Interest", showlegend=False)
st.plotly_chart(fig_cp, use_container_width=True)

# 3) OI per strike per expiratie
if selected_exps:
    for e in selected_exps[:5]:
        sub = df[df["expiration"] == e].groupby("strike", as_index=False)["open_interest"].sum().sort_values("strike")
        if sub.empty:
            continue
        fig = go.Figure(go.Bar(x=sub["strike"], y=sub["open_interest"]))
        fig.update_layout(title=f"Open Interest per Strike — Expiry {e}", xaxis_title="Strike", yaxis_title="Open Interest", height=380)
        st.plotly_chart(fig, use_container_width=True)

# 4) Heatmap (Volume of OI)
metric_choice = st.radio("Heatmap metric", ["volume", "open_interest"], horizontal=True, index=0)
bins = st.slider("Aantal strike-bins", 20, 100, 40, step=5)
q_low, q_hi = df["strike"].quantile([0.02, 0.98])
strike_bins = np.linspace(q_low, q_hi, bins+1)
labels = 0.5 * (strike_bins[:-1] + strike_bins[1:])
df_hm = df[(df["strike"] >= q_low) & (df["strike"] <= q_hi)].copy()
df_hm["strike_bin"] = pd.cut(df_hm["strike"], bins=strike_bins, labels=np.round(labels, 1), include_lowest=True)
pivot = df_hm.pivot_table(index="days_to_exp", columns="strike_bin", values=metric_choice, aggfunc="sum", fill_value=0)
fig_hm = go.Figure(data=go.Heatmap(z=pivot.values, x=[float(x) for x in pivot.columns.astype(float)], y=pivot.index))
fig_hm.update_layout(title=f"Heatmap — {metric_choice.capitalize()} over DTE × Strike", xaxis_title="Strike (bin)", yaxis_title="DTE", height=520)
st.plotly_chart(fig_hm, use_container_width=True)

# 5) PPD
colL, colR = st.columns(2)
with colL:
    fig_ppd_hist = go.Figure(go.Histogram(x=df["ppd"].dropna(), nbinsx=60))
    fig_ppd_hist.update_layout(title="PPD Distributie", xaxis_title="PPD", yaxis_title="Aantal", height=420)
    st.plotly_chart(fig_ppd_hist, use_container_width=True)

with colR:
    # ✅ Robuuste grouping met gelabelde kolom
    ts_ppd = (
        df.assign(snap_date=df["snapshot_date"].dt.date)
          .groupby("snap_date", as_index=False)["ppd"].mean()
          .rename(columns={"snap_date": "date"})
    )
    if ts_ppd.empty:
        st.info("Geen PPD-tijdreeks voor de huidige filters.")
    else:
        fig_ppd_ts = go.Figure(go.Scatter(x=ts_ppd["date"], y=ts_ppd["ppd"], mode="lines"))
        fig_ppd_ts.update_layout(title="Gemiddelde PPD per dag", xaxis_title="Snapshot date", yaxis_title="PPD", height=420)
        st.plotly_chart(fig_ppd_ts, use_container_width=True)

# 6) VIX vs IV
vix_vs_iv = (
    df.assign(snap_date=df["snapshot_date"].dt.date)
      .groupby("snap_date", as_index=False)
      .agg(vix=("vix", "mean"), iv=("implied_volatility", "mean"))
      .rename(columns={"snap_date": "date"})
)
if vix_vs_iv.empty:
    st.info("Geen VIX/IV-tijdreeks voor de huidige filters.")
else:
    fig_vix = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, subplot_titles=("VIX", "Gemiddelde IV"))
    fig_vix.add_trace(go.Scatter(x=vix_vs_iv["date"], y=vix_vs_iv["vix"], mode="lines", name="VIX"), row=1, col=1)
    fig_vix.add_trace(go.Scatter(x=vix_vs_iv["date"], y=vix_vs_iv["iv"], mode="lines", name="IV"), row=2, col=1)
    fig_vix.update_layout(height=520, title_text="VIX vs Gemiddelde Implied Volatility")
    st.plotly_chart(fig_vix, use_container_width=True)

# 7) Tabel — top contracts
metric_top = st.radio("Top contracts sorteer op", ["volume", "open_interest"], horizontal=True, index=0)
cols_view = ["snapshot_date","contract_symbol","type","expiration","days_to_exp","strike","underlying_price",
             "implied_volatility","last_price","bid","ask","mid_price","volume","open_interest","ppd"]
tbl = df[cols_view].sort_values(metric_top, ascending=False).head(200)
st.dataframe(tbl, use_container_width=True, hide_index=True)
