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
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return bigquery.ArrayQueryParameter(name, "STRING", [])
        elem = value[0]
        if isinstance(elem, int):   return bigquery.ArrayQueryParameter(name, "INT64", list(value))
        if isinstance(elem, float): return bigquery.ArrayQueryParameter(name, "FLOAT64", list(value))
        if isinstance(elem, (date, pd.Timestamp, datetime)):
            return bigquery.ArrayQueryParameter(name, "DATE", [str(pd.to_datetime(v).date()) for v in value])
        return bigquery.ArrayQueryParameter(name, "STRING", [str(v) for v in value])
    if isinstance(value, bool):                         return bigquery.ScalarQueryParameter(name, "BOOL", value)
    if isinstance(value, (int, np.integer)):            return bigquery.ScalarQueryParameter(name, "INT64", int(value))
    if isinstance(value, (float, np.floating)):         return bigquery.ScalarQueryParameter(name, "FLOAT64", float(value))
    if isinstance(value, datetime):                     return bigquery.ScalarQueryParameter(name, "TIMESTAMP", value)
    if isinstance(value, (date, pd.Timestamp)):         return bigquery.ScalarQueryParameter(name, "DATE", str(pd.to_datetime(value).date()))
    return bigquery.ScalarQueryParameter(name, "STRING", str(value))

def run_query(sql: str, params: dict | None = None) -> pd.DataFrame:
    job_config = None
    if params:
        job_config = bigquery.QueryJobConfig(query_parameters=[_bq_param(k, v) for k, v in params.items()])
    return _bq_client.query(sql, job_config=job_config).to_dataframe()

# ────────────────────────────────────────────────────────────────────────────────
# Page
# ────────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="SPX Options Dashboard", layout="wide")
st.title("🧩 SPX Options Dashboard")
VIEW = "marketdata.spx_options_enriched_v"

# ────────────────────────────────────────────────────────────────────────────────
# Basisfilters
# ────────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=600, show_spinner=False)
def load_date_bounds():
    df = run_query(f"SELECT MIN(CAST(snapshot_date AS DATE)) min_date, MAX(CAST(snapshot_date AS DATE)) max_date FROM `{VIEW}`")
    return df["min_date"].iloc[0], df["max_date"].iloc[0]

min_date, max_date = load_date_bounds()
default_start = max(min_date, max_date - timedelta(days=365))

colA, colB, colC, colD = st.columns([1.2, 1, 1, 1])
with colA:
    start_date, end_date = st.date_input(
        "Periode (snapshot_date)",
        value=(default_start, max_date),
        min_value=min_date, max_value=max_date, format="YYYY-MM-DD"
    )
with colB:
    opt_types = st.multiselect("Type", ["call", "put"], default=["call", "put"])
with colC:
    dte_range = st.slider("Days to Expiration (DTE)", 0, 365, (0, 60), step=1)
with colD:
    mny_range = st.slider("Moneyness (strike/underlying − 1)", -0.20, 0.20, (-0.10, 0.10), step=0.01)

@st.cache_data(ttl=600, show_spinner=False)
def load_expirations(start_date: date, end_date: date):
    df = run_query(f"""
        SELECT DISTINCT expiration
        FROM `{VIEW}`
        WHERE DATE(snapshot_date) BETWEEN @start AND @end
        ORDER BY expiration
    """, {"start": start_date, "end": end_date})
    return sorted(pd.to_datetime(df["expiration"]).dt.date.unique())

exps = load_expirations(start_date, end_date)
exp_default = [x for x in exps[:5]] if len(exps) > 0 else []
selected_exps = st.multiselect("Expiraties (optioneel, voor strike- en OI-grafiek)", exps, default=exp_default)

# ────────────────────────────────────────────────────────────────────────────────
# Data laden
# ────────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=600, show_spinner=True)
def load_filtered(start_date, end_date, types, dte_min, dte_max, mny_min, mny_max):
    sql = f"""
    WITH base AS (
      SELECT
        snapshot_date, contract_symbol, type, expiration, days_to_exp,
        strike, underlying_price,
        SAFE_DIVIDE(CAST(strike AS FLOAT64), NULLIF(underlying_price, 0)) - 1.0 AS moneyness,
        in_the_money, last_price, bid, ask, mid_price,
        implied_volatility, open_interest, volume, vix, ppd
      FROM `{VIEW}`
      WHERE DATE(snapshot_date) BETWEEN @start AND @end
        AND days_to_exp BETWEEN @dte_min AND @dte_max
        AND SAFE_DIVIDE(CAST(strike AS FLOAT64), NULLIF(underlying_price, 0)) - 1.0 BETWEEN @mny_min AND @mny_max
        AND LOWER(type) IN UNNEST(@types)
    )
    SELECT * FROM base
    """
    params = {
        "start": start_date, "end": end_date,
        "dte_min": int(dte_min), "dte_max": int(dte_max),
        "mny_min": float(mny_min), "mny_max": float(mny_max),
        "types": [t.lower() for t in types] if types else ["call", "put"],
    }
    df = run_query(sql, params=params)
    if not df.empty:
        df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
        df["expiration"]    = pd.to_datetime(df["expiration"]).dt.date
        for c in ["days_to_exp","implied_volatility","open_interest","volume","ppd","strike","underlying_price","last_price","mid_price","bid","ask"]:
            if c in df: df[c] = pd.to_numeric(df[c], errors="coerce")
        df["moneyness_pct"] = 100 * df["moneyness"]
    return df

df = load_filtered(start_date, end_date, opt_types, dte_range[0], dte_range[1], mny_range[0], mny_range[1])

if df.empty:
    st.warning("Geen data voor de huidige filters.")
    st.stop()

# ────────────────────────────────────────────────────────────────────────────────
# KPI's
# ────────────────────────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
with col1: st.metric("Records", f"{len(df):,}")
with col2: st.metric("Gem. IV", f"{df['implied_volatility'].mean():.2%}")
with col3: st.metric("Som Volume", f"{int(df['volume'].sum()):,}")
with col4: st.metric("Som Open Interest", f"{int(df['open_interest'].sum()):,}")
st.markdown("---")

# ────────────────────────────────────────────────────────────────────────────────
# A) SERIE-SELECTIE: één optiereeks volgen (type+strike+exp)
# ────────────────────────────────────────────────────────────────────────────────
st.subheader("Serie-selectie — volg één optiereeks door de tijd")

colS1, colS2, colS3, colS4 = st.columns([1, 1, 1, 1.6])
with colS1:
    series_type = st.selectbox("Serie Type", options=sorted(df["type"].str.lower().unique()), index=0)
with colS2:
    # strikes binnen de filters
    strikes = sorted(df.loc[df["type"].str.lower() == series_type, "strike"].dropna().unique().tolist())
    series_strike = st.selectbox("Serie Strike", options=strikes, index=len(strikes)//2 if strikes else 0)
with colS3:
    exps_for_type_strike = sorted(df[(df["type"].str.lower()==series_type) & (df["strike"]==series_strike)]["expiration"].dropna().unique().tolist())
    series_exp = st.selectbox("Serie Expiratie", options=exps_for_type_strike if exps_for_type_strike else exps, index=0 if exps_for_type_strike else 0)
with colS4:
    series_price_col = st.radio("Prijsbron", ["last_price","mid_price"], index=0, horizontal=True)

# data van de gekozen serie
serie = df[(df["type"].str.lower()==series_type) & (df["strike"]==series_strike) & (df["expiration"]==series_exp)].copy()
serie = serie.sort_values("snapshot_date")

if serie.empty:
    st.info("Geen ticks voor deze combinatie binnen de huidige filters.")
else:
    fig_ser = make_subplots(specs=[[{"secondary_y": True}]])
    fig_ser.add_trace(go.Scatter(x=serie["snapshot_date"], y=serie[series_price_col], name="Price", mode="lines+markers"), secondary_y=False)
    fig_ser.add_trace(go.Scatter(x=serie["snapshot_date"], y=serie["ppd"], name="PPD", mode="lines"), secondary_y=True)
    # underlying overlay
    fig_ser.add_trace(go.Scatter(x=serie["snapshot_date"], y=serie["underlying_price"], name="SP500", mode="lines", line=dict(dash="dot")), secondary_y=False)
    fig_ser.update_layout(
        title=f"Ontwikkeling — {series_type.upper()} {series_strike} exp {series_exp}",
        height=430, hovermode="x unified"
    )
    fig_ser.update_xaxes(title_text="Snapshot")
    fig_ser.update_yaxes(title_text="Price / SP500", secondary_y=False)
    fig_ser.update_yaxes(title_text="PPD", secondary_y=True)
    st.plotly_chart(fig_ser, use_container_width=True)

# ────────────────────────────────────────────────────────────────────────────────
# B) PPD vs DTE (ATM of rond gekozen strike)
# ────────────────────────────────────────────────────────────────────────────────
st.subheader("PPD vs DTE — opbouw van premium per dag")
mode_col, atm_col, win_col = st.columns([1.2, 1, 1])
with mode_col:
    ppd_mode = st.radio("Bereik", ["ATM-band (moneyness)", "Rond gekozen strike"], horizontal=False, index=0)
with atm_col:
    atm_band = st.slider("ATM-band (± moneyness %)", 0.01, 0.10, 0.02, step=0.01)
with win_col:
    strike_window = st.slider("Strike-venster rond gekozen strike (punten)", 10, 200, 50, step=10)

if ppd_mode.startswith("ATM"):
    df_ppd = df[np.abs(df["moneyness"]) <= atm_band].copy()
else:
    df_ppd = df[(df["strike"] >= series_strike - strike_window) & (df["strike"] <= series_strike + strike_window) &
                (df["type"].str.lower()==series_type)].copy()

ppd_curve = (df_ppd.groupby("days_to_exp", as_index=False)["ppd"].mean().sort_values("days_to_exp"))
fig_ppd_dte = go.Figure(go.Scatter(x=ppd_curve["days_to_exp"], y=ppd_curve["ppd"], mode="lines+markers"))
fig_ppd_dte.update_layout(
    title="PPD vs Days To Expiration",
    xaxis_title="Days to Expiration",
    yaxis_title="Gemiddelde PPD",
    height=400
)
st.plotly_chart(fig_ppd_dte, use_container_width=True)

with st.expander("Uitleg"):
    st.write(
        "- **Serie-selectie** volgt exact één contract (type × strike × expiratie) door de tijd en toont Price, PPD en de SP500.\n"
        "- **PPD vs DTE** toont hoe de *gemiddelde* PPD oploopt naarmate de resterende looptijd groter is. Kies ATM (±moneyness) of rond de gekozen strike."
    )

st.markdown("---")

# ────────────────────────────────────────────────────────────────────────────────
# Overige visualisaties (ongewijzigd)
# ────────────────────────────────────────────────────────────────────────────────
term = df.groupby(["days_to_exp", "type"], as_index=False)["implied_volatility"].mean().sort_values("days_to_exp")
fig_term = go.Figure()
for t in sorted(term["type"].unique()):
    sub = term[term["type"] == t]
    fig_term.add_trace(go.Scatter(x=sub["days_to_exp"], y=sub["implied_volatility"], mode="lines+markers", name=f"IV {t.upper()}"))
fig_term.update_layout(title="Term Structure — Gemiddelde IV", xaxis_title="DTE", yaxis_title="Implied Volatility", height=420)
st.plotly_chart(fig_term, use_container_width=True)

agg_cp = df.groupby("type", as_index=False)[["volume", "open_interest"]].sum().sort_values("type")
fig_cp = make_subplots(rows=1, cols=2, subplot_titles=("Volume", "Open Interest"))
fig_cp.add_trace(go.Bar(x=agg_cp["type"].str.upper(), y=agg_cp["volume"]), row=1, col=1)
fig_cp.add_trace(go.Bar(x=agg_cp["type"].str.upper(), y=agg_cp["open_interest"]), row=1, col=2)
fig_cp.update_layout(height=420, title_text="Calls vs Puts — Volume & Open Interest", showlegend=False)
st.plotly_chart(fig_cp, use_container_width=True)

if selected_exps:
    for e in selected_exps[:5]:
        sub = df[df["expiration"] == e].groupby("strike", as_index=False)["open_interest"].sum().sort_values("strike")
        if not sub.empty:
            fig = go.Figure(go.Bar(x=sub["strike"], y=sub["open_interest"]))
            fig.update_layout(title=f"Open Interest per Strike — Expiry {e}", xaxis_title="Strike", yaxis_title="Open Interest", height=380)
            st.plotly_chart(fig, use_container_width=True)

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

# PPD distributie & tijdreeks
colL, colR = st.columns(2)
with colL:
    fig_ppd_hist = go.Figure(go.Histogram(x=df["ppd"].dropna(), nbinsx=60))
    fig_ppd_hist.update_layout(title="PPD Distributie", xaxis_title="PPD", yaxis_title="Aantal", height=420)
    st.plotly_chart(fig_ppd_hist, use_container_width=True)
with colR:
    ts_ppd = (df.assign(snap_date=df["snapshot_date"].dt.date)
                .groupby("snap_date", as_index=False)["ppd"].mean()
                .rename(columns={"snap_date":"date"}))
    if ts_ppd.empty:
        st.info("Geen PPD-tijdreeks voor de huidige filters.")
    else:
        fig_ppd_ts = go.Figure(go.Scatter(x=ts_ppd["date"], y=ts_ppd["ppd"], mode="lines"))
        fig_ppd_ts.update_layout(title="Gemiddelde PPD per dag", xaxis_title="Snapshot date", yaxis_title="PPD", height=420)
        st.plotly_chart(fig_ppd_ts, use_container_width=True)

# VIX vs IV
vix_vs_iv = (df.assign(snap_date=df["snapshot_date"].dt.date)
               .groupby("snap_date", as_index=False)
               .agg(vix=("vix","mean"), iv=("implied_volatility","mean"))
               .rename(columns={"snap_date":"date"}))
if not vix_vs_iv.empty:
    fig_vix = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, subplot_titles=("VIX", "Gemiddelde IV"))
    fig_vix.add_trace(go.Scatter(x=vix_vs_iv["date"], y=vix_vs_iv["vix"], mode="lines", name="VIX"), row=1, col=1)
    fig_vix.add_trace(go.Scatter(x=vix_vs_iv["date"], y=vix_vs_iv["iv"], mode="lines", name="IV"), row=2, col=1)
    fig_vix.update_layout(height=520, title_text="VIX vs Gemiddelde Implied Volatility")
    st.plotly_chart(fig_vix, use_container_width=True)

# Top contracts
metric_top = st.radio("Top contracts sorteer op", ["volume", "open_interest"], horizontal=True, index=0)
cols_view = ["snapshot_date","contract_symbol","type","expiration","days_to_exp","strike","underlying_price",
             "implied_volatility","last_price","bid","ask","mid_price","volume","open_interest","ppd"]
tbl = df[cols_view].sort_values(metric_top, ascending=False).head(200)
st.dataframe(tbl, use_container_width=True, hide_index=True)
