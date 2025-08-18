# pages/3_SPX_Options.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ⇩ Gebruik dezelfde utils als op je andere pagina’s
try:
    from utils import run_query  # verwacht signatuur: run_query(sql: str, params: dict | None = None) -> pd.DataFrame
except Exception:
    # Fallback: rechtstreekse BigQuery client (alleen gebruiken als je geen utils hebt)
    from google.cloud import bigquery
    _bq_client = bigquery.Client()
    def run_query(sql: str, params: dict | None = None) -> pd.DataFrame:
        job_config = None
        if params:
            job_config = bigquery.QueryJobConfig(query_parameters=[
                bigquery.ScalarQueryParameter(k, "STRING" if isinstance(v, str) else
                                                  "INT64" if isinstance(v, int) else
                                                  "FLOAT64" if isinstance(v, float) else
                                                  "TIMESTAMP", v)
                for k, v in params.items()
            ])
        return _bq_client.query(sql, job_config=job_config).to_dataframe()

st.set_page_config(page_title="SPX Options Dashboard", layout="wide")
st.title("🧩 SPX Options Dashboard")

# ────────────────────────────────────────────────────────────────────────────────
# Data ophalen
# ────────────────────────────────────────────────────────────────────────────────
VIEW = "marketdata.spx_options_enriched_v"

@st.cache_data(ttl=600, show_spinner=False)
def load_date_bounds():
    sql = f"""
    SELECT
      MIN(CAST(snapshot_date AS DATE)) AS min_date,
      MAX(CAST(snapshot_date AS DATE)) AS max_date
    FROM `{VIEW}`
    """
    df = run_query(sql)
    mn = df["min_date"].iloc[0]
    mx = df["max_date"].iloc[0]
    if pd.isna(mn) or pd.isna(mx):
        today = date.today()
        return today - timedelta(days=30), today
    return mn, mx

min_date, max_date = load_date_bounds()

# Default laatste 1 jaar
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
    if isinstance(daterange, tuple):
        start_date, end_date = daterange
    else:
        start_date, end_date = default_start, max_date

with colB:
    opt_types = st.multiselect("Type", options=["call", "put"], default=["call", "put"])

with colC:
    dte_range = st.slider("Days to Expiration (DTE)", 0, 365, (0, 60), step=1)

with colD:
    moneyness_range = st.slider("Moneyness (strike/underlying − 1)", -0.20, 0.20, (-0.10, 0.10), step=0.01)

# Keuze van expiratie (neem top dichtstbijzijnde 10)
@st.cache_data(ttl=600, show_spinner=False)
def load_expirations(start_date: str, end_date: str):
    sql = f"""
    SELECT DISTINCT expiration
    FROM `{VIEW}`
    WHERE DATE(snapshot_date) BETWEEN @start AND @end
    ORDER BY expiration
    """
    df = run_query(sql, params={"start": str(start_date), "end": str(end_date)})
    return sorted(pd.to_datetime(df["expiration"]).dt.date.unique())

exps = load_expirations(start_date, end_date)
exp_default = [x for x in exps[:5]] if len(exps) > 0 else []
selected_exps = st.multiselect("Expiraties (optioneel, voor strike- en OI-grafiek)", exps, default=exp_default)

# ────────────────────────────────────────────────────────────────────────────────
# Query gefilterde dataset
# ────────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=600, show_spinner=True)
def load_filtered(start_date, end_date, types, dte_min, dte_max, mny_min, mny_max):
    # We berekenen moneyness in SQL voor performance
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
        "start": str(start_date),
        "end": str(end_date),
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

df = load_filtered(start_date, end_date, opt_types, dte_range[0], dte_range[1], moneyness_range[0], moneyness_range[1])

if df.empty:
    st.warning("Geen data voor de huidige filters.")
    st.stop()

# ────────────────────────────────────────────────────────────────────────────────
# KPI’s
# ────────────────────────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Records", f"{len(df):,}")
with col2:
    st.metric("Gem. IV", f"{df['implied_volatility'].mean():.2%}")
with col3:
    st.metric("Som Volume", f"{int(df['volume'].sum()):,}")
with col4:
    st.metric("Som Open Interest", f"{int(df['open_interest'].sum()):,}")

st.markdown("---")

# ────────────────────────────────────────────────────────────────────────────────
# 1) Term structure IV (gemiddeld per DTE en type)
# ────────────────────────────────────────────────────────────────────────────────
term = (
    df.groupby(["days_to_exp", "type"], as_index=False)["implied_volatility"]
    .mean()
    .sort_values("days_to_exp")
)

fig_term = go.Figure()
for t in sorted(term["type"].unique()):
    sub = term[term["type"] == t]
    fig_term.add_trace(go.Scatter(
        x=sub["days_to_exp"], y=sub["implied_volatility"],
        mode="lines+markers", name=f"IV {t.upper()}"
    ))
fig_term.update_layout(
    title="Term Structure — Gemiddelde Implied Volatility per DTE",
    xaxis_title="Days to Expiration",
    yaxis_title="Implied Volatility",
    hovermode="x unified",
    height=420
)
st.plotly_chart(fig_term, use_container_width=True)

with st.expander("Toelichting — Term Structure"):
    st.write(
        "- Stijgende IV bij kortere DTE kan duiden op nabijere event-risico’s (earnings, CPI, FED).\n"
        "- Vergelijking met VIX (lager op deze pagina) helpt om relatieve over/undervaluation te spotten."
    )

# ────────────────────────────────────────────────────────────────────────────────
# 2) Calls vs Puts — volume en OI (stacked)
# ────────────────────────────────────────────────────────────────────────────────
agg_cp = (
    df.groupby("type", as_index=False)[["volume", "open_interest"]]
    .sum()
    .sort_values("type")
)

fig_cp = make_subplots(rows=1, cols=2, subplot_titles=("Volume", "Open Interest"))
fig_cp.add_trace(go.Bar(x=agg_cp["type"].str.upper(), y=agg_cp["volume"], name="Volume"), row=1, col=1)
fig_cp.add_trace(go.Bar(x=agg_cp["type"].str.upper(), y=agg_cp["open_interest"], name="Open Interest"), row=1, col=2)
fig_cp.update_layout(height=420, title_text="Calls vs Puts — Volume & Open Interest", showlegend=False)
st.plotly_chart(fig_cp, use_container_width=True)

# ────────────────────────────────────────────────────────────────────────────────
# 3) OI per strike voor geselecteerde expiraties
# ────────────────────────────────────────────────────────────────────────────────
if selected_exps:
    tab_oi, tab_vol = st.tabs(["Open Interest per Strike (expiraties)", "Volume per Strike (expiraties)"])
    with tab_oi:
        for e in selected_exps[:10]:
            sub = df[df["expiration"] == e].groupby("strike", as_index=False)["open_interest"].sum().sort_values("strike")
            if sub.empty:
                continue
            fig = go.Figure(go.Bar(x=sub["strike"], y=sub["open_interest"], name=str(e)))
            fig.update_layout(title=f"Open Interest per Strike — Expiry {e}", xaxis_title="Strike", yaxis_title="Open Interest", height=380)
            st.plotly_chart(fig, use_container_width=True)
    with tab_vol:
        for e in selected_exps[:10]:
            sub = df[df["expiration"] == e].groupby("strike", as_index=False)["volume"].sum().sort_values("strike")
            if sub.empty:
                continue
            fig = go.Figure(go.Bar(x=sub["strike"], y=sub["volume"], name=str(e)))
            fig.update_layout(title=f"Volume per Strike — Expiry {e}", xaxis_title="Strike", yaxis_title="Volume", height=380)
            st.plotly_chart(fig, use_container_width=True)

# ────────────────────────────────────────────────────────────────────────────────
# 4) Heatmap: Volume (of OI) over DTE × Strike (gebinned)
# ────────────────────────────────────────────────────────────────────────────────
with st.expander("Instellingen Heatmap"):
    metric_choice = st.radio("Metric", ["volume", "open_interest"], horizontal=True, index=0)
    bins = st.slider("Aantal strike-bins", min_value=20, max_value=120, value=50, step=5)

# Strike binner (voorkomt zware pivot)
q_low, q_hi = df["strike"].quantile([0.02, 0.98])
strike_bins = np.linspace(q_low, q_hi, bins+1)
labels = 0.5 * (strike_bins[:-1] + strike_bins[1:])
df_hm = df[(df["strike"] >= q_low) & (df["strike"] <= q_hi)].copy()
df_hm["strike_bin"] = pd.cut(df_hm["strike"], bins=strike_bins, labels=np.round(labels, 1), include_lowest=True)

pivot = df_hm.pivot_table(index="days_to_exp", columns="strike_bin", values=metric_choice, aggfunc="sum", fill_value=0)
fig_hm = go.Figure(data=go.Heatmap(
    z=pivot.values,
    x=[float(x) for x in pivot.columns.astype(float)],
    y=pivot.index,
    colorbar_title=metric_choice
))
fig_hm.update_layout(
    title=f"Heatmap — {metric_choice.capitalize()} over DTE × Strike (gebinned)",
    xaxis_title="Strike (bin)",
    yaxis_title="Days to Expiration",
    height=520
)
st.plotly_chart(fig_hm, use_container_width=True)

with st.expander("Toelichting — Heatmap"):
    st.write(
        "- Donkere cellen markeren concentraties in volume/OI op specifieke strikes en DTE.\n"
        "- Handig om 'magneten' richting expiry te spotten."
    )

# ────────────────────────────────────────────────────────────────────────────────
# 5) PPD distributie + tijdreeks
# ────────────────────────────────────────────────────────────────────────────────
colL, colR = st.columns(2)
with colL:
    fig_ppd_hist = go.Figure(go.Histogram(x=df["ppd"].dropna(), nbinsx=60))
    fig_ppd_hist.update_layout(title="PPD Distributie", xaxis_title="PPD", yaxis_title="Count", height=420)
    st.plotly_chart(fig_ppd_hist, use_container_width=True)

with colR:
    ts_ppd = df.groupby(pd.to_datetime(df["snapshot_date"]).dt.date, as_index=False)["ppd"].mean()
    fig_ppd_ts = go.Figure(go.Scatter(x=ts_ppd["snapshot_date"], y=ts_ppd["ppd"], mode="lines"))
    fig_ppd_ts.update_layout(title="Gemiddelde PPD per dag", xaxis_title="Snapshot date", yaxis_title="PPD", height=420)
    st.plotly_chart(fig_ppd_ts, use_container_width=True)

with st.expander("Toelichting — PPD"):
    st.write(
        "- PPD (premium per dag) helpt om relatieve 'carry' te vergelijken tussen maturities/strikes.\n"
        "- Een stijgende PPD kan komen door hogere IV of grotere extrinsieke waarde."
    )

# ────────────────────────────────────────────────────────────────────────────────
# 6) VIX vs Gem. IV
# ────────────────────────────────────────────────────────────────────────────────
vix_vs_iv = (
    df.groupby(pd.to_datetime(df["snapshot_date"]).dt.date, as_index=False)
      .agg(vix=("vix", "mean"), iv=("implied_volatility", "mean"))
      .dropna()
      .sort_values("snapshot_date")
)

fig_vix = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=("VIX", "Gemiddelde IV"))
fig_vix.add_trace(go.Scatter(x=vix_vs_iv["snapshot_date"], y=vix_vs_iv["vix"], mode="lines", name="VIX"), row=1, col=1)
fig_vix.add_trace(go.Scatter(x=vix_vs_iv["snapshot_date"], y=vix_vs_iv["iv"], mode="lines", name="IV"), row=2, col=1)
fig_vix.update_layout(height=520, title_text="VIX vs Gemiddelde Implied Volatility")
st.plotly_chart(fig_vix, use_container_width=True)

# ────────────────────────────────────────────────────────────────────────────────
# 7) Tabel — Top contracts (Volume of OI)
# ────────────────────────────────────────────────────────────────────────────────
st.subheader("Top contracts")
metric_top = st.radio("Sorteer op", ["volume", "open_interest"], horizontal=True, index=0)
cols_view = [
    "snapshot_date", "contract_symbol", "type", "expiration", "days_to_exp",
    "strike", "underlying_price", "moneyness_pct", "implied_volatility",
    "last_price", "bid", "ask", "mid_price", "volume", "open_interest", "ppd"
]
tbl = (df[cols_view]
       .dropna(subset=[metric_top])
       .sort_values(metric_top, ascending=False)
       .head(200))
st.dataframe(tbl, use_container_width=True, hide_index=True)
