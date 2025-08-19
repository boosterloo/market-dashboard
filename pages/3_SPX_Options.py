# pages/3_SPX_Options.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly import colors as pc

from google.cloud import bigquery
from google.oauth2 import service_account

# ── BigQuery client via st.secrets ──────────────────────────────────────────────
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
        e = value[0]
        if isinstance(e, int):   return bigquery.ArrayQueryParameter(name, "INT64", list(value))
        if isinstance(e, float): return bigquery.ArrayQueryParameter(name, "FLOAT64", list(value))
        if isinstance(e, (date, pd.Timestamp, datetime)):
            return bigquery.ArrayQueryParameter(name, "DATE", [str(pd.to_datetime(v).date()) for v in value])
        return bigquery.ArrayQueryParameter(name, "STRING", [str(v) for v in value])
    if isinstance(value, bool):                 return bigquery.ScalarQueryParameter(name, "BOOL", value)
    if isinstance(value, (int, np.integer)):    return bigquery.ScalarQueryParameter(name, "INT64", int(value))
    if isinstance(value, (float, np.floating)): return bigquery.ScalarQueryParameter(name, "FLOAT64", float(value))
    if isinstance(value, datetime):             return bigquery.ScalarQueryParameter(name, "TIMESTAMP", value)
    if isinstance(value, (date, pd.Timestamp)): return bigquery.ScalarQueryParameter(name, "DATE", str(pd.to_datetime(value).date()))
    return bigquery.ScalarQueryParameter(name, "STRING", str(value))

def run_query(sql: str, params: dict | None = None) -> pd.DataFrame:
    job_config = None
    if params:
        job_config = bigquery.QueryJobConfig(query_parameters=[_bq_param(k, v) for k, v in params.items()])
    return _bq_client.query(sql, job_config=job_config).to_dataframe()

# ── Page ───────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="SPX Options Dashboard", layout="wide")
st.title("🧩 SPX Options Dashboard")
VIEW = "marketdata.spx_options_enriched_v"

# ── Basisfilters ────────────────────────────────────────────────────────────────
@st.cache_data(ttl=600, show_spinner=False)
def load_date_bounds():
    df = run_query(f"""
        SELECT MIN(CAST(snapshot_date AS DATE)) min_date,
               MAX(CAST(snapshot_date AS DATE)) max_date
        FROM `{VIEW}`
    """)
    return df["min_date"].iloc[0], df["max_date"].iloc[0]

min_date, max_date = load_date_bounds()
default_start = max(min_date, max_date - timedelta(days=365))

colA, colB, colC, colD, colE = st.columns([1.2, 0.8, 1, 1, 1])
with colA:
    start_date, end_date = st.date_input(
        "Periode (snapshot_date)",
        value=(default_start, max_date),
        min_value=min_date, max_value=max_date, format="YYYY-MM-DD"
    )
with colB:
    sel_type = st.radio("Type", ["call", "put"], index=1, horizontal=True)
with colC:
    dte_range = st.slider("Days to Expiration (DTE)", 0, 365, (0, 60), step=1)
with colD:
    mny_range = st.slider("Moneyness (strike/underlying − 1)", -0.20, 0.20, (-0.10, 0.10), step=0.01)
with colE:
    show_underlying = st.toggle("Toon S&P500 overlay", value=True)

@st.cache_data(ttl=600, show_spinner=False)
def load_expirations(start_date: date, end_date: date, sel_type: str):
    df = run_query(f"""
        SELECT DISTINCT expiration
        FROM `{VIEW}`
        WHERE DATE(snapshot_date) BETWEEN @start AND @end
          AND LOWER(type) = @t
        ORDER BY expiration
    """, {"start": start_date, "end": end_date, "t": sel_type})
    return sorted(pd.to_datetime(df["expiration"]).dt.date.unique())

exps = load_expirations(start_date, end_date, sel_type)
exp_default = [x for x in exps[:5]] if len(exps) > 0 else []
selected_exps = st.multiselect("Expiraties (voor OI/strike & matrix)", exps, default=exp_default)

# ── Data laden ─────────────────────────────────────────────────────────────────
@st.cache_data(ttl=600, show_spinner=True)
def load_filtered(start_date, end_date, sel_type, dte_min, dte_max, mny_min, mny_max):
    sql = f"""
    WITH base AS (
      SELECT
        snapshot_date, contract_symbol, type, expiration, days_to_exp,
        strike, underlying_price,
        SAFE_DIVIDE(CAST(strike AS FLOAT64), NULLIF(underlying_price, 0)) - 1.0 AS moneyness,
        (CAST(strike AS FLOAT64) - CAST(underlying_price AS FLOAT64)) AS dist_points,
        in_the_money, last_price, bid, ask, mid_price,
        implied_volatility, open_interest, volume, vix, ppd
      FROM `{VIEW}`
      WHERE DATE(snapshot_date) BETWEEN @start AND @end
        AND LOWER(type) = @t
        AND days_to_exp BETWEEN @dte_min AND @dte_max
        AND SAFE_DIVIDE(CAST(strike AS FLOAT64), NULLIF(underlying_price, 0)) - 1.0 BETWEEN @mny_min AND @mny_max
    )
    SELECT * FROM base
    """
    params = {
        "start": start_date, "end": end_date,
        "t": sel_type,
        "dte_min": int(dte_min), "dte_max": int(dte_max),
        "mny_min": float(mny_min), "mny_max": float(mny_max),
    }
    df = run_query(sql, params=params)
    if not df.empty:
        df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
        df["expiration"]    = pd.to_datetime(df["expiration"]).dt.date
        for c in ["days_to_exp","implied_volatility","open_interest","volume","ppd",
                  "strike","underlying_price","last_price","mid_price","bid","ask","dist_points"]:
            if c in df: df[c] = pd.to_numeric(df[c], errors="coerce")
        df["moneyness_pct"] = 100 * df["moneyness"]
        df["abs_dist_pct"]  = (np.abs(df["dist_points"]) / df["underlying_price"]) * 100.0
    return df

df = load_filtered(start_date, end_date, sel_type, dte_range[0], dte_range[1], mny_range[0], mny_range[1])
if df.empty:
    st.warning("Geen data voor de huidige filters.")
    st.stop()

# ── KPI's ──────────────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
with col1: st.metric("Records", f"{len(df):,}")
with col2: st.metric("Gem. IV", f"{df['implied_volatility'].mean():.2%}")
with col3: st.metric("Som Volume", f"{int(df['volume'].sum()):,}")
with col4: st.metric("Som Open Interest", f"{int(df['open_interest'].sum()):,}")
st.markdown("---")

# ── Outlier-instellingen ───────────────────────────────────────────────────────
st.caption("Outliers kunnen de schaal verstoren. Kies een methode.")
col_out1, col_out2 = st.columns([1.2, 1])
with col_out1:
    outlier_mode = st.radio("Outlier-methode", ["Geen", "Percentiel clip", "IQR filter", "Z‑score filter"],
                            horizontal=True, index=1)
with col_out2:
    pct_clip = st.slider("Percentiel clip (links/rechts)", 0, 10, 1, step=1,
                         help="Alleen gebruikt bij 'Percentiel clip'")

def apply_outlier(series: pd.Series, mode: str, pct: int) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").astype(float)
    if mode == "Geen":
        return s
    if mode == "Percentiel clip":
        if s.notna().any():
            lo, hi = np.nanpercentile(s, [pct, 100 - pct])
            return s.clip(lower=lo, upper=hi)
        return s
    if mode == "IQR filter":
        if s.notna().any():
            q1, q3 = np.nanpercentile(s, [25, 75])
            iqr = q3 - q1
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            return s.where((s >= lo) & (s <= hi), np.nan)
        return s
    if mode == "Z‑score filter":
        mu, sd = np.nanmean(s), np.nanstd(s)
        if sd == 0 or np.isnan(sd): return s
        z = (s - mu) / sd
        return s.where(np.abs(z) <= 3.0, np.nan)
    return s

# ── A) Serie-selectie — EXACT TWEE grafieken (Price↔SP500 & PPD↔SP500) ─────────
st.subheader("Serie-selectie — volg één optiereeks door de tijd")

colS1, colS2, colS3, colS4 = st.columns([1, 1, 1, 1.6])
with colS1:
    strikes = sorted(df["strike"].dropna().unique().tolist())
    series_strike = st.selectbox("Serie Strike", options=strikes,
                                 index=len(strikes)//2 if strikes else 0)
with colS2:
    exps_for_strike = sorted(df[df["strike"]==series_strike]["expiration"]
                             .dropna().unique().tolist())
    series_exp = st.selectbox("Serie Expiratie",
                              options=exps_for_strike if exps_for_strike else exps,
                              index=0)
with colS3:
    series_price_col = st.radio("Prijsbron", ["last_price","mid_price"], index=0, horizontal=True)
with colS4:
    pass

serie = df[(df["strike"]==series_strike) & (df["expiration"]==series_exp)].copy().sort_values("snapshot_date")

if serie.empty:
    st.info("Geen ticks voor deze combinatie binnen de huidige filters.")
else:
    c1, c2 = st.columns(2)

    # 1) Price ↔ SP500 (dubbele as)
    with c1:
        y_price = apply_outlier(serie[series_price_col], outlier_mode, pct_clip)
        fig_price = make_subplots(specs=[[{"secondary_y": True}]])
        fig_price.add_trace(go.Scatter(
            x=serie["snapshot_date"], y=y_price,
            name="Price", mode="lines+markers", connectgaps=True
        ), secondary_y=False)
        if show_underlying:
            fig_price.add_trace(go.Scatter(
                x=serie["snapshot_date"], y=serie["underlying_price"],
                name="SP500", mode="lines", line=dict(dash="dot"), connectgaps=True
            ), secondary_y=True)
        fig_price.update_layout(
            title=f"{sel_type.upper()} {series_strike} — exp {series_exp} | Price vs SP500",
            height=420, hovermode="x unified"
        )
        fig_price.update_xaxes(title_text="Meetmoment")
        fig_price.update_yaxes(title_text="Price", secondary_y=False, rangemode="tozero")
        fig_price.update_yaxes(title_text="SP500", secondary_y=True)
        st.plotly_chart(fig_price, use_container_width=True)

    # 2) PPD ↔ SP500 (dubbele as)
    with c2:
        y_ppd = apply_outlier(serie["ppd"], outlier_mode, pct_clip)
        fig_ppd = make_subplots(specs=[[{"secondary_y": True}]])
        fig_ppd.add_trace(go.Scatter(
            x=serie["snapshot_date"], y=y_ppd,
            name="PPD", mode="lines+markers", connectgaps=True
        ), secondary_y=False)
        if show_underlying:
            fig_ppd.add_trace(go.Scatter(
                x=serie["snapshot_date"], y=serie["underlying_price"],
                name="SP500", mode="lines", line=dict(dash="dot"), connectgaps=True
            ), secondary_y=True)
        fig_ppd.update_layout(
            title=f"{sel_type.upper()} {series_strike} — exp {series_exp} | PPD vs SP500",
            height=420, hovermode="x unified"
        )
        fig_ppd.update_xaxes(title_text="Meetmoment")
        fig_ppd.update_yaxes(title_text="PPD", secondary_y=False, rangemode="tozero")
        fig_ppd.update_yaxes(title_text="SP500", secondary_y=True)
        st.plotly_chart(fig_ppd, use_container_width=True)

# ── B) PPD & Afstand tot Uitoefenprijs ─────────────────────────────────────────
st.subheader("PPD & Afstand tot Uitoefenprijs (ATM→OTM/ITM)")
snapshots = sorted(df["snapshot_date"].dt.floor("min").unique())
default_idx = len(snapshots) - 1 if snapshots else 0
sel_snapshot = st.selectbox(
    "Peildatum (snapshot)",
    options=snapshots, index=default_idx,
    format_func=lambda x: pd.to_datetime(x).strftime("%Y-%m-%d %H:%M")
)
df_last = df[df["snapshot_date"] == sel_snapshot].copy()
df_last["abs_dist_pct"] = ((df_last["dist_points"].abs() / df_last["underlying_price"]) * 100.0).round(2)

ppd_vs_dist = (
    df_last.assign(ppd_f=apply_outlier(df_last["ppd"], outlier_mode, pct_clip))
           .groupby("abs_dist_pct", as_index=False)["ppd_f"].mean()
           .rename(columns={"ppd_f": "ppd"})
           .sort_values("abs_dist_pct")
)

fig_ppd_dist = go.Figure(go.Scatter(
    x=ppd_vs_dist["abs_dist_pct"], y=ppd_vs_dist["ppd"],
    mode="lines+markers", name="PPD", connectgaps=True
))
fig_ppd_dist.update_layout(
    title=f"PPD vs Afstand tot Strike — {pd.to_datetime(sel_snapshot).strftime('%Y-%m-%d %H:%M')}",
    xaxis_title="Afstand tot strike (|strike − underlying| / underlying, %)",
    yaxis_title="PPD",
    height=400
)
st.plotly_chart(fig_ppd_dist, use_container_width=True)

# ── C) Ontwikkeling Prijs per Expiratiedatum (peildatum) ───────────────────────
st.subheader("Ontwikkeling Prijs per Expiratiedatum (laatste snapshot)")
exp_curve_raw = (
    df_last[df_last["strike"] == series_strike]
      .groupby("expiration", as_index=False)
      .agg(price=("last_price", "mean"), ppd=("ppd", "mean"))
      .sort_values("expiration")
)
exp_curve = exp_curve_raw.copy()
exp_curve["price_f"] = apply_outlier(exp_curve["price"], outlier_mode, pct_clip)
exp_curve["ppd_f"]   = apply_outlier(exp_curve["ppd"], outlier_mode, pct_clip)

fig_exp = make_subplots(specs=[[{"secondary_y": True}]])
fig_exp.add_trace(go.Scatter(
    x=exp_curve["expiration"], y=exp_curve["price_f"],
    name="Price", mode="lines+markers", connectgaps=True
), secondary_y=False)
fig_exp.add_trace(go.Scatter(
    x=exp_curve["expiration"], y=exp_curve["ppd_f"],
    name="PPD", mode="lines+markers", connectgaps=True
), secondary_y=True)
fig_exp.update_layout(
    title=f"{sel_type.upper()} — Strike {series_strike} — peildatum {pd.to_datetime(sel_snapshot).strftime('%Y-%m-%d %H:%M')}",
    height=420, hovermode="x unified"
)
fig_exp.update_xaxes(title_text="Expiratiedatum")
fig_exp.update_yaxes(title_text="Price", secondary_y=False, rangemode="tozero")
fig_exp.update_yaxes(title_text="PPD",   secondary_y=True,  rangemode="tozero")
st.plotly_chart(fig_exp, use_container_width=True)

# ── D) PPD vs DTE ──────────────────────────────────────────────────────────────
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
    df_ppd = df[(df["strike"] >= series_strike - strike_window) &
                (df["strike"] <= series_strike + strike_window)].copy()

ppd_curve = (
    df_ppd.assign(ppd_f=apply_outlier(df_ppd["ppd"], outlier_mode, pct_clip))
         .groupby("days_to_exp", as_index=False)["ppd_f"].mean()
         .rename(columns={"ppd_f": "ppd"})
         .sort_values("days_to_exp")
)

fig_ppd_dte = go.Figure(go.Scatter(
    x=ppd_curve["days_to_exp"], y=ppd_curve["ppd"],
    mode="lines+markers", connectgaps=True
))
fig_ppd_dte.update_layout(
    title="PPD vs Days To Expiration",
    xaxis_title="Days to Expiration", yaxis_title="Gemiddelde PPD", height=400
)
st.plotly_chart(fig_ppd_dte, use_container_width=True)

st.markdown("---")

# ── E) Matrix — meetmoment × strike ────────────────────────────────────────────
st.subheader("Matrix — meetmoment × strike")
colM1, colM2, colM3, colM4 = st.columns([1, 1, 1, 1])
with colM1:
    matrix_exp = st.selectbox("Expiratie (matrix)", options=sorted(df["expiration"].unique().tolist()), index=0, key="mx_exp")
with colM2:
    matrix_metric = st.radio("Waarde", ["last_price", "mid_price", "ppd"], horizontal=False, index=0, key="mx_metric")
with colM3:
    max_rows = st.slider("Max. meetmomenten (recentste)", 50, 500, 200, step=50, key="mx_rows")
with colM4:
    pass

mx = df[df["expiration"]==matrix_exp].copy().sort_values("snapshot_date").tail(max_rows)
if mx.empty:
    st.info("Geen matrix-data voor de gekozen expiratie.")
else:
    mx["snap_s"] = mx["snapshot_date"].dt.strftime("%Y-%m-%d %H:%M")
    pivot = mx.pivot_table(index="snap_s", columns="strike", values=matrix_metric, aggfunc="mean")
    pivot = pivot.sort_index(ascending=False).round(2)

    arr_raw = pivot.values.astype(float)
    arr_clip = apply_outlier(pd.Series(arr_raw.flatten()), outlier_mode, pct_clip).values.reshape(arr_raw.shape)

    tab_hm, tab_tbl = st.tabs(["Heatmap", "Tabel (met kleur)"])

    with tab_hm:
        z = arr_clip
        x = pivot.columns.astype(float)
        y = pivot.index.tolist()
        fig_mx = go.Figure(data=go.Heatmap(
            z=z, x=x, y=y, colorbar_title=matrix_metric.capitalize(),
            hovertemplate="Snapshot: %{y}<br>Strike: %{x}<br>Value: %{z:.2f}<extra></extra>"
        ))
        fig_mx.update_layout(
            title=f"Heatmap — {sel_type.upper()} exp {matrix_exp} — {matrix_metric}",
            xaxis_title="Strike", yaxis_title="Meetmoment", height=520
        )
        st.plotly_chart(fig_mx, use_container_width=True)

    with tab_tbl:
        scale = "Blues" if matrix_metric in ("last_price", "mid_price") else "Oranges"
        vmin = np.nanmin(arr_clip) if np.isfinite(arr_clip).any() else 0.0
        vmax = np.nanmax(arr_clip) if np.isfinite(arr_clip).any() else 1.0
        denom = (vmax - vmin) if vmax != vmin else 1.0
        norm = (np.nan_to_num(arr_clip, nan=vmin) - vmin) / denom

        cell_colors = []
        for col_idx in range(norm.shape[1]):
            col_vals = norm[:, col_idx]
            col_colors = [pc.sample_colorscale(scale, float(v)) for v in col_vals]
            cell_colors.append(col_colors)

        header_vals = ["Snapshot"] + [str(c) for c in pivot.columns.tolist()]
        cell_vals = [pivot.index.tolist()] + [pivot[c].tolist() for c in pivot.columns.tolist()]
        header_color = pc.sample_colorscale(scale, 0.6)

        fig_tbl = go.Figure(data=[go.Table(
            header=dict(values=header_vals, fill_color=header_color, font=dict(color="white"), align="center"),
            cells=dict(values=cell_vals,
                       fill_color=[["white"]*len(pivot)] + cell_colors,
                       align="right", format=[None]+[".2f"]*len(pivot.columns))
        )])
        fig_tbl.update_layout(title=f"Tabel — {sel_type.upper()} exp {matrix_exp} — {matrix_metric}", height=520)
        st.plotly_chart(fig_tbl, use_container_width=True)

st.markdown("---")

# ── F) Overige visualisaties ───────────────────────────────────────────────────
term = df.groupby("days_to_exp", as_index=False)["implied_volatility"].mean().sort_values("days_to_exp")
fig_term = go.Figure(go.Scatter(x=term["days_to_exp"], y=term["implied_volatility"],
                                mode="lines+markers", name=f"IV {sel_type.upper()}"))
fig_term.update_layout(title="Term Structure — Gemiddelde IV", xaxis_title="DTE",
                       yaxis_title="Implied Volatility", height=420)
st.plotly_chart(fig_term, use_container_width=True)

if selected_exps:
    for e in selected_exps[:5]:
        sub = df[df["expiration"] == e].groupby("strike", as_index=False)["open_interest"].sum().sort_values("strike")
        if not sub.empty:
            fig = go.Figure(go.Bar(x=sub["strike"], y=sub["open_interest"]))
            fig.update_layout(title=f"Open Interest per Strike — {sel_type.upper()} exp {e}",
                              xaxis_title="Strike", yaxis_title="Open Interest", height=380)
            st.plotly_chart(fig, use_container_width=True)

vix_vs_iv = (df.assign(snap_date=df["snapshot_date"].dt.date)
               .groupby("snap_date", as_index=False)
               .agg(vix=("vix","mean"), iv=("implied_volatility","mean"))
               .rename(columns={"snap_date":"date"}))
if not vix_vs_iv.empty:
    fig_vix = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                            subplot_titles=("VIX", "Gemiddelde IV"))
    fig_vix.add_trace(go.Scatter(x=vix_vs_iv["date"], y=vix_vs_iv["vix"], mode="lines", name="VIX"), row=1, col=1)
    fig_vix.add_trace(go.Scatter(x=vix_vs_iv["date"], y=vix_vs_iv["iv"], mode="lines", name="IV"), row=2, col=1)
    fig_vix.update_layout(height=520, title_text=f"VIX vs IV ({sel_type.upper()})")
    st.plotly_chart(fig_vix, use_container_width=True)
