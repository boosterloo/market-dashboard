# pages/EU_Country_Spreads.py — EU landen yields & spreads (levels + fragmentatie)
# Vereist: utils.bq.run_query, st.secrets["gcp_service_account"]["project_id"]
# Data: marketdata.eu_yields_daily (kolommen o.a.: date, country, tenor, value, spread_to_DE, spread_to_EA, source, snapshot_date)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from utils.bq import run_query

st.set_page_config(page_title="EU Country Yields & Spreads", layout="wide")
st.title("🇪🇺 EU Country Yields & Fragmentation")

PROJECT_ID = st.secrets["gcp_service_account"]["project_id"]
TABLES     = st.secrets.get("tables", {})

EU_TABLE = TABLES.get("eu_yields_daily", f"{PROJECT_ID}.marketdata.eu_yields_daily")

# ============== helpers ==============
def pct_fmt(x, dp=2):
    return "—" if pd.isna(x) else f"{round(float(x), dp)}%"

def to_bp(x): 
    return None if pd.isna(x) else float(x)*100.0

def zscore_series(s: pd.Series, window: int) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    mu = s.rolling(window).mean()
    sd = s.rolling(window).std()
    return (s - mu) / sd

@st.cache_data(show_spinner=False, ttl=600)
def list_countries() -> list[str]:
    sql = f"SELECT DISTINCT country FROM `{EU_TABLE}` ORDER BY country"
    return run_query(sql, timeout=20)["country"].tolist()

@st.cache_data(show_spinner=False, ttl=600)
def list_tenors() -> list[str]:
    sql = f"SELECT DISTINCT tenor FROM `{EU_TABLE}` ORDER BY tenor"
    return run_query(sql, timeout=20)["tenor"].tolist()

@st.cache_data(show_spinner=False, ttl=300)
def overall_minmax() -> tuple[pd.Timestamp, pd.Timestamp]:
    sql = f"SELECT MIN(date) AS dmin, MAX(date) AS dmax FROM `{EU_TABLE}`"
    r = run_query(sql, timeout=20).iloc[0]
    return pd.to_datetime(r["dmin"]), pd.to_datetime(r["dmax"])

@st.cache_data(show_spinner=True, ttl=300)
def load_eu(countries: list[str], tenors: list[str],
            date_from: pd.Timestamp|None, date_to: pd.Timestamp|None) -> pd.DataFrame:
    """
    Robuuste fetch zonder BigQuery ARRAY-params.
    We geven CSV strings door en bouwen arrays met SPLIT() in SQL.
    """
    if not countries:
        return pd.DataFrame(columns=["date","country","tenor","value","spread_to_DE","spread_to_EA","source","snapshot_date"])
    if not tenors:
        tenors = ["10Y"]

    countries_csv = ",".join(sorted(set(countries)))
    tenors_csv    = ",".join(sorted(set(tenors)))

    params = {"countries_csv": countries_csv, "tenors_csv": tenors_csv}
    where = [
        "country IN UNNEST(SPLIT(@countries_csv))",
        "tenor   IN UNNEST(SPLIT(@tenors_csv))",
    ]
    if date_from is not None:
        params["dmin"] = str(pd.to_datetime(date_from).date())
        where.append("date >= @dmin")
    if date_to is not None:
        params["dmax"] = str(pd.to_datetime(date_to).date())
        where.append("date <= @dmax")

    sql = f"""
    SELECT
      date,
      country,
      tenor,
      SAFE_CAST(value AS FLOAT64)        AS value,
      SAFE_CAST(spread_to_DE AS FLOAT64) AS spread_to_DE,
      SAFE_CAST(spread_to_EA AS FLOAT64) AS spread_to_EA,
      source,
      snapshot_date
    FROM `{EU_TABLE}`
    WHERE {' AND '.join(where)}
    ORDER BY date
    """
    df = run_query(sql, params=params, timeout=120)
    df["date"] = pd.to_datetime(df["date"])
    return df

# ============== data & filters ==============
with st.spinner("Land- en tenorselecties laden…"):
    ALL_CN = list_countries() or ["EA","DE","FR","IT","NL","GR"]
    ALL_TN = list_tenors() or ["2Y","5Y","10Y","30Y"]

default_countries = [c for c in ["EA","DE","FR","IT","NL","GR"] if c in ALL_CN] or ALL_CN[:6]
default_tenor = "10Y" if "10Y" in ALL_TN else (ALL_TN[0] if ALL_TN else "10Y")

c1,c2,c3,c4 = st.columns([1.6, 1, 1, 1.2])
with c1:
    sel_countries = st.multiselect("Landen", options=ALL_CN, default=default_countries)
with c2:
    sel_tenor = st.selectbox("Tenor (voor grafieken)", options=sorted(ALL_TN), index=sorted(ALL_TN).index(default_tenor))
with c3:
    mode_z = st.toggle("Z-scores (rolling)", value=False)
with c4:
    z_window = st.number_input("Z-window (dagen)", min_value=20, max_value=260, value=90, step=10)

# Periode presets
st.subheader("Periode")
dmin, dmax = overall_minmax()
def clamp(ts: pd.Timestamp) -> pd.Timestamp:
    return min(max(ts, dmin), dmax)

pr = st.radio("Range", ["3M","6M","1Y","3Y","5Y","YTD","Max","Custom"], horizontal=True, index=2)
if pr == "3M":  start_date, end_date = clamp(dmax - pd.DateOffset(months=3)), dmax
elif pr == "6M":start_date, end_date = clamp(dmax - pd.DateOffset(months=6)), dmax
elif pr == "1Y":start_date, end_date = clamp(dmax - pd.DateOffset(years=1)), dmax
elif pr == "3Y":start_date, end_date = clamp(dmax - pd.DateOffset(years=3)), dmax
elif pr == "5Y":start_date, end_date = clamp(dmax - pd.DateOffset(years=5)), dmax
elif pr == "YTD":start_date, end_date = clamp(pd.Timestamp(dmax.year,1,1)), dmax
elif pr == "Max":start_date, end_date = dmin, dmax
else:
    date_range = st.slider("Selecteer periode (Custom)",
                           min_value=dmin.date(), max_value=dmax.date(),
                           value=(clamp(dmax - pd.DateOffset(years=1)).date(), dmax.date()))
    start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])

if not sel_countries:
    st.warning("Kies minimaal één land."); st.stop()

TENORS_FOR_LOAD = sorted(set([sel_tenor, "10Y"]))  # laad altijd 10Y mee (fragmentatie)
try:
    with st.spinner("BigQuery laden…"):
        DF = load_eu(sel_countries, TENORS_FOR_LOAD, start_date, end_date)
except Exception as e:
    st.error("BigQuery-query faalde. Hieronder de exception uit je app:")
    st.exception(e)
    st.stop()

if DF.empty:
    st.info("Geen data voor deze selectie."); st.stop()

# ============== Snapshot (laatste datum) ==============
st.subheader("Snapshot — Laatste datum")
latest_date = DF["date"].max()
DFl = DF[DF["date"] == latest_date]
DFl_t = DFl[DFl["tenor"] == sel_tenor].copy()

kpi_cols = st.columns(6)
def get_val(df, country, col):
    s = df[df["country"]==country][col]
    return None if s.empty else float(s.iloc[0])

for i, cc in enumerate(sel_countries[:6]):
    with kpi_cols[i]:
        y   = get_val(DFl_t, cc, "value")
        sDE = get_val(DFl_t, cc, "spread_to_DE")
        sEA = get_val(DFl_t, cc, "spread_to_EA")
        st.metric(f"{cc} {sel_tenor}", pct_fmt(y,2),
                  help=f"Δ vs DE: {round(to_bp(sDE) or 0,1)} bp • Δ vs EA: {round(to_bp(sEA) or 0,1)} bp")

fig_bar = go.Figure()
fig_bar.add_trace(go.Bar(x=DFl_t["country"], y=DFl_t["value"], name=f"{sel_tenor} yield"))
fig_bar.update_layout(margin=dict(l=10,r=10,t=10,b=10), yaxis_title="Yield (%)", xaxis_title="Country")
st.plotly_chart(fig_bar, use_container_width=True)

# ============== Tijdreeks — Levels (yield) ==============
st.subheader(f"Tijdreeks — Levels ({sel_tenor})")
DFt = DF[DF["tenor"] == sel_tenor].copy()
if mode_z:
    DFt = DFt.sort_values("date")
    DFt["value"] = DFt.groupby("country", group_keys=False)["value"].apply(lambda s: zscore_series(s, z_window))

fig_lvl = go.Figure()
for cc in sel_countries:
    dcc = DFt[DFt["country"]==cc]
    if dcc.empty: continue
    fig_lvl.add_trace(go.Scatter(x=dcc["date"], y=dcc["value"], name=cc, mode="lines"))
fig_lvl.update_layout(margin=dict(l=10,r=10,t=10,b=10),
                      yaxis_title=("Yield (z)" if mode_z else "Yield (%)"),
                      xaxis_title="Date")
fig_lvl.update_xaxes(range=[start_date, end_date])
st.plotly_chart(fig_lvl, use_container_width=True)

# ============== Tijdreeks — Spreads vs DE / EA ==============
st.subheader(f"Tijdreeks — Spreads vs DE & EA ({sel_tenor})")
DFs = DFt.copy()
mopt = st.radio("Spread baseline", ["vs DE","vs EA"], horizontal=True, index=0)
col_sp = "spread_to_DE" if mopt=="vs DE" else "spread_to_EA"

fig_sp = go.Figure()
for cc in sel_countries:
    dcc = DFs[DFs["country"]==cc]
    if dcc.empty: continue
    y = dcc[col_sp]* (1.0 if mode_z else 100.0)
    if mode_z:
        y = zscore_series(y, z_window)
    fig_sp.add_trace(go.Scatter(x=dcc["date"], y=y, name=cc, mode="lines"))
fig_sp.update_layout(margin=dict(l=10,r=10,t=10,b=10),
                     yaxis_title=("Spread (z)" if mode_z else "Spread (bp)"),
                     xaxis_title="Date")
fig_sp.update_xaxes(range=[start_date, end_date])
st.plotly_chart(fig_sp, use_container_width=True)

# ============== Fragmentatie — Bund cross spreads (10Y) ==============
st.subheader("Fragmentatie — Bund cross spreads (10Y)")
need_10y = DF[(DF["tenor"]=="10Y")].copy()
if not need_10y.empty:
    fig_frag = go.Figure()
    for cc in [c for c in sel_countries if c != "DE"]:
        dcc = need_10y[need_10y["country"]==cc]
        if dcc.empty: continue
        y = dcc["spread_to_DE"] * (1.0 if mode_z else 100.0)
        if mode_z: y = zscore_series(y, z_window)
        fig_frag.add_trace(go.Scatter(x=dcc["date"], y=y, name=f"{cc}−DE", mode="lines"))
    fig_frag.update_layout(margin=dict(l=10,r=10,t=10,b=10),
                           yaxis_title=("Spread (z)" if mode_z else "Spread (bp)"),
                           xaxis_title="Date")
    fig_frag.update_xaxes(range=[start_date, end_date])
    st.plotly_chart(fig_frag, use_container_width=True)
else:
    st.caption("Geen 10Y data beschikbaar voor fragmentatie-spreads.")

# ============== Δ1d distributies en tijdreeks (bp) ==============
st.subheader(f"Δ1d — Distributie & Tijdreeks ({sel_tenor})")
DFt_sorted = DFt.sort_values(["country","date"])
DFt_sorted["d1_bp"] = DFt_sorted.groupby("country")["value"].diff(1) * 100.0
d1 = DFt_sorted.dropna(subset=["d1_bp"])

cH1, cH2 = st.columns(2)
with cH1:
    H = go.Figure()
    for cc in sel_countries:
        s = d1[d1["country"]==cc]["d1_bp"].replace([np.inf,-np.inf], np.nan).dropna()
        if s.empty: continue
        H.add_trace(go.Histogram(x=s, nbinsx=40, name=cc, opacity=0.55))
    H.update_layout(title="Δ1d (bp) — histogram", barmode="overlay",
                    margin=dict(l=10,r=10,t=40,b=10), xaxis_title="Δ (bp)", yaxis_title="Count")
    st.plotly_chart(H, use_container_width=True)
with cH2:
    figd = go.Figure()
    for cc in sel_countries:
        s = d1[d1["country"]==cc]
        if s.empty: continue
        figd.add_trace(go.Bar(x=s["date"], y=s["d1_bp"], name=cc, opacity=0.5))
    figd.add_hline(y=0, line_width=1, line_color="gray", opacity=0.5)
    figd.update_layout(margin=dict(l=10,r=10,t=10,b=10), barmode="overlay",
                       yaxis_title="Δ1d (bp)", xaxis_title="Date")
    figd.update_xaxes(range=[start_date, end_date])
    st.plotly_chart(figd, use_container_width=True)

# ============== Heatmap — z-scores van spreads (country−DE) ==============
st.subheader(f"Heatmap — Z-scores spreads vs DE ({sel_tenor})")
HDF = DFt[DFt["country"]!="DE"].pivot(index="date", columns="country", values="spread_to_DE").sort_index()
if not HDF.empty:
    if mode_z:
        HZ = HDF.apply(lambda s: zscore_series(s, z_window))
    else:
        HZ = HDF * 100.0  # bp
    figH = go.Figure(data=go.Heatmap(
        z=HZ.values,
        x=HZ.columns.tolist(),
        y=HZ.index,
        coloraxis="coloraxis"
    ))
    figH.update_layout(margin=dict(l=10,r=10,t=10,b=10),
                       coloraxis_colorscale="RdBu", coloraxis_cmid=0,
                       xaxis_title="Country", yaxis_title="Date")
    st.plotly_chart(figH, use_container_width=True)
else:
    st.caption("Onvoldoende data om een heatmap te tonen.")

# ============== Tabel & download ==============
st.subheader("Gegevens (gefilterd)")
show_table = st.toggle("Toon tabel", value=False)
DF_show = DF[(DF["date"]>=start_date) & (DF["date"]<=end_date)].copy()
if not show_table:
    st.caption("Gebruik de downloadknop hieronder om data te exporteren.")
else:
    st.dataframe(DF_show.sort_values(["date","country","tenor"], ascending=[False, True, True]))

csv = DF_show.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download CSV (EU landen, gefilterd)", data=csv,
                   file_name=f"eu_country_yields_{sel_tenor}.csv", mime="text/csv")
