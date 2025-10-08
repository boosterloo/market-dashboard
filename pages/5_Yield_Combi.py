# pages/Yield_Combined.py — US & EU: Yield curve + Ontwikkeling, met KPI's en tijdreeks-toggle
# Vereist: utils.bq.run_query, utils.bq.bq_ping
# Secrets: 
#   [tables]
#   us_yield_view = "<project>.marketdata.us_yield_curve_enriched_v"
#   eu_yield_view = "<project>.marketdata.eu_yield_curve_enriched_v"

import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta, date as date_cls
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.bq import run_query, bq_ping

# ───────────────────── Page ─────────────────────
st.set_page_config(page_title="🧯 Yield — US & EU (combi)", layout="wide")
st.title("🧯 Yield — US & EU (combi)")

# ──────────────────── Views ─────────────────────
TABLES = st.secrets.get("tables", {})
US_VIEW = TABLES.get("us_yield_view", "nth-pier-468314-p7.marketdata.us_yield_curve_enriched_v")
EU_VIEW = TABLES.get("eu_yield_view", "nth-pier-468314-p7.marketdata.eu_yield_curve_enriched_v")

# ─────────────────── Health ─────────────────────
try:
    if not bq_ping():
        st.error("Geen BigQuery-verbinding."); st.stop()
except Exception as e:
    st.error("Geen BigQuery-verbinding."); st.caption(f"Details: {e}"); st.stop()

# ─────────────────── Helpers ────────────────────
TENOR_ORDER = ["y_3m","y_6m","y_1y","y_2y","y_3y","y_5y","y_7y","y_10y","y_20y","y_30y"]

@st.cache_data(ttl=1800, show_spinner=False)
def load_view(view: str) -> pd.DataFrame:
    return run_query(f"SELECT * FROM `{view}` ORDER BY date")

def to_long(df: pd.DataFrame, region: str) -> pd.DataFrame:
    """Zet brede y_* kolommen om naar long (date, region, tenor, value). Laat long al door."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["date","region","tenor","value"])
    ycols = [c for c in df.columns if str(c).startswith("y_")]
    if not ycols:
        if {"date","tenor","value"}.issubset(df.columns):
            out = df.copy()
            if "region" not in out.columns:
                out["region"] = region
            return out[["date","region","tenor","value"]]
        return pd.DataFrame(columns=["date","region","tenor","value"])
    out = (
        df[["date"] + ycols].melt("date", var_name="tenor", value_name="value")
        .dropna(subset=["value"]).assign(region=region)
    )
    return out[["date","region","tenor","value"]]

def nice_tenor(t: str) -> str:
    if not isinstance(t, str): return str(t)
    if t.startswith("y_"):
        core = t[2:]
        if core.endswith("m"): return core[:-1].upper()+"M"
        if core.endswith("y"): return core[:-1].upper()+"Y"
        return core.upper()
    return t.upper()

# Autoscaling: royale marge + minimale span (pp) zodat vlakke lijnen zichtbaar blijven
def axis_range(values: pd.Series, pad_pct: float, min_span_pp: float = 0.30, symmetric: bool=False):
    vals = pd.Series(values).astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if vals.empty:
        return None
    vmin, vmax = float(vals.min()), float(vals.max())
    span = max(vmax - vmin, 0.0)
    # afdwingen minimaal bereik
    if span < min_span_pp:
        mid = (vmax + vmin) / 2.0
        half = min_span_pp / 2.0
        vmin, vmax, span = mid - half, mid + half, min_span_pp
    if symmetric:
        a = max(abs(vmin), abs(vmax))
        pad = a * (pad_pct/100.0)
        return (-a - pad, a + pad)
    pad = span * (pad_pct/100.0)
    return (vmin - pad, vmax + pad)

def last_value(df: pd.DataFrame, region: str, tenor: str, up_to: date_cls | None = None):
    """Laatste waarde t/m up_to (of helemaal laatste) voor (region, tenor)."""
    sub = df[(df["region"]==region) & (df["tenor"]==tenor)].dropna(subset=["value"])
    if up_to is not None:
        sub = sub[sub["date"]<=up_to]
    if sub.empty:
        return np.nan, None
    last_row = sub.iloc[-1]
    return float(last_row["value"]), last_row["date"]

# ───────────────────── Data ─────────────────────
with st.spinner("Yield data laden…"):
    us_raw = load_view(US_VIEW)
    eu_raw = load_view(EU_VIEW)

for df in (us_raw, eu_raw):
    if df is not None and not df.empty and "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.date

us = to_long(us_raw, "US") if us_raw is not None else pd.DataFrame(columns=["date","region","tenor","value"])
eu = to_long(eu_raw, "EU") if eu_raw is not None else pd.DataFrame(columns=["date","region","tenor","value"])
all_df = pd.concat([us, eu], ignore_index=True)
if all_df.empty:
    st.warning("Geen bruikbare data gevonden."); st.stop()

# ─────────────────── Sidebar ────────────────────
with st.sidebar:
    st.header("Filters")
    st.caption("Samenstellen van je weergave")

    regions = st.multiselect("Regio’s", ["US","EU"], default=["US","EU"])

    default_tenors = ["y_2y","y_5y","y_10y","y_30y"]
    tenors = st.multiselect("Looptijden (voor beide grafieken)", TENOR_ORDER, default=default_tenors)

    # KPI-tenant (losse keuze zodat je altijd bv. 10Y kunt tonen)
    kpi_tenor = st.selectbox("KPI-tenor", TENOR_ORDER, index=TENOR_ORDER.index("y_10y") if "y_10y" in TENOR_ORDER else 0)

    # Periode voor tijdreeks — robuust met unieke datums
    date_series = all_df["date"].dropna().sort_values().unique()
    if len(date_series) < 2:
        min_d = max_d = date_series[0]
    else:
        min_d, max_d = date_series[0], date_series[-1]

    # Periode presets
    st.subheader("Periode")
    def clamp(ts: date_cls) -> date_cls:
        return min(max(ts, min_d), max_d)

    pr = st.radio("Range", ["3M","6M","1Y","3Y","5Y","YTD","Max","Custom"], horizontal=True, index=2)
    if pr == "3M":  start_date, end_date = clamp(max_d - timedelta(days=90)),  max_d
    elif pr == "6M":start_date, end_date = clamp(max_d - timedelta(days=182)), max_d
    elif pr == "1Y":start_date, end_date = clamp(max_d - timedelta(days=365)), max_d
    elif pr == "3Y":start_date, end_date = clamp(max_d - timedelta(days=365*3)), max_d
    elif pr == "5Y":start_date, end_date = clamp(max_d - timedelta(days=365*5)), max_d
    elif pr == "YTD":start_date, end_date = clamp(date_cls(max_d.year,1,1)), max_d
    elif pr == "Max":start_date, end_date = min_d, max_d
    else:
        # Custom slider
        start_default = clamp(max_d - timedelta(days=365))
        dr = st.slider("Selecteer periode (Custom)",
                       min_value=min_d, max_value=max_d,
                       value=(start_default, max_d), format="YYYY-MM-DD")
        start_date, end_date = dr[0], dr[1]

    # Peildatum voor Yield Curve — default rechterkant van de periode
    curve_date = st.date_input("Peildatum (yield curve)", value=end_date, min_value=min_d, max_value=max_d)

    # Tijdreeks-weergave
    tr_mode = st.radio("Weergave tijdreeks", ["Subplots per looptijd", "Overlay"], index=0, horizontal=False)

    # Royale y-assen
    ypad = st.slider("Y-as marge (royale ruimte, %)", 20, 120, 60, step=5,
                     help="Extra witruimte boven/onder. Min. span 0.30 pp voorkomt 'platte' lijnen.")

if not regions:
    st.info("Kies minimaal één regio."); st.stop()
if not tenors:
    st.info("Kies minimaal één looptijd."); st.stop()

# Filter & orden
f = all_df[(all_df["region"].isin(regions)) & (all_df["tenor"].isin(tenors))].copy()
f["tenor"] = pd.Categorical(f["tenor"], categories=TENOR_ORDER, ordered=True)
f = f.sort_values(["region","tenor","date"]).reset_index(drop=True)

# ─────────────────── KPI-blok ───────────────────
c1, c2, c3, c4 = st.columns([1,1,1,1.4])
us_last, us_d = last_value(all_df, "US", kpi_tenor, up_to=end_date)
eu_last, eu_d = last_value(all_df, "EU", kpi_tenor, up_to=end_date)
with c1:
    st.metric(f"US {nice_tenor(kpi_tenor)} (laatst)", f"{us_last:.2f}%" if not np.isnan(us_last) else "—")
with c2:
    st.metric(f"EU {nice_tenor(kpi_tenor)} (laatst)", f"{eu_last:.2f}%" if not np.isnan(eu_last) else "—")
with c3:
    if not np.isnan(us_last) and not np.isnan(eu_last):
        spread = us_last - eu_last
        st.metric(f"US–EU spread {nice_tenor(kpi_tenor)}", f"{spread:.2f} pp")
    else:
        st.metric(f"US–EU spread {nice_tenor(kpi_tenor)}", "—")
with c4:
    st.caption(f"Periode: {start_date} → {end_date}  |  Peildatum curve: {curve_date}")

st.divider()

# ─────────────── Grafiek 1: Yield Curve ───────────────
st.subheader("Yield curve (term structure)")

# Kies dichtstbijzijnde datum <= curve_date; anders eerste erna
curve_pool = f[f["date"] <= curve_date]
if curve_pool.empty:
    curve_pool = f[f["date"] >= curve_date]
use_d = curve_pool["date"].max() if not curve_pool.empty else None

fig_curve = go.Figure()
if use_d is None:
    st.info("Geen data rond de gekozen peildatum.")
else:
    curve = f[f["date"]==use_d]
    yvals = []
    for reg in sorted(curve["region"].unique()):
        sub = curve[curve["region"]==reg].sort_values("tenor")
        if sub.empty: continue
        yvals.append(sub["value"])
        fig_curve.add_trace(go.Scatter(
            x=[nice_tenor(t) for t in sub["tenor"]],
            y=sub["value"],
            mode="lines+markers",
            name=reg,
            legendgroup=reg,
            hovertemplate="%{x}: %{y:.2f}%<extra>"+reg+"</extra>"
        ))
    rng_y = axis_range(pd.concat(yvals, ignore_index=True) if yvals else pd.Series(dtype=float), pad_pct=ypad)
    if rng_y:
        fig_curve.update_yaxes(range=list(rng_y))
    fig_curve.update_layout(
        height=480,
        margin=dict(l=10,r=10,t=30,b=10),
        legend_title_text="Regio",
        title=f"Term-structure op {use_d}"
    )
    st.plotly_chart(fig_curve, use_container_width=True)

st.caption("Legenda: klik op US/EU om lijnen te (de)selecteren. Autoscaling houdt lijnen leesbaar.")

st.divider()

# ────── Grafiek 2: Ontwikkeling van de rente (tijdreeks) ──────
st.subheader("Ontwikkeling van de rente (tijdreeks)")
mask = (f["date"]>=start_date) & (f["date"]<=end_date)
ft = f.loc[mask].copy()

if ft.empty:
    st.info("Geen data in de gekozen periode.")
else:
    if tr_mode == "Subplots per looptijd":
        rows = len(tenors)
        fig_ts = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                               subplot_titles=[nice_tenor(t) for t in tenors])

        for i, t in enumerate(tenors, start=1):
            sub = ft[ft["tenor"]==t]
            if sub.empty: 
                continue
            buckets = []
            for reg in sorted(sub["region"].unique()):
                s2 = sub[sub["region"]==reg].dropna(subset=["value"])
                buckets.append(s2["value"])
                fig_ts.add_trace(
                    go.Scatter(
                        x=s2["date"], y=s2["value"], mode="lines",
                        name=f"{reg} — {nice_tenor(t)}",
                        legendgroup=reg,
                        showlegend=(i==1),
                        hovertemplate="%{x}<br>%{y:.2f}%<extra>"+reg+"</extra>"
                    ),
                    row=i, col=1
                )
            rng_y = axis_range(pd.concat(buckets, ignore_index=True) if buckets else pd.Series(dtype=float),
                               pad_pct=ypad, min_span_pp=0.30)
            if rng_y:
                fig_ts.update_yaxes(range=list(rng_y), row=i, col=1)

        fig_ts.update_layout(height=280*rows+60, margin=dict(l=10,r=10,t=40,b=10))
        st.plotly_chart(fig_ts, use_container_width=True)

    else:  # Overlay
        fig_ov = go.Figure()
        yvals_all = []
        # plot per regio + tenor, zodat legenda goed werkt
        for reg in sorted(ft["region"].unique()):
            for t in tenors:
                sub = ft[(ft["region"]==reg) & (ft["tenor"]==t)].dropna(subset=["value"])
                if sub.empty: 
                    continue
                yvals_all.append(sub["value"])
                fig_ov.add_trace(go.Scatter(
                    x=sub["date"], y=sub["value"], mode="lines",
                    name=f"{reg} — {nice_tenor(t)}", legendgroup=f"{reg}",
                    hovertemplate="%{x}<br>%{y:.2f}%<extra>"+f"{reg} {nice_tenor(t)}"+"</extra>"
                ))
        rng_y = axis_range(pd.concat(yvals_all, ignore_index=True) if yvals_all else pd.Series(dtype=float),
                           pad_pct=ypad, min_span_pp=0.30)
        if rng_y:
            fig_ov.update_yaxes(range=list(rng_y))
        fig_ov.update_layout(height=520, margin=dict(l=10,r=10,t=10,b=10), legend_title_text="Regio — Tenor")
        st.plotly_chart(fig_ov, use_container_width=True)

st.caption("Periode-links bepaalt de tijdreeks; peildatum bepaalt de yield curve. Toggle ‘Overlay’ voor 1 gecombineerde grafiek.")
