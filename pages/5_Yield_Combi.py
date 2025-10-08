# pages/Yield_Combined.py — US/EU combi met échte autoscaling, duidelijke regio‑weergave en periodekeuze
# Verbeteringen:
# - Y‑assen autoscaling per grafiek met royale marge + minimale span (bijna vlakke lijnen worden leesbaar)
# - Periode‑slider robuust (fallback als min==max, gebruik unieke datums)
# - Yield curve toont altijd US én/óf EU duidelijk; legenda toggles werken; hover info strak
# - Tijdreeks per tenor met dezelfde autoscaling‑logica en subplot‑titels

import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta, date as date_cls
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.bq import run_query, bq_ping

# -------------------- Pagina --------------------
st.set_page_config(page_title="🧯 Yield — US & EU (combi)", layout="wide")
st.title("🧯 Yield — US & EU (combi)")

# -------------------- Views ----------------------
TABLES = st.secrets.get("tables", {})
US_VIEW = TABLES.get("us_yield_view", "nth-pier-468314-p7.marketdata.us_yield_curve_enriched_v")
EU_VIEW = TABLES.get("eu_yield_view", "nth-pier-468314-p7.marketdata.eu_yield_curve_enriched_v")

# -------------------- Health ---------------------
try:
    if not bq_ping():
        st.error("Geen BigQuery-verbinding."); st.stop()
except Exception as e:
    st.error("Geen BigQuery-verbinding."); st.caption(f"Details: {e}"); st.stop()

# -------------------- Helpers --------------------
TENOR_ORDER = ["y_3m","y_6m","y_1y","y_2y","y_3y","y_5y","y_7y","y_10y","y_20y","y_30y"]

@st.cache_data(ttl=1800, show_spinner=False)
def load_view(view: str) -> pd.DataFrame:
    return run_query(f"SELECT * FROM `{view}` ORDER BY date")

def to_long(df: pd.DataFrame, region: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["date","region","tenor","value"]) 
    ycols = [c for c in df.columns if str(c).startswith("y_")]
    if not ycols:
        # al long-vorm?
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

# Autoscaling met royale marge en minimale span (in percentagepunt)
def axis_range(values: pd.Series, pad_pct: float, min_span_pp: float = 0.30, symmetric: bool=False):
    vals = pd.Series(values).astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if vals.empty:
        return None
    vmin, vmax = float(vals.min()), float(vals.max())
    span = max(vmax - vmin, 0.0)
    # als bijna vlak: forceer minimale span rond het midden
    if span < min_span_pp:
        mid = (vmax + vmin) / 2.0
        half = min_span_pp / 2.0
        vmin, vmax = mid - half, mid + half
        span = min_span_pp
    if symmetric:
        a = max(abs(vmin), abs(vmax))
        pad = a * (pad_pct/100.0)
        return (-a - pad, a + pad)
    pad = span * (pad_pct/100.0)
    return (vmin - pad, vmax + pad)

# -------------------- Data -----------------------
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

# -------------------- Sidebar --------------------
with st.sidebar:
    st.header("Filters")
    st.caption("Stel je weergave samen")

    regions = st.multiselect("Regio’s", ["US","EU"], default=["US","EU"]) 

    default_tenors = ["y_2y","y_5y","y_10y","y_30y"]
    tenors = st.multiselect("Looptijden", TENOR_ORDER, default=default_tenors)

    # Periode voor tijdreeks (tweede grafiek)
    # Gebruik unieke datums om slider te laten werken bij gaten
    date_series = all_df["date"].dropna().sort_values().unique()
    if len(date_series) < 2:
        min_d = max_d = date_series[0]
    else:
        min_d, max_d = date_series[0], date_series[-1]
    default_days = 365
    start_default = (max_d - timedelta(days=default_days)) if isinstance(max_d, date_cls) else max_d
    start_default = max(min_d, start_default)
    periode = st.slider(
        "Periode (tijdreeks)", min_value=min_d, max_value=max_d,
        value=(start_default, max_d), format="YYYY-MM-DD"
    ) if len(date_series) >= 2 else (min_d, max_d)

    # Peildatum voor yield curve (eerste grafiek) — standaard rechter eindpunt van periode
    last_in_range = periode[1]
    curve_date = st.date_input(
        "Peildatum (yield curve)", value=last_in_range,
        min_value=min_d, max_value=max_d,
        help="Kies de datum voor de term-structure."
    )

    # Royale y-assen: standaard 60% extra marge
    ypad = st.slider("Y-as marge (royale ruimte, %)", 20, 120, 60, step=5,
                     help="Extra witruimte boven/onder de datarange. Min. span 0.30 pp.")

if not regions:
    st.info("Kies minimaal één regio."); st.stop()
if not tenors:
    st.info("Kies minimaal één looptijd."); st.stop()

# filteren
f = all_df[(all_df["region"].isin(regions)) & (all_df["tenor"].isin(tenors))].copy()
f["tenor"] = pd.Categorical(f["tenor"], categories=TENOR_ORDER, ordered=True)
f = f.sort_values(["region","tenor","date"]).reset_index(drop=True)

# -------------------- Grafiek 1: Yield Curve --------------------
st.subheader("Yield curve (term structure)")

# Kies dichtstbijzijnde beschikbare datum <= curve_date
curve_pool = f[f["date"] <= curve_date]
if curve_pool.empty:
    # als alles na curve_date ligt: pak eerste datum erna
    curve_pool = f[f["date"] >= curve_date]
use_d = curve_pool["date"].max() if not curve_pool.empty else None

fig_curve = go.Figure()
if use_d is None:
    st.info("Geen data rond de gekozen peildatum.")
else:
    curve = f[f["date"]==use_d]
    if curve.empty:
        st.info("Geen data exact op peildatum.")
    else:
        yvals = []
        for reg in sorted(curve["region"].unique()):
            sub = curve[curve["region"]==reg].sort_values("tenor")
            if sub.empty: continue
            yvals.append(sub["value"]) 
            fig_curve.add_trace(go.Scatter(
                x=[nice_tenor(t) for t in sub["tenor"]],
                y=sub["value"], mode="lines+markers",
                name=f"{reg}", legendgroup=reg,
                hovertemplate="%{x}: %{y:.2f}%<extra>"+reg+"</extra>"
            ))
        rng_y = axis_range(pd.concat(yvals, ignore_index=True) if yvals else pd.Series(dtype=float), pad_pct=ypad)
        if rng_y:
            fig_curve.update_yaxes(range=list(rng_y))
        fig_curve.update_layout(
            height=480, margin=dict(l=10,r=10,t=30,b=10), legend_title_text="Regio",
            title=f"Term-structure op {use_d}"
        )
        st.plotly_chart(fig_curve, use_container_width=True)

st.caption("Legenda = toggle: klik om US of EU (de)selecteren. Lijnen schalen automatisch met royale marge.")

st.divider()

# -------------------- Grafiek 2: Ontwikkeling per looptijd --------------------
st.subheader("Ontwikkeling per looptijd (tijdreeks)")

mask = (f["date"]>=periode[0]) & (f["date"]<=periode[1])
ft = f.loc[mask].copy()
if ft.empty:
    st.info("Geen data in de gekozen periode.")
else:
    rows = len(tenors)
    fig_ts = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                           subplot_titles=[nice_tenor(t) for t in tenors])
    for i, t in enumerate(tenors, start=1):
        sub = ft[ft["tenor"]==t]
        if sub.empty: continue
        buckets = []
        for reg in sorted(sub["region"].unique()):
            s2 = sub[sub["region"]==reg].dropna(subset=["value"])
            buckets.append(s2["value"]) 
            fig_ts.add_trace(
                go.Scatter(
                    x=s2["date"], y=s2["value"], mode="lines",
                    name=f"{reg} — {nice_tenor(t)}", legendgroup=reg,
                    showlegend=(i==1), hovertemplate="%{x}<br>%{y:.2f}%<extra>"+reg+"</extra>"
                ), row=i, col=1
            )
        rng_y = axis_range(pd.concat(buckets, ignore_index=True) if buckets else pd.Series(dtype=float),
                           pad_pct=ypad, min_span_pp=0.30)
        if rng_y:
            fig_ts.update_yaxes(range=list(rng_y), row=i, col=1)

    fig_ts.update_layout(height=280*rows+60, margin=dict(l=10,r=10,t=40,b=10))
    st.plotly_chart(fig_ts, use_container_width=True)

st.caption("Tip: gebruik de slider links voor de periode. Y‑assen autoscalen royaal; bij vlakke reeksen wordt minimaal 0.30 pp span afgedwongen.")
