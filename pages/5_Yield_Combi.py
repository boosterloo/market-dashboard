# pages/Yield_Combined.py — US/EU in één pagina, simpele layout: 2 hoofdgrafieken + deselect via legenda
# - Linker (grijze) sidebar bevat alle keuzes (regio, looptijden, periode, y-as marge)
# - Grafiek 1: Yield curve (term structure) op gekozen peildatum
# - Grafiek 2: Ontwikkeling per looptijd (tijdreeks) — US & EU naast/over elkaar
# - Klik op de legenda om lijnen aan/uit te zetten (deselecteren)

import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
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

def axis_range(values: pd.Series, pad_pct: float, symmetric: bool=False):
    vals = pd.Series(values).dropna()
    if vals.empty:
        return None
    if symmetric:
        a = float(vals.abs().max())
        pad = a * (pad_pct/100.0)
        return (-a - pad, a + pad)
    vmin, vmax = float(vals.min()), float(vals.max())
    span = vmax - vmin
    base = max(abs(vmin), abs(vmax), span)
    pad = base * (pad_pct/100.0)
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
    min_d, max_d = all_df["date"].min(), all_df["date"].max()
    default_days = 365
    start_default = max(min_d, max_d - timedelta(days=default_days))
    rng = st.slider("Periode (tijdreeks)", min_value=min_d, max_value=max_d,
                    value=(start_default, max_d), format="YYYY-MM-DD")

    # Peildatum voor yield curve (eerste grafiek) — standaard laatste dag in de range
    last_in_range = max_d if rng is None else rng[1]
    curve_date = st.date_input("Peildatum (yield curve)", value=last_in_range,
                               min_value=min_d, max_value=max_d)

    # Royale y-assen
    ypad = st.slider("Y-as marge (royale ruimte, %)", 20, 120, 60, step=5)

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
curve = f[f["date"]==curve_date]
if curve.empty:
    # fallback: dichtstbijzijnde datum vóór peildatum binnen dataset
    prior = f[f["date"]<=curve_date]
    if not prior.empty:
        use_d = prior["date"].max()
        curve = prior[prior["date"]==use_d]
        st.caption(f"Geen exacte match op {curve_date}; toon {use_d}.")

fig_curve = go.Figure()
yvals = []
for reg in sorted(curve["region"].unique()):
    sub = curve[curve["region"]==reg].sort_values("tenor")
    if sub.empty: continue
    yvals.append(sub["value"]) 
    fig_curve.add_trace(go.Scatter(
        x=[nice_tenor(t) for t in sub["tenor"]],
        y=sub["value"], mode="lines+markers",
        name=f"{reg}", legendgroup=reg
    ))
if yvals:
    rng_y = axis_range(pd.concat(yvals, ignore_index=True), pad_pct=ypad)
    if rng_y:
        fig_curve.update_yaxes(range=list(rng_y))
fig_curve.update_layout(height=460, margin=dict(l=10,r=10,t=10,b=10), legend_title_text="Regio")
st.plotly_chart(fig_curve, use_container_width=True)

st.caption("Hint: klik op items in de legenda om lijnen te (de)selecteren.")

st.divider()

# -------------------- Grafiek 2: Ontwikkeling per looptijd --------------------
st.subheader("Ontwikkeling per looptijd (tijdreeks)")

mask = (f["date"]>=rng[0]) & (f["date"]<=rng[1])
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
        ybucket = []
        for reg in sorted(sub["region"].unique()):
            s2 = sub[sub["region"]==reg].dropna(subset=["value"])
            ybucket.append(s2["value"]) 
            fig_ts.add_trace(
                go.Scatter(x=s2["date"], y=s2["value"], mode="lines",
                           name=f"{reg} — {nice_tenor(t)}", legendgroup=reg,
                           showlegend=(i==1)),
                row=i, col=1
            )
        rng_y = axis_range(pd.concat(ybucket, ignore_index=True) if ybucket else pd.Series(dtype=float), pad_pct=ypad)
        if rng_y:
            fig_ts.update_yaxes(range=list(rng_y), row=i, col=1)

    fig_ts.update_layout(height=260*rows+60, margin=dict(l=10,r=10,t=40,b=10))
    st.plotly_chart(fig_ts, use_container_width=True)

st.caption("Legenda = toggle: je kunt per regio/looptijd lijnen verbergen voor snelle vergelijkingen.")
