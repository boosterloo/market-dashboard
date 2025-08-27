# pages/4_Macro.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
import plotly.graph_objects as go
from google.api_core.exceptions import NotFound, BadRequest

from utils.bq import run_query, bq_ping

st.set_page_config(page_title="📊 Grafieken per categorie", layout="wide")
st.title("Grafieken per categorie")

# ---------- Config ----------
DEFAULT_MACRO_VIEW = "nth-pier-468314-p7.marketdata.macro_series_wide_monthly_fill_v"
MACRO_VIEW = st.secrets.get("tables", {}).get("macro_view", DEFAULT_MACRO_VIEW)

# Kleuren (contrastrijk)
COL_BLUE = "#2563eb"   # primaire lijn
COL_CYAN = "#0891b2"   # secundaire lijn (links)
COL_RED  = "#dc2626"   # secundaire as (rechts)

def best_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns: return c
        if c.lower() in lower: return lower[c.lower()]
    return None

def padded_range(s: pd.Series, pad_ratio: float = 0.05):
    s = s.dropna()
    if s.empty: return None
    lo, hi = float(s.min()), float(s.max())
    if lo == hi:
        pad = abs(lo) * pad_ratio if lo != 0 else 1.0
        return [lo - pad, hi + pad]
    span = hi - lo
    pad = span * pad_ratio
    return [lo - pad, hi + pad]

def add_metric_chip(col, title: str, value: float, delta: float):
    color = "#16a34a" if delta > 0 else ("#dc2626" if delta < 0 else "#6b7280")
    col.markdown(f"**{title}**")
    col.markdown(
        f"""
        <div style="display:flex;gap:12px;align-items:baseline;">
          <div style="font-size:1.4rem;font-weight:700;">{value:.2f}</div>
          <div style="padding:2px 8px;border-radius:999px;background:{color}20;color:{color};font-weight:700;">
            {delta:+.2f}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------- BigQuery health ----------
try:
    with st.spinner("BigQuery check…"):
        if not bq_ping():
            st.error("Geen BigQuery-verbinding. Controleer Secrets ([gcp_service_account]).")
            st.stop()
except Exception as e:
    st.error("Geen BigQuery-verbinding. Controleer Secrets ([gcp_service_account]).")
    st.caption(f"Details: {e}")
    st.stop()

@st.cache_data(ttl=1800)
def load_macro():
    df = run_query(f"SELECT * FROM `{MACRO_VIEW}` ORDER BY date")
    if "date" not in df.columns:
        raise ValueError(f"Kolom 'date' niet aanwezig in view {MACRO_VIEW}")

    # Map kolommen naar namen die de grafiek verwacht
    cpi_h = best_col(df, ["cpi_headline","cpi_all","cpi","cpi_index"])
    cpi_c = best_col(df, ["cpi_core","core_cpi"])
    pce_h = best_col(df, ["pce_headline","pce_all","pce","pce_index"])

    ip    = best_col(df, ["industrial_production","ind_production","ind_prod","ip_index"])
    retail= best_col(df, ["retail_sales","retail"])
    house = best_col(df, ["housing_starts","housing","starts"])

    need = dict(cpi_headline=cpi_h, cpi_core=cpi_c, pce_headline=pce_h,
                industrial_production=ip, retail_sales=retail, housing_starts=house)
    missing = [k for k,v in need.items() if v is None]
    if missing:
        raise ValueError(f"Verplichte velden ontbreken in {MACRO_VIEW}: {', '.join(missing)}")

    df = df.rename(columns={
        cpi_h:"cpi_headline", cpi_c:"cpi_core", pce_h:"pce_headline",
        ip:"industrial_production", retail:"retail_sales", house:"housing_starts"
    })
    return df

# Laden met nette foutmeldingen
try:
    df = load_macro()
except NotFound:
    st.error(f"View niet gevonden: `{MACRO_VIEW}`. Pas [tables].macro_view in secrets.toml aan.")
    st.stop()
except (BadRequest, ValueError) as e:
    st.error(f"Macro-view probleem: {e}")
    st.stop()

# Datums
if not np.issubdtype(df["date"].dtype, np.datetime64):
    df["date"] = pd.to_datetime(df["date"])

# ---------- Periode over volle breedte ----------
all_min, all_max = df["date"].min().date(), df["date"].max().date()
default_start = all_max - timedelta(days=5*365)
start, end = st.slider("Periode", min_value=all_min, max_value=all_max,
                       value=(default_start, all_max), format="YYYY-MM-DD")

dfp = df[(df["date"].dt.date >= start) & (df["date"].dt.date <= end)].copy()

st.divider()

# ================== Inflatie (dual-axis) ==================
st.subheader("Inflatie (dual-axis)")
if dfp.empty:
    st.info("Geen inflatie-data in de gekozen periode.")
else:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dfp["date"], y=dfp["cpi_headline"],
                             mode="lines", name="CPI (headline)",
                             line=dict(color=COL_BLUE, width=3)))
    fig.add_trace(go.Scatter(x=dfp["date"], y=dfp["cpi_core"],
                             mode="lines", name="CPI (core)",
                             line=dict(color=COL_CYAN, width=2, dash="dash")))
    right_range = padded_range(dfp["pce_headline"])
    fig.add_trace(go.Scatter(x=dfp["date"], y=dfp["pce_headline"],
                             mode="lines", name="PCE (headline)",
                             line=dict(color=COL_RED, width=2, dash="dash"), yaxis="y2"))
    fig.update_layout(
        height=420, legend=dict(orientation="h"),
        margin=dict(l=0,r=0,t=10,b=0),
        yaxis=dict(title="Niveau (CPI)"),
        yaxis2=dict(title="Niveau (PCE)", overlaying="y", side="right", range=right_range),
        xaxis=dict(title="Datum")
    )
    st.plotly_chart(fig, use_container_width=True)

    last2 = dfp[["cpi_headline","cpi_core","pce_headline"]].tail(2)
    if len(last2) == 2:
        (c1, c2, c3) = st.columns(3)
        l, p = last2.iloc[-1], last2.iloc[-2]
        add_metric_chip(c1, "CPI (headline)", float(l["cpi_headline"]), float(l["cpi_headline"] - p["cpi_headline"]))
        add_metric_chip(c2, "CPI (core)",     float(l["cpi_core"]),     float(l["cpi_core"]     - p["cpi_core"]))
        add_metric_chip(c3, "PCE (headline)", float(l["pce_headline"]), float(l["pce_headline"] - p["pce_headline"]))

st.divider()

# ================== Activiteit (dual-axis) ==================
st.subheader("Activiteit (dual-axis)")
if dfp.empty:
    st.info("Geen activiteit-data in de gekozen periode.")
else:
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=dfp["date"], y=dfp["industrial_production"],
                              mode="lines", name="Industrial production",
                              line=dict(color=COL_BLUE, width=3)))
    fig2.add_trace(go.Scatter(x=dfp["date"], y=dfp["retail_sales"],
                              mode="lines", name="Retail sales",
                              line=dict(color=COL_CYAN, width=2, dash="dash")))
    right_range2 = padded_range(dfp["housing_starts"])
    fig2.add_trace(go.Scatter(x=dfp["date"], y=dfp["housing_starts"],
                              mode="lines", name="Housing starts",
                              line=dict(color=COL_RED, width=2, dash="dash"), yaxis="y2"))
    fig2.update_layout(
        height=420, legend=dict(orientation="h"),
        margin=dict(l=0,r=0,t=10,b=0),
        yaxis=dict(title="Niveau (IP / Retail)"),
        yaxis2=dict(title="Niveau (Housing starts)", overlaying="y", side="right", range=right_range2),
        xaxis=dict(title="Datum")
    )
    st.plotly_chart(fig2, use_container_width=True)

    last2 = dfp[["industrial_production","retail_sales","housing_starts"]].tail(2)
    if len(last2) == 2:
        (c1, c2, c3) = st.columns(3)
        l, p = last2.iloc[-1], last2.iloc[-2]
        add_metric_chip(c1, "Industrial production", float(l["industrial_production"]), float(l["industrial_production"] - p["industrial_production"]))
        add_metric_chip(c2, "Retail sales",          float(l["retail_sales"]),          float(l["retail_sales"]          - p["retail_sales"]))
        add_metric_chip(c3, "Housing starts",        float(l["housing_starts"]),        float(l["housing_starts"]        - p["housing_starts"]))
