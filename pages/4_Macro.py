# pages/macro_categories.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta
import plotly.graph_objects as go

from utils.bq import run_query, bq_ping

st.set_page_config(page_title="📊 Grafieken per categorie", layout="wide")
st.title("Grafieken per categorie")

# ---------------------------
# Config & helpers
# ---------------------------
DEFAULT_INFL_VIEW = "nth-pier-468314-p7.marketdata.macro_inflation_v"
DEFAULT_ACT_VIEW  = "nth-pier-468314-p7.marketdata.macro_activity_v"

INFL_VIEW = st.secrets.get("tables", {}).get("infl_view", DEFAULT_INFL_VIEW)
ACT_VIEW  = st.secrets.get("tables", {}).get("act_view",  DEFAULT_ACT_VIEW)

# Sterke, contrastrijke kleuren
COL_BLUE   = "#2563eb"  # CPI headline / Industrial production
COL_CYAN   = "#0891b2"  # CPI core / Retail sales
COL_RED    = "#dc2626"  # PCE headline / Housing starts  (secundaire as)

def best_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols_lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        if c.lower() in cols_lower:
            return cols_lower[c.lower()]
    return None

def padded_range(s: pd.Series, pad_ratio: float = 0.05) -> list[float]:
    s = s.dropna()
    if s.empty:
        return None
    lo, hi = float(s.min()), float(s.max())
    if lo == hi:
        pad = abs(lo) * pad_ratio if lo != 0 else 1.0
        return [lo - pad, hi + pad]
    span = hi - lo
    pad = span * pad_ratio
    return [lo - pad, hi + pad]

def add_metric_chip(col, title: str, value: float, delta: float):
    # Groen positief, rood negatief
    delta_str = f"{delta:+.2f}"
    color = "#16a34a" if delta > 0 else ("#dc2626" if delta < 0 else "#6b7280")
    col.markdown(f"**{title}**")
    col.markdown(
        f"""
        <div style="display:flex;gap:12px;align-items:baseline;">
          <div style="font-size:1.4rem;font-weight:700;">{value:.2f}</div>
          <div style="padding:2px 8px;border-radius:999px;background:{color}20;color:{color};font-weight:700;">
            {delta_str}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------
# BigQuery check & data
# ---------------------------
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
def load_infl():
    # Neem alles en map kolommen dynamisch
    df = run_query(f"SELECT * FROM `{INFL_VIEW}` ORDER BY date")
    if "date" not in df.columns:
        raise ValueError(f"Kolom 'date' niet aanwezig in view {INFL_VIEW}")

    cpi_h  = best_col(df, ["cpi_headline","cpi","cpi_index","cpi_h"])
    cpi_c  = best_col(df, ["cpi_core","core_cpi","cpi_ex_food_energy"])
    pce_h  = best_col(df, ["pce_headline","pce","pce_index","pce_h"])

    need = dict(cpi_headline=cpi_h, cpi_core=cpi_c, pce_headline=pce_h)
    missing = [k for k,v in need.items() if v is None]
    if missing:
        raise ValueError(f"Inflatie-velden ontbreken: {', '.join(missing)}")

    df = df.rename(columns={cpi_h:"cpi_headline", cpi_c:"cpi_core", pce_h:"pce_headline"})
    return df

@st.cache_data(ttl=1800)
def load_act():
    df = run_query(f"SELECT * FROM `{ACT_VIEW}` ORDER BY date")
    if "date" not in df.columns:
        raise ValueError(f"Kolom 'date' niet aanwezig in view {ACT_VIEW}")

    ip     = best_col(df, ["industrial_production","ind_prod","ip_index"])
    retail = best_col(df, ["retail_sales","retail","retail_index"])
    house  = best_col(df, ["housing_starts","housing","starts"])

    need = dict(industrial_production=ip, retail_sales=retail, housing_starts=house)
    missing = [k for k,v in need.items() if v is None]
    if missing:
        raise ValueError(f"Activiteit-velden ontbreken: {', '.join(missing)}")

    df = df.rename(columns={ip:"industrial_production", retail:"retail_sales", house:"housing_starts"})
    return df

with st.spinner("Data laden…"):
    df_infl = load_infl()
    df_act  = load_act()

for df in (df_infl, df_act):
    if not np.issubdtype(df["date"].dtype, np.datetime64):
        df["date"] = pd.to_datetime(df["date"])

# ---------------------------
# Periode-slider over volle breedte
# ---------------------------
all_min = min(df_infl["date"].min(), df_act["date"].min()).date()
all_max = max(df_infl["date"].max(), df_act["date"].max()).date()
default_start = all_max - timedelta(days=5*365)

start, end = st.slider(
    "Periode",
    min_value=all_min,
    max_value=all_max,
    value=(default_start, all_max),
    format="YYYY-MM-DD"
)

def subset(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df["date"].dt.date >= start) & (df["date"].dt.date <= end)].copy()

df_infl_p = subset(df_infl)
df_act_p  = subset(df_act)

st.divider()

# ===========================
# Inflatie (dual-axis)
# ===========================
st.subheader("Inflatie (dual-axis)")
if df_infl_p.empty:
    st.info("Geen inflatie-data in de gekozen periode.")
else:
    fig = go.Figure()

    # Linkeras: CPI headline + CPI core
    fig.add_trace(go.Scatter(
        x=df_infl_p["date"], y=df_infl_p["cpi_headline"],
        mode="lines", name="CPI (headline)", line=dict(color=COL_BLUE, width=3)
    ))
    fig.add_trace(go.Scatter(
        x=df_infl_p["date"], y=df_infl_p["cpi_core"],
        mode="lines", name="CPI (core)", line=dict(color=COL_CYAN, width=2, dash="dash")
    ))

    # Rechteras: PCE headline (ROOD) met dynamische schaal
    right_range = padded_range(df_infl_p["pce_headline"])
    fig.add_trace(go.Scatter(
        x=df_infl_p["date"], y=df_infl_p["pce_headline"],
        mode="lines", name="PCE (headline)", line=dict(color=COL_RED, width=2, dash="dash"),
        yaxis="y2"
    ))

    fig.update_layout(
        height=420,
        legend=dict(orientation="h"),
        margin=dict(l=0,r=0,t=10,b=0),
        yaxis=dict(title="Niveau (CPI)"),
        yaxis2=dict(
            title="Niveau (PCE)",
            overlaying="y",
            side="right",
            range=right_range  # <-- dynamische schaal
        ),
        xaxis=dict(title="Datum")
    )
    st.plotly_chart(fig, use_container_width=True)

    # Deltas (laatste vs vorige punt)
    latest = df_infl_p.tail(2).copy()
    if len(latest) >= 2:
        last, prev = latest.iloc[-1], latest.iloc[-2]
        c1, c2, c3 = st.columns(3)
        add_metric_chip(c1, "CPI (headline)", float(last["cpi_headline"]), float(last["cpi_headline"] - prev["cpi_headline"]))
        add_metric_chip(c2, "CPI (core)",     float(last["cpi_core"]),     float(last["cpi_core"]     - prev["cpi_core"]))
        add_metric_chip(c3, "PCE (headline)", float(last["pce_headline"]), float(last["pce_headline"] - prev["pce_headline"]))

st.divider()

# ===========================
# Activiteit (dual-axis)
# ===========================
st.subheader("Activiteit (dual-axis)")
if df_act_p.empty:
    st.info("Geen activiteit-data in de gekozen periode.")
else:
    fig2 = go.Figure()

    # Linkeras: Industrial production + Retail sales
    fig2.add_trace(go.Scatter(
        x=df_act_p["date"], y=df_act_p["industrial_production"],
        mode="lines", name="Industrial production", line=dict(color=COL_BLUE, width=3)
    ))
    fig2.add_trace(go.Scatter(
        x=df_act_p["date"], y=df_act_p["retail_sales"],
        mode="lines", name="Retail sales", line=dict(color=COL_CYAN, width=2, dash="dash")
    ))

    # Rechteras: Housing starts (ROOD) met dynamische schaal
    right_range2 = padded_range(df_act_p["housing_starts"])
    fig2.add_trace(go.Scatter(
        x=df_act_p["date"], y=df_act_p["housing_starts"],
        mode="lines", name="Housing starts", line=dict(color=COL_RED, width=2, dash="dash"),
        yaxis="y2"
    ))

    fig2.update_layout(
        height=420,
        legend=dict(orientation="h"),
        margin=dict(l=0,r=0,t=10,b=0),
        yaxis=dict(title="Niveau (IP / Retail)"),
        yaxis2=dict(
            title="Niveau (Housing starts)",
            overlaying="y",
            side="right",
            range=right_range2  # <-- dynamische schaal
        ),
        xaxis=dict(title="Datum")
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Deltas (laatste vs vorige punt)
    latest = df_act_p.tail(2).copy()
    if len(latest) >= 2:
        last, prev = latest.iloc[-1], latest.iloc[-2]
        c1, c2, c3 = st.columns(3)
        add_metric_chip(c1, "Industrial production", float(last["industrial_production"]), float(last["industrial_production"] - prev["industrial_production"]))
        add_metric_chip(c2, "Retail sales",          float(last["retail_sales"]),          float(last["retail_sales"]          - prev["retail_sales"]))
        add_metric_chip(c3, "Housing starts",        float(last["housing_starts"]),        float(last["housing_starts"]        - prev["housing_starts"]))
