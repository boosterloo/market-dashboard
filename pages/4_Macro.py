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

# Kleuren
COL_BLUE = "#2563eb"   # primaire lijn
COL_CYAN = "#0891b2"   # secundair (links)
COL_RED  = "#dc2626"   # secundair (rechts)
PALETTE  = ["#2563eb","#0891b2","#dc2626","#16a34a","#9333ea","#f59e0b","#0ea5e9","#ef4444","#14b8a6","#f97316"]

def padded_range(s: pd.Series, pad_ratio: float = 0.05):
    s = pd.to_numeric(s, errors="coerce").dropna()
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

def normalize_100(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    base = s.dropna().iloc[0] if s.notna().any() else np.nan
    return s / base * 100.0 if pd.notna(base) and base != 0 else s

def best_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns: return c
        if c.lower() in lower: return lower[c.lower()]
    return None

# ---------- BigQuery check ----------
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

    # Map vaste namen
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

# Laden
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

# ---------- Periode ----------
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
    fig.add_trace(go.Scatter(x=dfp["date"], y=pd.to_numeric(dfp["cpi_headline"], errors="coerce"),
                             mode="lines", name="CPI (headline)",
                             line=dict(color=COL_BLUE, width=3)))
    fig.add_trace(go.Scatter(x=dfp["date"], y=pd.to_numeric(dfp["cpi_core"], errors="coerce"),
                             mode="lines", name="CPI (core)",
                             line=dict(color=COL_CYAN, width=2, dash="dash")))
    r2 = padded_range(dfp["pce_headline"])
    fig.add_trace(go.Scatter(x=dfp["date"], y=pd.to_numeric(dfp["pce_headline"], errors="coerce"),
                             mode="lines", name="PCE (headline)",
                             line=dict(color=COL_RED, width=2, dash="dash"), yaxis="y2"))
    fig.update_layout(
        height=420, legend=dict(orientation="h"),
        margin=dict(l=0,r=0,t=10,b=0),
        yaxis=dict(title="Niveau (CPI)"),
        yaxis2=dict(title="Niveau (PCE)", overlaying="y", side="right", range=r2),
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

# ================== Activiteit (dual-axis + autoscale IP) ==================
st.subheader("Activiteit (dual-axis)")
if dfp.empty:
    st.info("Geen activiteit-data in de gekozen periode.")
else:
    fig2 = go.Figure()
    # y (links): Retail sales
    fig2.add_trace(go.Scatter(x=dfp["date"], y=pd.to_numeric(dfp["retail_sales"], errors="coerce"),
                              mode="lines", name="Retail sales",
                              line=dict(color=COL_CYAN, width=2, dash="dash")))
    # y2 (rechts): Housing starts
    y2r = padded_range(dfp["housing_starts"])
    fig2.add_trace(go.Scatter(x=dfp["date"], y=pd.to_numeric(dfp["housing_starts"], errors="coerce"),
                              mode="lines", name="Housing starts",
                              line=dict(color=COL_RED, width=2, dash="dash"), yaxis="y2"))
    # y3 (links overlay): Industrial production eigen schaal
    y3r = padded_range(dfp["industrial_production"])
    fig2.add_trace(go.Scatter(x=dfp["date"], y=pd.to_numeric(dfp["industrial_production"], errors="coerce"),
                              mode="lines", name="Industrial production",
                              line=dict(color=COL_BLUE, width=3), yaxis="y3"))
    fig2.update_layout(
        height=420, legend=dict(orientation="h"),
        margin=dict(l=0,r=0,t=10,b=0),
        xaxis=dict(title="Datum"),
        yaxis=dict(title="Niveau (Retail)"),
        yaxis2=dict(title="Niveau (Housing starts)", overlaying="y", side="right", range=y2r),
        yaxis3=dict(overlaying="y", side="left", range=y3r, showticklabels=False, showgrid=False)
    )
    st.plotly_chart(fig2, use_container_width=True)

    last2 = dfp[["industrial_production","retail_sales","housing_starts"]].tail(2)
    if len(last2) == 2:
        (c1, c2, c3) = st.columns(3)
        l, p = last2.iloc[-1], last2.iloc[-2]
        add_metric_chip(c1, "Industrial production", float(l["industrial_production"]), float(l["industrial_production"] - p["industrial_production"]))
        add_metric_chip(c2, "Retail sales",          float(l["retail_sales"]),          float(l["retail_sales"]          - p["retail_sales"]))
        add_metric_chip(c3, "Housing starts",        float(l["housing_starts"]),        float(l["housing_starts"]        - p["housing_starts"]))

st.divider()

# ================== Overige macro-indicatoren — geclusterd & opgesplitst ==================
st.subheader("Overige macro-indicatoren")

# Eén instelling voor ALLE overige grafieken:
mode = st.radio("Weergave overige grafieken", ["Genormaliseerd (index = 100)", "Eigen as per serie"], horizontal=True, index=0)
show_ma3 = st.checkbox("Toon MA3 (waar beschikbaar)", value=True)

# Bepaal beschikbare kolommen
have = set(dfp.columns)

# ---- Arbeidsmarkt ----
arb_cols = [c for c in ["unemployment","payrolls","init_claims"] if c in have]
if arb_cols:
    st.markdown("### Arbeidsmarkt")
    fig = go.Figure()
    if mode.startswith("Genormaliseerd"):
        for i, col in enumerate(arb_cols):
            fig.add_trace(go.Scatter(
                x=dfp["date"], y=normalize_100(dfp[col]),
                mode="lines", name=col, line=dict(width=2, color=PALETTE[i % len(PALETTE)])
            ))
        yr = padded_range(pd.concat([normalize_100(dfp[c]) for c in arb_cols], axis=0))
        fig.update_layout(height=420, legend=dict(orientation="h"), margin=dict(l=0,r=0,t=10,b=0),
                          yaxis=dict(title="Index (=100)", range=yr), xaxis=dict(title="Datum"))
    else:
        for i, col in enumerate(arb_cols):
            ax_id = "" if i == 0 else str(i+1)
            ax_name = f"yaxis{ax_id}" if ax_id else "yaxis"
            series = pd.to_numeric(dfp[col], errors="coerce")
            fig.add_trace(go.Scatter(x=dfp["date"], y=series, mode="lines",
                                     name=col, yaxis=f"y{ax_id}" if ax_id else "y",
                                     line=dict(width=2, color=PALETTE[i % len(PALETTE)])))
            rng = padded_range(series)
            if i == 0:
                fig.update_layout(yaxis=dict(title=col, range=rng))
            elif i == 1:
                fig.update_layout(yaxis2=dict(title=col, overlaying="y", side="right", range=rng))
            else:
                side = "left" if i % 2 == 0 else "right"
                fig[ "layout" ][ax_name] = dict(overlaying="y", side=side, range=rng, showticklabels=False, showgrid=False)
        fig.update_layout(height=420, legend=dict(orientation="h"), margin=dict(l=0,r=0,t=10,b=0), xaxis=dict(title="Datum"))
    st.plotly_chart(fig, use_container_width=True)

    last2 = dfp[arb_cols].tail(2)
    if len(last2) == 2:
        cols = st.columns(min(5, len(arb_cols)))
        for i, col in enumerate(arb_cols):
            l, p = float(last2[col].iloc[-1]), float(last2[col].iloc[-2])
            add_metric_chip(cols[i % len(cols)], col, l, l - p)

# ---- Geld: M2 niveaus ----
m2_level_cols = [c for c in ["m2","m2_real"] if c in have]
m2_ma_cols    = [c for c in ["m2_ma3","m2_real_ma3"] if c in have]
if m2_level_cols:
    st.markdown("### M2 — niveaus")
    fig = go.Figure()
    if mode.startswith("Genormaliseerd"):
        series_list = [normalize_100(dfp[c]) for c in m2_level_cols]
        if show_ma3 and m2_ma_cols:
            series_list += [normalize_100(dfp[c]) for c in m2_ma_cols]
        for i, c in enumerate(m2_level_cols + (m2_ma_cols if show_ma3 else [])):
            fig.add_trace(go.Scatter(x=dfp["date"], y=normalize_100(dfp[c]),
                                     mode="lines", name=c, line=dict(width=2, color=PALETTE[i % len(PALETTE)])))
        yr = padded_range(pd.concat(series_list, axis=0))
        fig.update_layout(height=380, legend=dict(orientation="h"), margin=dict(l=0,r=0,t=10,b=0),
                          yaxis=dict(title="Index (=100)", range=yr), xaxis=dict(title="Datum"))
    else:
        for i, c in enumerate(m2_level_cols + (m2_ma_cols if show_ma3 else [])):
            ax_id = "" if i == 0 else str(i+1)
            ax_name = f"yaxis{ax_id}" if ax_id else "yaxis"
            s = pd.to_numeric(dfp[c], errors="coerce")
            fig.add_trace(go.Scatter(x=dfp["date"], y=s, mode="lines",
                                     name=c, yaxis=f"y{ax_id}" if ax_id else "y",
                                     line=dict(width=2, color=PALETTE[i % len(PALETTE)])))
            rng = padded_range(s)
            if i == 0:
                fig.update_layout(yaxis=dict(title=c, range=rng))
            elif i == 1:
                fig.update_layout(yaxis2=dict(title=c, overlaying="y", side="right", range=rng))
            else:
                side = "left" if i % 2 == 0 else "right"
                fig["layout"][ax_name] = dict(overlaying="y", side=side, range=rng, showticklabels=False, showgrid=False)
        fig.update_layout(height=380, legend=dict(orientation="h"), margin=dict(l=0,r=0,t=10,b=0), xaxis=dict(title="Datum"))
    st.plotly_chart(fig, use_container_width=True)

    last2 = dfp[m2_level_cols].tail(2)
    if len(last2) == 2:
        cols = st.columns(min(5, len(m2_level_cols)))
        for i, c in enumerate(m2_level_cols):
            l, p = float(last2[c].iloc[-1]), float(last2[c].iloc[-2])
            add_metric_chip(cols[i % len(cols)], c, l, l - p)

# ---- Geld: M2 YoY ----
m2_yoy_cols = [c for c in ["m2_yoy","m2_real_yoy"] if c in have]
if m2_yoy_cols:
    st.markdown("### M2 — YoY (%)")
    fig = go.Figure()
    for i, c in enumerate(m2_yoy_cols):
        s = pd.to_numeric(dfp[c], errors="coerce")
        fig.add_trace(go.Scatter(x=dfp["date"], y=s, mode="lines",
                                 name=c, line=dict(width=2, color=PALETTE[i % len(PALETTE)])))
    yr = padded_range(pd.concat([pd.to_numeric(dfp[c], errors="coerce") for c in m2_yoy_cols], axis=0))
    fig.update_layout(height=320, legend=dict(orientation="h"), margin=dict(l=0,r=0,t=10,b=0),
                      yaxis=dict(title="%", range=yr), xaxis=dict(title="Datum"))
    st.plotly_chart(fig, use_container_width=True)

    last2 = dfp[m2_yoy_cols].tail(2)
    if len(last2) == 2:
        cols = st.columns(min(5, len(m2_yoy_cols)))
        for i, c in enumerate(m2_yoy_cols):
            l, p = float(last2[c].iloc[-1]), float(last2[c].iloc[-2])
            add_metric_chip(cols[i % len(cols)], c, l, l - p)

# ---- Velocity: niveau ----
vel_level_cols = [c for c in ["m2_vel"] if c in have]
vel_ma_cols    = [c for c in ["m2_vel_ma3"] if c in have]
if vel_level_cols:
    st.markdown("### Velocity — niveau")
    fig = go.Figure()
    if mode.startswith("Genormaliseerd"):
        for i, c in enumerate(vel_level_cols + (vel_ma_cols if show_ma3 else [])):
            fig.add_trace(go.Scatter(x=dfp["date"], y=normalize_100(dfp[c]),
                                     mode="lines", name=c, line=dict(width=2, color=PALETTE[i % len(PALETTE)])))
        yr = padded_range(pd.concat([normalize_100(dfp[c]) for c in vel_level_cols + (vel_ma_cols if show_ma3 else [])], axis=0))
        fig.update_layout(height=320, legend=dict(orientation="h"), margin=dict(l=0,r=0,t=10,b=0),
                          yaxis=dict(title="Index (=100)", range=yr), xaxis=dict(title="Datum"))
    else:
        for i, c in enumerate(vel_level_cols + (vel_ma_cols if show_ma3 else [])):
            ax_id = "" if i == 0 else str(i+1)
            ax_name = f"yaxis{ax_id}" if ax_id else "yaxis"
            s = pd.to_numeric(dfp[c], errors="coerce")
            fig.add_trace(go.Scatter(x=dfp["date"], y=s, mode="lines",
                                     name=c, yaxis=f"y{ax_id}" if ax_id else "y",
                                     line=dict(width=2, color=PALETTE[i % len(PALETTE)])))
            rng = padded_range(s)
            if i == 0:
                fig.update_layout(yaxis=dict(title=c, range=rng))
            elif i == 1:
                fig.update_layout(yaxis2=dict(title=c, overlaying="y", side="right", range=rng))
            else:
                side = "left" if i % 2 == 0 else "right"
                fig["layout"][ax_name] = dict(overlaying="y", side=side, range=rng, showticklabels=False, showgrid=False)
        fig.update_layout(height=320, legend=dict(orientation="h"), margin=dict(l=0,r=0,t=10,b=0), xaxis=dict(title="Datum"))
    st.plotly_chart(fig, use_container_width=True)

    last2 = dfp[vel_level_cols].tail(2)
    if len(last2) == 2:
        cols = st.columns(min(5, len(vel_level_cols)))
        for i, c in enumerate(vel_level_cols):
            l, p = float(last2[c].iloc[-1]), float(last2[c].iloc[-2])
            add_metric_chip(cols[i % len(cols)], c, l, l - p)

# ---- Velocity: YoY ----
vel_yoy_cols = [c for c in ["m2_vel_yoy"] if c in have]
if vel_yoy_cols:
    st.markdown("### Velocity — YoY (%)")
    fig = go.Figure()
    s = pd.to_numeric(dfp["m2_vel_yoy"], errors="coerce")
    fig.add_trace(go.Scatter(x=dfp["date"], y=s, mode="lines",
                             name="m2_vel_yoy", line=dict(width=2, color=PALETTE[0])))
    fig.update_layout(height=280, legend=dict(orientation="h"), margin=dict(l=0,r=0,t=10,b=0),
                      yaxis=dict(title="%", range=padded_range(s)), xaxis=dict(title="Datum"))
    st.plotly_chart(fig, use_container_width=True)

    last2 = dfp[vel_yoy_cols].tail(2)
    if len(last2) == 2:
        (c1,) = st.columns(1)
        l, p = float(last2["m2_vel_yoy"].iloc[-1]), float(last2["m2_vel_yoy"].iloc[-2])
        add_metric_chip(c1, "m2_vel_yoy", l, l - p)
