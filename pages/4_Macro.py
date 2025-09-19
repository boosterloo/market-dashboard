# pages/4_Macro.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
import plotly.graph_objects as go
from google.api_core.exceptions import NotFound, BadRequest

from utils.bq import run_query, bq_ping

st.set_page_config(page_title="📊 Macro (grafieken + deltas)", layout="wide")
st.title("Macro-dashboard")

# ========= Config =========
DEFAULT_MACRO_VIEW = "nth-pier-468314-p7.marketdata.macro_series_wide_monthly_fill_v"
MACRO_VIEW = st.secrets.get("tables", {}).get("macro_view", DEFAULT_MACRO_VIEW)

PALETTE  = ["#2563eb","#0891b2","#dc2626","#16a34a","#9333ea","#f59e0b",
            "#0ea5e9","#ef4444","#14b8a6","#f97316","#64748b","#d946ef"]

COL_POS = "#16a34a"  # groen
COL_NEG = "#dc2626"  # rood

# ========= Helpers =========
def best_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns: return c
        if c.lower() in lower: return lower[c.lower()]
    return None

def padded_range(s: pd.Series, pad_ratio: float = 0.05):
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty: return None
    lo, hi = float(s.min()), float(s.max())
    if not np.isfinite(lo) or not np.isfinite(hi):
        return None
    if lo == hi:
        pad = abs(lo) * pad_ratio if lo != 0 else 1.0
        return [lo - pad, hi + pad]
    span = hi - lo
    pad = span * pad_ratio
    return [lo - pad, hi + pad]

def normalize_100(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    base = s.dropna().iloc[0] if s.notna().any() else np.nan
    return s / base * 100.0 if pd.notna(base) and base != 0 else s

def yoy_from_index(s: pd.Series) -> pd.Series:
    s_num = pd.to_numeric(s, errors="coerce")
    if s_num.dropna().between(-10, 60).mean() > 0.8 and s_num.notna().sum() > 6:
        return s_num
    return (s_num / s_num.shift(12) - 1.0) * 100.0

def add_metric_chip(col, title: str, value: float, delta: float, unit: str = ""):
    color = COL_POS if delta > 0 else (COL_NEG if delta < 0 else "#6b7280")
    u = f" {unit}" if unit else ""
    col.markdown(f"**{title}**")
    col.markdown(
        f"""
        <div style="display:flex;gap:12px;align-items:baseline;">
          <div style="font-size:1.4rem;font-weight:700;">{value:.2f}{u}</div>
          <div style="padding:2px 8px;border-radius:999px;background:{color}20;color:{color};font-weight:700;">
            {delta:+.2f}{u}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def info_card(title: str, bullets: list[str], tone: str = "neutral"):
    colors = {"neutral":("#374151","#9CA3AF"), "bull":("#065f46","#10b981"), "bear":("#7f1d1d","#ef4444")}
    border, accent = colors.get(tone, colors["neutral"])
    items = "".join([f"<li>{b}</li>" for b in bullets])
    st.markdown(
        f"""
        <div style="border:1px solid {border}; border-radius:12px; padding:10px 14px; margin:6px 0;">
          <div style="font-weight:700; color:{accent}; margin-bottom:4px;">{title}</div>
          <ul style="margin:0 0 0 1.2rem; padding:0;">{items}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

def dynamic_conclusion(changes: dict[str, float]) -> tuple[str, str]:
    score = 0.0
    notes = []
    for k, v in changes.items():
        if v is None or not np.isfinite(v): 
            continue
        k_low = k.lower()
        if any(x in k_low for x in ["cpi","pce","ppi"]):
            score -= np.sign(v) * min(abs(v), 2.0); notes.append(f"{k}: {v:+.2f}pp")
        elif any(x in k_low for x in ["unemployment","claims"]):
            score -= np.sign(v) * min(abs(v), 2.0); notes.append(f"{k}: {v:+.2f}")
        elif any(x in k_low for x in ["payroll","retail","industrial","housing","starts"]):
            score += np.sign(v) * min(abs(v), 2.0); notes.append(f"{k}: {v:+.2f}")
    tone = "neutral"; 
    if score >= 1.0: tone = "bull"
    elif score <= -1.0: tone = "bear"
    summary = " • ".join(notes) if notes else "Weinig richtinggevende verandering."
    return tone, summary

def tiny_delta_chart(x: pd.Series, y: pd.Series, name: str):
    d = pd.to_numeric(y, errors="coerce").diff()
    colors = [COL_POS if (pd.notna(v) and v >= 0) else COL_NEG for v in d]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=x, y=d, name=f"Δ {name}", marker=dict(color=colors)))
    fig.update_layout(height=160, margin=dict(l=0,r=0,t=4,b=0),
                      xaxis=dict(title=""), yaxis=dict(title="Δ per stap", range=padded_range(d)))
    st.plotly_chart(fig, use_container_width=True)

# ========= BigQuery health =========
try:
    with st.spinner("BigQuery check…"):
        if not bq_ping():
            st.error("Geen BigQuery-verbinding. Controleer Secrets ([gcp_service_account])."); st.stop()
except Exception as e:
    st.error("Geen BigQuery-verbinding. Controleer Secrets ([gcp_service_account])."); st.caption(f"Details: {e}"); st.stop()

@st.cache_data(ttl=1800)
def load_macro() -> pd.DataFrame:
    df = run_query(f"SELECT * FROM `{MACRO_VIEW}` ORDER BY date")
    if "date" not in df.columns:
        raise ValueError(f"Kolom 'date' niet aanwezig in view {MACRO_VIEW}")
    return df

# ========= Data laden =========
try:
    df = load_macro()
except NotFound:
    st.error(f"View niet gevonden: `{MACRO_VIEW}`. Pas [tables].macro_view in secrets.toml aan."); st.stop()
except (BadRequest, ValueError) as e:
    st.error(f"Macro-view probleem: {e}"); st.stop()

if not np.issubdtype(df["date"].dtype, np.datetime64):
    df["date"] = pd.to_datetime(df["date"])

# ======== Column mapping ========
colmap = {
    "cpi_idx": best_col(df, ["cpi_all","cpi","cpi_index"]), "cpi_core_idx": best_col(df, ["cpi_core","cpi_core_index"]),
    "pce_idx": best_col(df, ["pce_all","pce","pce_index"]),
    "cpi_yoy": best_col(df, ["cpi_yoy","cpi_all_yoy"]), "cpi_core_yoy": best_col(df, ["cpi_core_yoy"]),
    "pce_yoy": best_col(df, ["pce_yoy","pce_all_yoy"]),
    "ppi_yoy": best_col(df, ["ppi_yoy","ppi_all_yoy","producer_price_yoy"]),
    "ppi_core_yoy": best_col(df, ["ppi_core_yoy"]),
    "ppi_idx": best_col(df, ["ppi_all","ppi","producer_price_index"]),
    "ppi_core_idx": best_col(df, ["ppi_core"]),
    "industrial_production": best_col(df, ["industrial_production","ip_index"]),
    "retail_sales": best_col(df, ["retail_sales","retail"]),
    "housing_starts": best_col(df, ["housing_starts","housing","starts"]),
    "unemployment": best_col(df, ["unemployment","unemp_rate"]),
    "payrolls": best_col(df, ["payrolls","nonfarm_payrolls"]),
    "init_claims": best_col(df, ["init_claims","initial_claims"]),
    "m2": best_col(df, ["m2"]), "m2_ma3": best_col(df, ["m2_ma3"]),
    "m2_real": best_col(df, ["m2_real"]), "m2_real_ma3": best_col(df, ["m2_real_ma3"]),
    "m2_yoy": best_col(df, ["m2_yoy"]), "m2_real_yoy": best_col(df, ["m2_real_yoy"]),
    "m2_vel": best_col(df, ["m2_vel"]), "m2_vel_ma3": best_col(df, ["m2_vel_ma3"]),
    "m2_vel_yoy": best_col(df, ["m2_vel_yoy"]),
    "maybe_ppi_in_indprod": best_col(df, ["ind_production","ppi_ind_production","ind_production_ppi"]),
}

# ======== CPI/PCE/PPI naar YoY % ========
df_pct = df.copy()
df_pct["cpi_headline"] = pd.to_numeric(df[colmap["cpi_yoy"]], errors="coerce") if colmap["cpi_yoy"] else (yoy_from_index(df[colmap["cpi_idx"]]) if colmap["cpi_idx"] else np.nan)
df_pct["cpi_core"]     = pd.to_numeric(df[colmap["cpi_core_yoy"]], errors="coerce") if colmap["cpi_core_yoy"] else (yoy_from_index(df[colmap["cpi_core_idx"]]) if colmap["cpi_core_idx"] else np.nan)
df_pct["pce_headline"] = pd.to_numeric(df[colmap["pce_yoy"]], errors="coerce") if colmap["pce_yoy"] else (yoy_from_index(df[colmap["pce_idx"]]) if colmap["pce_idx"] else np.nan)

ppi_headline_series = None
if colmap["ppi_yoy"]: ppi_headline_series = pd.to_numeric(df[colmap["ppi_yoy"]], errors="coerce")
elif colmap["ppi_idx"]: ppi_headline_series = yoy_from_index(df[colmap["ppi_idx"]])
elif colmap["maybe_ppi_in_indprod"]: pass
if ppi_headline_series is not None: df_pct["ppi_headline"] = ppi_headline_series
if colmap["ppi_core_yoy"]: df_pct["ppi_core"] = pd.to_numeric(df[colmap["ppi_core_yoy"]], errors="coerce")
elif colmap["ppi_core_idx"]: df_pct["ppi_core"] = yoy_from_index(df[colmap["ppi_core_idx"]])

# ========= UI =========
all_min, all_max = df["date"].min().date(), df["date"].max().date()
default_start = all_max - timedelta(days=5*365)
start, end = st.slider("Periode", min_value=all_min, max_value=all_max, value=(default_start, all_max), format="YYYY-MM-DD")

colA, colB = st.columns([1,1])
with colA:
    view_mode = st.radio("Weergave overige grafieken", ["Genormaliseerd (=100)", "Eigen schaal per serie"], horizontal=True, index=0)
with colB:
    daily_interp = st.checkbox("Interpoleer naar dag (experimenteel)", value=False, help="Lineaire verdeling van maandpunten naar dagelijkse waarden voor beter zicht op dagelijkse Δ.")

dfp = df_pct[(df_pct["date"].dt.date >= start) & (df_pct["date"].dt.date <= end)].copy()

def maybe_daily(dfin: pd.DataFrame) -> pd.DataFrame:
    if not daily_interp or dfin.empty: return dfin
    d = dfin.set_index("date").asfreq("D")
    d = d.interpolate(method="time").reset_index().rename(columns={"index":"date"})
    return d

# ========= Inflatie (CPI/PCE + PPI) =========
st.subheader("Inflatie (YoY %, CPI/PCE + PPI)")
info_card("Hoe lees je dit?", [
  "🔵 CPI (headline) & core + 🔴 PCE (headline) links (YoY %).",
  "🟣 PPI (headline/core) rechts (YoY %) met eigen schaal.",
  "Hoger = meer prijsdruk; lager = desinflatie."
])

ppi_present = any(c in dfp.columns for c in ["ppi_headline","ppi_core"])
if not ppi_present:
    numeric_cols = [c for c in df.columns if c != "date"]
    fallback_ppi = st.selectbox("Geen PPI-kolom gedetecteerd — kies optioneel een kolom om als PPI (YoY %) te tonen:",
                                options=["(geen)"] + numeric_cols, index=0)
    if fallback_ppi != "(geen)":
        try:
            tmp = df[["date", fallback_ppi]].copy()
            tmp["ppi_headline"] = yoy_from_index(tmp[fallback_ppi])
            dfp = dfp.merge(tmp[["date","ppi_headline"]], on="date", how="left")
        except Exception:
            st.warning(f"Kon YoY niet afleiden uit '{fallback_ppi}'.")

infl_left = [c for c in ["cpi_headline","cpi_core","pce_headline"] if c in dfp.columns]
infl_right = [c for c in ["ppi_headline","ppi_core"] if c in dfp.columns]

if not infl_left and not infl_right:
    st.info("Geen CPI/PCE/PPI kolommen gevonden.")
else:
    dfi_cols = ["date"] + infl_left + infl_right
    dfi = maybe_daily(dfp[dfi_cols].copy())
    fig = go.Figure()
    stack_left = []
    for i, c in enumerate(infl_left):
        s = pd.to_numeric(dfi[c], errors="coerce"); stack_left.append(s)
        fig.add_trace(go.Scatter(x=dfi["date"], y=s, mode="lines", name=c, yaxis="y",
                                 line=dict(width=2, color=PALETTE[i % len(PALETTE)])))
    y_left = padded_range(pd.concat(stack_left, axis=0)) if stack_left else None
    stack_right = []
    for j, c in enumerate(infl_right):
        s = pd.to_numeric(dfi[c], errors="coerce"); stack_right.append(s)
        fig.add_trace(go.Scatter(x=dfi["date"], y=s, mode="lines", name=c, yaxis="y2",
                                 line=dict(width=2, dash="dash", color=PALETTE[(j+4) % len(PALETTE)])))
    y_right = padded_range(pd.concat(stack_right, axis=0)) if stack_right else None
    fig.update_layout(height=460, legend=dict(orientation="h"), margin=dict(l=0,r=0,t=10,b=0),
                      xaxis=dict(title="Datum"),
                      yaxis=dict(title="YoY % (CPI/PCE)", range=y_left),
                      yaxis2=dict(title="YoY % (PPI)", overlaying="y", side="right", range=y_right))
    st.plotly_chart(fig, use_container_width=True)

    series_all = infl_left + infl_right
    last2 = dfi[series_all].dropna().tail(2)
    if len(last2) == 2:
        changes = {c: float(last2[c].iloc[-1] - last2[c].iloc[-2]) for c in series_all}
        tone, summary = dynamic_conclusion(changes)
        info_card("Dynamische conclusie (inflatie & producentenprijzen)", [summary], tone=tone)
    if series_all:
        cols = st.columns(min(5, len(series_all)))
        for i, c in enumerate(series_all):
            ser = dfi[c].dropna()
            if len(ser) >= 2:
                l, p = float(ser.iloc[-1]), float(ser.iloc[-2])
                add_metric_chip(cols[i % len(cols)], c, l, l - p, unit="%")
    st.markdown("**Dagelijkse delta per serie (pp)**")
    for c in series_all:
        st.caption(f"Δ {c}"); tiny_delta_chart(dfi["date"], dfi[c], c)

st.divider()

# ========= Activiteit =========
st.subheader("Activiteit (IP / Retail / Housing)")
act_cols = [c for c in ["industrial_production","retail_sales","housing_starts"] if c in dfp.columns]
if "industrial_production" not in dfp.columns and colmap["maybe_ppi_in_indprod"] and "ppi_headline" not in dfp.columns:
    try:
        dfp = dfp.merge(df[["date", colmap["maybe_ppi_in_indprod"]]], on="date", how="left")
        dfp = dfp.rename(columns={colmap["maybe_ppi_in_indprod"]: "industrial_production"})
    except Exception:
        st.warning(f"Kon fallback-kolom '{colmap['maybe_ppi_in_indprod']}' niet samenvoegen.")
act_cols = [c for c in ["industrial_production","retail_sales","housing_starts"] if c in dfp.columns]
if len(act_cols) == 0:
    st.info("Geen activiteit-kolommen gevonden (industrial_production / retail_sales / housing_starts).")
else:
    select_cols = ["date"] + [c for c in act_cols if c in dfp.columns]
    dfa = maybe_daily(dfp[select_cols].copy())
    fig2 = go.Figure()
    if view_mode.startswith("Genormaliseerd"):
        stack = []
        for i, c in enumerate(act_cols):
            if c not in dfa.columns: continue
            s = normalize_100(dfa[c]); stack.append(s)
            fig2.add_trace(go.Scatter(x=dfa["date"], y=s, mode="lines", name=c,
                                      line=dict(width=2, color=PALETTE[i % len(PALETTE)])))
        yr = padded_range(pd.concat(stack, axis=0)) if stack else None
        fig2.update_layout(height=420, legend=dict(orientation="h"), margin=dict(l=0,r=0,t=10,b=0),
                           yaxis=dict(title="Index (=100)", range=yr), xaxis=dict(title="Datum"))
    else:
        stacks = []
        for i, c in enumerate(act_cols):
            if c not in dfa.columns: continue
            s = pd.to_numeric(dfa[c], errors="coerce"); stacks.append(s)
            fig2.add_trace(go.Scatter(x=dfa["date"], y=s, mode="lines", name=c,
                                      line=dict(width=2, color=PALETTE[i % len(PALETTE)])))
        yr = padded_range(pd.concat(stacks, axis=0)) if stacks else None
        fig2.update_layout(height=420, legend=dict(orientation="h"), margin=dict(l=0,r=0,t=10,b=0),
                           xaxis=dict(title="Datum"), yaxis=dict(title="Niveau", range=yr))
    st.plotly_chart(fig2, use_container_width=True)

    last2 = dfa[[c for c in act_cols if c in dfa.columns]].dropna().tail(2)
    if len(last2) == 2 and last2.shape[1] > 0:
        changes = {c: float(last2[c].iloc[-1] - last2[c].iloc[-2]) for c in last2.columns}
        tone, summary = dynamic_conclusion(changes)
        info_card("Dynamische conclusie (activiteit)", [summary], tone=tone)
    st.markdown("**Dagelijkse delta per serie**")
    for c in act_cols:
        if c in dfa.columns:
            st.caption(f"Δ {c}"); tiny_delta_chart(dfa["date"], dfa[c], c)

st.divider()

# ========= Arbeidsmarkt (payrolls altijd op y2) =========
st.subheader("Arbeidsmarkt")
arb_cols = [c for c in ["unemployment","payrolls","init_claims"] if c in dfp.columns]
if arb_cols:
    dfl = maybe_daily(dfp[["date"] + arb_cols].copy())
    fig3 = go.Figure()
    left_cols  = [c for c in ["unemployment","init_claims"] if c in arb_cols]
    right_col  = "payrolls" if "payrolls" in arb_cols else None

    if view_mode.startswith("Genormaliseerd"):
        # Links: genormaliseerde unemployment/claims. Rechts: genormaliseerde payrolls (eigen range).
        left_stack = []
        for i, c in enumerate(left_cols):
            s = normalize_100(dfl[c]); left_stack.append(s)
            fig3.add_trace(go.Scatter(x=dfl["date"], y=s, mode="lines", name=c, yaxis="y",
                                      line=dict(width=2, color=PALETTE[i % len(PALETTE)])))
        y_left = padded_range(pd.concat(left_stack, axis=0)) if left_stack else None
        y_right = None
        if right_col:
            s_r = normalize_100(dfl[right_col]); y_right = padded_range(s_r)
            fig3.add_trace(go.Scatter(x=dfl["date"], y=s_r, mode="lines", name=right_col, yaxis="y2",
                                      line=dict(width=3, dash="dash", color=PALETTE[3])))
        fig3.update_layout(height=400, legend=dict(orientation="h"), margin=dict(l=0,r=0,t=10,b=0),
                           xaxis=dict(title="Datum"),
                           yaxis=dict(title="Index (=100) — werkloosheid/claims", range=y_left),
                           yaxis2=dict(title="Index (=100) — payrolls", overlaying="y", side="right", range=y_right))
    else:
        # Links: niveaus unemployment/claims. Rechts: niveau payrolls met eigen range.
        left_stack = []
        for i, c in enumerate(left_cols):
            s = pd.to_numeric(dfl[c], errors="coerce"); left_stack.append(s)
            fig3.add_trace(go.Scatter(x=dfl["date"], y=s, mode="lines", name=c, yaxis="y",
                                      line=dict(width=2, color=PALETTE[i % len(PALETTE)])))
        y_left = padded_range(pd.concat(left_stack, axis=0)) if left_stack else None
        y_right = None
        if right_col:
            s_r = pd.to_numeric(dfl[right_col], errors="coerce"); y_right = padded_range(s_r)
            fig3.add_trace(go.Scatter(x=dfl["date"], y=s_r, mode="lines", name=right_col, yaxis="y2",
                                      line=dict(width=3, dash="dash", color=PALETTE[3])))
        fig3.update_layout(height=420, legend=dict(orientation="h"), margin=dict(l=0,r=0,t=10,b=0),
                           xaxis=dict(title="Datum"),
                           yaxis=dict(title="Niveau — werkloosheid/claims", range=y_left),
                           yaxis2=dict(title="Niveau — payrolls", overlaying="y", side="right", range=y_right))
    st.plotly_chart(fig3, use_container_width=True)

    last2 = dfl[[c for c in arb_cols if c in dfl.columns]].dropna().tail(2)
    if len(last2) == 2 and last2.shape[1] > 0:
        changes = {c: float(last2[c].iloc[-1] - last2[c].iloc[-2]) for c in last2.columns}
        tone, summary = dynamic_conclusion(changes)
        info_card("Dynamische conclusie (arbeidsmarkt)", [summary], tone=tone)

    st.markdown("**Dagelijkse delta per serie**")
    for c in arb_cols:
        st.caption(f"Δ {c}"); tiny_delta_chart(dfl["date"], dfl[c], c)
else:
    st.info("Geen arbeidsmarkt-kolommen gevonden (unemployment/payrolls/init_claims).")

st.divider()

# ========= Geldhoeveelheid & velocity =========
st.subheader("Geldhoeveelheid en velocity")
have = set(dfp.columns)
m2_level_cols = [c for c in ["m2","m2_real"] if c in have]
m2_ma_cols    = [c for c in ["m2_ma3","m2_real_ma3"] if c in have]
m2_yoy_cols   = [c for c in ["m2_yoy","m2_real_yoy"] if c in have]
vel_level_cols= [c for c in ["m2_vel"] if c in have]
vel_ma_cols   = [c for c in ["m2_vel_ma3"] if c in have]
vel_yoy_cols  = [c for c in ["m2_vel_yoy"] if c in have]

def plot_block(title: str, cols: list[str], ma_cols: list[str] = None):
    if not cols: return
    st.markdown(f"### {title}")
    dfm = maybe_daily(dfp[["date"] + cols + (ma_cols or [])].copy())
    fig = go.Figure()
    if view_mode.startswith("Genormaliseerd"):
        stack = []
        series_cols = cols + (ma_cols or [])
        for i, c in enumerate(series_cols):
            s = normalize_100(dfm[c]); stack.append(s)
            fig.add_trace(go.Scatter(x=dfm["date"], y=s, mode="lines", name=c,
                                     line=dict(width=2, color=PALETTE[i % len(PALETTE)])))
        yr = padded_range(pd.concat(stack, axis=0))
        fig.update_layout(height=360, legend=dict(orientation="h"), margin=dict(l=0,r=0,t=10,b=0),
                          yaxis=dict(title="Index (=100)", range=yr), xaxis=dict(title="Datum"))
    else:
        series_cols = cols + (ma_cols or []); stacks = []
        for i, c in enumerate(series_cols):
            s = pd.to_numeric(dfm[c], errors="coerce"); stacks.append(s)
            fig.add_trace(go.Scatter(x=dfm["date"], y=s, mode="lines", name=c,
                                     line=dict(width=2, color=PALETTE[i % len(PALETTE)])))
        yr = padded_range(pd.concat(stacks, axis=0))
        fig.update_layout(height=360, legend=dict(orientation="h"), margin=dict(l=0,r=0,t=10,b=0),
                          xaxis=dict(title="Datum"), yaxis=dict(title="Niveau", range=yr))
    st.plotly_chart(fig, use_container_width=True)

    last2 = dfm[cols].dropna().tail(2)
    if len(last2) == 2:
        changes = {c: float(last2[c].iloc[-1] - last2[c].iloc[-2]) for c in cols}
        tone, summary = dynamic_conclusion(changes)
        info_card(f"Dynamische conclusie ({title})", [summary], tone=tone)

    st.markdown("**Dagelijkse delta per serie**")
    for c in cols:
        st.caption(f"Δ {c}"); tiny_delta_chart(dfm["date"], dfm[c], c)

plot_block("M2 — niveaus", m2_level_cols, m2_ma_cols)

if m2_yoy_cols:
    st.markdown("### M2 — YoY (%)")
    dfm = maybe_daily(dfp[["date"] + m2_yoy_cols].copy())
    fig = go.Figure(); stack = []
    for i, c in enumerate(m2_yoy_cols):
        s = pd.to_numeric(dfm[c], errors="coerce"); stack.append(s)
        fig.add_trace(go.Scatter(x=dfm["date"], y=s, mode="lines", name=c,
                                 line=dict(width=2, color=PALETTE[i % len(PALETTE)])))
    yr = padded_range(pd.concat(stack, axis=0))
    fig.update_layout(height=320, legend=dict(orientation="h"), margin=dict(l=0,r=0,t=10,b=0),
                      yaxis=dict(title="%", range=yr), xaxis=dict(title="Datum"))
    st.plotly_chart(fig, use_container_width=True)

    last2 = dfm[m2_yoy_cols].dropna().tail(2)
    if len(last2) == 2:
        changes = {c: float(last2[c].iloc[-1] - last2[c].iloc[-2]) for c in m2_yoy_cols}
        tone, summary = dynamic_conclusion(changes)
        info_card("Dynamische conclusie (M2 YoY)", [summary], tone=tone)

    st.markdown("**Dagelijkse delta per serie (pp)**")
    for c in m2_yoy_cols:
        st.caption(f"Δ {c}"); tiny_delta_chart(dfm["date"], dfm[c], c)

plot_block("Velocity — niveau", vel_level_cols, vel_ma_cols)

if vel_yoy_cols:
    st.markdown("### Velocity — YoY (%)")
    dfv = maybe_daily(dfp[["date"] + vel_yoy_cols].copy())
    s = pd.to_numeric(dfv[vel_yoy_cols[0]], errors="coerce")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dfv["date"], y=s, mode="lines", name=vel_yoy_cols[0],
                             line=dict(width=2, color=PALETTE[0])))
    fig.update_layout(height=280, legend=dict(orientation="h"), margin=dict(l=0,r=0,t=10,b=0),
                      yaxis=dict(title="%", range=padded_range(s)), xaxis=dict(title="Datum"))
    st.plotly_chart(fig, use_container_width=True)

    last2 = dfv[vel_yoy_cols].dropna().tail(2)
    if len(last2) == 2:
        changes = {vel_yoy_cols[0]: float(last2[vel_yoy_cols[0]].iloc[-1] - last2[vel_yoy_cols[0]].iloc[-2])}
        tone, summary = dynamic_conclusion(changes)
        info_card("Dynamische conclusie (Velocity YoY)", [summary], tone=tone)

    st.markdown("**Dagelijkse delta (pp)**")
    tiny_delta_chart(dfv["date"], dfv[vel_yoy_cols[0]], vel_yoy_cols[0])
