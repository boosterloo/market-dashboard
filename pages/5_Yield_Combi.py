# pages/Yield_Combined.py — Gecombineerd US/EU met royale y-assen, periode-schuif, term-structure & delta-bars
# Volledig compatibel met jouw utils (utils.bq.run_query, bq_ping) en secrets tables.

import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.bq import run_query, bq_ping

# -------------------- Page config --------------------
st.set_page_config(page_title="🧯 Yield Curve Dashboard (US & EU)", layout="wide")
st.title("🧯 Yield Curve — US & EU (gecombineerd)")

# -------------------- Secrets / Views ----------------
TABLES = st.secrets.get("tables", {})
US_VIEW = TABLES.get("us_yield_view", "nth-pier-468314-p7.marketdata.us_yield_curve_enriched_v")
EU_VIEW = TABLES.get("eu_yield_view", "nth-pier-468314-p7.marketdata.eu_yield_curve_enriched_v")

# -------------------- Health check -------------------
try:
    if not bq_ping():
        st.error("Geen BigQuery-verbinding."); st.stop()
except Exception as e:
    st.error("Geen BigQuery-verbinding."); st.caption(f"Details: {e}"); st.stop()

# -------------------- Helpers ------------------------
TENOR_ORDER = [
    "y_3m","y_6m","y_1y","y_2y","y_3y","y_5y","y_7y","y_10y","y_20y","y_30y"
]

def _wide_to_long(df: pd.DataFrame, region_label: str) -> pd.DataFrame:
    """Accepteert 'enriched' views met kolommen y_* en zet naar long (date, region, tenor, value)."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["date","region","tenor","value"])
    ycols = [c for c in df.columns if str(c).startswith("y_")]
    if not ycols:
        # Probeer al long
        needed = {"date","tenor","value"}
        if needed.issubset(set(df.columns)):
            out = df.copy()
            if "region" not in out.columns:
                out["region"] = region_label
            return out[["date","region","tenor","value"]]
        return pd.DataFrame(columns=["date","region","tenor","value"])
    out = (
        df[["date"] + ycols].melt(id_vars="date", var_name="tenor", value_name="value")
        .dropna(subset=["value"]).assign(region=region_label)
    )
    return out[["date","region","tenor","value"]]

@st.cache_data(ttl=1800, show_spinner=False)
def load_us() -> pd.DataFrame:
    q = f"SELECT * FROM `{US_VIEW}` ORDER BY date"
    return run_query(q)

@st.cache_data(ttl=1800, show_spinner=False)
def load_eu() -> pd.DataFrame:
    q = f"SELECT * FROM `{EU_VIEW}` ORDER BY date"
    return run_query(q)

def _nice_tenor_label(t: str) -> str:
    # y_3m -> 3M, y_2y -> 2Y
    if not isinstance(t, str): return str(t)
    if t.startswith("y_"):
        core = t[2:]
        if core.endswith("m"): return core[:-1].upper() + "M"
        if core.endswith("y"): return core[:-1].upper() + "Y"
        return core.upper()
    return t.upper()

def _bp(x):
    try: return float(x) * 100  # percentagepunt -> basispunten
    except: return np.nan

def _axis_range(values: pd.Series, pad_pct: float, symmetric: bool=False):
    vals = pd.Series(values).dropna()
    if vals.empty:
        return None
    if symmetric:
        a = float(vals.abs().max())
        pad = a * (pad_pct/100.0)
        return (-a - pad, a + pad)
    vmin, vmax = float(vals.min()), float(vals.max())
    span = vmax - vmin
    pad = max(abs(vmin), abs(vmax), span) * (pad_pct/100.0)
    return (vmin - pad, vmax + pad)

# -------------------- Sidebar filters ----------------
with st.sidebar:
    st.header("Filters")
    st.caption("Kies de periode en selectie")

    reg_sel = st.multiselect("Regio’s", ["US","EU"], default=["US","EU"])  # beide aan

    tenor_default = ["y_2y","y_5y","y_10y","y_30y"]
    tenor_sel = st.multiselect(
        "Looptijden", TENOR_ORDER, default=tenor_default,
        help="Selecteer de looptijden die je wilt tonen."
    )

    view_mode = st.radio("Weergave", ["Overlay", "Gestapeld"], index=0, horizontal=True)

    delta_horizon = st.select_slider(
        "Delta horizon", options=[1,7,30,60,90,180,365], value=1,
        help="Verschil t.o.v. N dagen geleden."
    )
    delta_unit = st.radio("Delta eenheid", ["bp", "%-punt"], index=0, horizontal=True)

    # Royale y-assen: default 60% extra marge
    ypad = st.slider("Y-as marge (royale ruimte, in %)", 20, 120, 60, step=5,
                     help="Extra witruimte boven/onder de datarange.")

# -------------------- Load data ----------------------
with st.spinner("US/EU yield data laden…"):
    df_us = load_us()
    df_eu = load_eu()

if (df_us is None or df_us.empty) and (df_eu is None or df_eu.empty):
    st.warning("Geen data gevonden in de opgegeven views."); st.stop()

# Zorg voor DATE dtype
for df in (df_us, df_eu):
    if df is not None and not df.empty and "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.date

# Maak long
us_long = _wide_to_long(df_us, "US") if df_us is not None else pd.DataFrame(columns=["date","region","tenor","value"])
eu_long = _wide_to_long(df_eu, "EU") if df_eu is not None else pd.DataFrame(columns=["date","region","tenor","value"])

df_all = pd.concat([us_long, eu_long], ignore_index=True)
if df_all.empty:
    st.warning("Geen bruikbare (tenor,value) data gevonden."); st.stop()

# Globale periode slider (op basis van beide)
min_d, max_d = df_all["date"].min(), df_all["date"].max()
# default window 365 dagen
default_days = 365
start_default = max(min_d, max_d - timedelta(days=default_days))
start_d, end_d = st.slider(
    "Periode", min_value=min_d, max_value=max_d, value=(start_default, max_d),
    format="YYYY-MM-DD"
)
mask = (df_all["date"] >= start_d) & (df_all["date"] <= end_d)
df_all = df_all.loc[mask].copy()

# Filter regio & tenors
if reg_sel:
    df_all = df_all[df_all["region"].isin(reg_sel)]
if tenor_sel:
    df_all = df_all[df_all["tenor"].isin(tenor_sel)]
else:
    st.info("Selecteer minimaal één looptijd."); st.stop()

# Sorteer tenors netjes
df_all["tenor"] = pd.Categorical(df_all["tenor"], categories=TENOR_ORDER, ordered=True)
df_all = df_all.sort_values(["region","tenor","date"]).reset_index(drop=True)

# -------------------- KPI blok -----------------------
col1, col2, col3, col4 = st.columns(4)

def _last_val(df, region, tenor):
    m = (df["region"]==region) & (df["tenor"]==tenor)
    sub = df.loc[m].dropna(subset=["value"])
    return np.nan if sub.empty else float(sub.iloc[-1]["value"])

lab = _nice_tenor_label(tenor_sel[0]) if tenor_sel else "—"
us_last = _last_val(df_all, "US", tenor_sel[0])
eu_last = _last_val(df_all, "EU", tenor_sel[0])
with col1:
    st.metric(f"Laatst ({lab}) — US", f"{us_last:.2f}%" if not np.isnan(us_last) else "—")
with col2:
    st.metric(f"Laatst ({lab}) — EU", f"{eu_last:.2f}%" if not np.isnan(eu_last) else "—")
with col3:
    if not np.isnan(us_last) and not np.isnan(eu_last):
        spread = us_last - eu_last
        st.metric(f"US–EU spread ({lab})", f"{spread:.2f} pp")
    else:
        st.metric(f"US–EU spread ({lab})", "—")
with col4:
    st.caption(f"Periode: {start_d} → {end_d}  |  Δ-horizon: {delta_horizon}d")

st.divider()

# -------------------- Term-structure (laatste datum) --------------------
st.subheader("Rente per looptijd (laatste observatie)")

left, right = st.columns([2,1], gap="large")

with left:
    fig_ts = go.Figure()
    yvals = []
    for region in sorted(df_all["region"].unique()):
        sub = df_all[df_all["region"]==region]
        if sub.empty: continue
        last_d = sub["date"].max()
        ts = sub[sub["date"]==last_d].sort_values("tenor")
        if ts.empty: continue
        yvals.append(ts["value"])  # voor y-as range
        fig_ts.add_trace(go.Scatter(
            x=[_nice_tenor_label(t) for t in ts["tenor"]],
            y=ts["value"], mode="lines+markers", name=region
        ))
    if yvals:
        rng = _axis_range(pd.concat(yvals, ignore_index=True), pad_pct=ypad, symmetric=False)
        if rng:
            fig_ts.update_yaxes(range=list(rng))
    fig_ts.update_layout(height=440, margin=dict(l=10,r=10,t=10,b=10), legend_title_text="Regio")
    st.plotly_chart(fig_ts, use_container_width=True)

with right:
    st.subheader("US–EU (laatste) per looptijd")
    last_all = []
    for region in ["US","EU"]:
        sub = df_all[df_all["region"]==region]
        if sub.empty: continue
        last_d = sub["date"].max()
        sub = sub[sub["date"]==last_d]
        p = sub.pivot_table(index="tenor", values="value", aggfunc="last")
        p.columns = [region]
        p["tenor"] = p.index
        last_all.append(p.reset_index(drop=True))
    if len(last_all)==2:
        comb = last_all[0].merge(last_all[1], on="tenor", how="inner")
        comb = comb[comb["tenor"].isin(tenor_sel)].sort_values("tenor")
        comb["spread_pp"] = comb["US"] - comb["EU"]
        fig_sp = go.Figure()
        fig_sp.add_trace(go.Bar(
            x=[_nice_tenor_label(t) for t in comb["tenor"]], y=comb["spread_pp"], name="US–EU (pp)"
        ))
        rng = _axis_range(comb["spread_pp"], pad_pct=ypad, symmetric=True)
        if rng:
            fig_sp.update_yaxes(range=list(rng))
        fig_sp.update_layout(height=440, margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig_sp, use_container_width=True)
    else:
        st.info("Onvoldoende data om US–EU spread te tonen.")

st.divider()

# -------------------- Tijdreeks per looptijd --------------------
st.subheader("Tijdreeks per looptijd")

if not tenor_sel:
    st.info("Selecteer minimaal één looptijd."); st.stop()

n_rows = len(tenor_sel)
subplot_titles = [f"{_nice_tenor_label(t)}" for t in tenor_sel]
fig_ts2 = make_subplots(rows=n_rows, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                        subplot_titles=subplot_titles)

# Bereid ranges per subplot
for i, tenor in enumerate(tenor_sel, start=1):
    sub = df_all[df_all["tenor"]==tenor]
    if sub.empty: continue
    # traces
    for region in sorted(sub["region"].unique()):
        s2 = sub[sub["region"]==region].dropna(subset=["value"])
        fig_ts2.add_trace(
            go.Scatter(x=s2["date"], y=s2["value"], mode="lines",
                       name=f"{region} — {_nice_tenor_label(tenor)}",
                       showlegend=(i==1)),
            row=i, col=1
        )
    # axis range (royale)
    rng = _axis_range(sub["value"], pad_pct=ypad, symmetric=False)
    if rng:
        fig_ts2.update_yaxes(range=list(rng), row=i, col=1)

fig_ts2.update_layout(height=260*n_rows + 60, margin=dict(l=10,r=10,t=40,b=10))
st.plotly_chart(fig_ts2, use_container_width=True)

st.divider()

# -------------------- Delta staafdiagrammen --------------------
st.subheader(f"Delta t.o.v. {delta_horizon}d geleden")

def compute_delta(df: pd.DataFrame, n: int, unit: str) -> pd.DataFrame:
    df = df.sort_values(["region","tenor","date"]).copy()
    df["value_prev"] = df.groupby(["region","tenor"])["value"].shift(n)
    df["delta_pp"] = df["value"] - df["value_prev"]
    if unit == "bp":
        df["delta_plot"] = df["delta_pp"].map(_bp)
        df["unit_lbl"] = "bp"
    else:
        df["delta_plot"] = df["delta_pp"]
        df["unit_lbl"] = "pp"
    return df

df_delta = compute_delta(df_all, delta_horizon, delta_unit)

# Laatste datum per regio/tenor binnen periode tonen als staaf
bars = []
for region in sorted(df_delta["region"].unique()):
    sub = df_delta[df_delta["region"]==region].dropna(subset=["delta_plot"])
    if sub.empty: 
        continue
    last_per_tenor = sub.sort_values("date").groupby("tenor").tail(1)
    last_per_tenor = last_per_tenor[last_per_tenor["tenor"].isin(tenor_sel)].sort_values("tenor")
    if last_per_tenor.empty:
        continue
    bars.append((region, last_per_tenor))

fig_d = go.Figure()
all_deltas = []
for region, frame in bars:
    fig_d.add_trace(go.Bar(
        x=[_nice_tenor_label(t) for t in frame["tenor"]],
        y=frame["delta_plot"],
        name=f"{region} Δ{delta_horizon}d ({frame.iloc[0]['unit_lbl']})"
    ))
    all_deltas.append(frame["delta_plot"])

rng = _axis_range(pd.concat(all_deltas, ignore_index=True) if all_deltas else pd.Series(dtype=float),
                  pad_pct=ypad, symmetric=True)
if rng:
    fig_d.update_yaxes(range=list(rng))

fig_d.update_layout(barmode=("relative" if view_mode=="Gestapeld" else "group"),
                    height=440, margin=dict(l=10,r=10,t=10,b=10))
st.plotly_chart(fig_d, use_container_width=True)

st.caption("Tip: verander ‘Delta horizon’ voor 1d/7d/30d… en schakel tussen Overlay/Gestapeld voor visuele vergelijking. Y-assen schalen royaal mee.")
