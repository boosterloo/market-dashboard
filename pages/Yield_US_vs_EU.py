# pages/Yield_US_EU_Compare.py — US vs EU vergelijking (enriched views)
# Features:
# - Gedeelde periode & peildata (US/EU apples-to-apples)
# - Term structure overlay + Δ-curve (US–EU, bp) ⇒ SAMENGEVOEGD MET DUBBELE Y-AS
# - KPI’s: 2Y/10Y/10Y–2Y + differentials
# - Tijdreeks: levels (2Y/10Y), spreads (10Y–2Y) + differential bars
# - Deltas: 1d/7d/30d — histogrammen (bp & %) + tijdreeks + US–EU differential
# - Rolling band: Δ(US–EU) met 90d μ ± 1σ (levels of z-scores)
# - Z-scores toggle (rolling) voor eerlijke cross-regio benchmark
# - Correlatiematrix Δ1d (US & EU), Lead/Lag heatmap, Δ-scatter met beta/R²
# - Optioneel: EU fragmentatie-spreads (OAT–Bund, BTP–Bund) als de kolommen bestaan

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.bq import run_query

st.set_page_config(page_title="US vs EU Yield — Compare", layout="wide")
st.title("🌍 US vs EU — Yield Curve Compare")

PROJECT_ID = st.secrets["gcp_service_account"]["project_id"]
TABLES     = st.secrets.get("tables", {})

US_VIEW = TABLES.get("us_yield_view", f"{PROJECT_ID}.marketdata.us_yield_curve_enriched_v")
EU_VIEW = TABLES.get("eu_yield_view", f"{PROJECT_ID}.marketdata.eu_yield_curve_enriched_v")

# ============== helpers ==============
def list_columns(fqtn: str) -> set[str]:
    proj, dset, tbl = fqtn.split(".")
    sql = f"""
    SELECT column_name
    FROM `{proj}.{dset}.INFORMATION_SCHEMA.COLUMNS`
    WHERE table_name = @tbl
    """
    dfc = run_query(sql, params={"tbl": tbl}, timeout=30)
    return set([c.lower() for c in dfc["column_name"].astype(str)])

def have(cols: set[str], c: str) -> bool:
    return c.lower() in cols

def pick_y2y(cols: set[str]) -> str | None:
    return "y_2y_synth" if have(cols,"y_2y_synth") else ("y_2y" if have(cols,"y_2y") else None)

def load_view(fqtn: str) -> pd.DataFrame:
    cols = list_columns(fqtn)
    y2y = pick_y2y(cols)
    if not y2y:
        st.error(f"`{fqtn}` mist 2Y kolom (y_2y_synth of y_2y)."); st.stop()
    sel = ["date"]
    for src, alias in [("y_3m","y_3m"), (y2y,"y_2y"), ("y_5y","y_5y"), ("y_10y","y_10y"), ("y_30y","y_30y")]:
        if have(cols, src): sel.append(f"SAFE_CAST({src} AS FLOAT64) AS {alias}")
    for extra in ["spread_10_2","spread_30_10","snapshot_date",
                  "y_3m_d1_bp","y_2y_d1_bp","y_5y_d1_bp","y_10y_d1_bp","y_30y_d1_bp"]:
        if have(cols, extra): sel.append(extra)
    for base in ["y_3m","y_2y","y_5y","y_10y","y_30y","spread_10_2","spread_30_10"]:
        if have(cols, f"{base}_d7"):  sel.append(f"SAFE_CAST({base}_d7  AS FLOAT64) AS {base}_d7")
        if have(cols, f"{base}_d30"): sel.append(f"SAFE_CAST({base}_d30 AS FLOAT64) AS {base}_d30")
    # Optionele fragmentatie-spreads
    optional_spreads = [
        "oat_bund_spread", "oat_bund_10y_spread", "oat_bund", "fr_de_10y_spread",
        "btp_bund_spread", "btp_bund_10y_spread", "it_de_10y_spread",
    ]
    for s in optional_spreads:
        if have(cols, s):
            sel.append(f"SAFE_CAST({s} AS FLOAT64) AS {s.lower()}")
    sql = f"SELECT {', '.join(sel)} FROM `{fqtn}` ORDER BY date"
    return run_query(sql, timeout=60)

def pct_fmt(x, dp=2):
    return "—" if pd.isna(x) else f"{round(float(x), dp)}%"

def to_zscores(df: pd.DataFrame, cols: list[str], window: int) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            s = pd.to_numeric(out[c], errors="coerce")
            mu = s.rolling(window).mean()
            sd = s.rolling(window).std()
            out[c] = (s - mu) / sd
    return out

# ============== data ==============
with st.spinner("BigQuery laden…"):
    df_us = load_view(US_VIEW)
    df_eu = load_view(EU_VIEW)

if df_us.empty or df_eu.empty:
    st.warning("Geen data in één van de views."); st.stop()

df_us["date"] = pd.to_datetime(df_us["date"])
df_eu["date"] = pd.to_datetime(df_eu["date"])
dates_common = sorted(set(df_us["date"]).intersection(set(df_eu["date"])))
if not dates_common:
    st.warning("Geen overlap in datums tussen US en EU."); st.stop()

# ============== bovenbalk ==============
top1, top2, top3, top4 = st.columns([1.1, 1.1, 1, 1])
with top1:
    round_dp = st.slider("Decimalen", 1, 4, 2)
with top2:
    show_table = st.toggle("Toon tabel onderaan", value=False)
with top3:
    strict = st.toggle("Strikt (alle looptijden)", value=False, help="Filter op datums met 3M/2Y/5Y/10Y/30Y in beide regio's.")
with top4:
    delta_h_sel = st.radio("Δ-horizon", ["1d","7d","30d"], horizontal=True, index=1)

# Extra toggles
x1, x2, x3 = st.columns([1, 1.2, 2])
with x1:
    mode_z = st.toggle("Z-scores (rolling)", value=False, help="Standariseer per reeks met rolling mean/std.")
with x2:
    z_window = st.number_input("Z-window (dagen)", min_value=20, max_value=260, value=90, step=10, help="Rolling window voor z-scores.")
with x3:
    preset = st.radio("Preset", ["Custom", "Rates Focus", "Curve Focus", "Vol & Regime"], horizontal=True, index=0)

# ============== Periode ==============
st.subheader("Periode")
dmin = max(min(dates_common), pd.to_datetime("1990-01-01"))
dmax = max(dates_common)
left, _ = st.columns([1.75, 1])
with left:
    pr = st.radio(
        "Presets",
        ["1D","1W","1M","3M","6M","1Y","3Y","5Y","10Y","YTD","Max","Custom"],
        horizontal=True, index=5
    )

def clamp(ts: pd.Timestamp) -> pd.Timestamp: return max(dmin, ts)

if pr == "1D":   start_date, end_date = clamp(dmax - pd.DateOffset(days=1)), dmax
elif pr == "1W": start_date, end_date = clamp(dmax - pd.DateOffset(weeks=1)), dmax
elif pr == "1M": start_date, end_date = clamp(dmax - pd.DateOffset(months=1)), dmax
elif pr == "3M": start_date, end_date = clamp(dmax - pd.DateOffset(months=3)), dmax
elif pr == "6M": start_date, end_date = clamp(dmax - pd.DateOffset(months=6)), dmax
elif pr == "1Y": start_date, end_date = clamp(dmax - pd.DateOffset(years=1)), dmax
elif pr == "3Y": start_date, end_date = clamp(dmax - pd.DateOffset(years=3)), dmax
elif pr == "5Y": start_date, end_date = clamp(dmax - pd.DateOffset(years=5)), dmax
elif pr == "10Y":start_date, end_date = clamp(dmax - pd.DateOffset(years=10)), dmax
elif pr == "YTD":start_date, end_date = clamp(pd.Timestamp(dmax.year,1,1)), dmax
elif pr == "Max":start_date, end_date = dmin, dmax
else:
    date_range = st.slider("Selecteer periode (Custom)",
                           min_value=dmin.date(), max_value=dmax.date(),
                           value=(clamp(dmax - pd.DateOffset(years=1)).date(), dmax.date()))
    start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])

mask_us = (df_us["date"].isin(dates_common)) & (df_us["date"]>=start_date) & (df_us["date"]<=end_date)
mask_eu = (df_eu["date"].isin(dates_common)) & (df_eu["date"]>=start_date) & (df_eu["date"]<=end_date)
US = df_us.loc[mask_us].copy()
EU = df_eu.loc[mask_eu].copy()

if strict:
    need = [c for c in ["y_3m","y_2y","y_5y","y_10y","y_30y"] if c in US.columns and c in EU.columns]
    if need:
        US = US.dropna(subset=need); EU = EU.dropna(subset=need)
        common = sorted(set(US["date"]).intersection(set(EU["date"])))
        US = US[US["date"].isin(common)]
        EU = EU[EU["date"].isin(common)]

if US.empty or EU.empty:
    st.info("Na filteren geen gemeenschappelijke data."); st.stop()

# ============== Views: levels of z-scores ==============
level_cols = [c for c in ["y_3m","y_2y","y_5y","y_10y","y_30y","spread_10_2","spread_30_10"] if c in US.columns and c in EU.columns]
US_view = US.copy(); EU_view = EU.copy()
if mode_z and level_cols:
    US_view = to_zscores(US_view, level_cols, z_window)
    EU_view = to_zscores(EU_view, level_cols, z_window)

# ============== Snapshots: peildata (US/EU gelijk) ==============
st.subheader("Term Structure — snapshot (US vs EU)")
snap_dates = sorted(set(US["date"]).intersection(set(EU["date"])))
latest = snap_dates[-1]
one_month_prior = min(snap_dates, key=lambda d: abs(pd.Timestamp(d) - (pd.Timestamp(latest) - pd.DateOffset(months=1))))

g1, g2, g3 = st.columns([1.4, 1.4, 1])
with g1:
    snap_primary = st.selectbox("Hoofd peildatum", options=snap_dates, index=len(snap_dates)-1,
                                format_func=lambda d: pd.Timestamp(d).strftime("%Y-%m-%d"))
with g2:
    enable_compare = st.checkbox("Vergelijk met 2e peildatum", value=True)
    snap_secondary = st.selectbox("2e peildatum", options=snap_dates,
                                  index=snap_dates.index(one_month_prior) if one_month_prior in snap_dates else max(0,len(snap_dates)-2),
                                  format_func=lambda d: pd.Timestamp(d).strftime("%Y-%m-%d"),
                                  disabled=not enable_compare)
with g3:
    st.caption("US & EU worden op **exact dezelfde datum** vergeleken.")

snapUS1 = US_view[US_view["date"]==snap_primary].tail(1)
snapEU1 = EU_view[EU_view["date"]==snap_primary].tail(1)
snapUS2 = US_view[US_view["date"]==snap_secondary].tail(1) if enable_compare else pd.DataFrame()
snapEU2 = EU_view[EU_view["date"]==snap_secondary].tail(1) if enable_compare else pd.DataFrame()

def curve_points(row: pd.Series):
    mats = ["3M","2Y","5Y","10Y","30Y"]
    vals = [row.get("y_3m"),row.get("y_2y"),row.get("y_5y"),row.get("y_10y"),row.get("y_30y")]
    m = [m for m,v in zip(mats,vals) if pd.notna(v)]
    v = [v for v in vals if pd.notna(v)]
    return m,v

# KPI’s (exact wat er in view staat: levels of z)
k1,k2,k3,k4,k5,k6 = st.columns(6)
def getv(dfrow, col):
    return None if dfrow.empty or col not in dfrow.columns else float(dfrow[col].values[0])
y2_us, y10_us = getv(snapUS1,"y_2y"), getv(snapUS1,"y_10y")
y2_eu, y10_eu = getv(snapEU1,"y_2y"), getv(snapEU1,"y_10y")
sp_us = (y10_us - y2_us) if (y10_us is not None and y2_us is not None) else None
sp_eu = (y10_eu - y2_eu) if (y10_eu is not None and y2_eu is not None) else None
diff_10 = (y10_us - y10_eu) if (y10_us is not None and y10_eu is not None) else None
diff_2  = (y2_us  - y2_eu ) if (y2_us  is not None and y2_eu  is not None) else None
diff_sp = (sp_us  - sp_eu ) if (sp_us  is not None and sp_eu  is not None) else None

lbl_unit = "z" if mode_z else "%"
k1.metric("US 2Y",  "—" if y2_us is None else (f"{round(y2_us,2)} z" if mode_z else pct_fmt(y2_us, round_dp)))
k2.metric("US 10Y", "—" if y10_us is None else (f"{round(y10_us,2)} z" if mode_z else pct_fmt(y10_us, round_dp)))
k3.metric("US 10Y–2Y", "—" if sp_us is None else (f"{round(sp_us,2)} z" if mode_z else f"{round(sp_us,2)} pp"))
# differentials in bp/z
if mode_z:
    k4.metric("Δ(US–EU) 2Y",  "—" if diff_2  is None else f"{round(diff_2,2)} z")
    k5.metric("Δ(US–EU) 10Y", "—" if diff_10 is None else f"{round(diff_10,2)} z")
    k6.metric("Δ(US–EU) (10Y–2Y)", "—" if diff_sp is None else f"{round(diff_sp,2)} z")
else:
    k4.metric("Δ(US–EU) 2Y",  "—" if diff_2  is None else f"{round(diff_2*100,1)} bp")
    k5.metric("Δ(US–EU) 10Y", "—" if diff_10 is None else f"{round(diff_10*100,1)} bp")
    k6.metric("Δ(US–EU) (10Y–2Y)", "—" if diff_sp is None else f"{round(diff_sp*100,1)} bp")

# ======= SAMENGEVOEGDE GRAFIEK: Term structure + Δ(US–EU) op dubbele y-as =======
rowA = snapUS1.iloc[0] if not snapUS1.empty else pd.Series()
rowB = snapEU1.iloc[0] if not snapEU1.empty else pd.Series()
mA, vA = curve_points(rowA)
mB, vB = curve_points(rowB)

def align_and_diff(m1, v1, m2, v2, as_bp: bool):
    d = {}
    for m,val in zip(m1,v1): d[m]=[val, None]
    for m,val in zip(m2,v2):
        if m in d: d[m][1]=val
        else: d[m]=[None, val]
    xs, ys = [], []
    for m,(a,b) in d.items():
        if a is not None and b is not None:
            xs.append(m); ys.append((a-b)*(100.0 if as_bp else 1.0))
    order = {k:i for i,k in enumerate(["3M","2Y","5Y","10Y","30Y"])}
    p = sorted(zip(xs,ys), key=lambda t: order.get(t[0], 99))
    return [x for x,_ in p], [y for _,y in p]

fig_ts = make_subplots(specs=[[{"secondary_y": True}]])
# US/EU op primaire Y
if mA and vA:
    fig_ts.add_trace(go.Scatter(x=mA, y=vA, mode="lines+markers",
                                name=f"US {pd.Timestamp(snap_primary).date()}"), secondary_y=False)
if mB and vB:
    fig_ts.add_trace(go.Scatter(x=mB, y=vB, mode="lines+markers",
                                name=f"EU {pd.Timestamp(snap_primary).date()}",
                                line=dict(dash="dash")), secondary_y=False)

# Optionele vergelijking met 2e peildatum (transparanter)
if enable_compare and not snapUS2.empty and not snapEU2.empty:
    mA2, vA2 = curve_points(snapUS2.iloc[0]); mB2, vB2 = curve_points(snapEU2.iloc[0])
    if mA2 and vA2:
        fig_ts.add_trace(go.Scatter(x=mA2, y=vA2, mode="lines+markers",
                                    name=f"US {pd.Timestamp(snap_secondary).date()}",
                                    opacity=0.4), secondary_y=False)
    if mB2 and vB2:
        fig_ts.add_trace(go.Scatter(x=mB2, y=vB2, mode="lines+markers",
                                    name=f"EU {pd.Timestamp(snap_secondary).date()}",
                                    line=dict(dash="dash"), opacity=0.4), secondary_y=False)

# US–EU differential op secundaire Y
xm, ydiff = align_and_diff(mA or [], vA or [], mB or [], vB or [], as_bp=(not mode_z))
if xm:
    fig_ts.add_trace(go.Scatter(x=xm, y=ydiff, mode="lines+markers",
                                name=("US–EU Δ (bp)" if not mode_z else "US–EU Δ (z)")),
                     secondary_y=True)

# Dynamische schaal per as (autorange per as onafhankelijk)
fig_ts.update_yaxes(title_text=("Yield (z)" if mode_z else "Yield (%)"),
                    secondary_y=False, autorange=True)
fig_ts.update_yaxes(title_text=("Δ (z)" if mode_z else "Δ (bp)"),
                    secondary_y=True, autorange=True)
fig_ts.update_xaxes(title_text="Maturity")
fig_ts.update_layout(title="Term structure + US–EU Δ (dubbele y-as)",
                     margin=dict(l=10,r=10,t=35,b=10), legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
st.plotly_chart(fig_ts, use_container_width=True)

# ============== Tijdreeks — Levels & Spreads ==============
st.subheader("Tijdreeks — Levels & Spreads (overlay)")

# 10Y en 2Y overlay
fig1 = go.Figure()
if "y_10y" in US_view.columns: fig1.add_trace(go.Scatter(x=US_view["date"], y=US_view["y_10y"], name=("US 10Y z" if mode_z else "US 10Y"), mode="lines"))
if "y_10y" in EU_view.columns: fig1.add_trace(go.Scatter(x=EU_view["date"], y=EU_view["y_10y"], name=("EU 10Y z" if mode_z else "EU 10Y"), mode="lines", line=dict(dash="dash")))
if "y_2y"  in US_view.columns: fig1.add_trace(go.Scatter(x=US_view["date"], y=US_view["y_2y"],  name=("US 2Y z"  if mode_z else "US 2Y"),  mode="lines"))
if "y_2y"  in EU_view.columns: fig1.add_trace(go.Scatter(x=EU_view["date"], y=EU_view["y_2y"],  name=("EU 2Y z"  if mode_z else "EU 2Y"),  mode="lines", line=dict(dash="dash")))
fig1.update_layout(margin=dict(l=10,r=10,t=10,b=10),
                   yaxis_title=("z-score" if mode_z else "Yield (%)"), xaxis_title="Date")
fig1.update_xaxes(range=[start_date, end_date])
st.plotly_chart(fig1, use_container_width=True)

# 10Y–2Y: US, EU & differential
st.subheader("Tijdreeks — 10Y–2Y (US, EU) & differential")
fig2 = go.Figure()
if "spread_10_2" in US_view.columns: fig2.add_trace(go.Scatter(x=US_view["date"], y=US_view["spread_10_2"], name=("US 10Y–2Y z" if mode_z else "US 10Y–2Y"), mode="lines"))
if "spread_10_2" in EU_view.columns: fig2.add_trace(go.Scatter(x=EU_view["date"], y=EU_view["spread_10_2"], name=("EU 10Y–2Y z" if mode_z else "EU 10Y–2Y"), mode="lines", line=dict(dash="dash")))
if ("spread_10_2" in US_view.columns) and ("spread_10_2" in EU_view.columns):
    df_join = pd.merge(US_view[["date","spread_10_2"]], EU_view[["date","spread_10_2"]],
                       on="date", suffixes=("_us","_eu"))
    if mode_z:
        diff_series = df_join["spread_10_2_us"] - df_join["spread_10_2_eu"]
        fig2.add_trace(go.Bar(x=df_join["date"], y=diff_series, name="US–EU (z)", opacity=0.35))
        fig2.update_yaxes(title_text="Spread (z) & Δ (z)")
    else:
        diff_bp = (df_join["spread_10_2_us"] - df_join["spread_10_2_eu"]) * 100.0
        fig2.add_trace(go.Bar(x=df_join["date"], y=diff_bp, name="US–EU (bp)", opacity=0.35))
        fig2.update_yaxes(title_text="Spread (pp) & Δ (bp)")
fig2.update_layout(margin=dict(l=10,r=10,t=10,b=10), xaxis_title="Date", barmode="overlay")
fig2.update_xaxes(range=[start_date, end_date])
st.plotly_chart(fig2, use_container_width=True)

# ============== US–EU Differential Rolling Band (levels/z) ==============
st.subheader("US–EU Differential — rolling band")
diff_target = "y_10y" if "y_10y" in level_cols else (level_cols[0] if level_cols else None)
if diff_target:
    JJ = pd.merge(US_view[["date", diff_target]], EU_view[["date", diff_target]],
                  on="date", suffixes=("_us","_eu"))
    JJ["diff"] = JJ[f"{diff_target}_us"] - JJ[f"{diff_target}_eu"]
    mu = JJ["diff"].rolling(90).mean()
    sd = JJ["diff"].rolling(90).std()
    figB = go.Figure()
    figB.add_trace(go.Scatter(x=JJ["date"], y=JJ["diff"], name="US–EU", mode="lines"))
    figB.add_trace(go.Scatter(x=JJ["date"], y=(mu+sd), name="+1σ", line=dict(dash="dot")))
    figB.add_trace(go.Scatter(x=JJ["date"], y=(mu-sd), name="-1σ", line=dict(dash="dot"), fill="tonexty", opacity=0.15))
    figB.update_layout(margin=dict(l=10,r=10,t=10,b=10),
                       yaxis_title=("Δ z-score (US–EU)" if mode_z else "Δ (pp)"),
                       xaxis_title="Date")
    figB.update_xaxes(range=[start_date, end_date])
    st.plotly_chart(figB, use_container_width=True)

# ============== EU fragmentatie-spreads (optioneel) ==============
st.subheader("EU fragmentatie — OAT–Bund / BTP–Bund (indien aanwezig)")
def first_available(df: pd.DataFrame, names: list[str]) -> str | None:
    for n in names:
        if n in df.columns: return n
    return None

oat_candidates = ["oat_bund_spread","oat_bund_10y_spread","oat_bund","fr_de_10y_spread"]
btp_candidates = ["btp_bund_spread","btp_bund_10y_spread","it_de_10y_spread"]
oat_col = first_available(EU, oat_candidates)
btp_col = first_available(EU, btp_candidates)

if (oat_col is None) and (btp_col is None):
    st.caption("Geen fragmentatie-kolommen gevonden in de EU-view (optioneel).")
else:
    F = go.Figure()
    if oat_col is not None:
        F.add_trace(go.Scatter(x=EU["date"], y=EU[oat_col], name=oat_col.upper().replace("_"," "), mode="lines"))
    if btp_col is not None:
        F.add_trace(go.Scatter(x=EU["date"], y=EU[btp_col], name=btp_col.upper().replace("_"," "), mode="lines"))
    F.update_layout(margin=dict(l=10,r=10,t=10,b=10),
                    yaxis_title="Spread t.o.v. Bund (pp)", xaxis_title="Date")
    F.update_xaxes(range=[start_date, end_date])
    st.plotly_chart(F, use_container_width=True)
    st.caption("NB: Bij fiscale/landenspecifieke stress zie je dit vaak eerder in *kruis-spreads* dan in de aggregate eurocurve-slope.")

# ============== Deltas (1d/7d/30d) ==============
st.subheader("Deltas — verdeling & tijdreeks")
if delta_h_sel == "1d":
    suf = "_d1_bp"; is_bp = True
elif delta_h_sel == "7d":
    suf = "_d7";    is_bp = False
else:
    suf = "_d30";   is_bp = False

bases = [("y_3m","3M"),("y_2y","2Y"),("y_5y","5Y"),("y_10y","10Y"),("y_30y","30Y"),
         ("spread_10_2","10Y-2Y"),("spread_30_10","30Y-10Y")]
cands = []
for b,l in bases:
    col = f"{b}{suf}"
    if (delta_h_sel == "1d") or (col in US.columns and col in EU.columns):
        cands.append((b,l))
if not cands:
    st.info("Geen delta-kolommen voor deze horizon.")
else:
    def_idx = next((i for i,(b,_) in enumerate(cands) if b=="y_10y"), 0)
    b_sel, label_sel = st.selectbox("Metric", cands, index=def_idx, format_func=lambda t: t[1])

    def get_delta_series(df: pd.DataFrame, base: str) -> pd.Series:
        if suf == "_d1_bp":
            if f"{base}_d1_bp" in df.columns:
                return pd.to_numeric(df[f"{base}_d1_bp"], errors="coerce")
            else:
                return pd.to_numeric(df[base], errors="coerce").diff() * 100.0
        else:
            s = pd.to_numeric(df[f"{base}{suf}"], errors="coerce")  # pp
            return s * 100.0  # → bp

    USd = get_delta_series(US, b_sel)
    EUd = get_delta_series(EU, b_sel)

    # Relatieve %: Δpp / vorige pp * 100
    if suf == "_d1_bp":
        dpp_US = USd / 100.0
        dpp_EU = EUd / 100.0
        baseUS = pd.to_numeric(US[b_sel], errors="coerce")
        baseEU = pd.to_numeric(EU[b_sel], errors="coerce")
    else:
        dpp_US = pd.to_numeric(US[f"{b_sel}{suf}"], errors="coerce")
        dpp_EU = pd.to_numeric(EU[f"{b_sel}{suf}"], errors="coerce")
        baseUS = pd.to_numeric(US[b_sel], errors="coerce")
        baseEU = pd.to_numeric(EU[b_sel], errors="coerce")

    pctUS = (dpp_US / baseUS.shift(1).replace(0,np.nan)) * 100.0
    pctEU = (dpp_EU / baseEU.shift(1).replace(0,np.nan)) * 100.0

   
    # Δ tijdreeks (US, EU en differential)
    figd = go.Figure()
    figd.add_trace(go.Bar(x=US["date"], y=USd, name=f"US Δ{delta_h_sel} ({label_sel})", opacity=0.6))
    figd.add_trace(go.Bar(x=EU["date"], y=EUd, name=f"EU Δ{delta_h_sel} ({label_sel})", opacity=0.6))
    dfJ = pd.DataFrame({"date":US["date"].values, "US":USd.values}).merge(
          pd.DataFrame({"date":EU["date"].values, "EU":EUd.values}), on="date", how="inner")
    dfJ["USminusEU"] = dfJ["US"] - dfJ["EU"]
    figd.add_trace(go.Scatter(x=dfJ["date"], y=dfJ["USminusEU"], name="Δ(US–EU)", mode="lines", line=dict(width=2)))
    figd.add_hline(y=0, line_width=1, line_color="gray", opacity=0.5)
    figd.update_layout(margin=dict(l=10,r=10,t=10,b=10), barmode="overlay", yaxis_title="Δ (bp)", xaxis_title="Date")
    figd.update_xaxes(range=[start_date, end_date])
    st.plotly_chart(figd, use_container_width=True)

