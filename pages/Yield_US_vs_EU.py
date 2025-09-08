# pages/Yield_US_EU_Compare.py — US vs EU vergelijking op enriched views
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

# ---------- helpers ----------
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
    if not y2y: st.error(f"`{fqtn}` mist 2Y kolom (y_2y_synth/y_2y)."); st.stop()
    sel = ["date"]
    for src, alias in [("y_3m","y_3m"), (y2y,"y_2y"), ("y_5y","y_5y"), ("y_10y","y_10y"), ("y_30y","y_30y")]:
        if have(cols, src): sel.append(f"SAFE_CAST({src} AS FLOAT64) AS {alias}")
    for extra in ["spread_10_2","spread_30_10","snapshot_date",
                  "y_3m_d1_bp","y_2y_d1_bp","y_5y_d1_bp","y_10y_d1_bp","y_30y_d1_bp"]:
        if have(cols, extra): sel.append(extra)
    for base in ["y_3m","y_2y","y_5y","y_10y","y_30y","spread_10_2","spread_30_10"]:
        if have(cols, f"{base}_d7"):  sel.append(f"SAFE_CAST({base}_d7  AS FLOAT64) AS {base}_d7")
        if have(cols, f"{base}_d30"): sel.append(f"SAFE_CAST({base}_d30 AS FLOAT64) AS {base}_d30")
    sql = f"SELECT {', '.join(sel)} FROM `{fqtn}` ORDER BY date"
    return run_query(sql, timeout=60)

def pct_fmt(x, dp=2):
    return "—" if pd.isna(x) else f"{round(float(x), dp)}%"

# ---------- data ----------
with st.spinner("BigQuery laden…"):
    df_us = load_view(US_VIEW)
    df_eu = load_view(EU_VIEW)
if df_us.empty or df_eu.empty:
    st.warning("Geen data in één van de views."); st.stop()

# Gemeenschappelijke periode
df_us["date"] = pd.to_datetime(df_us["date"])
df_eu["date"] = pd.to_datetime(df_eu["date"])
dates_common = sorted(set(df_us["date"]).intersection(set(df_eu["date"])))
if not dates_common:
    st.warning("Geen overlap in datums tussen US en EU."); st.stop()

# ============== bovenbalk: opties ==============
top1, top2, top3, top4 = st.columns([1.1, 1.1, 1, 1])
with top1:
    round_dp = st.slider("Decimalen", 1, 4, 2)
with top2:
    show_table = st.toggle("Toon tabel onderaan", value=False)
with top3:
    # filters: strikt = alle looptijden aanwezig in beide regio's
    strict = st.toggle("Strikt (alle looptijden)", value=False)
with top4:
    # Δ horizon voor signals/hist
    delta_h_sel = st.radio("Δ-horizon", ["1d","7d","30d"], horizontal=True, index=1)

# Periode presets
st.subheader("Periode")
dmin = max(min(dates_common), pd.to_datetime("1990-01-01"))
dmax = max(dates_common)
left, _ = st.columns([1.75, 1])
with left:
    preset = st.radio(
        "Presets",
        ["1D","1W","1M","3M","6M","1Y","3Y","5Y","10Y","YTD","Max","Custom"],
        horizontal=True, index=5
    )

def clamp(ts: pd.Timestamp) -> pd.Timestamp: return max(dmin, ts)

if preset == "1D":   start_date, end_date = clamp(dmax - pd.DateOffset(days=1)), dmax
elif preset == "1W": start_date, end_date = clamp(dmax - pd.DateOffset(weeks=1)), dmax
elif preset == "1M": start_date, end_date = clamp(dmax - pd.DateOffset(months=1)), dmax
elif preset == "3M": start_date, end_date = clamp(dmax - pd.DateOffset(months=3)), dmax
elif preset == "6M": start_date, end_date = clamp(dmax - pd.DateOffset(months=6)), dmax
elif preset == "1Y": start_date, end_date = clamp(dmax - pd.DateOffset(years=1)), dmax
elif preset == "3Y": start_date, end_date = clamp(dmax - pd.DateOffset(years=3)), dmax
elif preset == "5Y": start_date, end_date = clamp(dmax - pd.DateOffset(years=5)), dmax
elif preset == "10Y":start_date, end_date = clamp(dmax - pd.DateOffset(years=10)), dmax
elif preset == "YTD":start_date, end_date = clamp(pd.Timestamp(dmax.year,1,1)), dmax
elif preset == "Max":start_date, end_date = dmin, dmax
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
        # sync datums na dropna
        common = sorted(set(US["date"]).intersection(set(EU["date"])))
        US = US[US["date"].isin(common)]
        EU = EU[EU["date"].isin(common)]

if US.empty or EU.empty:
    st.info("Na filteren geen gemeenschappelijke data."); st.stop()

# ============== snapshots: peildata US/EU gelijk ==============
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

snapUS1 = US[US["date"]==snap_primary].tail(1)
snapEU1 = EU[EU["date"]==snap_primary].tail(1)
snapUS2 = US[US["date"]==snap_secondary].tail(1) if enable_compare else pd.DataFrame()
snapEU2 = EU[EU["date"]==snap_secondary].tail(1) if enable_compare else pd.DataFrame()

def curve_points(row: pd.Series):
    mats = ["3M","2Y","5Y","10Y","30Y"]
    vals = [row.get("y_3m"),row.get("y_2y"),row.get("y_5y"),row.get("y_10y"),row.get("y_30y")]
    m = [m for m,v in zip(mats,vals) if pd.notna(v)]
    v = [v for v in vals if pd.notna(v)]
    return m,v

# KPI’s
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

k1.metric("US 2Y",  pct_fmt(y2_us, round_dp))
k2.metric("US 10Y", pct_fmt(y10_us, round_dp))
k3.metric("US 10Y–2Y", "—" if sp_us is None else f"{round(sp_us,2)} pp")
k4.metric("Δ(US–EU) 2Y",  "—" if diff_2  is None else f"{round(diff_2*100,1)} bp")
k5.metric("Δ(US–EU) 10Y", "—" if diff_10 is None else f"{round(diff_10*100,1)} bp")
k6.metric("Δ(US–EU) (10Y–2Y)", "—" if diff_sp is None else f"{round(diff_sp*100,1)} bp")

# Term structure overlay & Δ-curve (US–EU)
rowA = snapUS1.iloc[0] if not snapUS1.empty else pd.Series()
rowB = snapEU1.iloc[0] if not snapEU1.empty else pd.Series()
mA, vA = curve_points(rowA)
mB, vB = curve_points(rowB)

ts = make_subplots(rows=1, cols=2, subplot_titles=("Term structure", "Δ-curve (US–EU) in bp"), column_widths=[0.6,0.4])
if mA and vA: ts.add_trace(go.Scatter(x=mA, y=vA, mode="lines+markers", name=f"US {pd.Timestamp(snap_primary).date()}"), row=1, col=1)
if mB and vB: ts.add_trace(go.Scatter(x=mB, y=vB, mode="lines+markers", name=f"EU {pd.Timestamp(snap_primary).date()}", line=dict(dash="dash")), row=1, col=1)

# Δ-curve = (US - EU) * 100 bp
def align_and_diff(m1, v1, m2, v2):
    d = {}
    for m,val in zip(m1,v1): d[m]=[val, None]
    for m,val in zip(m2,v2):
        if m in d: d[m][1]=val
        else: d[m]=[None, val]
    xs, ys = [], []
    for m,(a,b) in d.items():
        if a is not None and b is not None:
            xs.append(m); ys.append((a-b)*100.0)
    order = {k:i for i,k in enumerate(["3M","2Y","5Y","10Y","30Y"])}
    p = sorted(zip(xs,ys), key=lambda t: order.get(t[0], 99))
    return [x for x,_ in p], [y for _,y in p]

xm, ybp = align_and_diff(mA,vA,mB,vB)
if xm:
    ts.add_trace(go.Scatter(x=xm, y=ybp, mode="lines+markers", name="US–EU (bp)"), row=1, col=2)

ts.update_yaxes(title_text="Yield (%)", row=1, col=1)
ts.update_yaxes(title_text="Δ (bp)", row=1, col=2)
ts.update_xaxes(title_text="Maturity", row=1, col=1)
ts.update_xaxes(title_text="Maturity", row=1, col=2)
ts.update_layout(margin=dict(l=10,r=10,t=30,b=10))
st.plotly_chart(ts, use_container_width=True)

# ============== spreads & levels tijdreeks ==============
st.subheader("Tijdreeks — Levels & Spreads")

# 10Y en 2Y overlay
fig1 = go.Figure()
if "y_10y" in US.columns: fig1.add_trace(go.Scatter(x=US["date"], y=US["y_10y"], name="US 10Y", mode="lines"))
if "y_10y" in EU.columns: fig1.add_trace(go.Scatter(x=EU["date"], y=EU["y_10y"], name="EU 10Y", mode="lines", line=dict(dash="dash")))
if "y_2y" in US.columns:  fig1.add_trace(go.Scatter(x=US["date"], y=US["y_2y"],  name="US 2Y",  mode="lines"))
if "y_2y" in EU.columns:  fig1.add_trace(go.Scatter(x=EU["date"], y=EU["y_2y"],  name="EU 2Y",  mode="lines", line=dict(dash="dash")))
fig1.update_layout(margin=dict(l=10,r=10,t=10,b=10), yaxis_title="Yield (%)", xaxis_title="Date")
fig1.update_xaxes(range=[start_date, end_date])
st.plotly_chart(fig1, use_container_width=True)

# 10Y–2Y: US, EU & differential
fig2 = go.Figure()
if "spread_10_2" in US.columns: fig2.add_trace(go.Scatter(x=US["date"], y=US["spread_10_2"], name="US 10Y–2Y", mode="lines"))
if "spread_10_2" in EU.columns: fig2.add_trace(go.Scatter(x=EU["date"], y=EU["spread_10_2"], name="EU 10Y–2Y", mode="lines", line=dict(dash="dash")))
if "spread_10_2" in US.columns and "spread_10_2" in EU.columns:
    df_join = pd.merge(US[["date","spread_10_2"]], EU[["date","spread_10_2"]], on="date", suffixes=("_us","_eu"))
    df_join["diff_bp"] = (df_join["spread_10_2_us"] - df_join["spread_10_2_eu"]) * 100.0
    fig2.add_trace(go.Bar(x=df_join["date"], y=df_join["diff_bp"], name="US–EU (10Y–2Y) bp", opacity=0.5))
fig2.update_layout(margin=dict(l=10,r=10,t=10,b=10), yaxis_title="Spread (pp) & Δ (bp)", xaxis_title="Date", barmode="overlay")
fig2.update_xaxes(range=[start_date, end_date])
st.plotly_chart(fig2, use_container_width=True)

# ============== Δ’s (1d/7d/30d) ==============
st.subheader("Deltas — verdeling & tijdreeks")

# horizon kolommen
if delta_h_sel == "1d":
    suf = "_d1_bp"; units = "bp"; is_bp = True
elif delta_h_sel == "7d":
    suf = "_d7"; units = "bp (pp*100)"; is_bp = False
else:
    suf = "_d30"; units = "bp (pp*100)"; is_bp = False

# kies metric
bases = [("y_3m","3M"),("y_2y","2Y"),("y_5y","5Y"),("y_10y","10Y"),("y_30y","30Y"),("spread_10_2","10Y-2Y"),("spread_30_10","30Y-10Y")]
cands = []
for b,l in bases:
    col = f"{b}{suf}"
    # 1d: laten we altijd toe; fallback zo nodig
    if (delta_h_sel == "1d") or (col in US.columns and col in EU.columns):
        cands.append((b,l))
if not cands:
    st.info("Geen delta-kolommen voor deze horizon."); st.stop()

def_idx = next((i for i,(b,_) in enumerate(cands) if b=="y_10y"), 0)
b_sel, label_sel = st.selectbox("Metric", cands, index=def_idx, format_func=lambda t: t[1])

# Bouw Δ-series US & EU
def get_delta_series(df: pd.DataFrame, base: str) -> pd.Series:
    if suf == "_d1_bp":
        if f"{base}_d1_bp" in df.columns:
            return pd.to_numeric(df[f"{base}_d1_bp"], errors="coerce")
        else:
            return pd.to_numeric(df[base], errors="coerce").diff() * 100.0
    else:
        # pp → bp
        s = pd.to_numeric(df[f"{base}{suf}"], errors="coerce")
        return s * 100.0

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

# Histograms naast elkaar (abs bp & %)
h1, h2 = st.columns(2)
with h1:
    H = go.Figure()
    H.add_trace(go.Histogram(x=USd.replace([np.inf,-np.inf],np.nan).dropna(), nbinsx=40, name="US", opacity=0.6))
    H.add_trace(go.Histogram(x=EUd.replace([np.inf,-np.inf],np.nan).dropna(), nbinsx=40, name="EU", opacity=0.6))
    H.update_layout(title=f"Δ {label_sel} — abs ({units})", barmode="overlay",
                    margin=dict(l=10,r=10,t=40,b=10), xaxis_title=f"Δ ({units})", yaxis_title="Aantal dagen")
    st.plotly_chart(H, use_container_width=True)
with h2:
    H2 = go.Figure()
    H2.add_trace(go.Histogram(x=pctUS.replace([np.inf,-np.inf],np.nan).dropna(), nbinsx=40, name="US", opacity=0.6))
    H2.add_trace(go.Histogram(x=pctEU.replace([np.inf,-np.inf],np.nan).dropna(), nbinsx=40, name="EU", opacity=0.6))
    H2.update_layout(title=f"Δ {label_sel} — relatief (%)", barmode="overlay",
                     margin=dict(l=10,r=10,t=40,b=10), xaxis_title="Δ (%)", yaxis_title="Aantal dagen")
    st.plotly_chart(H2, use_container_width=True)

# Δ tijdreeks (US, EU en differential)
figd = go.Figure()
figd.add_trace(go.Bar(x=US["date"], y=USd, name=f"US Δ{delta_h_sel} ({label_sel})", opacity=0.6))
figd.add_trace(go.Bar(x=EU["date"], y=EUd, name=f"EU Δ{delta_h_sel} ({label_sel})", opacity=0.6))
# differential
dfJ = pd.DataFrame({"date":US["date"].values, "US":USd.values}).merge(
      pd.DataFrame({"date":EU["date"].values, "EU":EUd.values}), on="date", how="inner")
dfJ["USminusEU"] = dfJ["US"] - dfJ["EU"]
figd.add_trace(go.Scatter(x=dfJ["date"], y=dfJ["USminusEU"], name="Δ(US–EU)", mode="lines", line=dict(width=2)))
figd.add_hline(y=0, line_width=1, line_color="gray", opacity=0.5)
figd.update_layout(margin=dict(l=10,r=10,t=10,b=10), barmode="overlay", yaxis_title="Δ (bp)", xaxis_title="Date")
figd.update_xaxes(range=[start_date, end_date])
st.plotly_chart(figd, use_container_width=True)

# ============== Heatmap: US–EU level-differentials ==============
st.subheader("Level-differentials — US–EU (bp) over tijd")
mats_cols = [c for c in ["y_3m","y_2y","y_5y","y_10y","y_30y"] if c in US.columns and c in EU.columns]
if mats_cols:
    JJ = pd.merge(US[["date"]+mats_cols], EU[["date"]+mats_cols], on="date", suffixes=("_us","_eu"))
    for m in mats_cols:
        JJ[f"{m}_diff_bp"] = (JJ[f"{m}_us"] - JJ[f"{m}_eu"]) * 100.0
    heat = JJ[["date"] + [f"{m}_diff_bp" for m in mats_cols]].set_index("date")
    ylabels = [c.replace("y_","").replace("_diff_bp","").upper() for c in heat.columns]
    Hm = go.Figure(data=go.Heatmap(
        z=heat.T.values, x=heat.index.astype(str), y=ylabels, coloraxis="coloraxis"
    ))
    Hm.update_layout(margin=dict(l=10,r=10,t=10,b=10),
                     coloraxis_colorscale="RdBu", coloraxis_cmid=0)
    st.plotly_chart(Hm, use_container_width=True)
else:
    st.info("Onvoldoende overlappende looptijden voor de heatmap.")

# ============== Tabel & download ==============
if show_table:
    st.subheader("Tabel (samengevoegd, gemeenschappelijke datums)")
    merged = pd.merge(US, EU, on="date", suffixes=("_us","_eu"))
    st.dataframe(merged.sort_values("date", ascending=False).round(round_dp))

csv = pd.merge(US, EU, on="date", suffixes=("_us","_eu")).to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download CSV (US & EU, gefilterd)", data=csv,
                   file_name="yield_compare_us_eu.csv", mime="text/csv")
