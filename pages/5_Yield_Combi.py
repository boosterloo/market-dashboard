# pages/Yield_US_EU_Compare.py — Simple & Direct BQ version (no utils)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- BigQuery client (direct) ---
from google.cloud import bigquery
from google.oauth2 import service_account

st.set_page_config(page_title="US vs EU — Yield Compare (Simple)", layout="wide")
st.title("🌍 US vs EU — Yield Compare")

# ------------ Config from secrets ------------
SECRETS_SA = st.secrets.get("gcp_service_account", None)
TABLES     = st.secrets.get("tables", {})

PROJECT_ID = (SECRETS_SA or {}).get("project_id") or st.secrets.get("project_id") or ""
US_VIEW = TABLES.get("us_yield_view", f"{PROJECT_ID}.marketdata.us_yield_curve_enriched_v")
EU_VIEW = TABLES.get("eu_yield_view", f"{PROJECT_ID}.marketdata.eu_yield_curve_enriched_v")

# ------------ BQ Client ------------
def make_bq_client():
    if SECRETS_SA:
        creds = service_account.Credentials.from_service_account_info(SECRETS_SA)
        return bigquery.Client(project=PROJECT_ID, credentials=creds)
    # Valt terug op Application Default Credentials (lokaal / GCP)
    return bigquery.Client(project=PROJECT_ID or None)

CLIENT = make_bq_client()

@st.cache_data(ttl=1800, show_spinner=False)
def query_df(sql: str, params: dict | None = None) -> pd.DataFrame:
    job_config = bigquery.QueryJobConfig()
    if params:
        job_config.query_parameters = [bigquery.ScalarQueryParameter(k, "STRING", v) for k, v in params.items()]
    return CLIENT.query(sql, job_config=job_config).to_dataframe()

@st.cache_data(ttl=1800, show_spinner=False)
def list_columns(fqtn: str) -> set[str]:
    proj, dset, tbl = fqtn.split(".")
    sql = f"""
    SELECT LOWER(column_name) AS column_name
    FROM `{proj}.{dset}.INFORMATION_SCHEMA.COLUMNS`
    WHERE table_name = @tbl
    """
    dfc = query_df(sql, {"tbl": tbl})
    return set(dfc["column_name"].tolist())

def pick_y2y(cols: set[str]) -> str | None:
    return "y_2y_synth" if "y_2y_synth" in cols else ("y_2y" if "y_2y" in cols else None)

@st.cache_data(ttl=1800, show_spinner=True)
def load_view(fqtn: str) -> pd.DataFrame:
    cols = list_columns(fqtn)
    y2y = pick_y2y(cols)
    if not y2y:
        st.error(f"`{fqtn}` mist 2Y kolom (y_2y_synth of y_2y).")
        return pd.DataFrame()
    sel = ["date"]
    for src, alias in [("y_3m","y_3m"), (y2y,"y_2y"), ("y_5y","y_5y"), ("y_10y","y_10y"), ("y_30y","y_30y")]:
        if src in cols: sel.append(f"SAFE_CAST({src} AS FLOAT64) AS {alias}")
    for extra in ["spread_10_2","spread_30_10","snapshot_date",
                  "y_3m_d1_bp","y_2y_d1_bp","y_5y_d1_bp","y_10y_d1_bp","y_30y_d1_bp"]:
        if extra in cols: sel.append(extra)
    for base in ["y_3m","y_2y","y_5y","y_10y","y_30y","spread_10_2","spread_30_10"]:
        if f"{base}_d7"  in cols: sel.append(f"SAFE_CAST({base}_d7  AS FLOAT64) AS {base}_d7")
        if f"{base}_d30" in cols: sel.append(f"SAFE_CAST({base}_d30 AS FLOAT64) AS {base}_d30")
    # optioneel: fragmentatie-spreads
    for s in ["oat_bund_spread","oat_bund_10y_spread","oat_bund","fr_de_10y_spread",
              "btp_bund_spread","btp_bund_10y_spread","it_de_10y_spread"]:
        if s in cols: sel.append(f"SAFE_CAST({s} AS FLOAT64) AS {s.lower()}")

    sql = f"SELECT {', '.join(sel)} FROM `{fqtn}` ORDER BY date"
    df = query_df(sql)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df

# ------------ Load data ------------
with st.spinner("BigQuery laden…"):
    US = load_view(US_VIEW)
    EU = load_view(EU_VIEW)

if US.empty or EU.empty:
    st.warning("Geen data in één van de views.")
    st.stop()

# ------------ Controls (compact) ------------
c1, c2, c3, c4 = st.columns([1.1, 1.1, 1.1, 1])
with c1:
    round_dp = st.slider("Decimalen", 1, 4, 2)
with c2:
    strict = st.toggle("Strikt (alle looptijden)", value=False,
                       help="Filter op datums met 3M/2Y/5Y/10Y/30Y in beide regio's.")
with c3:
    delta_h = st.radio("Δ-horizon", ["1d","7d","30d"], horizontal=True, index=1)
with c4:
    show_table = st.toggle("Tabel onderaan", value=False)

# ------------ Gemeenschappelijke periode ------------
dates_common = sorted(set(US["date"]).intersection(set(EU["date"])))
US = US[US["date"].isin(dates_common)].copy()
EU = EU[EU["date"].isin(dates_common)].copy()

if strict:
    need = [c for c in ["y_3m","y_2y","y_5y","y_10y","y_30y"] if c in US.columns and c in EU.columns]
    if need:
        US = US.dropna(subset=need)
        EU = EU.dropna(subset=need)
        common = sorted(set(US["date"]).intersection(set(EU["date"])))
        US = US[US["date"].isin(common)]
        EU = EU[EU["date"].isin(common)]

if US.empty or EU.empty:
    st.info("Na filteren geen gemeenschappelijke data.")
    st.stop()

# ------------ Periode presets ------------
st.subheader("Periode")
dmin = max(min(US["date"]), pd.to_datetime("1990-01-01"))
dmax = max(US["date"])
left, _ = st.columns([1.75, 1])
with left:
    preset = st.radio(
        "Presets",
        ["1D","1W","1M","3M","6M","1Y","3Y","5Y","10Y","YTD","Max","Custom"],
        horizontal=True, index=5
    )

def clamp(ts): return max(dmin, ts)

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

US = US[(US["date"]>=start_date) & (US["date"]<=end_date)].copy()
EU = EU[(EU["date"]>=start_date) & (EU["date"]<=end_date)].copy()
if US.empty or EU.empty:
    st.info("Geen data in de gekozen periode.")
    st.stop()

# ------------ Snapshots ------------
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
    st.caption("US & EU worden op **dezelfde datum** vergeleken.")

def curve_points(row: pd.Series):
    mats = ["3M","2Y","5Y","10Y","30Y"]
    vals = [row.get("y_3m"), row.get("y_2y"), row.get("y_5y"), row.get("y_10y"), row.get("y_30y")]
    m = [m for m, v in zip(mats, vals) if pd.notna(v)]
    v = [v for v in vals if pd.notna(v)]
    return m, v

snapUS1 = US[US["date"]==snap_primary].tail(1)
snapEU1 = EU[EU["date"]==snap_primary].tail(1)

# KPI’s
def getv(dfrow, col):
    return None if dfrow.empty or col not in dfrow.columns else float(dfrow[col].values[0])
y2_us, y10_us = getv(snapUS1,"y_2y"), getv(snapUS1,"y_10y")
y2_eu, y10_eu = getv(snapEU1,"y_2y"), getv(snapEU1,"y_10y")
sp_us = (y10_us - y2_us) if (y10_us is not None and y2_us is not None) else None
sp_eu = (y10_eu - y2_eu) if (y10_eu is not None and y2_eu is not None) else None
diff_10 = (y10_us - y10_eu) if (y10_us is not None and y10_eu is not None) else None
diff_2  = (y2_us  - y2_eu ) if (y2_us  is not None and y2_eu  is not None) else None
diff_sp = (sp_us  - sp_eu ) if (sp_us  is not None and sp_eu  is not None) else None

k1,k2,k3,k4,k5,k6 = st.columns(6)
fmt_pct = lambda x: ("—" if pd.isna(x) else f"{round(float(x), round_dp)}%")
k1.metric("US 2Y",  fmt_pct(y2_us))
k2.metric("US 10Y", fmt_pct(y10_us))
k3.metric("US 10Y–2Y", "—" if sp_us is None else f"{round(sp_us,2)} pp")
k4.metric("Δ(US–EU) 2Y",  "—" if diff_2  is None else f"{round(diff_2*100,1)} bp")
k5.metric("Δ(US–EU) 10Y", "—" if diff_10 is None else f"{round(diff_10*100,1)} bp")
k6.metric("Δ(US–EU) (10Y–2Y)", "—" if diff_sp is None else f"{round(diff_sp*100,1)} bp")

# Term structure + Δ-curve
rowA = snapUS1.iloc[0] if not snapUS1.empty else pd.Series()
rowB = snapEU1.iloc[0] if not snapEU1.empty else pd.Series()
mA, vA = curve_points(rowA)
mB, vB = curve_points(rowB)

ts = make_subplots(rows=1, cols=2, subplot_titles=("Term structure", "Δ-curve (US–EU) in bp"),
                   column_widths=[0.6,0.4])
if mA: ts.add_trace(go.Scatter(x=mA, y=vA, mode="lines+markers", name=f"US {pd.Timestamp(snap_primary).date()}"), row=1, col=1)
if mB: ts.add_trace(go.Scatter(x=mB, y=vB, mode="lines+markers", name=f"EU {pd.Timestamp(snap_primary).date()}", line=dict(dash="dash")), row=1, col=1)

def align_and_diff(m1, v1, m2, v2):
    d = {}
    for m,val in zip(m1,v1): d[m]=[val, None]
    for m,val in zip(m2,v2): d[m]=[d.get(m,[None,None])[0], val]
    order = {k:i for i,k in enumerate(["3M","2Y","5Y","10Y","30Y"])}
    xs, ys = [], []
    for k in sorted(d.keys(), key=lambda x: order.get(x, 99)):
        a,b = d[k]
        if a is not None and b is not None:
            xs.append(k); ys.append((a-b)*100.0)
    return xs, ys

xm, ybp = align_and_diff(mA,vA,mB,vB)
if xm:
    ts.add_trace(go.Scatter(x=xm, y=ybp, mode="lines+markers", name="US–EU (bp)"), row=1, col=2)
ts.update_yaxes(title_text="Yield (%)", row=1, col=1)
ts.update_yaxes(title_text="Δ (bp)", row=1, col=2)
ts.update_xaxes(title_text="Maturity", row=1, col=1)
ts.update_xaxes(title_text="Maturity", row=1, col=2)
ts.update_layout(margin=dict(l=10,r=10,t=30,b=10))
st.plotly_chart(ts, use_container_width=True)

# ------------ Tijdreeks Levels (2Y/10Y) ------------
st.subheader("Tijdreeks — Levels")
fig1 = go.Figure()
if "y_10y" in US.columns: fig1.add_trace(go.Scatter(x=US["date"], y=US["y_10y"], name="US 10Y", mode="lines"))
if "y_10y" in EU.columns: fig1.add_trace(go.Scatter(x=EU["date"], y=EU["y_10y"], name="EU 10Y", mode="lines", line=dict(dash="dash")))
if "y_2y"  in US.columns: fig1.add_trace(go.Scatter(x=US["date"], y=US["y_2y"],  name="US 2Y",  mode="lines"))
if "y_2y"  in EU.columns: fig1.add_trace(go.Scatter(x=EU["date"], y=EU["y_2y"],  name="EU 2Y",  mode="lines", line=dict(dash="dash")))
fig1.update_layout(margin=dict(l=10,r=10,t=10,b=10), yaxis_title="Yield (%)", xaxis_title="Date")
fig1.update_xaxes(range=[start_date, end_date])
st.plotly_chart(fig1, use_container_width=True)

# ------------ 10Y–2Y + differential ------------
st.subheader("Tijdreeks — 10Y–2Y & US–EU differential")
fig2 = go.Figure()
if "spread_10_2" in US.columns: fig2.add_trace(go.Scatter(x=US["date"], y=US["spread_10_2"], name="US 10Y–2Y", mode="lines"))
if "spread_10_2" in EU.columns: fig2.add_trace(go.Scatter(x=EU["date"], y=EU["spread_10_2"], name="EU 10Y–2Y", mode="lines", line=dict(dash="dash")))
if "spread_10_2" in US.columns and "spread_10_2" in EU.columns:
    j = pd.merge(US[["date","spread_10_2"]], EU[["date","spread_10_2"]], on="date", suffixes=("_us","_eu"))
    j["diff_bp"] = (j["spread_10_2_us"] - j["spread_10_2_eu"]) * 100.0
    fig2.add_trace(go.Bar(x=j["date"], y=j["diff_bp"], name="US–EU (bp)", opacity=0.4))
fig2.update_layout(margin=dict(l=10,r=10,t=10,b=10), yaxis_title="Spread (pp) & Δ (bp)", xaxis_title="Date", barmode="overlay")
fig2.update_xaxes(range=[start_date, end_date])
st.plotly_chart(fig2, use_container_width=True)

# ------------ EU fragmentatie (optioneel) ------------
st.subheader("EU fragmentatie — OAT–Bund / BTP–Bund (indien aanwezig)")
def first_available(df: pd.DataFrame, names: list[str]) -> str | None:
    for n in names:
        if n in df.columns: return n
    return None
oat = first_available(EU, ["oat_bund_spread","oat_bund_10y_spread","oat_bund","fr_de_10y_spread"])
btp = first_available(EU, ["btp_bund_spread","btp_bund_10y_spread","it_de_10y_spread"])
if oat or btp:
    F = go.Figure()
    if oat: F.add_trace(go.Scatter(x=EU["date"], y=EU[oat], name=oat.upper().replace("_"," "), mode="lines"))
    if btp: F.add_trace(go.Scatter(x=EU["date"], y=EU[btp], name=btp.upper().replace("_"," "), mode="lines"))
    F.update_layout(margin=dict(l=10,r=10,t=10,b=10), yaxis_title="Spread t.o.v. Bund (pp)", xaxis_title="Date")
    F.update_xaxes(range=[start_date, end_date])
    st.plotly_chart(F, use_container_width=True)
else:
    st.caption("Geen fragmentatie-kolommen gevonden in de EU-view (optioneel).")

# ------------ Deltas (1d/7d/30d) ------------
st.subheader("Deltas — histogram & tijdreeks")

if   delta_h == "1d": suf="_d1_bp"; is_bp=True
elif delta_h == "7d": suf="_d7";    is_bp=False
else:                  suf="_d30";   is_bp=False

bases = [("y_3m","3M"),("y_2y","2Y"),("y_5y","5Y"),("y_10y","10Y"),("y_30y","30Y"),
         ("spread_10_2","10Y-2Y"),("spread_30_10","30Y-10Y")]

# kies default metric (10Y als beschikbaar)
def_idx = next((i for i,(b,_) in enumerate(bases) if b=="y_10y"), 0)
b_sel, label_sel = st.selectbox("Metric", bases, index=def_idx, format_func=lambda t: t[1])

def get_delta_series(df: pd.DataFrame, base: str) -> pd.Series:
    if suf == "_d1_bp":
        if f"{base}_d1_bp" in df.columns:
            return pd.to_numeric(df[f"{base}_d1_bp"], errors="coerce")
        return pd.to_numeric(df[base], errors="coerce").diff() * 100.0
    else:
        # pp -> bp
        col = f"{base}{suf}"
        if col not in df.columns:
            return pd.Series(index=df.index, dtype=float)
        return pd.to_numeric(df[col], errors="coerce") * 100.0

USd = get_delta_series(US, b_sel)
EUd = get_delta_series(EU, b_sel)

# relatieve %: Δpp / vorige pp * 100
if suf == "_d1_bp":
    dpp_US = USd / 100.0
    dpp_EU = EUd / 100.0
    baseUS = pd.to_numeric(US[b_sel], errors="coerce")
    baseEU = pd.to_numeric(EU[b_sel], errors="coerce")
else:
    dpp_US = pd.to_numeric(US.get(f"{b_sel}{suf}", pd.Series(index=US.index)), errors="coerce")
    dpp_EU = pd.to_numeric(EU.get(f"{b_sel}{suf}", pd.Series(index=EU.index)), errors="coerce")
    baseUS = pd.to_numeric(US[b_sel], errors="coerce")
    baseEU = pd.to_numeric(EU[b_sel], errors="coerce")

pctUS = (dpp_US / baseUS.shift(1).replace(0,np.nan)) * 100.0
pctEU = (dpp_EU / baseEU.shift(1).replace(0,np.nan)) * 100.0

h1, h2 = st.columns(2)
with h1:
    H = go.Figure()
    H.add_trace(go.Histogram(x=USd.replace([np.inf,-np.inf],np.nan).dropna(), nbinsx=40, name="US", opacity=0.6))
    H.add_trace(go.Histogram(x=EUd.replace([np.inf,-np.inf],np.nan).dropna(), nbinsx=40, name="EU", opacity=0.6))
    H.update_layout(title=f"Δ {label_sel} — abs (bp)", barmode="overlay",
                    margin=dict(l=10,r=10,t=40,b=10), xaxis_title="Δ (bp)", yaxis_title="Aantal dagen")
    st.plotly_chart(H, use_container_width=True)
with h2:
    H2 = go.Figure()
    H2.add_trace(go.Histogram(x=pctUS.replace([np.inf,-np.inf],np.nan).dropna(), nbinsx=40, name="US", opacity=0.6))
    H2.add_trace(go.Histogram(x=pctEU.replace([np.inf,-np.inf],np.nan).dropna(), nbinsx=40, name="EU", opacity=0.6))
    H2.update_layout(title=f"Δ {label_sel} — relatief (%)", barmode="overlay",
                     margin=dict(l=10,r=10,t=40,b=10), xaxis_title="Δ (%)", yaxis_title="Aantal dagen")
    st.plotly_chart(H2, use_container_width=True)

# Δ tijdreeks + differential
dfJ = pd.DataFrame({"date":US["date"].values, "US":USd.values}).merge(
      pd.DataFrame({"date":EU["date"].values, "EU":EUd.values}), on="date", how="inner")
dfJ["USminusEU"] = dfJ["US"] - dfJ["EU"]
figd = go.Figure()
figd.add_trace(go.Bar(x=dfJ["date"], y=dfJ["US"], name=f"US Δ{delta_h}", opacity=0.6))
figd.add_trace(go.Bar(x=dfJ["date"], y=dfJ["EU"], name=f"EU Δ{delta_h}", opacity=0.6))
figd.add_trace(go.Scatter(x=dfJ["date"], y=dfJ["USminusEU"], name="Δ(US–EU)", mode="lines", line=dict(width=2)))
figd.add_hline(y=0, line_width=1, line_color="gray", opacity=0.5)
figd.update_layout(margin=dict(l=10,r=10,t=10,b=10), barmode="overlay", yaxis_title="Δ (bp)", xaxis_title="Date")
figd.update_xaxes(range=[start_date, end_date])
st.plotly_chart(figd, use_container_width=True)

# ------------ Tabel & download ------------
if show_table:
    st.subheader("Tabel (US & EU, gemeenschappelijke datums)")
    merged = pd.merge(US, EU, on="date", suffixes=("_us","_eu"))
    st.dataframe(merged.sort_values("date", ascending=False).round(round_dp))
csv = pd.merge(US, EU, on="date", suffixes=("_us","_eu")).to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download CSV (US & EU, gefilterd)", data=csv,
                   file_name="yield_compare_us_eu.csv", mime="text/csv")
