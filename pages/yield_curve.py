# pages/yield_curve.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from utils.bq import run_query

st.set_page_config(page_title="Yield Curve Dashboard", layout="wide")
st.title("🧯 Yield Curve Dashboard")

PROJECT_ID = st.secrets["gcp_service_account"]["project_id"]
TABLES     = st.secrets.get("tables", {})
YIELD_VIEW = TABLES.get("yield_view", f"{PROJECT_ID}.marketdata.yield_curve_analysis_wide")

# --- Kolommen ophalen & SELECT dynamisch opbouwen ---
def list_columns(fqtn: str) -> set[str]:
    proj, dset, tbl = fqtn.split(".")
    sql = f"""
    SELECT column_name
    FROM `{proj}.{dset}.INFORMATION_SCHEMA.COLUMNS`
    WHERE table_name = @tbl
    """
    dfc = run_query(sql, params={"tbl": tbl}, timeout=30)
    return set([c.lower() for c in dfc["column_name"].astype(str)])

cols = list_columns(YIELD_VIEW)

def have(c: str) -> bool: return c.lower() in cols
y2y_col = "y_2y_synth" if have("y_2y_synth") else ("y_2y" if have("y_2y") else None)
if not y2y_col:
    st.error(f"`{YIELD_VIEW}` bevat geen `y_2y_synth` of `y_2y`.")
    st.stop()

select_parts = ["date"]
for src, alias in [("y_3m","y_3m"), (y2y_col,"y_2y"), ("y_5y","y_5y"), ("y_10y","y_10y"), ("y_30y","y_30y")]:
    if have(src):
        select_parts.append(f"SAFE_CAST({src} AS FLOAT64) AS {alias}")
if have("spread_10_2"): select_parts.append("SAFE_CAST(spread_10_2 AS FLOAT64) AS spread_10_2")
if have("spread_30_10"): select_parts.append("SAFE_CAST(spread_30_10 AS FLOAT64) AS spread_30_10")
if have("snapshot_date"): select_parts.append("snapshot_date")

sql = f"SELECT\n  {', '.join(select_parts)}\nFROM `{YIELD_VIEW}`\nORDER BY date"

with st.spinner("Data ophalen uit BigQuery…"):
    df = run_query(sql, timeout=60)
if df.empty:
    st.warning("Geen data gevonden.")
    st.stop()

# ---------- Filters ----------
topA, topB, topC = st.columns([1.2,1,1])
with topA:
    strict = st.toggle("Strikt filter (alle looptijden aanwezig)", value=False,
                       help="Als uit, filtert alleen op aanwezigheid van 2Y & 10Y.")
with topB:
    round_dp = st.slider("Decimalen", 1, 4, 2)
with topC:
    show_table = st.toggle("Tabel tonen", value=False)

df_f = df.copy()
needed = ["y_3m","y_2y","y_5y","y_10y","y_30y"]
if strict:
    present = [c for c in needed if c in df_f.columns]
    if present: df_f = df_f.dropna(subset=present)
else:
    subset = [c for c in ["y_2y","y_10y"] if c in df_f.columns]
    if subset: df_f = df_f.dropna(subset=subset)

if df_f.empty:
    st.info("Na filteren geen data over.")
    st.stop()

# ---------- Periode-schuif (standaard laatste 1 jaar) ----------
dmin = pd.to_datetime(min(df_f["date"]))
dmax = pd.to_datetime(max(df_f["date"]))
default_start = max(dmin, dmax - pd.DateOffset(years=1))

st.subheader("Periode")
date_range = st.slider(
    "Selecteer periode",
    min_value=dmin.to_pydatetime().date(),
    max_value=dmax.to_pydatetime().date(),
    value=(default_start.to_pydatetime().date(), dmax.to_pydatetime().date()),
)
start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
mask = (pd.to_datetime(df_f["date"]) >= start_date) & (pd.to_datetime(df_f["date"]) <= end_date)
df_range = df_f.loc[mask].copy()
if df_range.empty:
    st.info("Geen data in de gekozen periode.")
    st.stop()

# ---------- Snapshot-keuze (onafhankelijk van periode) ----------
st.sidebar.header("Snapshot")
all_dates = list(df_f["date"].dropna().unique())
sel_date = st.sidebar.selectbox("Kies datum", all_dates, index=len(all_dates)-1, format_func=str)
snap = df_f[df_f["date"] == sel_date].tail(1)

# ---------- KPI’s ----------
def fmt(x): 
    return "—" if pd.isna(x) else f"{round(float(x), round_dp)}%"
k1, k2, k3, k4, k5 = st.columns(5)
for col, box in zip(["y_3m","y_2y","y_5y","y_10y","y_30y"], [k1,k2,k3,k4,k5]):
    val = fmt(snap[col].values[0]) if (col in snap.columns and not snap.empty) else "—"
    box.metric(col.upper().replace("_",""), val)

# ---------- Grafieken (onder elkaar) ----------

# 1) Term Structure (snapshot)
st.subheader(f"Term Structure • {sel_date}")
maturities, values = [], []
order = [("y_3m","3M"), ("y_2y","2Y"), ("y_5y","5Y"), ("y_10y","10Y"), ("y_30y","30Y")]
for col, label in order:
    if col in snap.columns:
        maturities.append(label)
        values.append(snap[col].values[0] if not snap.empty else None)
ts_fig = go.Figure()
ts_fig.add_trace(go.Scatter(x=maturities, y=values, mode="lines+markers", name="Snapshot"))
ts_fig.update_layout(margin=dict(l=10,r=10,t=10,b=10), yaxis_title="Yield (%)", xaxis_title="Maturity")
st.plotly_chart(ts_fig, use_container_width=True)

# 2) Spreads (tijdreeks, gefilterd op periode)
st.subheader("Spreads")
if "spread_10_2" in df_range.columns or "spread_30_10" in df_range.columns:
    sp = go.Figure()
    if "spread_10_2" in df_range.columns:
        sp.add_trace(go.Scatter(x=df_range["date"], y=df_range["spread_10_2"], name="10Y - 2Y"))
    if "spread_30_10" in df_range.columns:
        sp.add_trace(go.Scatter(x=df_range["date"], y=df_range["spread_30_10"], name="30Y - 10Y"))
    sp.update_layout(margin=dict(l=10,r=10,t=10,b=10), yaxis_title="Spread (pp)", xaxis_title="Date")
    st.plotly_chart(sp, use_container_width=True)
else:
    st.info("Spreads niet beschikbaar in de view.")

# 3) Rentes per looptijd (tijdreeks, gefilterd op periode)
st.subheader("Rentes per looptijd (tijdreeks)")
avail_yields = [c for c in ["y_3m","y_2y","y_5y","y_10y","y_30y"] if c in df_range.columns]
default_sel = [c for c in ["y_2y","y_10y","y_30y"] if c in avail_yields] or avail_yields[:2]
sel = st.multiselect("Selecteer looptijden", avail_yields, default=default_sel)
if sel:
    yf = go.Figure()
    for col in sel:
        yf.add_trace(go.Scatter(x=df_range["date"], y=df_range[col], name=col.upper()))
    yf.update_layout(margin=dict(l=10,r=10,t=10,b=10), yaxis_title="Yield (%)", xaxis_title="Date")
    st.plotly_chart(yf, use_container_width=True)

# 4) Heatmap (periode)
st.subheader("Heatmap van rentes")
hm = df_range[["date"] + avail_yields].set_index("date")
hfig = go.Figure(data=go.Heatmap(
    z=hm[avail_yields].T.values,
    x=hm.index.astype(str),
    y=[c.replace("y_","").upper() for c in avail_yields],
    coloraxis="coloraxis"
))
hfig.update_layout(margin=dict(l=10,r=10,t=10,b=10), coloraxis_colorscale="Viridis")
st.plotly_chart(hfig, use_container_width=True)

# Tabel + download (periode)
if show_table:
    st.subheader("Tabel")
    st.dataframe(df_range.sort_values("date", ascending=False).round(round_dp))

csv = df_range.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download CSV (gefilterd op periode)", data=csv,
                   file_name="yield_curve_filtered.csv", mime="text/csv")

with st.expander("Debug: kolommen in view"):
    st.write(sorted(list(cols)))
