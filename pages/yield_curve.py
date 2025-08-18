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

# --- Haal kolommen van de view op (via INFORMATION_SCHEMA) ---
def list_columns(fqtn: str) -> set[str]:
    proj, dset, tbl = fqtn.split(".")
    sql = f"""
    SELECT column_name
    FROM `{proj}.{dset}.INFORMATION_SCHEMA.COLUMNS`
    WHERE table_name = @tbl
    """
    dfc = run_query(sql, params={"tbl": tbl}, timeout=30)
    return set([c.lower() for c in dfc["column_name"].astype(str)])

try:
    cols = list_columns(YIELD_VIEW)
except Exception as e:
    st.error(f"Kon kolommen niet ophalen voor `{YIELD_VIEW}`.\nDetails: {type(e).__name__}")
    st.stop()

# --- Bepaal welke kolommen we veilig kunnen selecteren ---
have = lambda c: c.lower() in cols

y2y_col = "y_2y_synth" if have("y_2y_synth") else ("y_2y" if have("y_2y") else None)
if not y2y_col:
    st.error(f"`{YIELD_VIEW}` bevat geen `y_2y_synth` of `y_2y`. Voeg één van beide toe.")
    st.stop()

select_parts = ["date"]
for name in [("y_3m","y_3m"), (y2y_col,"y_2y"), ("y_5y","y_5y"), ("y_10y","y_10y"), ("y_30y","y_30y")]:
    src, alias = name
    if have(src):
        select_parts.append(f"SAFE_CAST({src} AS FLOAT64) AS {alias}")

# spreads en snapshot_date alleen als ze bestaan
if have("spread_10_2"):
    select_parts.append("SAFE_CAST(spread_10_2 AS FLOAT64) AS spread_10_2")
if have("spread_30_10"):
    select_parts.append("SAFE_CAST(spread_30_10 AS FLOAT64) AS spread_30_10")
if have("snapshot_date"):
    select_parts.append("snapshot_date")

select_sql = ",\n  ".join(select_parts)
sql = f"SELECT\n  {select_sql}\nFROM `{YIELD_VIEW}`\nORDER BY date"

# --- Data laden ---
try:
    with st.spinner("Data ophalen uit BigQuery…"):
        df = run_query(sql, timeout=60)
except Exception as e:
    st.error(f"Kon de view niet lezen: {YIELD_VIEW}\nDetails: {type(e).__name__}")
    with st.expander("Debug: gegenereerde SELECT"):
        st.code(sql, language="sql")
    st.stop()

if df.empty:
    st.warning("Geen data gevonden (na query).")
    st.stop()

# --- Filters ---
cA, cB, cC, cD = st.columns([1.2,1.2,1,1.2])
with cA:
    last_n = st.number_input("Laatste N dagen (0 = alles)", value=365, min_value=0, step=50)
with cB:
    strict = st.toggle("Strikt filter (alle looptijden aanwezig)", value=False)
with cC:
    round_dp = st.slider("Decimalen", 1, 4, 2)
with cD:
    show_table = st.toggle("Tabel tonen", value=False)

df_f = df.copy()

needed = ["y_3m","y_2y","y_5y","y_10y","y_30y"]
present = [c for c in needed if c in df_f.columns]

if strict and present:
    df_f = df_f.dropna(subset=present)
else:
    subset = [c for c in ["y_2y","y_10y"] if c in df_f.columns]
    if subset:
        df_f = df_f.dropna(subset=subset)

if last_n and last_n > 0:
    df_f = df_f.iloc[-last_n:]

if df_f.empty:
    st.info("Na filteren geen data over.")
    st.stop()

# --- Snapshot ---
st.sidebar.header("Snapshot")
dates = list(df_f["date"].dropna().unique())
sel_date = st.sidebar.selectbox("Kies datum", dates, index=len(dates)-1, format_func=str)
snap = df_f[df_f["date"] == sel_date].tail(1)

# --- KPI’s (alleen tonen als kolom bestaat) ---
def fmt(x): 
    return "—" if pd.isna(x) else f"{round(float(x), round_dp)}%"

kcols = st.columns(5)
labels = ["y_3m","y_2y","y_5y","y_10y","y_30y"]
for i, lab in enumerate(labels):
    val = fmt(snap[lab].values[0]) if (lab in snap.columns and not snap.empty) else "—"
    kcols[i].metric(lab.upper().replace("_",""), val)

# --- Charts ---
c1, c2 = st.columns([1.4,1])

# Term structure (snapshot)
with c1:
    st.subheader(f"Term Structure • {sel_date}")
    maturities, values = [], []
    order = [("y_3m","3M"), ("y_2y","2Y"), ("y_5y","5Y"), ("y_10y","10Y"), ("y_30y","30Y")]
    for col, label in order:
        if col in snap.columns:
            maturities.append(label)
            values.append(snap[col].values[0] if not snap.empty else None)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=maturities, y=values, mode="lines+markers", name="Snapshot"))
    fig.update_layout(margin=dict(l=10,r=10,t=10,b=10), yaxis_title="Yield (%)", xaxis_title="Maturity")
    st.plotly_chart(fig, use_container_width=True)

# Spreads (indien beschikbaar)
with c2:
    if "spread_10_2" in df_f.columns or "spread_30_10" in df_f.columns:
        st.subheader("Spreads")
        sp = go.Figure()
        if "spread_10_2" in df_f.columns:
            sp.add_trace(go.Scatter(x=df_f["date"], y=df_f["spread_10_2"], name="10Y - 2Y"))
        if "spread_30_10" in df_f.columns:
            sp.add_trace(go.Scatter(x=df_f["date"], y=df_f["spread_30_10"], name="30Y - 10Y"))
        sp.update_layout(margin=dict(l=10,r=10,t=10,b=10), yaxis_title="Spread (pp)", xaxis_title="Date")
        st.plotly_chart(sp, use_container_width=True)
    else:
        st.subheader("Spreads")
        st.info("Spreads niet beschikbaar in de view.")

# Yields (tijdreeks)
st.subheader("Rentes per looptijd (tijdreeks)")
avail_yields = [c for c in ["y_3m","y_2y","y_5y","y_10y","y_30y"] if c in df_f.columns]
default_sel = [c for c in ["y_2y","y_10y","y_30y"] if c in avail_yields] or avail_yields[:2]
sel = st.multiselect("Selecteer looptijden", avail_yields, default=default_sel)
if sel:
    yf = go.Figure()
    for col in sel:
        yf.add_trace(go.Scatter(x=df_f["date"], y=df_f[col], name=col.upper()))
    yf.update_layout(margin=dict(l=10,r=10,t=10,b=10), yaxis_title="Yield (%)", xaxis_title="Date")
    st.plotly_chart(yf, use_container_width=True)

# Tabel + download
if show_table:
    st.subheader("Tabel")
    st.dataframe(df_f.sort_values("date", ascending=False).round(round_dp))

csv = df_f.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download CSV (gefilterd)", data=csv, file_name="yield_curve_filtered.csv", mime="text/csv")

with st.expander("Debug: beschikbare kolommen"):
    st.write(sorted(list(cols)))
