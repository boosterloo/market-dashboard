# pages/4_Yield_Curve.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from utils.bq import run_query

st.set_page_config(page_title="Yield Curve Dashboard", layout="wide")
st.title("🧯 Yield Curve Dashboard")

PROJECT_ID = st.secrets["gcp_service_account"]["project_id"]
TABLES = st.secrets.get("tables", {})
# Lees de view uit secrets; val terug op de standaardnaam als niet gezet:
YIELD_VIEW = TABLES.get(
    "yield_view",
    f"{PROJECT_ID}.marketdata.yield_curve_analysis_wide"
)

# ------------------------- Data ophalen -------------------------
with st.spinner("Data ophalen uit BigQuery…"):
    sql = f"""
    SELECT
      date,
      SAFE_CAST(y_3m AS FLOAT64)           AS y_3m,
      SAFE_CAST(y_2y_synth AS FLOAT64)     AS y_2y,
      SAFE_CAST(y_5y AS FLOAT64)           AS y_5y,
      SAFE_CAST(y_10y AS FLOAT64)          AS y_10y,
      SAFE_CAST(y_30y AS FLOAT64)          AS y_30y,
      SAFE_CAST(spread_10_2 AS FLOAT64)    AS spread_10_2,
      SAFE_CAST(spread_30_10 AS FLOAT64)   AS spread_30_10,
      snapshot_date
    FROM `{YIELD_VIEW}`
    ORDER BY date
    """
    df = run_query(sql)

if df.empty:
    st.warning("Geen data gevonden in de view. Controleer of de view gevuld is.")
    st.stop()

# ------------------------- Filters -------------------------
cA, cB, cC, cD = st.columns([1.2,1.2,1,1.2])
with cA:
    last_n = st.number_input("Laatste N dagen (0 = alles)", value=365, min_value=0, step=50)
with cB:
    filter_mode = st.selectbox(
        "Filter lege rijen",
        ["Strikt (alle looptijden aanwezig)", "Licht (minstens 2Y & 10Y)"],
        index=0
    )
with cC:
    round_dp = st.slider("Decimalen", 1, 4, 2)
with cD:
    show_table = st.toggle("Tabel tonen", value=False)

df_f = df.copy()

if filter_mode.startswith("Strikt"):
    df_f = df_f.dropna(subset=["y_3m","y_2y","y_5y","y_10y","y_30y"])
else:
    df_f = df_f.dropna(subset=["y_2y","y_10y"])

if last_n and last_n > 0:
    df_f = df_f.iloc[-last_n:]

if df_f.empty:
    st.info("Na filteren is er geen data over.")
    st.stop()

# ------------------------- Snapshot keuze -------------------------
st.sidebar.header("Snapshot")
dates = list(df_f["date"].dropna().unique())
sel_date = st.sidebar.selectbox("Kies datum", dates, index=len(dates)-1, format_func=str)
snap = df_f[df_f["date"] == sel_date].tail(1)

# ------------------------- KPI’s -------------------------
def fmt(x): 
    return "—" if pd.isna(x) else f"{round(float(x), round_dp)}%"

k1, k2, k3, k4, k5 = st.columns(5)
with k1: st.metric("3M", fmt(snap["y_3m"].values[0] if not snap.empty else None))
with k2: st.metric("2Y", fmt(snap["y_2y"].values[0] if not snap.empty else None))
with k3: st.metric("5Y", fmt(snap["y_5y"].values[0] if not snap.empty else None))
with k4: st.metric("10Y", fmt(snap["y_10y"].values[0] if not snap.empty else None))
with k5: st.metric("30Y", fmt(snap["y_30y"].values[0] if not snap.empty else None))

# ------------------------- Charts -------------------------
c1, c2 = st.columns([1.4,1])

# Term structure (snapshot)
with c1:
    st.subheader(f"Term Structure • {sel_date}")
    maturities = ["3M","2Y","5Y","10Y","30Y"]
    values = [
        snap["y_3m"].values[0] if not snap.empty else None,
        snap["y_2y"].values[0] if not snap.empty else None,
        snap["y_5y"].values[0] if not snap.empty else None,
        snap["y_10y"].values[0] if not snap.empty else None,
        snap["y_30y"].values[0] if not snap.empty else None,
    ]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=maturities, y=values, mode="lines+markers", name="Snapshot"))
    fig.update_layout(margin=dict(l=10,r=10,t=10,b=10), yaxis_title="Yield (%)", xaxis_title="Maturity")
    st.plotly_chart(fig, use_container_width=True)

# Spreads (tijdreeks)
with c2:
    st.subheader("Spreads (10Y-2Y, 30Y-10Y)")
    sp = go.Figure()
    sp.add_trace(go.Scatter(x=df_f["date"], y=df_f["spread_10_2"], name="10Y - 2Y"))
    sp.add_trace(go.Scatter(x=df_f["date"], y=df_f["spread_30_10"], name="30Y - 10Y"))
    sp.update_layout(margin=dict(l=10,r=10,t=10,b=10), yaxis_title="Spread (pp)", xaxis_title="Date")
    st.plotly_chart(sp, use_container_width=True)

# Yields (tijdreeks)
st.subheader("Rentes per looptijd (tijdreeks)")
select_cols = st.multiselect(
    "Selecteer looptijden",
    ["y_3m","y_2y","y_5y","y_10y","y_30y"],
    default=["y_2y","y_10y","y_30y"]
)
if select_cols:
    yf = go.Figure()
    for col in select_cols:
        yf.add_trace(go.Scatter(x=df_f["date"], y=df_f[col], name=col.upper()))
    yf.update_layout(margin=dict(l=10,r=10,t=10,b=10), yaxis_title="Yield (%)", xaxis_title="Date")
    st.plotly_chart(yf, use_container_width=True)

# Heatmap
st.subheader("Heatmap van rentes")
hm = df_f[["date","y_3m","y_2y","y_5y","y_10y","y_30y"]].set_index("date")
hfig = go.Figure(data=go.Heatmap(
    z=hm.T.values,
    x=hm.index.astype(str),
    y=["3M","2Y","5Y","10Y","30Y"],
    coloraxis="coloraxis"
))
hfig.update_layout(margin=dict(l=10,r=10,t=10,b=10), coloraxis_colorscale="Viridis")
st.plotly_chart(hfig, use_container_width=True)

# Tabel + download
if show_table:
    st.subheader("Tabel")
    st.dataframe(df_f.sort_values("date", ascending=False).round(round_dp))

csv = df_f.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download CSV (gefilterd)", data=csv, file_name="yield_curve_filtered.csv", mime="text/csv")

st.caption(f"Bron: {YIELD_VIEW} • Lege datums gefilterd volgens gekozen modus.")
