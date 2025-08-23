# pages/4_Crypto.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from utils.bq import run_query, bq_ping

st.set_page_config(page_title="Crypto Dashboard", layout="wide")
st.title("🪙 Crypto Dashboard")

# ---- BigQuery check ----
if not bq_ping():
    st.error("Geen BigQuery-verbinding (controleer secrets).")
    st.stop()

# ---- Viewnaam ----
CRYPTO_WIDE_VIEW = st.secrets.get("tables", {}).get(
    "crypto_daily_wide",
    "nth-pier-468314-p7.marketdata.crypto_daily_wide"  # fallback
)

# ---- Data laden ----
sql = f"SELECT * FROM `{CRYPTO_WIDE_VIEW}` ORDER BY date"
df = run_query(sql)
if df.empty:
    st.warning("Geen data in de view.")
    st.stop()

df["date"] = pd.to_datetime(df["date"]).dt.date

# ---- Assets afleiden uit kolommen ----
all_assets = sorted({c.split("_", 1)[1] for c in df.columns if c.startswith("price_")})
default_assets = [a for a in ["btc", "eth", "sol", "bnb"] if a in all_assets] or all_assets[:4]

# ---- Filters ----
min_d, max_d = df["date"].min(), df["date"].max()
c1, c2 = st.columns([2, 1])
with c1:
    start, end = st.slider(
        "Periode",
        min_value=min_d, max_value=max_d,
        value=(max(max_d - timedelta(days=365), min_d), max_d),
        step=timedelta(days=1), format="YYYY-MM-DD",
    )
with c2:
    pick = st.multiselect(
        "Assets",
        options=[a.upper() for a in all_assets],
        default=[a.upper() for a in default_assets],
    )

mask = (df["date"] >= start) & (df["date"] <= end)
d = df.loc[mask].copy()

# ---- Helper ----
def col(asset: str, metric: str) -> str:
    return f"{metric}_{asset}"

def last_non_null(series: pd.Series):
    idx = series.last_valid_index()
    return series.loc[idx] if idx is not None else np.nan

# ---- Cards (laatste dag in selectie) ----
st.subheader("Overzicht")
cols = st.columns(4)
for i, name in enumerate(pick):
    a = name.lower()
    price_col = col(a, "price")
    delta_col = col(a, "delta_pct")
    if price_col not in d.columns:
        continue
    p = last_non_null(d[price_col])
    dp = last_non_null(d.get(delta_col, pd.Series(dtype=float)))
    with cols[i % 4]:
        st.metric(label=name, value=f"{p:.2f}" if pd.notna(p) else "—",
                  delta=f"{dp:.2f}%" if pd.notna(dp) else "—")

# ---- Detail: Price + MA7/MA30 (tabs) ----
st.subheader("Detailgrafiek (prijs + MA7/MA30)")
tabs = st.tabs(pick)
for t, name in zip(tabs, pick):
    a = name.lower()
    needed = [col(a, "price"), col(a, "ma7"), col(a, "ma30")]
    if not all(c in d.columns for c in needed):
        continue
    sub = d[["date"] + needed].copy()
    fig = make_subplots(specs=[[{"secondary_y": False}]])
    fig.add_trace(go.Scatter(x=sub["date"], y=sub[col(a, "price")], name=f"{name} prijs"))
    fig.add_trace(go.Scatter(x=sub["date"], y=sub[col(a, "ma7")],   name="MA7"))
    fig.add_trace(go.Scatter(x=sub["date"], y=sub[col(a, "ma30")],  name="MA30"))
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10))
    t.plotly_chart(fig, use_container_width=True)

# ---- Vergelijking genormaliseerd (start=100) ----
st.subheader("Vergelijking (genormaliseerd naar 100 op startdatum)")
norm = pd.DataFrame({"date": d["date"]}).drop_duplicates().set_index("date")
for name in pick:
    a = name.lower()
    pc = col(a, "price")
    if pc in d.columns:
        s = d.set_index("date")[pc].astype(float)
        s = s.dropna()
        if len(s):
            base = s.iloc[0]
            if base != 0:
                norm[name] = (d.set_index("date")[pc] / base) * 100.0
if norm.shape[1]:
    st.line_chart(norm, use_container_width=True)
else:
    st.info("Geen series om te normaliseren in deze periode.")

# ---- Dagelijkse Δ% ----
st.subheader("Dagelijkse verandering (%)")
delta_wide = pd.DataFrame(index=d["date"])
for name in pick:
    a = name.lower()
    colname = col(a, "delta_pct")
    if colname in d.columns:
        delta_wide[name] = d[colname]
if delta_wide.shape[1]:
    st.line_chart(delta_wide.set_index("date"), use_container_width=True)

# ---- YTD% ----
st.subheader("YTD performance (%)")
ytd_wide = pd.DataFrame(index=d["date"])
for name in pick:
    a = name.lower()
    colname = col(a, "ytd_pct")
    if colname in d.columns:
        ytd_wide[name] = d[colname]
if ytd_wide.shape[1]:
    st.line_chart(ytd_wide.set_index("date"), use_container_width=True)

# ---- Tabel laatste 30 rijen (eerste asset) ----
if pick:
    a0 = pick[0].lower()
    cols_tbl = [c for c in [col(a0, "price"), col(a0, "ma7"), col(a0, "ma30"),
                            col(a0, "delta_pct"), col(a0, "ytd_pct")] if c in d.columns]
    if cols_tbl:
        tbl = d[["date"] + cols_tbl].tail(30).copy()
        tbl.columns = ["date"] + [c.split("_", 1)[0].upper() for c in cols_tbl]
        st.subheader(f"Laatst 30 rijen – {pick[0]}")
        st.dataframe(tbl, use_container_width=True)
