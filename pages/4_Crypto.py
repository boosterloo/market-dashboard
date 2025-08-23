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

# ---- Helpers ----
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

# ---- Per asset: twee grafieken naast elkaar ----
st.subheader("Per asset")
for name in pick:
    a = name.lower()
    st.markdown(f"### {name}")

    c_left, c_right = st.columns(2)

    # Links: prijs + MA7/MA30 (lijnen)
    price_c = col(a, "price")
    ma7_c   = col(a, "ma7")
    ma30_c  = col(a, "ma30")
    if all(c in d.columns for c in [price_c, ma7_c, ma30_c]):
        sub = d[["date", price_c, ma7_c, ma30_c]].copy()
        with c_left:
            fig1 = make_subplots(specs=[[{"secondary_y": False}]])
            fig1.add_trace(go.Scatter(x=sub["date"], y=sub[price_c], name=f"{name} prijs"))
            fig1.add_trace(go.Scatter(x=sub["date"], y=sub[ma7_c],   name="MA7"))
            fig1.add_trace(go.Scatter(x=sub["date"], y=sub[ma30_c],  name="MA30"))
            fig1.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig1, use_container_width=True)
    else:
        with c_left:
            st.info("Nog geen prijs/MA-data voor deze selectie.")

    # Rechts: Δ% als staven + YTD% als lijn (secundaire as)
    dcol  = col(a, "delta_pct")
    ytd_c = col(a, "ytd_pct")
    has_delta = dcol in d.columns
    has_ytd   = ytd_c in d.columns

    if has_delta or has_ytd:
        with c_right:
            fig2 = make_subplots(specs=[[{"secondary_y": True}]])
            if has_delta:
                s = d.set_index("date")[dcol].astype(float)
                s = s[~s.index.duplicated(keep="last")]
                fig2.add_trace(go.Bar(x=s.index, y=s.values, name="Δ% per dag"), secondary_y=False)
                try:
                    fig2.add_hline(y=0, line_dash="dot", opacity=0.5)
                except Exception:
                    pass
                fig2.update_yaxes(title_text="Δ% dag", secondary_y=False)
            if has_ytd:
                y = d.set_index("date")[ytd_c].astype(float)
                # reindex op delta-index zodat assen gelijk lopen als beide bestaan
                if has_delta:
                    y = y.reindex(s.index)
                fig2.add_trace(go.Scatter(x=y.index, y=y.values, name="YTD%"), secondary_y=True)
                fig2.update_yaxes(title_text="YTD%", secondary_y=True)

            fig2.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig2, use_container_width=True)
    else:
        with c_right:
            st.info("Geen Δ%/YTD beschikbaar voor deze selectie.")

# ---- Tabel laatste 30 rijen (eerste asset) ----
if pick:
    a0 = pick[0].lower()
    cols_tbl = [c for c in [col(a0, "price"), col(a0, "ma7"), col(a0, "ma30"),
                            col(a0, "delta_pct"), col(a0, "ytd_pct")] if c in d.columns]
    if cols_tbl:
        tbl = d[["date"] + cols_tbl].tail(30).copy()
        rename_map = {
            col(a0, "price"): "PRICE",
            col(a0, "ma7"): "MA7",
            col(a0, "ma30"): "MA30",
            col(a0, "delta_pct"): "DELTA_%",  # dag %
            col(a0, "ytd_pct"): "YTD_%"
        }
        tbl = tbl.rename(columns=rename_map)
        st.subheader(f"Laatst 30 rijen – {pick[0]}")
        st.dataframe(tbl, use_container_width=True)
