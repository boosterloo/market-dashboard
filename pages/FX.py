# pages/fx_dashboard.py
# Verwachte kolommen per pair:
# <pair>_close, <pair>_delta_abs, <pair>_delta_pct, <pair>_ma50, <pair>_ma200, optioneel <pair>_rv20, <pair>_atr14

import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
import plotly.graph_objects as go

# ---------- BigQuery helpers ----------
try:
    from utils.bq import run_query, bq_ping
except Exception:
    import google.cloud.bigquery as bq
    from google.oauth2 import service_account

    @st.cache_resource(show_spinner=False)
    def _bq_client_from_secrets():
        creds = st.secrets["gcp_service_account"]
        credentials = service_account.Credentials.from_service_account_info(creds)
        return bq.Client(credentials=credentials, project=creds["project_id"])

    @st.cache_data(ttl=600, show_spinner=False)
    def run_query(sql: str) -> pd.DataFrame:
        client = _bq_client_from_secrets()
        return client.query(sql).to_dataframe()

    def bq_ping() -> bool:
        try:
            _bq_client_from_secrets().query("SELECT 1").result(timeout=10)
            return True
        except Exception:
            return False

# ---------- Page config ----------
st.set_page_config(page_title="🌍 FX Dashboard", layout="wide")
st.title("🌍 FX Dashboard")

FX_VIEW = st.secrets.get("tables", {}).get(
    "fx_wide_view",
    "nth-pier-468314-p7.marketdata.fx_rates_dashboard_v"
)

# ---------- Health check ----------
if not bq_ping():
    st.error("Geen BigQuery-verbinding. Controleer Secrets ([gcp_service_account]).")
    st.stop()

# ---------- Data laden ----------
@st.cache_data(ttl=600, show_spinner=False)
def load_data() -> pd.DataFrame:
    df = run_query(f"SELECT * FROM `{FX_VIEW}` ORDER BY date")
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"]).dt.date
    for c in df.columns:
        if c != "date":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

df_wide = load_data()
if df_wide.empty:
    st.warning("Geen data uit FX-view.")
    st.stop()

# ---------- Pairs & ordering ----------
close_cols = [c for c in df_wide.columns if c.endswith("_close")]
pairs_all = sorted([c.removesuffix("_close") for c in close_cols])

# Relevantievolgorde (hoog → laag). Rest volgt alfabetisch.
priority = ["eurusd", "gbpusd", "usdjpy", "usdchf", "audusd", "nzdusd"]
ordered = [p for p in priority if p in pairs_all] + [p for p in pairs_all if p not in priority]

# ---------- UI: periode (full width) + MA type ----------
min_d, max_d = df_wide["date"].min(), df_wide["date"].max()
default_start = max_d - timedelta(days=365)
default_start = default_start if default_start > min_d else min_d

with st.container():
    start, end = st.slider(
        "📅 Periode",
        min_value=min_d, max_value=max_d,
        value=(default_start, max_d),
        step=timedelta(days=1),
        format="YYYY-MM-DD",
    )

avg_mode = st.radio("Gemiddelde", ["EMA", "SMA"], index=0, horizontal=True)

# ---------- Filter ----------
mask = (df_wide["date"] >= start) & (df_wide["date"] <= end)
df = df_wide.loc[mask].sort_values("date").copy()

# ---------- Helpers ----------
def cols_for(p: str) -> dict:
    return {
        "close": f"{p}_close",
        "d_abs": f"{p}_delta_abs",
        "d_pct": f"{p}_delta_pct",
        "ma50": f"{p}_ma50",
        "ma200": f"{p}_ma200",
        "rv20": f"{p}_rv20",
        "atr14": f"{p}_atr14",
    }

def compute_ma(series: pd.Series, kind: str, window: int) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").astype(float)
    if kind == "EMA":
        return s.ewm(span=window, adjust=False, min_periods=window).mean()
    return s.rolling(window=window, min_periods=window).mean()

def _f(s):  # to float
    return pd.to_numeric(s, errors="coerce").astype(float)

# Kleuren
COLOR_PRICE   = "#111111"  # near-black
COLOR_MA50    = "#009E73"  # green
COLOR_MA200   = "#0072B2"  # blue
COLOR_BAR_POS = "#009E73"  # green
COLOR_BAR_NEG = "#D55E00"  # red

# ---------- Render per pair (price top, delta onder) ----------
for p in ordered:
    c = cols_for(p)
    has_close = c["close"] in df.columns and df[c["close"]].notna().any()

    # Sla pair over als er geen sluitingsprijs is
    if not has_close:
        continue

    st.markdown(f"## {p.upper()}")

    # --- Koers + MA/EMA(50/200) ---
    sub = df[["date", c["close"]]].dropna().copy()
    sub[c["close"]] = _f(sub[c["close"]])

    ma50  = compute_ma(sub[c["close"]], avg_mode, 50)
    ma200 = compute_ma(sub[c["close"]], avg_mode, 200)

    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(
        x=sub["date"], y=sub[c["close"]],
        name="Close", line=dict(width=2, color=COLOR_PRICE)
    ))
    if ma50.notna().any():
        fig_price.add_trace(go.Scatter(
            x=sub["date"], y=ma50.values,
            name=("EMA50" if avg_mode == "EMA" else "MA50"),
            line=dict(width=2, color=COLOR_MA50)
        ))
    if ma200.notna().any():
        fig_price.add_trace(go.Scatter(
            x=sub["date"], y=ma200.values,
            name=("EMA200" if avg_mode == "EMA" else "MA200"),
            line=dict(width=2, color=COLOR_MA200)
        ))

    fig_price.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    )
    st.plotly_chart(fig_price, use_container_width=True)

    # --- Δ% bars ---
    if c["d_pct"] in df.columns and df[c["d_pct"]].notna().any():
        bars = df[["date", c["d_pct"]]].dropna().copy()
        bars[c["d_pct"]] = _f(bars[c["d_pct"]]) * 100.0
    else:
        tmp = df[["date", c["close"]]].dropna().copy()
        tmp[c["close"]] = _f(tmp[c["close"]])
        tmp["pct"] = tmp[c["close"]].pct_change() * 100.0
        bars = tmp[["date", "pct"]].dropna().rename(columns={"pct": c["d_pct"]})

    if not bars.empty:
        colors = [COLOR_BAR_POS if v >= 0 else COLOR_BAR_NEG for v in bars[c["d_pct"]].values]
        fig_delta = go.Figure()
        fig_delta.add_trace(go.Bar(
            x=bars["date"], y=bars[c["d_pct"]],
            name="Δ% per dag", marker_color=colors, opacity=0.9
        ))
        # 0-lijn
        fig_delta.add_hline(y=0, line_dash="dot", opacity=0.6)

        fig_delta.update_yaxes(title_text="Δ% dag")
        fig_delta.update_layout(
            height=300,
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
        )
        st.plotly_chart(fig_delta, use_container_width=True)

    st.markdown("---")

# ---------- Optioneel: compacte tabel met laatste rijen ----------
st.subheader("Laatste rijen (gefilterd bereik)")
show_cols = ["date"]
for p in ordered:
    c = cols_for(p)
    show_cols += [c["close"], c["d_abs"], c["d_pct"], c["ma50"], c["ma200"], c["rv20"], c["atr14"]]
show_cols = [c for c in show_cols if c in df.columns]
st.dataframe(df[show_cols].tail(200), use_container_width=True)
