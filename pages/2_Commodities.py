# pages/2_Commodities.py
# Werkt met wide-view `marketdata.commodity_prices_wide_v`
# Verwachte kolommen per instrument: <pfx>_close, _delta_abs, _delta_pct, _ma20, _ma50, _ma200

import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---- BigQuery helpers ----
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

    def bq_ping() -> bool:
        try:
            _bq_client_from_secrets().query("SELECT 1").result(timeout=10)
            return True
        except Exception:
            return False

    @st.cache_data(ttl=300, show_spinner=False)
    def run_query(sql: str) -> pd.DataFrame:
        client = _bq_client_from_secrets()
        return client.query(sql).to_dataframe()

# ---- Page config ----
st.set_page_config(page_title="Commodities", layout="wide")
st.title("🛢️ Commodities Dashboard")

COM_WIDE_VIEW = st.secrets.get("tables", {}).get(
    "commodities_wide_view",
    "nth-pier-468314-p7.marketdata.commodity_prices_wide_v"
)

# ---- Health check ----
if not bq_ping():
    st.error("Geen BigQuery-verbinding. Controleer Secrets ([gcp_service_account]).")
    st.stop()

# ---- Data laden ----
@st.cache_data(ttl=300, show_spinner=False)
def load_data() -> pd.DataFrame:
    df = run_query(f"SELECT * FROM `{COM_WIDE_VIEW}` ORDER BY date")
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"]).dt.date
    # alles behalve 'date' naar float
    for c in df.columns:
        if c != "date":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

df_wide = load_data()
if df_wide.empty:
    st.warning("Geen data in commodity_prices_wide_v.")
    st.stop()

# ---- Instrumenten afleiden ----
close_cols = [c for c in df_wide.columns if c.endswith("_close")]
prefixes = sorted([c.removesuffix("_close") for c in close_cols])

LABELS = {
    "wti": "WTI",
    "brent": "Brent",
    "gold": "Gold",
    "silver": "Silver",
    "gasoline": "Gasoline (RBOB)",
    "heatingoil": "Heating Oil",
    "natgas": "Natural Gas",
    "copper": "Copper",
}
def label_of(pfx: str) -> str:
    return LABELS.get(pfx, pfx.upper())

# ---- UI: filters & opties ----
min_d, max_d = df_wide["date"].min(), df_wide["date"].max()
default_start = max(max_d - timedelta(days=365), min_d)

c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
with c1:
    start, end = st.slider(
        "Periode",
        min_value=min_d, max_value=max_d,
        value=(default_start, max_d),
        step=timedelta(days=1),
        format="YYYY-MM-DD",
    )
with c2:
    # standaard: ALLE instrumenten
    sel = st.multiselect(
        "Instrumenten (standaard alle)",
        options=[(p, label_of(p)) for p in prefixes],
        default=[(p, label_of(p)) for p in prefixes],
        format_func=lambda t: t[1],
    )
    sel = [p for p, _ in sel] or prefixes[:]  # fallback naar alle
with c3:
    avg_mode = st.radio("Gemiddelde", ["EMA", "SMA"], index=0, horizontal=True)
with c4:
    show_pairs = st.checkbox("Paarvergelijking (dubbele y-as)", value=False)

st.caption("Elke selectie toont z’n eigen 2 grafieken: links prijs + MA/EMA 20/50/200, rechts Δ% bars + YTD% (secundaire as).")

# ---- Filter data ----
mask = (df_wide["date"] >= start) & (df_wide["date"] <= end)
df = df_wide.loc[mask].sort_values("date").copy()

# ---- Helpers ----
def cols_for(pfx: str) -> dict:
    return {
        "close": f"{pfx}_close",
        "d_abs": f"{pfx}_delta_abs",
        "d_pct": f"{pfx}_delta_pct",
        "ma20": f"{pfx}_ma20",
        "ma50": f"{pfx}_ma50",
        "ma200": f"{pfx}_ma200",
    }

def compute_ma_ema(series: pd.Series, kind: str, window: int) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").astype(float)
    if kind == "EMA":
        return s.ewm(span=window, adjust=False, min_periods=window).mean()
    # SMA
    return s.rolling(window=window, min_periods=window).mean()

def ytd_percent_series(dates: pd.Series, values: pd.Series) -> pd.Series:
    """YTD% per jaar: (value / first_value_of_year - 1) * 100"""
    s = pd.Series(values.values, index=pd.to_datetime(dates), dtype="float64")
    years = s.index.year
    base = s.groupby(years).transform(lambda x: x.dropna().iloc[0] if len(x.dropna()) else np.nan)
    return (s / base - 1.0) * 100.0

# ---- Kleuren (Okabe-Ito) ----
COLOR_PRICE   = "#111111"  # bijna zwart
COLOR_MA20    = "#E69F00"  # oranje
COLOR_MA50    = "#009E73"  # groen
COLOR_MA200   = "#0072B2"  # blauw
COLOR_BAR_POS = "#009E73"  # groen bars
COLOR_BAR_NEG = "#D55E00"  # rood bars
COLOR_YTD     = "#CC79A7"  # paars

# ---- KPI's ----
st.subheader("Kerncijfers")
kpi_cols = st.columns(min(len(sel), 4) or 1)
for i, pfx in enumerate(sel or prefixes[:1]):
    c = cols_for(pfx)
    label = label_of(pfx)
    sub = df[["date", c["close"], c["d_abs"], c["d_pct"]]].dropna(subset=[c["close"]]).copy()
    with kpi_cols[i % len(kpi_cols)]:
        if sub.empty:
            st.metric(label, value="—", delta="—")
        else:
            last = sub.iloc[-1]
            val = float(last[c["close"]])
            d_abs = float(last[c["d_abs"]]) if pd.notnull(last[c["d_abs"]]) else 0.0
            d_pct = float(last[c["d_pct"]]) if pd.notnull(last[c["d_pct"]]) else 0.0
            st.metric(label, value=f"{val:,.2f}", delta=f"{d_abs:+.2f} ({d_pct*100:+.2f}%)")

st.markdown("---")

# ---- Per instrument: 2 grafieken naast elkaar (standaard voor ALLE selectie)
st.subheader("Per instrument")
for pfx in (sel or prefixes[:1]):
    c = cols_for(pfx)
    name = label_of(pfx)
    st.markdown(f"### {name}")
    left, right = st.columns(2)

    # Links: Close + MA/EMA 20/50/200
    needed = [c["close"]]
    if not all(col in df.columns for col in needed):
        with left:
            st.info("Geen prijsdata voor dit instrument.")
    else:
        sub = df[["date", c["close"]]].dropna().copy()
        sub[c["close"]] = pd.to_numeric(sub[c["close"]], errors="coerce").astype(float)

        # Bepaal MA/EMA: voor SMA gebruiken we databasekolommen als ze bestaan; anders berekenen
        if avg_mode == "SMA":
            ma20 = df[c["ma20"]] if c["ma20"] in df.columns else compute_ma_ema(sub[c["close"]], "SMA", 20)
            ma50 = df[c["ma50"]] if c["ma50"] in df.columns else compute_ma_ema(sub[c["close"]], "SMA", 50)
            ma200 = df[c["ma200"]] if c["ma200"] in df.columns else compute_ma_ema(sub[c["close"]], "SMA", 200)
            # align op sub
            ma20 = pd.to_numeric(ma20, errors="coerce").astype(float).reindex(df.index).loc[sub.index]
            ma50 = pd.to_numeric(ma50, errors="coerce").astype(float).reindex(df.index).loc[sub.index]
            ma200 = pd.to_numeric(ma200, errors="coerce").astype(float).reindex(df.index).loc[sub.index]
        else:
            ma20 = compute_ma_ema(sub[c["close"]], "EMA", 20)
            ma50 = compute_ma_ema(sub[c["close"]], "EMA", 50)
            ma200 = compute_ma_ema(sub[c["close"]], "EMA", 200)

        with left:
            fig1 = make_subplots(specs=[[{"secondary_y": False}]])
            fig1.add_trace(go.Scatter(
                x=sub["date"], y=sub[c["close"]], name="Close",
                line=dict(width=2, color=COLOR_PRICE)
            ))
            if ma20.notna().any():
                fig1.add_trace(go.Scatter(
                    x=sub["date"], y=ma20.values, name=("EMA20" if avg_mode=="EMA" else "MA20"),
                    line=dict(width=2, color=COLOR_MA20)
                ))
            if ma50.notna().any():
                fig1.add_trace(go.Scatter(
                    x=sub["date"], y=ma50.values, name=("EMA50" if avg_mode=="EMA" else "MA50"),
                    line=dict(width=2, color=COLOR_MA50)
                ))
            if ma200.notna().any():
                fig1.add_trace(go.Scatter(
                    x=sub["date"], y=ma200.values, name=("EMA200" if avg_mode=="EMA" else "MA200"),
                    line=dict(width=2, color=COLOR_MA200)
                ))
            fig1.update_layout(
                height=420, margin=dict(l=10, r=10, t=40, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
            )
            st.plotly_chart(fig1, use_container_width=True)

    # Rechts: Δ% bars + YTD% (secundaire as)
    with right:
        if c["d_pct"] not in df.columns:
            st.info("Geen dagverandering beschikbaar voor dit instrument.")
        else:
            bars = df[["date", c["d_pct"]]].dropna().copy()
            bars[c["d_pct"]] = pd.to_numeric(bars[c["d_pct"]], errors="coerce").astype(float) * 100.0
            bar_colors = [COLOR_BAR_POS if v >= 0 else COLOR_BAR_NEG for v in bars[c["d_pct"]].values]

            fig2 = make_subplots(specs=[[{"secondary_y": True}]])
            fig2.add_trace(
                go.Bar(x=bars["date"], y=bars[c["d_pct"]], name="Δ% per dag",
                       marker=dict(color=bar_colors), opacity=0.9),
                secondary_y=False
            )
            try:
                fig2.add_hline(y=0, line_dash="dot", opacity=0.5)
            except Exception:
                pass
            fig2.update_yaxes(title_text="Δ% dag", secondary_y=False)

            if c["close"] in df.columns:
                subc = df[["date", c["close"]]].dropna().copy()
                subc[c["close"]] = pd.to_numeric(subc[c["close"]], errors="coerce").astype(float)
                ytd = ytd_percent_series(subc["date"], subc[c["close"]])
                # align op bar-x
                ytd = ytd.reindex(pd.to_datetime(bars["date"]))
                fig2.add_trace(
                    go.Scatter(x=bars["date"], y=ytd.values, name="YTD%",
                               line=dict(width=2, color=COLOR_YTD)),
                    secondary_y=True
                )
                fig2.update_yaxes(title_text="YTD%", secondary_y=True)

            fig2.update_layout(
                height=420, margin=dict(l=10, r=10, t=40, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
            )
            st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# ---- Optionele paarvergelijkingen (dubbele y-as)
PAIR_CANDIDATES = [("wti", "brent"), ("gold", "silver"), ("gasoline", "heatingoil")]
if show_pairs:
    st.subheader("Paarvergelijkingen (dubbele y-as)")
    for a1, a2 in PAIR_CANDIDATES:
        if a1 not in prefixes or a2 not in prefixes:
            continue
        c1 = cols_for(a1); c2 = cols_for(a2)
        lbl1, lbl2 = label_of(a1), label_of(a2)
        sub = df[["date", c1["close"], c2["close"]]].dropna(how="all").copy()
        if sub.empty:
            continue
        for col in [c1["close"], c2["close"]]:
            if col in sub.columns:
                sub[col] = pd.to_numeric(sub[col], errors="coerce").astype(float)

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        if c1["close"] in sub.columns:
            fig.add_trace(go.Scatter(
                x=sub["date"], y=sub[c1["close"]], name=f"{lbl1} Close",
                line=dict(width=2, color=COLOR_PRICE)
            ), secondary_y=False)
        if c2["close"] in sub.columns:
            fig.add_trace(go.Scatter(
                x=sub["date"], y=sub[c2["close"]], name=f"{lbl2} Close",
                line=dict(width=2, color=COLOR_MA200)
            ), secondary_y=True)

        fig.update_yaxes(title_text=lbl1, secondary_y=False)
        fig.update_yaxes(title_text=lbl2, secondary_y=True)
        fig.update_layout(
            height=420, margin=dict(l=10, r=10, t=40, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            title=f"{lbl1} vs {lbl2}"
        )
        st.plotly_chart(fig, use_container_width=True)
