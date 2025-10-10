# pages/2_Commodities.py
# Data bron (wide view): marketdata.commodity_prices_wide_v
# Verwachte kolommen per instrument: <pfx>_close, _delta_abs, _delta_pct, _ma20, _ma50, _ma200

import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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

    @st.cache_data(ttl=300, show_spinner=False)
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
st.set_page_config(page_title="🛢️ Commodities", layout="wide")

# ---------- Titel ----------
st.title("🛢️ Commodities Dashboard")

COM_WIDE_VIEW = st.secrets.get("tables", {}).get(
    "commodities_wide_view",
    "nth-pier-468314-p7.marketdata.commodity_prices_wide_v"
)

# ---------- Health check ----------
try:
    if not bq_ping():
        st.error("Geen BigQuery-verbinding. Controleer Secrets ([gcp_service_account]).")
        st.stop()
except Exception as e:
    st.error("Geen BigQuery-verbinding.")
    st.caption(f"Details: {e}")
    st.stop()

# ---------- Load data ----------
@st.cache_data(ttl=300, show_spinner=False)
def load_data() -> pd.DataFrame:
    df = run_query(f"SELECT * FROM `{COM_WIDE_VIEW}` ORDER BY date")
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"]).dt.date
    for c in df.columns:
        if c != "date":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

df_wide = load_data()
if df_wide.empty:
    st.warning("Geen data gevonden in commodity_prices_wide_v.")
    st.stop()

# ---------- Instrumenten & labels ----------
close_cols = [c for c in df_wide.columns if c.endswith("_close")]
prefixes_all = sorted([c[:-6] for c in close_cols])  # strip "_close"

# Relevantievolgorde (hoog → laag). Rest volgt alfabetisch.
priority = ["wti", "brent", "gold", "silver", "copper", "natgas", "heatingoil", "gasoline"]
ordered_all = [p for p in priority if p in prefixes_all] + [p for p in prefixes_all if p not in priority]

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
label_of = lambda p: LABELS.get(p, p.upper())

def cols_for(pfx: str) -> dict:
    return {
        "close": f"{pfx}_close",
        "d_abs": f"{pfx}_delta_abs",
        "d_pct": f"{pfx}_delta_pct",
        "ma20": f"{pfx}_ma20",
        "ma50": f"{pfx}_ma50",
        "ma200": f"{pfx}_ma200",
    }

def _f(s):  # to float
    return pd.to_numeric(s, errors="coerce").astype(float)

def compute_ma(series: pd.Series, kind: str, window: int) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").astype(float)
    if kind == "EMA":
        return s.ewm(span=window, adjust=False, min_periods=window).mean()
    return s.rolling(window=window, min_periods=window).mean()

def pct_as_str(x: float) -> str:
    if pd.isna(x):
        return "—"
    return f"{x:.2f}%"

def arrow_and_color(x: float) -> tuple[str, str]:
    if pd.isna(x):
        return ("", "default")
    if x > 0:
        return ("▲", "green")
    if x < 0:
        return ("▼", "red")
    return ("■", "default")

# ---------- Sidebar ----------
with st.sidebar:
    st.header("⚙️ Instellingen")

    min_d, max_d = df_wide["date"].min(), df_wide["date"].max()
    default_start = max(max_d - timedelta(days=365), min_d)

    date_range = st.slider(
        "📅 Periode",
        min_value=min_d, max_value=max_d,
        value=(default_start, max_d),
        step=timedelta(days=1),
        format="YYYY-MM-DD",
    )
    avg_mode   = st.radio("Gemiddelde", ["EMA", "SMA"], index=0, horizontal=True)
    show_delta = st.checkbox("Δ%-grafieken tonen", value=True)
    show_combos = st.checkbox("Combinatiegrafieken tonen (Energy & Metals)", value=True)
    collapse   = st.checkbox("Secties inklapbaar maken", value=False)

    selected = st.multiselect(
        "Instrumenten",
        options=ordered_all,
        default=ordered_all[:6] if len(ordered_all) >= 6 else ordered_all,
        format_func=label_of,
        help="Selecteer welke instrumenten je onderaan in detail wilt zien."
    )

# ---------- Filter ----------
start, end = date_range
mask = (df_wide["date"] >= start) & (df_wide["date"] <= end)
df = df_wide.loc[mask].sort_values("date").copy()

# ---------- KPI rij ----------
st.markdown("### KPI's")

def latest_row_for(pfx: str) -> tuple[float, float]:
    """Return (last close, last d_pct*100) if available, else (nan, nan)."""
    c = cols_for(pfx)
    if c["close"] not in df.columns:
        return (np.nan, np.nan)
    sub = df[[c["close"], c["d_pct"]]].dropna(how="all").tail(1)
    if sub.empty:
        return (np.nan, np.nan)
    last_close = float(sub[c["close"]].iloc[0]) if c["close"] in sub else np.nan
    last_pct = float(sub[c["d_pct"]].iloc[0] * 100.0) if c["d_pct"] in sub and not pd.isna(sub[c["d_pct"]].iloc[0]) else np.nan
    return (last_close, last_pct)

# Kies KPI's die bijna altijd beschikbaar zijn:
kpi_list = [p for p in ["wti", "brent", "gold", "silver", "natgas", "copper"] if p in prefixes_all]
cols = st.columns(len(kpi_list) + 1)  # +1 voor marktbreedte

for i, p in enumerate(kpi_list):
    close, d_pct = latest_row_for(p)
    arrow, color = arrow_and_color(d_pct)
    label = label_of(p)
    delta_str = f"{arrow} {pct_as_str(d_pct)}"
    # Streamlit metric: value (close), delta (%)
    cols[i].metric(label=label, value=("—" if pd.isna(close) else f"{close:,.2f}"), delta=delta_str)

# Marktbreedte (laatste dag in filter)
try:
    last_day = df["date"].max()
    df_last = df.loc[df["date"] == last_day]
    adv = 0
    dec = 0
    for p in prefixes_all:
        c = cols_for(p)
        if c["d_pct"] in df_last.columns:
            val = df_last[c["d_pct"]].dropna()
            if not val.empty:
                adv += int(val.iloc[0] > 0)
                dec += int(val.iloc[0] < 0)
    breadth = f"↑{adv} / ↓{dec}"
except Exception:
    breadth = "—"
cols[-1].metric(label="Marktbreedte (laatste dag)", value=breadth)

st.divider()

# ---------- Per instrument (detail) ----------
st.subheader("Per instrument")

# Kleuren (Okabe–Ito)
COLOR_PRICE   = "#111111"  # near-black
COLOR_MA20    = "#E69F00"  # orange
COLOR_MA50    = "#009E73"  # green
COLOR_MA200   = "#0072B2"  # blue
COLOR_BAR_POS = "#009E73"  # green bars
COLOR_BAR_NEG = "#D55E00"  # red bars
COLOR_GOLD    = "#E69F00"
COLOR_SILVER  = "#56B4E9"
COLOR_WTI     = "#111111"
COLOR_BRENT   = "#0072B2"
COLOR_GAS     = "#009E73"

def plot_price_and_ma(df_in: pd.DataFrame, pfx: str):
    c = cols_for(pfx)
    sub = df_in[["date", c["close"]]].dropna().copy()
    if sub.empty:
        return None
    sub[c["close"]] = _f(sub[c["close"]])
    ma20  = compute_ma(sub[c["close"]], avg_mode, 20)
    ma50  = compute_ma(sub[c["close"]], avg_mode, 50)
    ma200 = compute_ma(sub[c["close"]], avg_mode, 200)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sub["date"], y=sub[c["close"]], name="Close",
                             line=dict(width=2, color=COLOR_PRICE)))
    if ma20.notna().any():
        fig.add_trace(go.Scatter(x=sub["date"], y=ma20.values,
                                 name=("EMA20" if avg_mode == "EMA" else "MA20"),
                                 line=dict(width=2, color=COLOR_MA20)))
    if ma50.notna().any():
        fig.add_trace(go.Scatter(x=sub["date"], y=ma50.values,
                                 name=("EMA50" if avg_mode == "EMA" else "MA50"),
                                 line=dict(width=2, color=COLOR_MA50)))
    if ma200.notna().any():
        fig.add_trace(go.Scatter(x=sub["date"], y=ma200.values,
                                 name=("EMA200" if avg_mode == "EMA" else "MA200"),
                                 line=dict(width=2, color=COLOR_MA200)))
    fig.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    )
    return fig

def plot_delta_bars(df_in: pd.DataFrame, pfx: str):
    c = cols_for(pfx)
    # Gebruik _d_pct indien aanwezig, anders on-the-fly
    if c["d_pct"] in df_in.columns and df_in[c["d_pct"]].notna().any():
        bars = df_in[["date", c["d_pct"]]].dropna().copy()
        if bars.empty:
            return None
        bars[c["d_pct"]] = _f(bars[c["d_pct"]]) * 100.0
        series = bars[c["d_pct"]]
    else:
        tmp = df_in[["date", c["close"]]].dropna().copy()
        if tmp.empty:
            return None
        tmp[c["close"]] = _f(tmp[c["close"]])
        tmp["pct"] = tmp[c["close"]].pct_change() * 100.0
        bars = tmp[["date", "pct"]].dropna().rename(columns={"pct": c["d_pct"]})
        if bars.empty:
            return None
        series = bars[c["d_pct"]]

    colors = [COLOR_BAR_POS if v >= 0 else COLOR_BAR_NEG for v in series.values]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=bars["date"], y=series, name="Δ% per dag",
                         marker_color=colors, opacity=0.9))
    fig.add_hline(y=0, line_dash="dot", opacity=0.6)
    fig.update_yaxes(title_text="Δ% dag")
    fig.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    )
    return fig

# Render detail secties voor selectie
for pfx in [p for p in ordered_all if p in selected]:
    name = label_of(pfx)
    # Container keuze: expander indien collapse True
    header = f"## {name}"
    container = st.expander(header) if collapse else st.container()
    with container:
        if not collapse:
            st.markdown(header)

        # Price + MA
        fig_price = plot_price_and_ma(df, pfx)
        if fig_price:
            st.plotly_chart(fig_price, use_container_width=True)
        else:
            st.info(f"Geen prijsdata voor {name}.")
            st.markdown("---")
            continue

        # Δ%-bars
        if show_delta:
            fig_delta = plot_delta_bars(df, pfx)
            if fig_delta:
                st.plotly_chart(fig_delta, use_container_width=True)

        st.markdown("---")

# ---------- Combo-grafieken ----------
if show_combos:
    st.subheader("Combinatiegrafieken")

    # Energy: WTI & Brent (links) + Natural Gas (rechts)
    need_e = ["wti_close", "brent_close", "natgas_close"]
    if all(n in df.columns for n in need_e):
        e = df[["date"] + need_e].dropna(how="all").copy()
        if not e.empty:
            e["wti_close"]    = _f(e["wti_close"])
            e["brent_close"]  = _f(e["brent_close"])
            e["natgas_close"] = _f(e["natgas_close"])

            fig_eng = make_subplots(specs=[[{"secondary_y": True}]])
            fig_eng.add_trace(go.Scatter(x=e["date"], y=e["wti_close"], name="WTI (USD/bbl)",
                                         line=dict(width=2, color=COLOR_WTI)), secondary_y=False)
            fig_eng.add_trace(go.Scatter(x=e["date"], y=e["brent_close"], name="Brent (USD/bbl)",
                                         line=dict(width=2, color=COLOR_BRENT)), secondary_y=False)
            fig_eng.add_trace(go.Scatter(x=e["date"], y=e["natgas_close"], name="NatGas (USD/MMBtu)",
                                         line=dict(width=2, color=COLOR_GAS)), secondary_y=True)
            fig_eng.update_yaxes(title_text="Oil price (USD/bbl)", secondary_y=False)
            fig_eng.update_yaxes(title_text="Natural Gas (USD/MMBtu)", secondary_y=True)
            fig_eng.update_layout(height=460, margin=dict(l=10, r=10, t=40, b=10),
                                  title="WTI & Brent vs Natural Gas",
                                  legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
            st.plotly_chart(fig_eng, use_container_width=True)
    else:
        miss = [n for n in need_e if n not in df.columns]
        st.info(f"Energy-combo: ontbrekende kolommen: {', '.join(miss)}")

    # Metals: Gold (links) & Silver (rechts)
    need_m = ["gold_close", "silver_close"]
    if all(n in df.columns for n in need_m):
        m = df[["date"] + need_m].dropna(how="all").copy()
        if not m.empty:
            m["gold_close"]   = _f(m["gold_close"])
            m["silver_close"] = _f(m["silver_close"])

            fig_met = make_subplots(specs=[[{"secondary_y": True}]])
            fig_met.add_trace(go.Scatter(x=m["date"], y=m["gold_close"], name="Gold (USD/oz)",
                                         line=dict(width=2, color=COLOR_GOLD)), secondary_y=False)
            fig_met.add_trace(go.Scatter(x=m["date"], y=m["silver_close"], name="Silver (USD/oz)",
                                         line=dict(width=2, color=COLOR_SILVER)), secondary_y=True)
            fig_met.update_yaxes(title_text="Gold (USD/oz)", secondary_y=False)
            fig_met.update_yaxes(title_text="Silver (USD/oz)", secondary_y=True)
            fig_met.update_layout(height=460, margin=dict(l=10, r=10, t=40, b=10),
                                  title="Gold vs Silver",
                                  legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
            st.plotly_chart(fig_met, use_container_width=True)
    else:
        miss = [n for n in need_m if n not in df.columns]
        st.info(f"Metals-combo: ontbrekende kolommen: {', '.join(miss)}")

# ---------- Tabel ----------
st.subheader("Laatste rijen (gefilterd bereik)")
show_cols = ["date"]
for pfx in ordered_all:
    cc = cols_for(pfx)
    show_cols += [cc["close"], cc["d_abs"], cc["d_pct"], cc["ma20"], cc["ma50"], cc["ma200"]]
show_cols = [c for c in show_cols if c in df.columns]

# Kleine prettige formatting: Δ% als %
df_tail = df[show_cols].tail(200).copy()
for c in [c for c in df_tail.columns if c.endswith("_delta_pct")]:
    df_tail[c] = (df_tail[c] * 100).round(2)

st.dataframe(df_tail, use_container_width=True)
